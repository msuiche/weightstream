//! The 4-rank test runner: `ferroseek-transport bench --ring nodes.json`.
//!
//! Any node launches it; ranks rendezvous over the management network and
//! the head (rank 0) drives the phases: linkup (QP bring-up), probe
//! (RDMA ping/pong per link per rail), gate (bit-exact all-reduce), soak
//! (100k x 4 KB all-reduce with loss accounting). Fail-stop throughout: a
//! 100 ms wait timeout on any collective aborts the whole run.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use crate::counter;
use crate::ctrl::{Head, Phase, Worker};
use crate::nodes::{Node, NodesFile};
use crate::ring::Ring;
use crate::stats::summarize;
use crate::verbs::Verbs;

pub struct BenchArgs {
    pub ring_path: String,
    pub port: u16,
    pub rank: Option<u8>,
    pub wait_ms: u64,
    pub ping_iters: usize,
    pub ping_warmup: usize,
    pub soak_iters: usize,
    pub gate_rounds: usize,
    pub verify_every: usize,
    pub elems: usize,
}

impl Default for BenchArgs {
    fn default() -> Self {
        BenchArgs {
            ring_path: String::new(),
            port: 29871,
            rank: None,
            wait_ms: 100,
            ping_iters: 2000,
            ping_warmup: 200,
            soak_iters: 100_000,
            gate_rounds: 8,
            verify_every: 1000,
            elems: 1024, // 4 KB fp32
        }
    }
}

pub fn usage() -> String {
    "usage: ferroseek-transport bench --ring <nodes-ring4.local.json> \
     [--port 29871] [--rank N] [--wait-ms 100] [--ping-iters 2000] \
     [--soak-iters 100000] [--gate-rounds 8] [--verify-every 1000]"
        .into()
}

fn num<T: std::str::FromStr>(args: &[String], i: usize, what: &str) -> Result<T, String> {
    args.get(i)
        .ok_or_else(|| format!("{what} needs a value"))?
        .parse()
        .map_err(|_| format!("{what}: invalid value {:?}", args[i]))
}

pub fn parse_args(args: &[String]) -> Result<BenchArgs, String> {
    let mut a = BenchArgs::default();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--ring" => {
                i += 1;
                a.ring_path = args.get(i).ok_or("--ring needs a path")?.clone();
            }
            "--port" => {
                i += 1;
                a.port = num(args, i, "--port")?;
            }
            "--rank" => {
                i += 1;
                a.rank = Some(num(args, i, "--rank")?);
            }
            "--wait-ms" => {
                i += 1;
                a.wait_ms = num(args, i, "--wait-ms")?;
            }
            "--ping-iters" => {
                i += 1;
                a.ping_iters = num(args, i, "--ping-iters")?;
            }
            "--soak-iters" => {
                i += 1;
                a.soak_iters = num(args, i, "--soak-iters")?;
            }
            "--gate-rounds" => {
                i += 1;
                a.gate_rounds = num(args, i, "--gate-rounds")?;
            }
            "--verify-every" => {
                i += 1;
                a.verify_every = num(args, i, "--verify-every")?;
            }
            _ => return Err(format!("unknown argument {:?}\n{}", args[i], usage())),
        }
        i += 1;
    }
    if a.ring_path.is_empty() {
        return Err(format!("--ring is required\n{}", usage()));
    }
    Ok(a)
}

fn hostname() -> String {
    let mut buf = [0u8; 256];
    unsafe {
        if libc::gethostname(buf.as_mut_ptr() as *mut libc::c_char, buf.len()) != 0 {
            return String::new();
        }
    }
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..end]).into_owned()
}

// ---------------------------------------------------------------- vectors

fn splitmix64(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Gate vector for (round, rank, index): different per rank, includes
/// denormals and inf/nan-free extremes (f32::MAX, -f32::MAX, smallest
/// subnormal, 2^93-scale), guaranteed never to overflow the 4-rank
/// rank-ordered sum.
pub fn gate_input(round: usize, rank: u8, i: usize) -> f32 {
    let h =
        splitmix64((round as u64) << 48 ^ (rank as u64) << 40 ^ (i as u64).wrapping_mul(0x9E37));
    let carrier = rank as usize == round % 4;
    if i == 0 {
        if carrier {
            f32::MAX
        } else {
            f32::from_bits((h & 0x7f) as u32 + 1)
        }
    } else if i == 1 {
        if carrier {
            -f32::MAX
        } else {
            f32::from_bits((h & 0x7f) as u32 + 1)
        }
    } else if i == 2 {
        if carrier {
            f32::from_bits(1) // smallest subnormal
        } else {
            0.0
        }
    } else {
        match h % 8 {
            0 => f32::from_bits((h % 0x0080_0000) as u32), // subnormal or zero
            1 => f32::from_bits(0x6E00_0000 | ((h >> 8) & 0x007f_ffff) as u32), // ~2^93
            2 => -f32::from_bits(0x6E00_0000 | ((h >> 8) & 0x007f_ffff) as u32),
            3 => f32::from_bits(0x3F80_0000 | ((h >> 8) & 0x007f_ffff) as u32), // [1,2)
            4 => -f32::from_bits(0x3F80_0000 | ((h >> 8) & 0x007f_ffff) as u32),
            _ => ((h % 1001) as f32) - 500.0,
        }
    }
}

/// Soak vector: cheap, deterministic, distinct per (iter, rank, index).
pub fn soak_input(iter: usize, rank: u8, i: usize) -> f32 {
    let v = (i as u64)
        .wrapping_mul(0x9E37_79B9)
        .wrapping_add((iter as u64) << 20)
        .wrapping_add((rank as u64) << 32);
    ((v % 977) as f32) / 977.0
}

fn fnv1a(state: &mut u64, bits: u32) {
    *state = (*state ^ bits as u64).wrapping_mul(0x0000_0100_0000_01B3);
}

fn expected_gate(nranks: usize, round: usize, i: usize) -> f32 {
    let mut acc = gate_input(round, 0, i);
    for r in 1..nranks {
        acc += gate_input(round, r as u8, i);
    }
    acc
}

fn expected_soak(nranks: usize, iter: usize, i: usize) -> f32 {
    let mut acc = soak_input(iter, 0, i);
    for r in 1..nranks {
        acc += soak_input(iter, r as u8, i);
    }
    acc
}

// ---------------------------------------------------------------- phases

fn run_gate(ring: &mut Ring, rounds: usize) -> Result<serde_json::Value, String> {
    let (elems, nranks, me) = (ring.elems, ring.nranks, ring.me);
    let mut input = vec![0f32; elems];
    let mut mismatches = 0usize;
    let mut hashes = Vec::new();
    let mut first_bad: Option<String> = None;
    for round in 0..rounds {
        for (i, v) in input.iter_mut().enumerate() {
            *v = gate_input(round, me, i);
        }
        let out = ring.allreduce(&input)?;
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for i in 0..elems {
            let want = expected_gate(nranks, round, i);
            if out[i].to_bits() != want.to_bits() {
                mismatches += 1;
                if first_bad.is_none() {
                    first_bad = Some(format!(
                        "round {round} elem {i}: got {:e} ({:#010x}), want {:e} ({:#010x})",
                        out[i],
                        out[i].to_bits(),
                        want,
                        want.to_bits()
                    ));
                }
            }
            fnv1a(&mut h, out[i].to_bits());
        }
        // all-gather: every rank must see the rank-ordered concatenation of
        // the same contributions, bit-exact
        let gathered = ring.allgather(&input)?;
        if gathered.len() != nranks * elems {
            mismatches += 1;
            first_bad.get_or_insert_with(|| "allgather length".to_string());
        }
        for r in 0..nranks {
            for i in 0..elems {
                let want = gate_input(round, r as u8, i);
                if gathered[r * elems + i].to_bits() != want.to_bits() {
                    mismatches += 1;
                    if first_bad.is_none() {
                        first_bad = Some(format!(
                            "round {round} gather rank {r} elem {i}: got {:#010x}, want {:#010x}",
                            gathered[r * elems + i].to_bits(),
                            want.to_bits()
                        ));
                    }
                }
                fnv1a(&mut h, gathered[r * elems + i].to_bits());
            }
        }
        hashes.push(format!("{h:016x}"));
    }
    Ok(serde_json::json!({
        "pass": mismatches == 0,
        "mismatches": mismatches,
        "first_bad": first_bad,
        "hashes": hashes,
    }))
}

fn run_soak(
    ring: &mut Ring,
    iters: usize,
    verify_every: usize,
) -> Result<serde_json::Value, String> {
    let (elems, nranks, me) = (ring.elems, ring.nranks, ring.me);
    let targets: Vec<(String, u8)> = ring
        .links
        .values()
        .map(|l| (l.ibdev.clone(), l.port))
        .collect();
    let before = counter::snapshot(&targets);
    let mut input = vec![0f32; elems];
    let mut lat = Vec::with_capacity(iters);
    let mut mismatches = 0usize;
    for it in 0..iters {
        for (i, v) in input.iter_mut().enumerate() {
            *v = soak_input(it, me, i);
        }
        let t0 = Instant::now();
        let out = ring.allreduce(&input)?;
        lat.push(t0.elapsed().as_secs_f64() * 1e6);
        if it % verify_every == 0 {
            for i in 0..elems {
                let want = expected_soak(nranks, it, i);
                if out[i].to_bits() != want.to_bits() {
                    mismatches += 1;
                }
            }
        }
    }
    let after = counter::snapshot(&targets);
    let d = counter::diff(&before, &after);
    let mut loss = serde_json::Map::new();
    let mut congestion = serde_json::Map::new();
    let mut other = serde_json::Map::new();
    for (k, v) in d {
        if counter::is_loss(&k) {
            loss.insert(k, serde_json::json!(v));
        } else if counter::is_congestion(&k) {
            congestion.insert(k, serde_json::json!(v));
        } else {
            other.insert(k, serde_json::json!(v));
        }
    }
    Ok(serde_json::json!({
        "latency": summarize(lat),
        "mismatches": mismatches,
        "loss": loss,
        "congestion": congestion,
        "other_counters": other,
    }))
}

// ---------------------------------------------------------------- driver

pub fn run(args: &BenchArgs) -> Result<(), String> {
    let json = std::fs::read_to_string(&args.ring_path)
        .map_err(|e| format!("read {}: {e}", args.ring_path))?;
    let map = NodesFile::parse(&json).map_err(|e| e.to_string())?;
    let me = map
        .my_rank(&hostname(), args.rank)
        .map_err(|e| e.to_string())?;
    let my_node = map.node_of(me).map_err(|e| e.to_string())?.clone();
    let cfg = map.ring_cfg(me).map_err(|e| e.to_string())?;
    eprintln!(
        "ferroseek-transport: rank {me} ({}) head={} port {} elems {} wait {}ms",
        my_node.name, my_node.head, args.port, args.elems, args.wait_ms
    );

    let verbs = Verbs::open().map_err(|e| e.to_string())?;
    let (mut ring, my_ep) = Ring::new(verbs, &cfg, args.elems, Duration::from_millis(args.wait_ms))
        .map_err(|e| e.to_string())?;
    eprintln!(
        "rank {me}: {} QPs (succ {} on {} / {}, pred {} on {} / {})",
        ring.qp_count(),
        cfg.succ.rank,
        cfg.succ.rails[0].subnet,
        cfg.succ.rails[1].subnet,
        cfg.pred.rank,
        cfg.pred.rails[0].subnet,
        cfg.pred.rails[1].subnet
    );

    if my_node.head {
        run_head(args, me, &my_node, &cfg, &mut ring, my_ep)
    } else {
        run_worker(args, me, &cfg, &mut ring, my_ep)
    }
}

fn run_head(
    args: &BenchArgs,
    me: u8,
    my_node: &Node,
    cfg: &crate::nodes::RingCfg,
    ring: &mut Ring,
    my_ep: serde_json::Value,
) -> Result<(), String> {
    let _ = cfg;
    let mut head = Head::bind(&my_node.management_ip, args.port, me).map_err(|e| e.to_string())?;
    head.wait_workers(4, Duration::from_secs(180))
        .map_err(|e| e.to_string())?;
    eprintln!("head {me}: all 4 ranks rendezvoused over the management net");

    let phase_secs = 60;

    // ---- linkup: exchange QP endpoints
    head.send_go(Phase::Linkup, serde_json::json!({}))
        .map_err(|e| e.to_string())?;
    let mut all = vec![(me, my_ep)];
    for data in head
        .collect_ready(Duration::from_secs(phase_secs))
        .map_err(|e| e.to_string())?
    {
        let rank = data["rank"]
            .as_u64()
            .ok_or("endpoint payload without rank")? as u8;
        all.push((rank, data));
    }
    all.sort_by_key(|(r, _)| *r);
    if all.len() != 4 {
        let msg = format!("only {} ranks in endpoint exchange", all.len());
        head.abort_all(&msg).ok();
        return Err(msg);
    }
    let ring_map: Vec<serde_json::Value> = all.iter().map(|(_, e)| e.clone()).collect();

    // ---- RTR then RTS (barrier between the two via the head)
    head.send_go(Phase::Rtr, serde_json::json!({ "ring": ring_map }))
        .map_err(|e| e.to_string())?;
    ring.to_rtr(&serde_json::Value::Array(ring_map.clone()))?;
    head.collect_ready(Duration::from_secs(phase_secs))
        .map_err(|e| e.to_string())?;
    head.send_go(Phase::Rts, serde_json::json!({}))
        .map_err(|e| e.to_string())?;
    ring.to_rts()?;
    head.collect_ready(Duration::from_secs(phase_secs))
        .map_err(|e| e.to_string())?;
    println!("QP bring-up: 4 links x 2 rails, RoCE v2 RC, path MTU 4096 — OK");

    // ---- probe: ping/pong per link per rail
    println!(
        "\nRoCE link probe (RDMA ping/pong, 8-byte write-with-imm, {} iters):",
        args.ping_iters
    );
    println!(
        "{:<8} {:>4} {:>12} {:>12} {:>10} {:>12}",
        "link", "rail", "rtt p50 us", "rtt p99 us", "min us", "1-way p50 us"
    );
    for a in 0..4u8 {
        let b = (a + 1) % 4;
        for rail in 0..2u64 {
            let cmd = serde_json::json!({
                "a": a, "b": b, "rail": rail,
                "iters": args.ping_iters, "warmup": args.ping_warmup,
            });
            head.send_go(Phase::Probe, cmd).map_err(|e| e.to_string())?;
            let mine = if me == a {
                Some(ring.ping(b, rail as usize, true, args.ping_iters, args.ping_warmup)?)
            } else if me == b {
                ring.ping(a, rail as usize, false, args.ping_iters, args.ping_warmup)?;
                None
            } else {
                None
            };
            let datas = head
                .collect_ready(Duration::from_secs(120))
                .map_err(|e| e.to_string())?;
            let s = if let Some(rtts) = mine {
                summarize(rtts)
            } else {
                // the initiator (rank a) is a worker; its payload has "ping"
                let d = datas
                    .iter()
                    .find(|d| d["ping"].is_object())
                    .ok_or("no ping summary from initiator")?;
                let p = &d["ping"];
                crate::stats::Summary {
                    n: p["n"].as_u64().unwrap_or(0) as usize,
                    min_us: p["min_us"].as_f64().unwrap_or(f64::NAN),
                    p50_us: p["p50_us"].as_f64().unwrap_or(f64::NAN),
                    p99_us: p["p99_us"].as_f64().unwrap_or(f64::NAN),
                    max_us: p["max_us"].as_f64().unwrap_or(f64::NAN),
                    mean_us: p["mean_us"].as_f64().unwrap_or(f64::NAN),
                }
            };
            if s.n == 0 {
                let msg = format!("link {a}-{b} rail {rail}: no ping samples (fail-stop)");
                head.abort_all(&msg).ok();
                return Err(msg);
            }
            println!(
                "{:<8} {:>4} {:>12.2} {:>12.2} {:>10.2} {:>12.2}",
                format!("{a}-{b}"),
                rail,
                s.p50_us,
                s.p99_us,
                s.min_us,
                s.p50_us / 2.0
            );
        }
    }

    // ---- gate: bit-exact all-reduce
    println!(
        "\nBit-exactness gate ({} rounds, 4 KB, denormals + f32::MAX extremes):",
        args.gate_rounds
    );
    head.send_go(
        Phase::Gate,
        serde_json::json!({ "rounds": args.gate_rounds }),
    )
    .map_err(|e| e.to_string())?;
    let my_gate = run_gate(ring, args.gate_rounds)?;
    let datas = head
        .collect_ready(Duration::from_secs(120))
        .map_err(|e| e.to_string())?;
    let mut all_data = vec![(me, my_gate)];
    for d in datas {
        let rank = d["rank"].as_u64().unwrap_or(255) as u8;
        all_data.push((rank, d["gate"].clone()));
    }
    all_data.sort_by_key(|(r, _)| *r);
    let mut gate_ok = true;
    for (rank, g) in &all_data {
        let pass = g["pass"].as_bool().unwrap_or(false);
        println!(
            "  rank {rank}: {} ({} mismatches) {}",
            if pass { "PASS" } else { "FAIL" },
            g["mismatches"].as_u64().unwrap_or(0),
            g["first_bad"].as_str().unwrap_or("")
        );
        gate_ok &= pass;
    }
    for round in 0..args.gate_rounds {
        let mut hash: Option<&str> = None;
        for (_, g) in &all_data {
            let h = g["hashes"][round].as_str().unwrap_or("-");
            match hash {
                None => hash = Some(h),
                Some(prev) if prev != h => {
                    gate_ok = false;
                    println!("  round {round}: rank results differ ({prev} vs {h})");
                }
                _ => {}
            }
        }
    }
    if !gate_ok {
        let msg = "bit-exactness gate FAILED — refusing to soak".to_string();
        head.abort_all(&msg).ok();
        return Err(msg);
    }
    println!("  all 4 ranks bit-identical and equal to the host-computed rank-ordered sums — PASS");

    // ---- soak
    println!(
        "\nSoak: {} iterations of a 4 KB ({} x fp32) all-reduce, verify every {}, fail-stop {} ms:",
        args.soak_iters, args.elems, args.verify_every, args.wait_ms
    );
    head.send_go(
        Phase::Soak,
        serde_json::json!({ "iters": args.soak_iters, "verify_every": args.verify_every }),
    )
    .map_err(|e| e.to_string())?;
    let t0 = Instant::now();
    let my_soak = run_soak(ring, args.soak_iters, args.verify_every)?;
    let datas = head
        .collect_ready(Duration::from_secs(300))
        .map_err(|e| e.to_string())?;
    println!("  wall time {:.1} s", t0.elapsed().as_secs_f64());

    let mut per_rank: BTreeMap<u8, serde_json::Value> = BTreeMap::new();
    per_rank.insert(me, my_soak["latency"].clone());
    for d in &datas {
        let rank = d["rank"].as_u64().unwrap_or(255) as u8;
        per_rank.insert(rank, d["soak"]["latency"].clone());
    }
    println!("\nAll-reduce latency per rank (us, includes the rank-ordered sum):");
    println!(
        "{:>6} {:>10} {:>12} {:>12} {:>10}",
        "rank", "n", "p50", "p99", "max"
    );
    let mut worst_p99 = 0.0f64;
    for (rank, p) in &per_rank {
        let p50 = p["p50_us"].as_f64().unwrap_or(f64::NAN);
        let p99 = p["p99_us"].as_f64().unwrap_or(f64::NAN);
        let mx = p["max_us"].as_f64().unwrap_or(f64::NAN);
        worst_p99 = worst_p99.max(p99);
        println!(
            "{:>6} {:>10} {:>12.2} {:>12.2} {:>10.2}",
            rank,
            p["n"].as_u64().unwrap_or(0),
            p50,
            p99,
            mx
        );
    }
    println!("  worst-rank p99: {worst_p99:.2} us");

    // loss accounting across all ranks
    let mut all_loss: BTreeMap<String, i64> = BTreeMap::new();
    let mut all_cong: BTreeMap<String, i64> = BTreeMap::new();
    let mut mismatches = 0u64;
    for (_me2, s) in [(me, &my_soak)] {
        mismatches += s["mismatches"].as_u64().unwrap_or(0);
        for (k, v) in s["loss"].as_object().unwrap_or(&serde_json::Map::new()) {
            *all_loss.entry(k.clone()).or_insert(0) += v.as_i64().unwrap_or(0);
        }
        for (k, v) in s["congestion"]
            .as_object()
            .unwrap_or(&serde_json::Map::new())
        {
            *all_cong.entry(k.clone()).or_insert(0) += v.as_i64().unwrap_or(0);
        }
    }
    for d in &datas {
        mismatches += d["soak"]["mismatches"].as_u64().unwrap_or(0);
        for (k, v) in d["soak"]["loss"]
            .as_object()
            .unwrap_or(&serde_json::Map::new())
        {
            *all_loss.entry(k.clone()).or_insert(0) += v.as_i64().unwrap_or(0);
        }
        for (k, v) in d["soak"]["congestion"]
            .as_object()
            .unwrap_or(&serde_json::Map::new())
        {
            *all_cong.entry(k.clone()).or_insert(0) += v.as_i64().unwrap_or(0);
        }
    }
    println!("\nPacket-loss accounting (direct links, before/after soak):");
    if all_loss.is_empty() {
        println!("  loss counters: all zero — zero packet loss");
    } else {
        for (k, v) in &all_loss {
            println!("  LOSS {k}: {v}");
        }
    }
    for (k, v) in &all_cong {
        println!("  congestion {k}: {v}");
    }
    println!("  bit-exact verify mismatches: {mismatches}");

    head.send_bye();

    let ok = all_loss.is_empty() && mismatches == 0;
    if ok {
        println!("\nRESULT: PASS (gate + soak, zero loss)");
        Ok(())
    } else {
        Err("RESULT: FAIL (loss or mismatches)".into())
    }
}

fn run_worker(
    args: &BenchArgs,
    me: u8,
    cfg: &crate::nodes::RingCfg,
    ring: &mut Ring,
    my_ep: serde_json::Value,
) -> Result<(), String> {
    let _ = args;
    let mut worker = Worker::connect(&cfg.head_ip, args.port, me, Duration::from_secs(180))
        .map_err(|e| e.to_string())?;
    loop {
        let Some((phase, cmd)) = worker.wait_go().map_err(|e| e.to_string())? else {
            return Ok(());
        };
        let result = match phase {
            Phase::Linkup => my_ep.clone(),
            Phase::Rtr => {
                ring.to_rtr(&cmd["ring"])?;
                serde_json::json!({ "rank": me })
            }
            Phase::Rts => {
                ring.to_rts()?;
                serde_json::json!({ "rank": me })
            }
            Phase::Probe => {
                let a = cmd["a"].as_u64().unwrap() as u8;
                let b = cmd["b"].as_u64().unwrap() as u8;
                let rail = cmd["rail"].as_u64().unwrap() as usize;
                let iters = cmd["iters"].as_u64().unwrap() as usize;
                let warmup = cmd["warmup"].as_u64().unwrap() as usize;
                let mut out = serde_json::json!({ "rank": me });
                if me == a {
                    let rtts = ring.ping(b, rail, true, iters, warmup)?;
                    out["ping"] = serde_json::to_value(summarize(rtts)).unwrap();
                } else if me == b {
                    ring.ping(a, rail, false, iters, warmup)?;
                }
                out
            }
            Phase::Gate => {
                let rounds = cmd["rounds"].as_u64().unwrap_or(8) as usize;
                let g = run_gate(ring, rounds)?;
                serde_json::json!({ "rank": me, "gate": g })
            }
            Phase::Soak => {
                let iters = cmd["iters"].as_u64().unwrap() as usize;
                let ve = cmd["verify_every"].as_u64().unwrap_or(1000) as usize;
                let s = run_soak(ring, iters, ve)?;
                serde_json::json!({ "rank": me, "soak": s })
            }
            // engine-only phases; the bench never issues them
            Phase::Start | Phase::Next => serde_json::json!({ "rank": me }),
        };
        worker.send_ready(result).map_err(|e| e.to_string())?;
    }
}
