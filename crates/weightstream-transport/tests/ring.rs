//! Tests that need real ibverbs hardware. Gated on the `WEIGHTSTREAM_RING`
//! environment variable (off-cluster `cargo test` skips them) and compiled
//! only on Linux.
//!
//! - `WEIGHTSTREAM_RING=self`: single-node QP loopback smoke test (two RC QPs
//!   connected to each other, one RDMA write-with-imm) — validates the FFI
//!   end to end on any one node.
//! - `WEIGHTSTREAM_RING=run` plus `WEIGHTSTREAM_RING_NODES=<json>`: the full
//!   4-node bench, all four nodes launched with the same invocation.

#[cfg(target_os = "linux")]
mod linux_tests {
    use weightstream_transport::verbs::{PollBatch, PollKind, Verbs};

    #[test]
    fn qp_loopback_smoke() {
        if std::env::var("WEIGHTSTREAM_RING").ok().as_deref() != Some("self") {
            eprintln!("skipping (off-ring; set WEIGHTSTREAM_RING=self to run)");
            return;
        }
        let verbs = Verbs::open().expect("open devices");
        let d = 0;
        let pa = verbs.probe_first_port(d).expect("first active RoCE port");
        let mut buf = vec![0u8; 4096];
        let qp1 = verbs.create_qp(&pa).expect("qp1");
        let qp2 = verbs.create_qp(&pa).expect("qp2");
        let mr = verbs.register(d, &mut buf).expect("register MR");
        qp1.to_rtr(qp2.qpn, pa.gid).expect("qp1 RTR");
        qp2.to_rtr(qp1.qpn, pa.gid).expect("qp2 RTR");
        qp1.to_rts().expect("qp1 RTS");
        qp2.to_rts().expect("qp2 RTS");
        qp1.post_recv(4).expect("qp1 recv");
        qp2.post_recv(4).expect("qp2 recv");

        // qp1 writes 8 bytes of buf into qp2's view of the same buffer,
        // with imm 7
        qp1.write_imm(
            buf.as_ptr() as u64,
            8,
            mr.lkey,
            buf.as_ptr() as u64,
            mr.rkey,
            7,
        )
        .expect("write");

        let mut batch = PollBatch::default();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        let mut seen = false;
        while !seen {
            for ev in batch.poll(verbs.devices[d].cq_ptr()).expect("poll") {
                if let PollKind::Recv { imm } = ev.kind {
                    assert_eq!(imm, 7);
                    assert_eq!(ev.qp_num, qp2.qpn);
                    seen = true;
                }
            }
            assert!(std::time::Instant::now() < deadline, "recv timeout");
        }
        println!("qp loopback: RDMA write-with-imm OK");
    }

    #[test]
    fn full_ring_bench() {
        if std::env::var("WEIGHTSTREAM_RING").ok().as_deref() != Some("run") {
            eprintln!("skipping (off-ring; launch all 4 nodes with WEIGHTSTREAM_RING=run)");
            return;
        }
        let args = weightstream_transport::bench::BenchArgs {
            ring_path: std::env::var("WEIGHTSTREAM_RING_NODES")
                .expect("WEIGHTSTREAM_RING_NODES must point at nodes-ring4.local.json"),
            ..Default::default()
        };
        weightstream_transport::bench::run(&args).expect("ring bench failed");
    }
}

#[cfg(not(target_os = "linux"))]
mod tests {
    #[test]
    fn ring_tests_are_linux_only() {
        // off-ring / off-platform: nothing to do; keeps `cargo test` green
        // on the Mac.
    }
}
