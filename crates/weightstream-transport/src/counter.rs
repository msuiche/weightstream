//! NIC loss accounting from sysfs — snapshot before, diff after.
//!
//! Two sources per link interface: the netdev statistics
//! (`/sys/class/net/<iface>/statistics/*`) and the RoCE hardware counters
//! (`/sys/class/infiniband/<dev>/ports/<port>/hw_counters/*`). We diff
//! everything, then classify: drop/error-ish counters are failures,
//! ECN/CNP counters are congestion information (expected while production
//! vLLM traffic shares the cable).

use std::collections::BTreeMap;
use std::fs;

#[derive(Debug, Clone)]
pub struct Counters(pub BTreeMap<String, u64>);

/// Snapshot per (ibverbs device, port): the RoCE hw_counters of the port,
/// plus the statistics of the netdev(s) attached to the device (resolved
/// via /sys/class/infiniband/<dev>/device/net/ — netdev names do not
/// necessarily match the ibverbs device name).
pub fn snapshot(targets: &[(String, u8)]) -> Counters {
    let mut m = BTreeMap::new();
    for (ibdev, port) in targets {
        let dir = format!("/sys/class/infiniband/{ibdev}/ports/{port}/hw_counters");
        read_dir_into(&dir, &format!("hw:{ibdev}/p{port}:"), &mut m);
        let netdevs = format!("/sys/class/infiniband/{ibdev}/device/net");
        if let Ok(entries) = fs::read_dir(&netdevs) {
            for e in entries.flatten() {
                if let Ok(ifname) = e.file_name().into_string() {
                    let dir = format!("/sys/class/net/{ifname}/statistics");
                    read_dir_into(&dir, &format!("net:{ifname}:"), &mut m);
                }
            }
        }
    }
    Counters(m)
}

fn read_dir_into(dir: &str, prefix: &str, m: &mut BTreeMap<String, u64>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let Ok(name) = e.file_name().into_string() else {
            continue;
        };
        let Ok(s) = fs::read_to_string(e.path()) else {
            continue;
        };
        if let Ok(v) = s.trim().parse::<u64>() {
            m.insert(format!("{prefix}{name}"), v);
        }
    }
}

/// nonzero deltas, after - before
pub fn diff(before: &Counters, after: &Counters) -> BTreeMap<String, i64> {
    let mut out = BTreeMap::new();
    for (k, v) in &after.0 {
        let b = before.0.get(k).copied().unwrap_or(0);
        let d = *v as i64 - b as i64;
        if d != 0 {
            out.insert(k.clone(), d);
        }
    }
    out
}

/// Does this counter name signal packet loss (as opposed to volume or
/// congestion)?
pub fn is_loss(key: &str) -> bool {
    let k = key.to_lowercase();
    k.contains("dropped")
        || k.contains("error")
        || k.contains("missed")
        || k.contains("out_of_buffer")
        || k.contains("out_of_sequence")
        || k.contains("outofsequence")
        || k.contains("duplicate")
        || k.contains("rnr_nak")
        || k.contains("local_ack_timeout")
        || k.contains("excessive_retimer")
}

/// Congestion signalling (informational, expected under shared traffic).
pub fn is_congestion(key: &str) -> bool {
    let k = key.to_lowercase();
    k.contains("ecn") || k.contains("cnp")
}
