//! Node-map parsing for the 4-node DGX Spark ring.
//!
//! The on-disk format is the recipe's `nodes-ring4.local.json` (schema
//! version 1): per node, the management IP, the RoCE GID index, the two RoCE
//! interfaces to use toward each ring neighbor (one per rail), and the /24
//! each interface lives in. Rank order follows the ring.

use serde::Deserialize;
use std::collections::BTreeMap;
use std::fmt;

#[derive(Debug, Deserialize, Clone)]
pub struct NodesFile {
    pub schema_version: u32,
    #[serde(default)]
    pub ssh_user: String,
    pub nodes: Vec<Node>,
}

/// One entry of the nodes JSON.
#[derive(Debug, Deserialize, Clone)]
pub struct Node {
    pub name: String,
    pub rank: u8,
    pub management_ip: String,
    pub head: bool,
    pub roce_gid_index: u8,
    /// JSON keys are rank strings: peer rank -> [rail0 iface, rail1 iface]
    /// on THIS node for the link toward that peer.
    pub roce_peer_hcas: BTreeMap<String, Vec<String>>,
    /// iface name -> "a.b.c.d/len"
    pub roce_subnets: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IfacePair {
    /// my interface name for this rail
    pub local: String,
    /// peer's interface name for this rail
    pub remote: String,
    /// the /24 both ends share
    pub subnet: String,
}

/// The link toward one ring neighbor: two rails, each a direct ConnectX-7
/// cable on its own /24.
#[derive(Debug, Clone)]
pub struct NeighborLink {
    pub rank: u8,
    pub rails: [IfacePair; 2],
}

/// Everything one rank needs to know about the ring.
#[derive(Debug, Clone)]
pub struct RingCfg {
    pub nranks: usize,
    pub me: u8,
    pub gid_index: u8,
    pub head_ip: String,
    pub succ: NeighborLink,
    pub pred: NeighborLink,
}

#[derive(Debug)]
pub enum NodeMapError {
    Parse(String),
    BadMap(String),
}

impl fmt::Display for NodeMapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NodeMapError::Parse(s) => write!(f, "node-map parse: {s}"),
            NodeMapError::BadMap(s) => write!(f, "node-map: {s}"),
        }
    }
}
impl std::error::Error for NodeMapError {}

impl NodesFile {
    pub fn parse(json: &str) -> Result<NodesFile, NodeMapError> {
        let f: NodesFile =
            serde_json::from_str(json).map_err(|e| NodeMapError::Parse(e.to_string()))?;
        if f.schema_version != 1 {
            return Err(NodeMapError::BadMap(format!(
                "unsupported schema_version {}",
                f.schema_version
            )));
        }
        Ok(f)
    }

    pub fn node_of(&self, rank: u8) -> Result<&Node, NodeMapError> {
        self.nodes
            .iter()
            .find(|n| n.rank == rank)
            .ok_or_else(|| NodeMapError::BadMap(format!("no node with rank {rank}")))
    }

    fn head(&self) -> Result<&Node, NodeMapError> {
        let heads: Vec<&Node> = self.nodes.iter().filter(|n| n.head).collect();
        match heads.len() {
            1 => Ok(heads[0]),
            _ => Err(NodeMapError::BadMap(format!(
                "expected exactly 1 head node, found {}",
                heads.len()
            ))),
        }
    }

    /// Rank of this machine: `override_rank` if given, else matched by
    /// hostname (short name, as in the JSON `name` field).
    pub fn my_rank(&self, hostname: &str, override_rank: Option<u8>) -> Result<u8, NodeMapError> {
        if let Some(r) = override_rank {
            self.node_of(r)?;
            return Ok(r);
        }
        let short = hostname.split('.').next().unwrap_or(hostname);
        self.nodes
            .iter()
            .find(|n| n.name == short)
            .map(|n| n.rank)
            .ok_or_else(|| NodeMapError::BadMap(format!("hostname {short:?} not in node map")))
    }

    /// Build my ring view: successor and predecessor links with both rails
    /// resolved to (my iface, peer iface, shared subnet).
    pub fn ring_cfg(&self, me: u8) -> Result<RingCfg, NodeMapError> {
        let nranks = self.nodes.len();
        if nranks < 2 {
            return Err(NodeMapError::BadMap("need >= 2 nodes".into()));
        }
        let ranks: Vec<u8> = {
            let mut r: Vec<u8> = self.nodes.iter().map(|n| n.rank).collect();
            r.sort();
            if r.iter().enumerate().any(|(i, &v)| v != i as u8) {
                return Err(NodeMapError::BadMap("ranks must be 0..n-1".into()));
            }
            r
        };
        if ranks.len() != nranks {
            return Err(NodeMapError::BadMap("duplicate ranks".into()));
        }
        let succ = (me as usize + 1) % nranks;
        let pred = (me as usize + nranks - 1) % nranks;
        let my_node = self.node_of(me)?;
        let gid_index = my_node.roce_gid_index;

        let link_to = |peer: u8| -> Result<NeighborLink, NodeMapError> {
            let peer_node = self.node_of(peer)?;
            let mine = my_node
                .roce_peer_hcas
                .get(&peer.to_string())
                .ok_or_else(|| {
                    NodeMapError::BadMap(format!(
                        "rank {me} has no roce_peer_hcas entry for peer {peer}"
                    ))
                })?;
            let theirs = peer_node
                .roce_peer_hcas
                .get(&me.to_string())
                .ok_or_else(|| {
                    NodeMapError::BadMap(format!(
                        "rank {peer} has no roce_peer_hcas entry for peer {me}"
                    ))
                })?;
            if mine.len() != 2 || theirs.len() != 2 {
                return Err(NodeMapError::BadMap(format!(
                    "link {me}-{peer}: expected 2 rails per side, got {}/{}",
                    mine.len(),
                    theirs.len()
                )));
            }
            let mut rails = Vec::with_capacity(2);
            for my_if in mine {
                let subnet = my_node
                    .roce_subnets
                    .get(my_if)
                    .ok_or_else(|| NodeMapError::BadMap(format!("no subnet for iface {my_if}")))?;
                // the peer interface on the same /24 (names may differ)
                let peer_if = theirs
                    .iter()
                    .find(|c| peer_node.roce_subnets.get(*c) == Some(subnet))
                    .ok_or_else(|| {
                        NodeMapError::BadMap(format!(
                            "no interface on rank {peer} shares subnet {subnet} for {my_if}"
                        ))
                    })?
                    .clone();
                rails.push(IfacePair {
                    local: my_if.clone(),
                    remote: peer_if,
                    subnet: subnet.clone(),
                });
            }
            if rails[0].subnet == rails[1].subnet {
                return Err(NodeMapError::BadMap(format!(
                    "link {me}-{peer}: both rails on one subnet {}",
                    rails[0].subnet
                )));
            }
            Ok(NeighborLink {
                rank: peer,
                rails: [rails[0].clone(), rails[1].clone()],
            })
        };

        Ok(RingCfg {
            nranks,
            me,
            gid_index,
            head_ip: self.head()?.management_ip.clone(),
            succ: link_to(succ as u8)?,
            pred: link_to(pred as u8)?,
        })
    }
}

/// Is `ip` inside `cidr` ("a.b.c.d/len")?
pub fn in_subnet(ip: &str, cidr: &str) -> bool {
    let parse = |s: &str| -> Option<[u8; 4]> {
        let mut out = [0u8; 4];
        for (i, part) in s.split('.').enumerate() {
            out[i] = part.parse().ok()?;
        }
        Some(out)
    };
    let (base, len) = match cidr.split_once('/') {
        Some(x) => x,
        None => return false,
    };
    let (Some(a), Some(b)) = (parse(ip), parse(base)) else {
        return false;
    };
    let len: u32 = match len.parse() {
        Ok(l) if l <= 32 => l,
        _ => return false,
    };
    if len == 0 {
        return true;
    }
    let shift = 32 - len;
    u32::from_be_bytes(a) >> shift == u32::from_be_bytes(b) >> shift
}

#[cfg(test)]
mod tests {
    use super::*;

    // The real nodes-ring4.local.json (spark-4687:~/projects/spark-ds41f/config/).
    const REAL: &str = r#"{
  "schema_version": 1,
  "ssh_user": "msuiche",
  "nodes": [
    {
      "name": "spark-4687",
      "rank": 0,
      "management_ip": "192.168.1.232",
      "head": true,
      "roce_gid_index": 3,
      "roce_peer_hcas": {
        "1": ["rocep1s0f1", "roceP2p1s0f1"],
        "3": ["rocep1s0f0", "roceP2p1s0f0"]
      },
      "roce_subnets": {
        "rocep1s0f0": "10.100.227.0/24",
        "roceP2p1s0f0": "10.101.227.0/24",
        "rocep1s0f1": "10.100.224.0/24",
        "roceP2p1s0f1": "10.101.224.0/24"
      }
    },
    {
      "name": "spark-5bc3",
      "rank": 1,
      "management_ip": "192.168.1.171",
      "head": false,
      "roce_gid_index": 3,
      "roce_peer_hcas": {
        "0": ["rocep1s0f1", "roceP2p1s0f1"],
        "2": ["rocep1s0f0", "roceP2p1s0f0"]
      },
      "roce_subnets": {
        "rocep1s0f0": "10.100.225.0/24",
        "roceP2p1s0f0": "10.101.225.0/24",
        "rocep1s0f1": "10.100.224.0/24",
        "roceP2p1s0f1": "10.101.224.0/24"
      }
    },
    {
      "name": "spark-7425",
      "rank": 2,
      "management_ip": "192.168.1.180",
      "head": false,
      "roce_gid_index": 3,
      "roce_peer_hcas": {
        "1": ["rocep1s0f0", "roceP2p1s0f0"],
        "3": ["rocep1s0f1", "roceP2p1s0f1"]
      },
      "roce_subnets": {
        "rocep1s0f0": "10.100.225.0/24",
        "roceP2p1s0f0": "10.101.225.0/24",
        "rocep1s0f1": "10.100.226.0/24",
        "roceP2p1s0f1": "10.101.226.0/24"
      }
    },
    {
      "name": "spark-7275",
      "rank": 3,
      "management_ip": "192.168.1.206",
      "head": false,
      "roce_gid_index": 3,
      "roce_peer_hcas": {
        "0": ["rocep1s0f0", "roceP2p1s0f0"],
        "2": ["rocep1s0f1", "roceP2p1s0f1"]
      },
      "roce_subnets": {
        "rocep1s0f0": "10.100.227.0/24",
        "roceP2p1s0f0": "10.101.227.0/24",
        "rocep1s0f1": "10.100.226.0/24",
        "roceP2p1s0f1": "10.101.226.0/24"
      }
    }
  ]
}"#;

    fn parsed() -> NodesFile {
        NodesFile::parse(REAL).unwrap()
    }

    #[test]
    fn parses_real_map() {
        let f = parsed();
        assert_eq!(f.nodes.len(), 4);
        assert_eq!(f.nodes[0].roce_gid_index, 3);
        assert_eq!(f.ssh_user, "msuiche");
    }

    #[test]
    fn rank_by_hostname() {
        let f = parsed();
        assert_eq!(f.my_rank("spark-7425", None).unwrap(), 2);
        assert_eq!(f.my_rank("spark-5bc3.example", None).unwrap(), 1);
        assert!(f.my_rank("other-host", None).is_err());
        assert_eq!(f.my_rank("whatever", Some(3)).unwrap(), 3);
    }

    #[test]
    fn ring_links_rank0() {
        let f = parsed();
        let cfg = f.ring_cfg(0).unwrap();
        assert_eq!(cfg.head_ip, "192.168.1.232");
        assert_eq!(cfg.succ.rank, 1);
        assert_eq!(cfg.pred.rank, 3);
        // 4687-5bc3: rail ifaces f1 on subnets 10.100.224/24, 10.101.224/24
        assert_eq!(cfg.succ.rails[0].local, "rocep1s0f1");
        assert_eq!(cfg.succ.rails[0].remote, "rocep1s0f1");
        assert_eq!(cfg.succ.rails[0].subnet, "10.100.224.0/24");
        assert_eq!(cfg.succ.rails[1].local, "roceP2p1s0f1");
        assert_eq!(cfg.succ.rails[1].subnet, "10.101.224.0/24");
        // 7275-4687: f0 pair on 10.100.227/24, 10.101.227/24
        assert_eq!(cfg.pred.rails[0].subnet, "10.100.227.0/24");
        assert_eq!(cfg.pred.rails[1].subnet, "10.101.227.0/24");
    }

    #[test]
    fn ring_links_rank3() {
        let f = parsed();
        let cfg = f.ring_cfg(3).unwrap();
        assert_eq!(cfg.succ.rank, 0);
        assert_eq!(cfg.pred.rank, 2);
        // 7425-7275 uses the f1 pair, 10.100.226/24
        assert_eq!(cfg.pred.rails[0].local, "rocep1s0f1");
        assert_eq!(cfg.pred.rails[0].remote, "rocep1s0f1");
        assert_eq!(cfg.pred.rails[0].subnet, "10.100.226.0/24");
    }

    #[test]
    fn all_rings_close() {
        let f = parsed();
        for r in 0..4u8 {
            let cfg = f.ring_cfg(r).unwrap();
            assert_eq!(cfg.nranks, 4);
            assert_eq!((cfg.succ.rank as usize) % 4, (r as usize + 1) % 4);
            assert_eq!(cfg.pred.rank as usize, (r as usize + 3) % 4);
        }
    }

    #[test]
    fn rejects_broken_map() {
        // peer entry missing (rank 1 loses its link toward rank 0)
        let bad = REAL.replace("\"0\": [\"rocep1s0f1\", \"roceP2p1s0f1\"],", "");
        let f = NodesFile::parse(&bad).unwrap();
        assert!(f.ring_cfg(1).is_err());
        // duplicate rank
        let dup = REAL.replace("\"rank\": 2,", "\"rank\": 0,");
        let f = NodesFile::parse(&dup).unwrap();
        assert!(f.ring_cfg(0).is_err());
    }

    #[test]
    fn subnet_math() {
        assert!(in_subnet("10.100.224.5", "10.100.224.0/24"));
        assert!(!in_subnet("10.100.225.5", "10.100.224.0/24"));
        assert!(in_subnet("192.168.1.232", "192.168.1.0/24"));
        assert!(!in_subnet("10.100.224.5", "10.101.224.0/24"));
    }
}
