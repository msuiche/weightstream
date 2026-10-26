//! Control channel: plain TCP over the management network, newline-delimited
//! JSON messages. Small by design — RDMA data never flows here, only phase
//! coordination (rendezvous, QP endpoint exchange, barriers, results).

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Phase {
    #[serde(rename = "linkup")]
    Linkup,
    #[serde(rename = "rtr")]
    Rtr,
    #[serde(rename = "rts")]
    Rts,
    #[serde(rename = "probe")]
    Probe,
    #[serde(rename = "gate")]
    Gate,
    #[serde(rename = "soak")]
    Soak,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Msg {
    /// worker -> head, first message on connect
    Hello { rank: u8 },
    /// head -> worker: run this phase (cmd carries per-phase parameters)
    Go { phase: Phase, cmd: serde_json::Value },
    /// worker -> head: phase done; data always carries "rank"
    Ready { data: serde_json::Value },
    /// either way: something failed; every rank must fail-stop
    Abort { reason: String },
    /// head -> worker: all done, exit 0
    Bye,
}

#[derive(Debug)]
pub struct CtrlError(pub String);
impl std::fmt::Display for CtrlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ctrl: {}", self.0)
    }
}
impl std::error::Error for CtrlError {}
impl From<std::io::Error> for CtrlError {
    fn from(e: std::io::Error) -> Self {
        CtrlError(e.to_string())
    }
}
impl From<serde_json::Error> for CtrlError {
    fn from(e: serde_json::Error) -> Self {
        CtrlError(e.to_string())
    }
}

pub type CtrlResult<T> = Result<T, CtrlError>;

fn send(stream: &mut TcpStream, m: &Msg) -> CtrlResult<()> {
    let line = serde_json::to_string(m)?;
    stream.write_all(line.as_bytes())?;
    stream.write_all(b"\n")?;
    stream.flush()?;
    Ok(())
}

pub struct WorkerConn {
    pub rank: u8,
    stream: TcpStream,
    reader: BufReader<TcpStream>,
}

/// Head side of the control channel. Rank 0 is the head and participates in
/// the ring itself; workers are ranks 1..n-1.
pub struct Head {
    pub workers: Vec<WorkerConn>,
    listener: Option<TcpListener>,
    my_rank: u8,
    nranks_hint: usize,
}

impl Head {
    /// Bind on the management IP (non-blocking accept loop is armed by
    /// `wait_workers`).
    pub fn bind(mgmt_ip: &str, port: u16, my_rank: u8) -> CtrlResult<Head> {
        let listener = TcpListener::bind((mgmt_ip, port))?;
        listener.set_nonblocking(true)?;
        Ok(Head {
            workers: Vec::new(),
            listener: Some(listener),
            my_rank,
            nranks_hint: 0,
        })
    }

    pub fn local_port(&self) -> CtrlResult<u16> {
        Ok(self
            .listener
            .as_ref()
            .ok_or(CtrlError("listener closed".into()))?
            .local_addr()?
            .port())
    }

    /// Accept until all worker ranks said hello. Fails if the full ring
    /// hasn't joined within `timeout`.
    pub fn wait_workers(&mut self, nranks: usize, timeout: Duration) -> CtrlResult<()> {
        let deadline = Instant::now() + timeout;
        self.nranks_hint = nranks;
        while self.workers.len() < nranks - 1 {
            if Instant::now() >= deadline {
                return Err(CtrlError(format!(
                    "only {}/{} worker ranks joined within {} s",
                    self.workers.len(),
                    nranks - 1,
                    timeout.as_secs()
                )));
            }
            let listener = self
                .listener
                .as_ref()
                .ok_or(CtrlError("listener closed".into()))?;
            match listener.accept() {
                Ok((stream, _)) => {
                    // macOS propagates the listener's O_NONBLOCK to the
                    // accepted socket; Linux does not. Clear it either way.
                    stream.set_nonblocking(false).ok();
                    stream.set_nodelay(true).ok();
                    let mut reader = BufReader::new(stream.try_clone()?);
                    let Msg::Hello { rank } = recv(&mut reader)? else {
                        return Err(CtrlError("expected Hello as first message".into()));
                    };
                    if rank == self.my_rank || rank as usize >= nranks {
                        return Err(CtrlError(format!("invalid hello rank {rank}")));
                    }
                    if self.workers.iter().any(|w| w.rank == rank) {
                        return Err(CtrlError(format!("duplicate rank {rank}")));
                    }
                    self.workers.push(WorkerConn { rank, stream, reader });
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(100));
                }
                Err(e) => return Err(e.into()),
            }
        }
        self.workers.sort_by_key(|w| w.rank);
        Ok(())
    }

    pub fn send_go(&mut self, phase: Phase, cmd: serde_json::Value) -> CtrlResult<()> {
        for w in &mut self.workers {
            send(&mut w.stream, &Msg::Go { phase, cmd: cmd.clone() })?;
        }
        Ok(())
    }

    pub fn send_bye(&mut self) {
        for w in &mut self.workers {
            let _ = send(&mut w.stream, &Msg::Bye);
        }
    }

    /// Collect one Ready from every worker (hard timeout; abort broadcast on
    /// failure). Returns rank -> data, ordered by rank.
    pub fn collect_ready(&mut self, timeout: Duration) -> CtrlResult<Vec<serde_json::Value>> {
        let deadline = Instant::now() + timeout;
        let mut out = Vec::new();
        for w in &mut self.workers {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                let _ = self.abort_all("head timeout waiting for Ready");
                return Err(CtrlError("timeout waiting for Ready".into()));
            }
            w.stream.set_read_timeout(Some(remaining)).ok();
            match recv(&mut w.reader) {
                Ok(Msg::Ready { data }) => out.push(data),
                Ok(Msg::Abort { reason }) => {
                    let _ = self.abort_all(&reason);
                    return Err(CtrlError(format!("worker aborted: {reason}")));
                }
                Ok(_) => {
                    let _ = self.abort_all("expected Ready");
                    return Err(CtrlError("expected Ready".into()));
                }
                Err(e) => {
                    let _ = self.abort_all(&e.to_string());
                    return Err(e);
                }
            }
        }
        Ok(out)
    }

    pub fn abort_all(&mut self, reason: &str) -> CtrlResult<()> {
        for w in &mut self.workers {
            let _ = send(&mut w.stream, &Msg::Abort { reason: reason.to_string() });
        }
        Ok(())
    }
}

/// Worker side of the control channel (ranks 1..n-1).
pub struct Worker {
    pub rank: u8,
    stream: TcpStream,
    reader: BufReader<TcpStream>,
}

impl Worker {
    pub fn connect(head_ip: &str, port: u16, rank: u8, wait: Duration) -> CtrlResult<Worker> {
        let deadline = Instant::now() + wait;
        let mut stream = None;
        while stream.is_none() {
            match TcpStream::connect((head_ip, port)) {
                Ok(s) => stream = Some(s),
                Err(e) => {
                    if Instant::now() >= deadline {
                        return Err(CtrlError(format!("connect to {head_ip}:{port}: {e}")));
                    }
                    std::thread::sleep(Duration::from_millis(500));
                }
            }
        }
        let stream = stream.unwrap();
        stream.set_nodelay(true).ok();
        stream.set_read_timeout(Some(wait)).ok();
        let reader = BufReader::new(stream.try_clone()?);
        let mut w = Worker { rank, stream, reader };
        send(&mut w.stream, &Msg::Hello { rank })?;
        Ok(w)
    }

    /// Block until the head sends the next Go. `None` = Bye (clean exit).
    pub fn wait_go(&mut self) -> CtrlResult<Option<(Phase, serde_json::Value)>> {
        loop {
            match recv(&mut self.reader)? {
                Msg::Go { phase, cmd } => return Ok(Some((phase, cmd))),
                Msg::Abort { reason } => return Err(CtrlError(format!("head aborted: {reason}"))),
                Msg::Bye => return Ok(None),
                Msg::Hello { .. } | Msg::Ready { .. } => continue,
            }
        }
    }

    pub fn send_ready(&mut self, data: serde_json::Value) -> CtrlResult<()> {
        send(&mut self.stream, &Msg::Ready { data })
    }

    pub fn send_abort(&mut self, reason: &str) -> CtrlResult<()> {
        send(&mut self.stream, &Msg::Abort { reason: reason.to_string() })
    }
}

fn recv(reader: &mut BufReader<TcpStream>) -> CtrlResult<Msg> {
    let mut line = String::new();
    let n = reader.read_line(&mut line)?;
    if n == 0 {
        return Err(CtrlError("control channel closed by peer".into()));
    }
    Ok(serde_json::from_str(line.trim())?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn msg_roundtrip() {
        let m = Msg::Go {
            phase: Phase::Probe,
            cmd: serde_json::json!({"rail": 1, "initiator": 0, "responder": 1}),
        };
        let s = serde_json::to_string(&m).unwrap();
        let back: Msg = serde_json::from_str(&s).unwrap();
        assert!(matches!(back, Msg::Go { phase: Phase::Probe, .. }));
        let r: Msg = serde_json::from_str(r#"{"type":"ready","data":{"ok":true}}"#).unwrap();
        assert!(matches!(r, Msg::Ready { .. }));
        let a: Msg = serde_json::from_str(r#"{"type":"abort","reason":"x"}"#).unwrap();
        assert!(matches!(a, Msg::Abort { .. }));
    }

    #[test]
    fn head_worker_rendezvous() {
        // loopback smoke test: two workers join, one Go/Ready round, Bye.
        let mut head = Head::bind("127.0.0.1", 0, 0).unwrap();
        let port = head.local_port().unwrap();
        let mut workers = Vec::new();
        for r in 1..3u8 {
            workers.push(Worker::connect("127.0.0.1", port, r, Duration::from_secs(5)).unwrap());
        }
        head.wait_workers(3, Duration::from_secs(5)).unwrap();
        let t = std::thread::spawn(move || {
            let mut head = head;
            head.send_go(Phase::Gate, serde_json::json!({"x": 1})).unwrap();
            let ready = head.collect_ready(Duration::from_secs(5)).unwrap();
            assert_eq!(ready.len(), 2);
            head.send_bye();
        });
        let handles: Vec<_> = workers
            .into_iter()
            .map(|mut w| {
                std::thread::spawn(move || {
                    let (phase, cmd) = w.wait_go().unwrap().unwrap();
                    assert_eq!(phase, Phase::Gate);
                    assert_eq!(cmd["x"], 1);
                    w.send_ready(serde_json::json!({"rank": w.rank})).unwrap();
                    assert!(w.wait_go().unwrap().is_none());
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        t.join().unwrap();
    }
}
