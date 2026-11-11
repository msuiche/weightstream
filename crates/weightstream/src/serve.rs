//! `weightstream serve` (docs/transport.md milestone 4): the OpenAI-compatible
//! endpoint on the TP4 head rank.
//!
//! One decode loop owns the engine (the ring collectives are strictly
//! lockstep — every rank decodes the same request at the same step), so
//! requests serialize through it by design. The HTTP side is a minimal
//! std-only server: a listener thread parses `POST /v1/completions`
//! (prompt, max_tokens, stream), hands the request to the decode loop,
//! and streams the generated tokens back as they land (SSE for
//! stream:true, one JSON body otherwise). The head broadcasts each
//! request to the workers with the tp4 control channel (`Phase::Next`);
//! workers just loop on `next_request`, so nothing here runs on them.
//!
//! Non-goals (later milestones): batching several concurrent requests
//! into one lockstep step, prefix-cache reuse, sampling parameters —
//! greedy only, the v1 contract.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};

/// A request as the HTTP side sees it.
#[derive(Clone)]
pub struct ServeRequest {
    pub prompt: String,
    pub max_tokens: u32,
    pub stream: bool,
    pub temperature: f32,
    pub top_p: f32,
    pub seed: u64,
}

/// The decode loop's answer: generated tokens, streamed one at a time.
pub enum TokenEvent {
    Token(String),
    Done,
}

/// Parse `POST /v1/completions` — headers up to the blank line, then a
/// Content-Length JSON body. Returns (request, http_version) or a
/// (status, reason) error for the caller to write.
pub fn read_request(stream: &mut TcpStream) -> Result<(ServeRequest, String), (u16, &'static str)> {
    let mut reader = BufReader::new(stream.try_clone().map_err(|_| (500u16, "clone"))?);
    let mut line = String::new();
    reader
        .read_line(&mut line)
        .map_err(|_| (400u16, "request line"))?;
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("");
    let path = parts.next().unwrap_or("");
    let version = parts.next().unwrap_or("HTTP/1.1").to_string();
    if method != "POST" {
        return Err((405u16, "method not allowed"));
    }
    if path != "/v1/completions" && path != "/v1/chat/completions" {
        return Err((404u16, "not found"));
    }
    let mut content_length = 0usize;
    loop {
        let mut h = String::new();
        reader.read_line(&mut h).map_err(|_| (400u16, "headers"))?;
        let t = h.trim();
        if t.is_empty() {
            break;
        }
        if let Some((k, v)) = t.split_once(':') {
            if k.eq_ignore_ascii_case("content-length") {
                content_length = v.trim().parse().unwrap_or(0);
            }
        }
    }
    if content_length == 0 || content_length > 1 << 20 {
        return Err((400u16, "content-length"));
    }
    let mut body = vec![0u8; content_length];
    reader.read_exact(&mut body).map_err(|_| (400u16, "body"))?;
    let json: serde_json::Value = serde_json::from_slice(&body).map_err(|_| (400u16, "json"))?;

    // OpenAI compat: completions takes "prompt"; chat/completions takes a
    // "messages" array — concatenated into one prompt (greedy, no chat
    // template yet; the model is a base checkpoint).
    let prompt = if let Some(p) = json.get("prompt").and_then(|v| v.as_str()) {
        p.to_string()
    } else if let Some(msgs) = json.get("messages").and_then(|v| v.as_array()) {
        msgs.iter()
            .filter_map(|m| m.get("content").and_then(|c| c.as_str()))
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        return Err((400u16, "prompt"));
    };
    let max_tokens = json
        .get("max_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(64) as u32;
    let stream = json
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let temperature = json
        .get("temperature")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as f32;
    let top_p = json.get("top_p").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
    let seed = json
        .get("seed")
        .and_then(|v| v.as_u64())
        .unwrap_or(weightstream_ref::REFERENCE_SEED);
    if prompt.is_empty() {
        return Err((400u16, "empty prompt"));
    }
    Ok((
        ServeRequest {
            prompt,
            max_tokens,
            stream,
            temperature,
            top_p,
            seed,
        },
        version,
    ))
}
pub fn write_status(
    stream: &mut TcpStream,
    code: u16,
    reason: &str,
    version: &str,
) -> std::io::Result<()> {
    write!(
        stream,
        "{version} {code} {reason}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
    )
}

/// Streaming (SSE) response writer: each token becomes one
/// `data: {...}` line; the final `data: [DONE]` closes.
pub struct SseWriter {
    stream: TcpStream,
}
impl SseWriter {
    pub fn open(mut stream: TcpStream, version: &str) -> std::io::Result<Self> {
        write!(stream, "{version} 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n")?;
        Ok(SseWriter { stream })
    }
    pub fn token(&mut self, text: &str) -> std::io::Result<()> {
        let payload = serde_json::json!({ "choices": [{ "text": text }] });
        write!(self.stream, "data: {payload}\n\n")
    }
    pub fn done(&mut self) -> std::io::Result<()> {
        write!(self.stream, "data: [DONE]\n\n")
    }
}

/// The non-streaming response: one OpenAI-shaped completions body.
pub fn write_completion(
    stream: &mut TcpStream,
    text: &str,
    version: &str,
    model: &str,
) -> std::io::Result<()> {
    let body = serde_json::json!({
        "object": "text_completion",
        "model": model,
        "choices": [{ "text": text, "finish_reason": "stop" }],
    });
    let body = body.to_string();
    write!(stream, "{version} 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len())
}

/// Run the accept loop: one connection at a time (requests serialize on
/// the decode loop anyway; keep-alive is refused). `submit` hands the
/// request to the decode owner; the returned receiver streams its tokens.
pub fn serve_forever<F>(addr: &str, submit: F) -> std::io::Result<()>
where
    F: Fn(ServeRequest) -> std::io::Result<Receiver<TokenEvent>> + Send + Sync + 'static,
{
    let listener = TcpListener::bind(addr)?;
    eprintln!("weightstream serve: listening on {addr}");
    let submit = Arc::new(submit);
    for conn in listener.incoming() {
        let mut sock = match conn {
            Ok(c) => c,
            Err(_) => continue,
        };
        let (req, version) = match read_request(&mut sock) {
            Ok(r) => r,
            Err((code, reason)) => {
                let _ = write_status(&mut sock, code, reason, "HTTP/1.1");
                continue;
            }
        };
        let rx = match (submit)(req.clone()) {
            Ok(rx) => rx,
            Err(e) => {
                eprintln!("serve: {e}");
                let _ = write_status(&mut sock, 503u16, "engine busy", &version);
                continue;
            }
        };
        // consume the token stream on the connection thread
        let res = if req.stream {
            let mut w = match SseWriter::open(sock.try_clone()?, &version) {
                Ok(w) => w,
                Err(_) => continue,
            };
            let mut ok = true;
            while let Ok(ev) = rx.recv() {
                match ev {
                    TokenEvent::Token(t) => ok = w.token(&t).is_ok() && ok,
                    TokenEvent::Done => break,
                }
            }
            ok = w.done().is_ok() && ok;
            if ok {
                Ok(())
            } else {
                Err(std::io::Error::new(std::io::ErrorKind::BrokenPipe, "sse"))
            }
        } else {
            let mut text = String::new();
            while let Ok(ev) = rx.recv() {
                match ev {
                    TokenEvent::Token(t) => text.push_str(&t),
                    TokenEvent::Done => break,
                }
            }
            write_completion(&mut sock, &text, &version, "deepseek-v4.1-flash")
        };
        if let Err(e) = res {
            eprintln!("serve: write: {e}");
        }
    }
    Ok(())
}

/// Queue shared with the decode owner: the HTTP thread parks here.
pub type SubmitQueue = Arc<Mutex<Option<Sender<ServeRequest>>>>;

pub fn make_pair() -> (SubmitQueue, Receiver<ServeRequest>) {
    let (tx, rx) = channel();
    (Arc::new(Mutex::new(Some(tx))), rx)
}
