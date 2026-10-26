fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    #[cfg(target_os = "linux")]
    {
        let code = run_linux(&args);
        std::process::exit(code);
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = args;
        eprintln!("ferroseek-transport: this binary requires Linux (libibverbs)");
        std::process::exit(1);
    }
}

#[cfg(target_os = "linux")]
fn run_linux(args: &[String]) -> i32 {
    if args.first().map(|s| s.as_str()) != Some("bench") {
        eprintln!("{}", ferroseek_transport::bench::usage());
        return 2;
    }
    let parsed = match ferroseek_transport::bench::parse_args(&args[1..]) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("ferroseek-transport: {e}");
            return 2;
        }
    };
    match ferroseek_transport::bench::run(&parsed) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("ferroseek-transport: FATAL: {e}");
            1
        }
    }
}
