fn main() -> std::process::ExitCode {
    match rag_cli::run() {
        Ok(code) => code,
        Err(error) => {
            eprintln!("{error:#}");
            std::process::ExitCode::from(2)
        }
    }
}
