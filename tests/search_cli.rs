use std::process::Command;

#[test]
fn search_help_documents_source_grouping() {
    let output = Command::new(env!("CARGO_BIN_EXE_rag"))
        .args(["search", "--help"])
        .output()
        .expect("rag search --help should run");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("help output should be UTF-8");
    assert!(stdout.contains("--group-by-source"));
    assert!(stdout.contains("at most one result from each source file"));
}
