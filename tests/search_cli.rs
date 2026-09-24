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
    assert!(stdout.contains("--hybrid"));
}

#[test]
fn keyword_scans_live_files_with_rg_style_status_and_globs() {
    let root = std::env::temp_dir().join(format!(
        "rag-keyword-cli-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(root.join("notes")).unwrap();
    std::fs::create_dir_all(root.join(".rag")).unwrap();
    std::fs::write(root.join("notes/hit.md"), "Foo.Bar\n").unwrap();
    std::fs::write(root.join("notes/INDEX.md"), "Foo.Bar\n").unwrap();
    std::fs::write(root.join("notes/other.jsonl"), "Foo.Bar\n").unwrap();
    std::fs::write(root.join(".rag/hidden.md"), "Foo.Bar\n").unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_rag"))
        .args([
            "keyword",
            "-e",
            "foo.bar",
            "-F",
            "-i",
            "--glob",
            "*.md",
            "--glob",
            "!**/INDEX.md",
            "-l",
        ])
        .arg(&root)
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(0));
    assert_eq!(
        String::from_utf8(output.stdout).unwrap().trim(),
        root.join("notes/hit.md").display().to_string()
    );

    let no_hits = Command::new(env!("CARGO_BIN_EXE_rag"))
        .args(["keyword", "-e", "missing", "-l"])
        .arg(&root)
        .output()
        .unwrap();
    assert_eq!(no_hits.status.code(), Some(1));
    let bad_regex = Command::new(env!("CARGO_BIN_EXE_rag"))
        .args(["keyword", "-e", "["])
        .arg(&root)
        .output()
        .unwrap();
    assert_eq!(bad_regex.status.code(), Some(2));
    std::fs::remove_dir_all(root).unwrap();
}
