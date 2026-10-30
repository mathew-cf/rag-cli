use std::process::Command;

#[test]
fn paths_combines_sources_without_admitting_unselected_siblings() {
    let root = std::env::temp_dir().join(format!("rag-paths-{}", std::process::id()));
    for dir in [
        "repos/a/docs",
        "repos/b/.cloudflare/docs",
        "repos/unselected",
        "repos/a/docs-old",
        "base",
    ] {
        std::fs::create_dir_all(root.join(dir)).unwrap();
    }
    for path in [
        "repos/a/docs/a.md",
        "repos/b/.cloudflare/docs/b.md",
        "repos/unselected/private.md",
        "repos/a/docs-old/old.md",
        "repos/a/top.md",
    ] {
        std::fs::write(root.join(path), "needle\n").unwrap();
    }
    let config = root.join("base/rag.toml");
    std::fs::write(&config, "[[index]]\nname = 'combined'\npaths = ['../repos/a/docs', '../repos/b/.cloudflare/docs', '../repos/a/docs']\nhidden = true\ninclude = ['unselected']\n").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_rag"))
        .args([
            "keyword",
            "-e",
            "needle",
            "-l",
            "--glob",
            "*.md",
            "--no-ignore",
            "--config",
        ])
        .arg(&config)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap().replace('\\', "/");
    assert!(text.contains("a/docs/a.md"));
    assert!(text.contains("b/.cloudflare/docs/b.md"));
    assert_eq!(text.lines().count(), 2, "{text}");
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink("../../unselected", root.join("repos/a/docs/linked")).unwrap();
        std::os::unix::fs::symlink(
            "../../unselected/private.md",
            root.join("repos/a/docs/leak.md"),
        )
        .unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_rag"))
            .args(["keyword", "-e", "needle", "-l", "--config"])
            .arg(&config)
            .output()
            .unwrap();
        assert!(output.status.success());
        assert_eq!(String::from_utf8(output.stdout).unwrap().lines().count(), 2);
    }
    // A new sibling must stay outside the corpus without regenerating config.
    std::fs::create_dir_all(root.join("repos/new")).unwrap();
    std::fs::write(root.join("repos/new/new.md"), "needle\n").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_rag"))
        .args(["keyword", "-e", "needle", "-l", "--config"])
        .arg(&config)
        .output()
        .unwrap();
    assert!(output.status.success());
    assert_eq!(String::from_utf8(output.stdout).unwrap().lines().count(), 2);
    for invalid in [
        "",
        "path = '../repos'\npaths = ['../repos/a/docs']",
        "paths = []",
        "paths = ['missing']",
    ] {
        std::fs::write(&config, format!("[[index]]\nname = 'bad'\n{invalid}\n")).unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_rag"))
            .args(["keyword", "-e", "needle", "--config"])
            .arg(&config)
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(2), "{invalid}");
    }
    std::fs::remove_dir_all(root).unwrap();
}

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
    assert!(stdout.contains("--semantic-weight"));
    assert!(stdout.contains("--keyword-weight"));
    assert!(stdout.contains("--keyword"));
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
        String::from_utf8(output.stdout)
            .unwrap()
            .trim()
            .replace('\\', "/"),
        root.join("notes/hit.md")
            .display()
            .to_string()
            .replace('\\', "/")
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

#[test]
fn keyword_searches_all_configured_sources_and_can_select_one() {
    let root = std::env::temp_dir().join(format!(
        "rag-keyword-federated-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(root.join("docs")).unwrap();
    std::fs::create_dir_all(root.join("reference")).unwrap();
    std::fs::create_dir_all(root.join(".git")).unwrap();
    std::fs::create_dir_all(root.join("docs/dist")).unwrap();
    std::fs::create_dir_all(root.join("docs/.hidden")).unwrap();
    std::fs::write(root.join("docs/one.md"), "cache hit\n").unwrap();
    std::fs::write(root.join("docs/ignored.md"), "cache ignored\n").unwrap();
    std::fs::write(root.join("docs/skip.md"), "cache skipped\n").unwrap();
    std::fs::write(root.join("docs/extra.mdx"), "cache extra\n").unwrap();
    std::fs::write(root.join("docs/code.rs"), "cache code\n").unwrap();
    std::fs::write(root.join("docs/dist/built.md"), "cache built\n").unwrap();
    std::fs::write(root.join("docs/.hidden/secret.md"), "cache hidden\n").unwrap();
    std::fs::write(root.join("docs/.secret.md"), "cache hidden file\n").unwrap();
    std::fs::write(root.join("docs/.gitignore"), "ignored.md\n").unwrap();
    std::fs::write(root.join("reference/two.md"), "cache miss\n").unwrap();
    std::fs::write(
        root.join("rag.toml"),
        "[[index]]\nname = 'docs'\npath = 'docs'\nextensions = ['mdx']\nexclude = ['skip.md']\ninclude = ['dist']\n\n[[index]]\nname = 'reference'\npath = 'reference'\n",
    )
    .unwrap();

    let all = Command::new(env!("CARGO_BIN_EXE_rag"))
        .current_dir(&root)
        .args(["keyword", "-e", "cache", "-l"])
        .output()
        .unwrap();
    assert!(all.status.success());
    let output = String::from_utf8(all.stdout).unwrap().replace('\\', "/");
    assert!(output.contains("docs/one.md"));
    assert!(output.contains("docs/extra.mdx"));
    assert!(output.contains("docs/dist/built.md"));
    assert!(output.contains("reference/two.md"));
    for excluded in ["ignored.md", "skip.md", "code.rs", "secret.md"] {
        assert!(!output.contains(excluded), "{excluded} was included");
    }

    let selected = Command::new(env!("CARGO_BIN_EXE_rag"))
        .current_dir(&root)
        .args(["keyword", "-e", "cache", "--only", "reference", "-l"])
        .output()
        .unwrap();
    assert!(selected.status.success());
    let output = String::from_utf8(selected.stdout)
        .unwrap()
        .replace('\\', "/");
    assert!(!output.contains("docs/one.md"));
    assert!(output.contains("reference/two.md"));

    let expanded = Command::new(env!("CARGO_BIN_EXE_rag"))
        .current_dir(&root)
        .args([
            "keyword",
            "-e",
            "cache",
            "--only",
            "docs",
            "--no-ignore",
            "--hidden",
            "-l",
        ])
        .output()
        .unwrap();
    assert!(expanded.status.success());
    let output = String::from_utf8(expanded.stdout)
        .unwrap()
        .replace('\\', "/");
    assert!(output.contains("docs/ignored.md"));
    assert!(output.contains("docs/.hidden/secret.md"));
    assert!(output.contains("docs/.secret.md"));
    assert!(!output.contains("docs/skip.md"));
    std::fs::remove_dir_all(root).unwrap();
}
