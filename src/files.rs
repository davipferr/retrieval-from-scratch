//! Indexing local files: your projects, notes and docs.

use std::fs;
use std::path::{Path, PathBuf};

use reqwest::Url;

use crate::html;

const TEXT_EXTENSIONS: &[&str] = &[
    "md", "markdown", "txt", "rst", "org", "adoc", "rs", "py", "js", "mjs", "ts", "tsx", "jsx",
    "go", "java", "kt", "c", "h", "cpp", "hpp", "cs", "rb", "php", "swift", "lua", "sh", "ps1",
    "sql", "toml", "yaml", "yml", "ini", "css", "scss", "vue", "svelte", "html", "htm",
];
const SKIPPED_DIRS: &[&str] = &[
    "target",
    "node_modules",
    "dist",
    "build",
    "out",
    "venv",
    "__pycache__",
    "vendor",
];
const MAX_FILE_BYTES: u64 = 1024 * 1024;

pub struct FileDoc {
    pub path: PathBuf,
    pub title: String,
    pub text: String,
}

/// Recursively collects indexable text files under `root`.
pub fn collect(root: &Path) -> std::io::Result<Vec<FileDoc>> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir)? {
            let entry = entry?;
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().into_owned();
            let file_type = entry.file_type()?;
            if file_type.is_dir() {
                if !name.starts_with('.') && !SKIPPED_DIRS.contains(&name.as_str()) {
                    stack.push(path);
                }
            } else if file_type.is_file()
                && entry.metadata()?.len() <= MAX_FILE_BYTES
                && let Some(doc) = read_file(root, &path)
            {
                out.push(doc);
            }
        }
    }
    out.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(out)
}

fn read_file(root: &Path, path: &Path) -> Option<FileDoc> {
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    if !TEXT_EXTENSIONS.contains(&ext.as_str()) {
        return None;
    }
    let content = fs::read_to_string(path).ok()?; // skips non-UTF-8 files
    let relative = path
        .strip_prefix(root)
        .unwrap_or(path)
        .display()
        .to_string();

    if ext == "html" || ext == "htm" {
        let base = Url::from_file_path(fs::canonicalize(path).ok()?).ok()?;
        let page = html::parse(&base, &content);
        return Some(FileDoc {
            path: path.to_path_buf(),
            title: page.title,
            text: page.text,
        });
    }

    // For Markdown, the first heading makes a better title than the file name.
    let heading = matches!(ext.as_str(), "md" | "markdown")
        .then(|| content.lines().find_map(|l| l.strip_prefix("# ")))
        .flatten()
        .map(|h| format!("{} — {relative}", h.trim()));
    Some(FileDoc {
        path: path.to_path_buf(),
        title: heading.unwrap_or(relative),
        text: content,
    })
}
