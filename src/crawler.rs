//! A polite, concurrent, single-site crawler.
//!
//! Breadth-first from a start URL, staying on the same host (and by default
//! under the start URL's directory), honoring `robots.txt`, with at most
//! `concurrency` requests in flight.

use std::collections::{HashSet, VecDeque};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::time::Duration;

use anyhow::Result;
use reqwest::{Client, Url, header};
use tokio::task::JoinSet;

use crate::html;

const USER_AGENT: &str = concat!(
    "mini-search/",
    env!("CARGO_PKG_VERSION"),
    " (learning project)"
);
const MAX_BODY_BYTES: u64 = 5 * 1024 * 1024;
const SKIPPED_EXTENSIONS: &[&str] = &[
    "png", "jpg", "jpeg", "gif", "svg", "webp", "ico", "pdf", "zip", "gz", "tar", "mp3", "mp4",
    "webm", "woff", "woff2", "ttf", "css", "js", "json", "xml", "rss", "exe", "dmg",
];

pub struct CrawlConfig {
    pub start: Url,
    pub max_pages: usize,
    pub max_depth: usize,
    pub concurrency: usize,
    /// Follow links anywhere on the host, not only under the start URL's directory.
    pub whole_host: bool,
}

pub struct Page {
    pub url: String,
    pub title: String,
    pub text: String,
}

pub async fn crawl(cfg: &CrawlConfig) -> Result<Vec<Page>> {
    let client = Client::builder()
        .user_agent(USER_AGENT)
        .timeout(Duration::from_secs(15))
        .build()?;
    let robots = Robots::fetch(&client, &cfg.start).await;
    let host = cfg.start.host_str().unwrap_or_default().to_string();
    // "/book/ch01.html" -> "/book/"
    let scope = if cfg.whole_host {
        "/".to_string()
    } else {
        let path = cfg.start.path();
        path[..=path.rfind('/').unwrap_or(0)].to_string()
    };
    let scope = Scope {
        host: &host,
        path: &scope,
        robots: &robots,
    };

    let mut start = cfg.start.clone();
    start.set_fragment(None);
    let mut seen: HashSet<String> = HashSet::from([start.to_string()]);
    let mut queue: VecDeque<(Url, usize)> = VecDeque::from([(start, 0)]);
    let mut tasks = JoinSet::new();
    let mut pages = Vec::new();
    // The same page is often reachable under several URLs (`/stable/x` and `/x`).
    let mut seen_content: HashSet<u64> = HashSet::new();

    loop {
        // Keep the pipeline full, but never schedule more than we still need.
        while tasks.len() < cfg.concurrency && pages.len() + tasks.len() < cfg.max_pages {
            let Some((url, depth)) = queue.pop_front() else {
                break;
            };
            let client = client.clone();
            tasks.spawn(async move {
                let result = fetch(&client, &url).await;
                (url, depth, result)
            });
        }
        let Some(joined) = tasks.join_next().await else {
            break;
        };
        let (url, depth, result) = joined?;

        let (final_url, body) = match result {
            Ok(Some(ok)) => ok,
            Ok(None) => continue, // not HTML
            Err(e) => {
                eprintln!("  ! {url}: {e}");
                continue;
            }
        };
        if final_url.host_str() != Some(host.as_str()) {
            continue; // redirected off-site
        }
        seen.insert(final_url.to_string());

        let parsed = html::parse(&final_url, &body);
        if depth < cfg.max_depth {
            for link in parsed.links {
                if scope.allows(&link) && seen.insert(link.to_string()) {
                    queue.push_back((link, depth + 1));
                }
            }
        }
        let mut hasher = DefaultHasher::new();
        parsed.text.hash(&mut hasher);
        if !parsed.text.is_empty() && seen_content.insert(hasher.finish()) {
            eprintln!("  [{:>4}] {final_url}", pages.len() + 1);
            pages.push(Page {
                url: final_url.to_string(),
                title: parsed.title,
                text: parsed.text,
            });
        }
    }
    Ok(pages)
}

/// Downloads a page. `Ok(None)` for responses that aren't HTML.
async fn fetch(client: &Client, url: &Url) -> Result<Option<(Url, String)>> {
    let resp = client.get(url.clone()).send().await?.error_for_status()?;
    let is_html = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_none_or(|ct| ct.contains("text/html") || ct.contains("application/xhtml"));
    if !is_html
        || resp
            .content_length()
            .is_some_and(|len| len > MAX_BODY_BYTES)
    {
        return Ok(None);
    }
    let final_url = resp.url().clone();
    Ok(Some((final_url, resp.text().await?)))
}

/// Which links the crawler may follow.
struct Scope<'a> {
    host: &'a str,
    path: &'a str,
    robots: &'a Robots,
}

impl Scope<'_> {
    fn allows(&self, url: &Url) -> bool {
        if !matches!(url.scheme(), "http" | "https") || url.host_str() != Some(self.host) {
            return false;
        }
        let path = url.path();
        if !path.starts_with(self.path) {
            return false;
        }
        let extension = path
            .rsplit_once('.')
            .map(|(_, ext)| ext.to_ascii_lowercase());
        if extension.is_some_and(|ext| SKIPPED_EXTENSIONS.contains(&ext.as_str())) {
            return false;
        }
        self.robots.allows(path)
    }
}

/// The subset of robots.txt that matters here: `Allow` / `Disallow` prefixes
/// in the groups addressed to `*` or to us. Longest matching rule wins.
#[derive(Default)]
struct Robots {
    rules: Vec<(String, bool)>, // (path prefix, allowed)
}

impl Robots {
    async fn fetch(client: &Client, start: &Url) -> Robots {
        let Ok(robots_url) = start.join("/robots.txt") else {
            return Robots::default();
        };
        let body = async {
            let resp = client
                .get(robots_url)
                .send()
                .await
                .ok()?
                .error_for_status()
                .ok()?;
            resp.text().await.ok()
        };
        body.await.map(|b| Robots::parse(&b)).unwrap_or_default()
    }

    fn parse(body: &str) -> Robots {
        let mut rules = Vec::new();
        let mut applies = false;
        let mut in_agent_lines = false;
        for line in body.lines() {
            let line = line.split('#').next().unwrap_or("").trim();
            let Some((key, value)) = line.split_once(':') else {
                continue;
            };
            let (key, value) = (key.trim().to_ascii_lowercase(), value.trim());
            match key.as_str() {
                "user-agent" => {
                    let ours = value == "*" || value.eq_ignore_ascii_case("mini-search");
                    // consecutive user-agent lines share one group
                    applies = if in_agent_lines {
                        applies || ours
                    } else {
                        ours
                    };
                    in_agent_lines = true;
                }
                "allow" | "disallow" => {
                    in_agent_lines = false;
                    let prefix = value.trim_end_matches('*');
                    if applies && !prefix.is_empty() {
                        rules.push((prefix.to_string(), key == "allow"));
                    }
                }
                _ => in_agent_lines = false,
            }
        }
        Robots { rules }
    }

    fn allows(&self, path: &str) -> bool {
        self.rules
            .iter()
            .filter(|(prefix, _)| path.starts_with(prefix.as_str()))
            .max_by_key(|(prefix, allow)| (prefix.len(), *allow))
            .is_none_or(|(_, allow)| *allow)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn robots_rules() {
        let robots = Robots::parse(
            "User-agent: googlebot\nDisallow: /\n\n\
             User-agent: *\nDisallow: /private/\nAllow: /private/public\nDisallow: /tmp*\n",
        );
        assert!(robots.allows("/"));
        assert!(robots.allows("/docs/page"));
        assert!(!robots.allows("/private/secret"));
        assert!(robots.allows("/private/public/page"));
        assert!(!robots.allows("/tmp/x"));
    }

    #[test]
    fn link_filter() {
        let robots = Robots::default();
        let scope = Scope {
            host: "example.com",
            path: "/docs/",
            robots: &robots,
        };
        let ok = |s: &str| scope.allows(&Url::parse(s).unwrap());
        assert!(ok("https://example.com/docs/"));
        assert!(ok("https://example.com/docs/a/b.html"));
        assert!(!ok("https://example.com/blog/"));
        assert!(!ok("https://other.com/docs/"));
        assert!(!ok("https://example.com/docs/logo.PNG"));
        assert!(!ok("mailto:me@example.com"));
    }
}
