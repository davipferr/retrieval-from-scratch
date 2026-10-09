use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use reqwest::Url;

use mini_search::crawler::{self, CrawlConfig};
use mini_search::index::{Index, IndexBuilder, WriteStats};
use mini_search::search::{self, Highlight, SearchOptions};
use mini_search::{files, server};

#[derive(Parser)]
#[command(
    version,
    about = "A mini search engine: crawl, index, rank with BM25, forgive typos."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Crawl a website and index its pages.
    Crawl {
        url: Url,
        /// Index directory to write.
        #[arg(short, long, default_value = "index")]
        out: PathBuf,
        #[arg(long, default_value_t = 200)]
        max_pages: usize,
        /// How many links deep to follow from the start page.
        #[arg(long, default_value_t = 30)]
        depth: usize,
        /// Requests in flight at once.
        #[arg(long, default_value_t = 8)]
        concurrency: usize,
        /// Follow links anywhere on the site, not just under the start URL's directory.
        #[arg(long)]
        whole_host: bool,
    },
    /// Index local files (projects, notes, docs) under one or more directories.
    Index {
        #[arg(required = true)]
        paths: Vec<PathBuf>,
        #[arg(short, long, default_value = "index")]
        out: PathBuf,
    },
    /// Search the index from the command line.
    Search {
        #[arg(required = true)]
        query: Vec<String>,
        #[arg(short, long, default_value = "index")]
        index: PathBuf,
        #[arg(short, long, default_value_t = 10)]
        limit: usize,
        /// Disable typo tolerance.
        #[arg(long)]
        exact: bool,
    },
    /// Show how the Levenshtein automaton compares to brute force for one word.
    Fuzzy {
        word: String,
        #[arg(short, long, default_value = "index")]
        index: PathBuf,
        /// Max edits (default: 0 for â‰¤2 chars, 1 for â‰¤5, else 2).
        #[arg(short, long)]
        max_distance: Option<u8>,
    },
    /// Print index statistics.
    Stats {
        #[arg(short, long, default_value = "index")]
        index: PathBuf,
    },
    /// Serve the search page and JSON API.
    Serve {
        #[arg(short, long, default_value = "index")]
        index: PathBuf,
        #[arg(short, long, default_value = "127.0.0.1:3000")]
        addr: SocketAddr,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Crawl {
            url,
            out,
            max_pages,
            depth,
            concurrency,
            whole_host,
        } => {
            println!("crawling {url} (max {max_pages} pages, depth {depth})");
            let started = Instant::now();
            let cfg = CrawlConfig {
                start: url,
                max_pages,
                max_depth: depth,
                concurrency,
                whole_host,
            };
            let pages = crawler::crawl(&cfg).await?;
            println!(
                "fetched {} pages in {:.1}s",
                pages.len(),
                started.elapsed().as_secs_f64()
            );

            let mut builder = IndexBuilder::new();
            for page in &pages {
                builder.add_document(&page.url, &page.title, &page.text);
            }
            report(builder.write(&out)?, &out);
        }
        Command::Index { paths, out } => {
            let mut builder = IndexBuilder::new();
            for root in &paths {
                let docs =
                    files::collect(root).with_context(|| format!("reading {}", root.display()))?;
                println!("{}: {} files", root.display(), docs.len());
                for doc in docs {
                    builder.add_document(&doc.path.display().to_string(), &doc.title, &doc.text);
                }
            }
            report(builder.write(&out)?, &out);
        }
        Command::Search {
            query,
            index,
            limit,
            exact,
        } => {
            let index = Index::open(&index)?;
            let query = query.join(" ");
            let started = Instant::now();
            let opts = SearchOptions {
                limit,
                fuzzy: !exact,
                ..Default::default()
            };
            let results = search::search(&index, &query, &opts);
            let took = started.elapsed();

            for t in &results.terms {
                let matches: Vec<String> = t
                    .matches
                    .iter()
                    .take(6)
                    .map(|m| {
                        if m.distance == 0 {
                            m.term.clone()
                        } else {
                            format!("{}~{}", m.term, m.distance)
                        }
                    })
                    .collect();
                let shown = if matches.is_empty() {
                    "(no match)".into()
                } else {
                    matches.join(", ")
                };
                println!("  {} -> {shown}", t.term);
            }
            println!(
                "{} results in {:.2} ms\n",
                results.total,
                took.as_secs_f64() * 1000.0
            );

            let matched = results.matched_terms();
            for (rank, hit) in results.hits.iter().enumerate() {
                println!(
                    "{:>2}. \x1b[1m{}\x1b[0m  ({:.3})",
                    rank + 1,
                    hit.title,
                    hit.score
                );
                println!("    \x1b[2m{}\x1b[0m", hit.url);
                println!(
                    "    {}\n",
                    search::snippet(&index, hit.doc, &matched, Highlight::Ansi)
                );
            }
        }
        Command::Fuzzy {
            word,
            index,
            max_distance,
        } => {
            let index = Index::open(&index)?;
            let r = search::fuzzy_report(&index, &word, max_distance);
            println!(
                "term {:?}, max distance {}, dictionary of {} terms\n",
                r.term, r.max_distance, r.dictionary_size
            );
            println!(
                "brute force:   {:>9.1} Âµs  (DP table against all {} terms)",
                r.bruteforce_us, r.dictionary_size
            );
            println!(
                "automaton:     {:>9.1} Âµs  ({} DFA states built in {:.1} Âµs; visited {} terms, \
                 {} transitions, {} prefix skips)",
                r.automaton_us,
                r.dfa_states,
                r.dfa_build_us,
                r.terms_visited,
                r.transitions,
                r.skips
            );
            println!("\n{} matches:", r.matches.len());
            for m in r.matches.iter().take(30) {
                println!(
                    "  {:<24} distance {}  in {} docs",
                    m.term, m.distance, m.doc_freq
                );
            }
        }
        Command::Stats { index } => {
            let index = Index::open(&index)?;
            println!("documents:       {}", index.num_docs());
            println!("terms:           {}", index.num_terms());
            println!("avg doc length:  {:.1} terms", index.avg_doc_len());
            for (file, size) in index.sizes() {
                println!("{file:<16} {}", human(size));
            }
        }
        Command::Serve { index, addr } => {
            let index = Index::open(&index)?;
            println!(
                "{} documents, {} terms",
                index.num_docs(),
                index.num_terms()
            );
            server::serve(index, addr).await?;
        }
    }
    Ok(())
}

fn report(stats: WriteStats, out: &std::path::Path) {
    println!(
        "wrote {}: {} docs, {} terms, {} postings",
        out.display(),
        stats.docs,
        stats.terms,
        stats.postings
    );
    if stats.postings_raw_bytes > 0 {
        println!(
            "posting lists: {} compressed vs {} as plain u32 pairs ({:.0}%)",
            human(stats.postings_bytes),
            human(stats.postings_raw_bytes),
            100.0 * stats.postings_bytes as f64 / stats.postings_raw_bytes as f64
        );
    }
}

fn human(bytes: usize) -> String {
    match bytes {
        b if b >= 1 << 20 => format!("{:.1} MB", b as f64 / (1 << 20) as f64),
        b if b >= 1 << 10 => format!("{:.1} KB", b as f64 / (1 << 10) as f64),
        b => format!("{b} B"),
    }
}
