use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};

use hybrid::retriever::{self, Mode, Retriever};
use hybrid::server;
use vector_db::embed::onnx::OnnxEmbedder;

#[derive(Parser)]
#[command(
    version,
    about = "Hybrid BM25 + vector retrieval over your repos and notes."
)]
struct Cli {
    /// Embedding model directory (model.onnx + vocab.txt).
    #[arg(long, global = true, default_value = "models/bge-small-en-v1.5")]
    model: PathBuf,
    /// Where the indexes live.
    #[arg(long, global = true, default_value = "data")]
    data: PathBuf,
    #[command(subcommand)]
    command: Command,
}

#[derive(Clone, Copy, ValueEnum)]
enum ModeArg {
    Bm25,
    Vector,
    Hybrid,
}

impl From<ModeArg> for Mode {
    fn from(m: ModeArg) -> Mode {
        match m {
            ModeArg::Bm25 => Mode::Bm25,
            ModeArg::Vector => Mode::Vector,
            ModeArg::Hybrid => Mode::Hybrid,
        }
    }
}

#[derive(Subcommand)]
enum Command {
    /// Chunk, embed and index every text file under the given folders.
    Ingest {
        #[arg(required = true)]
        paths: Vec<PathBuf>,
        /// Skip files whose path contains this text (repeatable).
        #[arg(long)]
        exclude: Vec<String>,
    },
    /// Search from the command line.
    Search {
        #[arg(required = true)]
        query: Vec<String>,
        #[arg(short, default_value_t = 5)]
        k: usize,
        #[arg(long, value_enum, default_value_t = ModeArg::Hybrid)]
        mode: ModeArg,
    },
    /// Show BM25, vector and hybrid results for the same query side by side.
    Compare {
        #[arg(required = true)]
        query: Vec<String>,
        #[arg(short, default_value_t = 5)]
        k: usize,
    },
    /// Serve the search API for the RAG app.
    Serve {
        #[arg(long, default_value = "127.0.0.1:4000")]
        addr: SocketAddr,
    },
}

fn load_embedder(model: &Path) -> Result<Box<OnnxEmbedder>> {
    Ok(Box::new(OnnxEmbedder::load(
        model,
        &vector_db::default_ort_path(),
    )?))
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Ingest { paths, exclude } => {
            let embedder = load_embedder(&cli.model)?;
            let started = Instant::now();
            let report =
                retriever::ingest(&paths, &cli.data, embedder.as_ref(), &exclude, |msg| {
                    eprintln!("  {msg}");
                })?;
            println!(
                "indexed {} files as {} chunks in {:.1}s (embedding {:.1}s, HNSW {:.1}s) -> {}",
                report.files,
                report.chunks,
                started.elapsed().as_secs_f64(),
                report.embed_secs,
                report.hnsw_secs,
                cli.data.display()
            );
        }
        Command::Search { query, k, mode } => {
            let r = Retriever::open(&cli.data, load_embedder(&cli.model)?)?;
            let res = r.search(&query.join(" "), k, mode.into())?;
            println!("{:.1} ms total\n", res.timings.total_ms);
            for (i, h) in res.hits.iter().enumerate() {
                let ranks = format!(
                    "bm25 #{} · vector #{}",
                    h.bm25_rank.map_or("-".into(), |r| r.to_string()),
                    h.vector_rank.map_or("-".into(), |r| r.to_string())
                );
                println!(
                    "{:>2}. {}:{}-{}  ({ranks})",
                    i + 1,
                    h.chunk.path,
                    h.chunk.start_line,
                    h.chunk.end_line
                );
                let preview: String = h
                    .text
                    .lines()
                    .filter(|l| !l.trim().is_empty())
                    .take(3)
                    .collect::<Vec<_>>()
                    .join(" ⏎ ");
                println!("    {}\n", preview.chars().take(160).collect::<String>());
            }
        }
        Command::Compare { query, k } => {
            let r = Retriever::open(&cli.data, load_embedder(&cli.model)?)?;
            let q = query.join(" ");
            for mode in [Mode::Bm25, Mode::Vector, Mode::Hybrid] {
                let res = r.search(&q, k, mode)?;
                println!("── {mode:?} ({:.1} ms)", res.timings.total_ms);
                for h in &res.hits {
                    println!(
                        "   {}:{}-{}",
                        h.chunk.path, h.chunk.start_line, h.chunk.end_line
                    );
                }
            }
        }
        Command::Serve { addr } => {
            let r = Retriever::open(&cli.data, load_embedder(&cli.model)?)?;
            println!("{} chunks from {} files", r.len(), r.files());
            server::serve(r, addr).await?;
        }
    }
    Ok(())
}
