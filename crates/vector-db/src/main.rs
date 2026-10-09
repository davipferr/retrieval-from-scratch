use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use clap::{Parser, Subcommand};

use vector_db::embed::Embedder;
use vector_db::embed::onnx::OnnxEmbedder;
use vector_db::hnsw::{self, HnswParams};
use vector_db::vectors::VectorSource;
use vector_db::{Collection, distance, flat, synthetic};

#[derive(Parser)]
#[command(
    version,
    about = "A mini vector database: SIMD distances, HNSW, mmap, local embeddings."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Benchmark distance kernels, brute force vs HNSW, and recall vs ef on synthetic data.
    Bench {
        #[arg(long, default_value_t = 20_000)]
        n: usize,
        #[arg(long, default_value_t = 384)]
        dim: usize,
        #[arg(long, default_value_t = 200)]
        queries: usize,
        #[arg(short, default_value_t = 10)]
        k: usize,
    },
    /// Measure HNSW recall on a stored collection, using its own vectors as queries.
    Recall {
        collection: PathBuf,
        #[arg(short, default_value_t = 10)]
        k: usize,
        #[arg(long, default_value_t = 200)]
        queries: usize,
    },
    /// Embed texts and print their pairwise cosine similarities.
    Embed {
        #[arg(required_unless_present = "file")]
        texts: Vec<String>,
        /// Embed this file cut into 1000-character pieces instead (a throughput test).
        #[arg(long)]
        file: Option<PathBuf>,
        #[arg(long, default_value = "models/bge-small-en-v1.5")]
        model: PathBuf,
    },
    /// Show how WordPiece splits a text.
    Tokenize {
        text: String,
        #[arg(long, default_value = "models/bge-small-en-v1.5")]
        model: PathBuf,
    },
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Bench { n, dim, queries, k } => bench(n, dim, queries, k),
        Command::Recall {
            collection,
            k,
            queries,
        } => {
            let c = Collection::open(&collection)?;
            println!("{} vectors of {} dims", c.len(), c.dim());
            let step = (c.len() / queries.max(1)).max(1);
            let qs: Vec<Vec<f32>> = (0..c.len())
                .step_by(step)
                .take(queries)
                .map(|i| c.vector(i as u32).to_vec())
                .collect();
            for ef in [k, 16, 32, 64, 128, 256] {
                println!("ef {ef:>4}: recall@{k} = {:.3}", c.recall(&qs, k, ef));
            }
            Ok(())
        }
        Command::Embed {
            mut texts,
            file,
            model,
        } => {
            if let Some(file) = file {
                let chars: Vec<char> = std::fs::read_to_string(file)?.chars().collect();
                texts = chars.chunks(1000).map(|c| c.iter().collect()).collect();
            }
            let started = Instant::now();
            let embedder = OnnxEmbedder::load(&model, &vector_db::default_ort_path())?;
            println!(
                "loaded {} ({} dims) in {:.2}s",
                embedder.model_name,
                embedder.dim(),
                started.elapsed().as_secs_f64()
            );
            let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
            let started = Instant::now();
            let vectors = embedder.embed_documents(&refs)?;
            let secs = started.elapsed().as_secs_f64();
            println!(
                "embedded {} texts in {:.1} ms ({:.1}/s)\n",
                texts.len(),
                secs * 1e3,
                texts.len() as f64 / secs
            );
            for (i, a) in vectors.iter().enumerate().take(8) {
                let row: Vec<String> = vectors
                    .iter()
                    .take(8)
                    .map(|b| format!("{:.3}", distance::dot(a, b)))
                    .collect();
                println!("{}  {:.50}", row.join("  "), texts[i]);
            }
            Ok(())
        }
        Command::Tokenize { text, model } => {
            let wp = vector_db::embed::wordpiece::WordPiece::from_file(&model.join("vocab.txt"))?;
            println!("{}", wp.pieces(&text).join(" "));
            Ok(())
        }
    }
}

fn bench(n: usize, dim: usize, n_queries: usize, k: usize) -> Result<()> {
    println!("generating {n} clustered vectors of {dim} dims…");
    // Queries come from the same distribution but are held out of the index.
    let all = synthetic::clustered(n + n_queries, dim, 42);
    let mut store = vector_db::vectors::VecStore::new(dim);
    for id in 0..n as u32 {
        store.push(all.get(id));
    }
    let queries: Vec<Vec<f32>> = (n..n + n_queries)
        .map(|i| all.get(i as u32).to_vec())
        .collect();

    // --- distance kernels ---
    println!("\ndot product over all {n} vectors (one query):");
    let q = &queries[0];
    let time = |f: &dyn Fn(&[f32], &[f32]) -> f32| {
        let started = Instant::now();
        let mut sink = 0.0;
        for _ in 0..5 {
            for id in 0..n as u32 {
                sink += f(q, store.get(id));
            }
        }
        std::hint::black_box(sink);
        started.elapsed().as_secs_f64() * 1e3 / 5.0
    };
    let scalar = time(&distance::dot_scalar);
    println!("  scalar       {scalar:>8.2} ms");
    let wide = time(&distance::dot_wide);
    println!("  wide f32x8   {wide:>8.2} ms   ({:.1}x)", scalar / wide);
    if distance::dot_avx2(q, q).is_some() {
        let avx = time(&|a, b| distance::dot_avx2(a, b).unwrap());
        println!("  avx2 + fma   {avx:>8.2} ms   ({:.1}x)", scalar / avx);
    }
    println!("  (dispatch uses: {})", distance::dot_impl_name());

    // --- build ---
    let params = HnswParams::default();
    println!(
        "\nbuilding HNSW (M={}, ef_construction={})…",
        params.m, params.ef_construction
    );
    let started = Instant::now();
    let graph = hnsw::build(&store, params, |done| {
        if done % 5000 == 0 {
            eprint!("\r  {done}/{n}");
        }
    });
    eprintln!();
    println!(
        "  built in {:.2}s, {} layers, avg degree on layer 0: {:.1}",
        started.elapsed().as_secs_f64(),
        graph_levels(&graph),
        graph.avg_degree(0)
    );

    // --- search ---
    let started = Instant::now();
    for q in &queries {
        std::hint::black_box(flat::search(&store, q, k));
    }
    let flat_ms = started.elapsed().as_secs_f64() * 1e3 / n_queries as f64;
    println!(
        "\n{:<14} {:>10} {:>12} {:>14}",
        "", "recall@k", "ms/query", "dist evals"
    );
    println!(
        "{:<14} {:>10.3} {:>12.3} {:>14}",
        "brute force", 1.0, flat_ms, n
    );
    for ef in [16, 32, 64, 128, 256] {
        let started = Instant::now();
        let mut evals = 0;
        for q in &queries {
            evals += hnsw::search(&graph, &store, q, k, ef).1.distance_evals;
        }
        let ms = started.elapsed().as_secs_f64() * 1e3 / n_queries as f64;
        let recall = hnsw::recall(&graph, &store, &queries, k, ef);
        println!(
            "{:<14} {:>10.3} {:>12.3} {:>14}   ({:.0}x faster)",
            format!("hnsw ef={ef}"),
            recall,
            ms,
            evals / n_queries,
            flat_ms / ms
        );
    }
    Ok(())
}

fn graph_levels(graph: &hnsw::HnswGraph) -> usize {
    use vector_db::hnsw::Graph;
    graph.max_level() + 1
}
