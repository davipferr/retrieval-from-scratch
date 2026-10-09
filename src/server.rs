//! HTTP API (axum) over a memory-mapped index.
//!
//! - `GET /`                                  search page
//! - `GET /api/search?q=...&limit=10&fuzzy=true`
//! - `GET /api/fuzzy?term=...`                 automaton vs brute force for one word
//! - `GET /api/stats`

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use axum::extract::{Query, State};
use axum::response::Html;
use axum::routing::get;
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use crate::index::Index;
use crate::search::{self, FuzzyReport, Highlight, QueryTerm, SearchOptions};

type AppState = Arc<Index>;

pub async fn serve(index: Index, addr: SocketAddr) -> Result<()> {
    let app = Router::new()
        .route("/", get(|| async { Html(include_str!("ui.html")) }))
        .route("/api/search", get(search_handler))
        .route("/api/fuzzy", get(fuzzy_handler))
        .route("/api/stats", get(stats_handler))
        .with_state(Arc::new(index));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    println!("listening on http://{}", listener.local_addr()?);
    axum::serve(listener, app).await?;
    Ok(())
}

#[derive(Deserialize)]
struct SearchParams {
    q: String,
    limit: Option<usize>,
    fuzzy: Option<bool>,
}

#[derive(Serialize)]
struct ApiHit {
    url: String,
    title: String,
    score: f32,
    /// HTML: escaped text with `<mark>` around matches.
    snippet: String,
}

#[derive(Serialize)]
struct SearchResponse {
    query: String,
    took_ms: f64,
    total: usize,
    terms: Vec<QueryTerm>,
    hits: Vec<ApiHit>,
}

async fn search_handler(
    State(index): State<AppState>,
    Query(p): Query<SearchParams>,
) -> Json<SearchResponse> {
    let started = Instant::now();
    let opts = SearchOptions {
        limit: p.limit.unwrap_or(10).clamp(1, 100),
        fuzzy: p.fuzzy.unwrap_or(true),
        ..Default::default()
    };
    let results = search::search(&index, &p.q, &opts);
    let matched = results.matched_terms();
    let hits = results
        .hits
        .iter()
        .map(|h| ApiHit {
            url: h.url.clone(),
            title: h.title.clone(),
            score: h.score,
            snippet: search::snippet(&index, h.doc, &matched, Highlight::Html),
        })
        .collect();
    Json(SearchResponse {
        query: p.q,
        took_ms: started.elapsed().as_secs_f64() * 1000.0,
        total: results.total,
        terms: results.terms,
        hits,
    })
}

#[derive(Deserialize)]
struct FuzzyParams {
    term: String,
    max: Option<u8>,
}

async fn fuzzy_handler(
    State(index): State<AppState>,
    Query(p): Query<FuzzyParams>,
) -> Json<FuzzyReport> {
    Json(search::fuzzy_report(&index, &p.term, p.max))
}

#[derive(Serialize)]
struct Stats {
    docs: usize,
    terms: usize,
    avg_doc_len: f32,
}

async fn stats_handler(State(index): State<AppState>) -> Json<Stats> {
    Json(Stats {
        docs: index.num_docs(),
        terms: index.num_terms(),
        avg_doc_len: index.avg_doc_len(),
    })
}
