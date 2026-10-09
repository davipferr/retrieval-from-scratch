//! HTTP API for the RAG app (or anything else).
//!
//! - `GET /api/search?q=...&k=8&mode=hybrid|bm25|vector`
//! - `GET /api/stats`
//! - `GET /health`

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::Result;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::routing::get;
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use crate::retriever::{Mode, Retriever, SearchResponse};

type AppState = Arc<Retriever>;

pub async fn serve(retriever: Retriever, addr: SocketAddr) -> Result<()> {
    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/api/search", get(search))
        .route("/api/stats", get(stats))
        .with_state(Arc::new(retriever));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    println!("listening on http://{}", listener.local_addr()?);
    axum::serve(listener, app).await?;
    Ok(())
}

#[derive(Deserialize)]
struct SearchParams {
    q: String,
    k: Option<usize>,
    #[serde(default)]
    mode: Mode,
}

type ApiError = (StatusCode, Json<serde_json::Value>);

fn error(status: StatusCode, msg: impl ToString) -> ApiError {
    (
        status,
        Json(serde_json::json!({ "error": msg.to_string() })),
    )
}

async fn search(
    State(retriever): State<AppState>,
    Query(p): Query<SearchParams>,
) -> Result<Json<SearchResponse>, ApiError> {
    if p.q.trim().is_empty() {
        return Err(error(StatusCode::BAD_REQUEST, "q must not be empty"));
    }
    let k = p.k.unwrap_or(8).clamp(1, 50);
    // Embedding the query runs the transformer: CPU-bound work, so keep it
    // off the async worker threads.
    let result = tokio::task::spawn_blocking(move || retriever.search(&p.q, k, p.mode))
        .await
        .map_err(|e| error(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    result
        .map(Json)
        .map_err(|e| error(StatusCode::INTERNAL_SERVER_ERROR, e))
}

#[derive(Serialize)]
struct Stats {
    chunks: usize,
    files: usize,
    dim: usize,
}

async fn stats(State(r): State<AppState>) -> Json<Stats> {
    Json(Stats {
        chunks: r.len(),
        files: r.files(),
        dim: r.dim(),
    })
}
