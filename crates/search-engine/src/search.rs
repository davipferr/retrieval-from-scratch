//! Query execution: analyze the query, expand typos, score with BM25, pick the
//! top results and cut snippets around the matches.

use std::collections::{HashMap, HashSet};

use serde::Serialize;

use crate::bm25::Bm25;
use crate::index::Index;
use crate::levenshtein::{self, LevenshteinDfa};
use crate::tokenizer;

#[derive(Debug, Clone)]
pub struct SearchOptions {
    pub limit: usize,
    /// Also match index terms within a few edits of each query term.
    pub fuzzy: bool,
    /// At most this many fuzzy expansions per query term (closest and most common first).
    pub max_expansions: usize,
    pub bm25: Bm25,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            limit: 10,
            fuzzy: true,
            max_expansions: 20,
            bm25: Bm25::default(),
        }
    }
}

/// Fuzzy matches score less than exact ones: indexed by edit distance.
const DISTANCE_WEIGHT: [f32; 3] = [1.0, 0.6, 0.35];

#[derive(Debug, Clone, Serialize)]
pub struct Expansion {
    pub term: String,
    pub distance: u8,
    pub doc_freq: u32,
}

/// What one query word turned into.
#[derive(Debug, Clone, Serialize)]
pub struct QueryTerm {
    /// The analyzed (normalized + stemmed) query term.
    pub term: String,
    /// Index terms it matched; empty if none.
    pub matches: Vec<Expansion>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Hit {
    pub doc: u32,
    pub score: f32,
    pub url: String,
    pub title: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchResults {
    pub terms: Vec<QueryTerm>,
    /// Number of documents that matched at least one term.
    pub total: usize,
    pub hits: Vec<Hit>,
}

impl SearchResults {
    /// Every index term the query matched, for highlighting.
    pub fn matched_terms(&self) -> HashSet<String> {
        self.terms
            .iter()
            .flat_map(|t| t.matches.iter().map(|m| m.term.clone()))
            .collect()
    }
}

/// Finds the index terms a query term should match.
fn expand(index: &Index, term: &str, opts: &SearchOptions) -> Vec<(usize, u8)> {
    let max = if opts.fuzzy {
        levenshtein::auto_max_distance(term)
    } else {
        0
    };
    if max == 0 {
        return index.find(term).map(|i| vec![(i, 0)]).unwrap_or_default();
    }
    let dfa = LevenshteinDfa::new(term, max);
    let (matches, _) = levenshtein::fuzzy_automaton(index, &dfa);
    let mut found: Vec<(usize, u8)> = matches.into_iter().map(|m| (m.term, m.distance)).collect();
    // closest first, then the most common (most likely what was meant)
    found.sort_by_key(|&(i, d)| (d, std::cmp::Reverse(index.doc_freq(i))));
    found.truncate(opts.max_expansions);
    found
}

pub fn search(index: &Index, query: &str, opts: &SearchOptions) -> SearchResults {
    let mut query_terms: Vec<String> = tokenizer::analyze(query);
    if query_terms.is_empty() {
        // a query made only of stopwords ("the who"): search for them anyway
        query_terms = query
            .split_whitespace()
            .map(tokenizer::analyze_word)
            .collect();
    }
    let mut seen = HashSet::new();
    query_terms.retain(|t| !t.is_empty() && seen.insert(t.clone()));

    let n_docs = index.num_docs();
    let avg_len = index.avg_doc_len();
    let mut scores: HashMap<u32, f32> = HashMap::new();
    let mut terms = Vec::with_capacity(query_terms.len());

    for qt in query_terms {
        let expansions = expand(index, &qt, opts);

        // A doc may contain several expansions of the same query word ("search"
        // and "seared" for "serch"); count only the best one, or typo-heavy
        // queries would favor docs that happen to contain many near-misses.
        //
        // All expansions share one idf, computed from the most common of them
        // (Lucene's "blended" frequencies). Otherwise rare misspellings — which
        // is what most near-misses are — would get the highest idf and outrank
        // the word the user actually meant.
        let max_df = expansions.iter().map(|&(ti, _)| index.doc_freq(ti)).max();
        let idf = Bm25::idf(n_docs, max_df.unwrap_or(0));
        let mut best: HashMap<u32, f32> = HashMap::new();
        for &(ti, distance) in &expansions {
            let weight = DISTANCE_WEIGHT[distance as usize];
            for p in index.postings(ti) {
                let s = weight * opts.bm25.score(idf, p.tf, index.doc(p.doc).len, avg_len);
                let slot = best.entry(p.doc).or_default();
                *slot = slot.max(s);
            }
        }
        for (doc, s) in best {
            *scores.entry(doc).or_default() += s;
        }

        terms.push(QueryTerm {
            term: qt,
            matches: expansions
                .iter()
                .map(|&(ti, distance)| Expansion {
                    term: index.term(ti).to_string(),
                    distance,
                    doc_freq: index.doc_freq(ti),
                })
                .collect(),
        });
    }

    let total = scores.len();
    let mut ranked: Vec<(u32, f32)> = scores.into_iter().collect();
    let by_score = |a: &(u32, f32), b: &(u32, f32)| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0));
    // Partial selection: O(n) to find the top `limit`, then sort only those.
    if opts.limit < ranked.len() {
        ranked.select_nth_unstable_by(opts.limit, by_score);
        ranked.truncate(opts.limit);
    }
    ranked.sort_unstable_by(by_score);

    let hits = ranked
        .into_iter()
        .map(|(doc, score)| {
            let meta = index.doc(doc);
            Hit {
                doc,
                score,
                url: meta.url.clone(),
                title: meta.title.clone(),
            }
        })
        .collect();

    SearchResults { terms, total, hits }
}

/// Side-by-side comparison of the two typo-tolerance strategies for one word.
#[derive(Debug, Clone, Serialize)]
pub struct FuzzyReport {
    pub term: String,
    pub max_distance: u8,
    pub dictionary_size: usize,
    pub dfa_states: usize,
    pub dfa_build_us: f64,
    pub automaton_us: f64,
    pub terms_visited: usize,
    pub transitions: usize,
    pub skips: usize,
    pub bruteforce_us: f64,
    pub matches: Vec<Expansion>,
}

pub fn fuzzy_report(index: &Index, word: &str, max_distance: Option<u8>) -> FuzzyReport {
    use std::time::Instant;
    let term = tokenizer::analyze_word(word);
    let max = max_distance
        .unwrap_or_else(|| levenshtein::auto_max_distance(&term))
        .min(3);

    let t = Instant::now();
    let dfa = LevenshteinDfa::new(&term, max);
    let dfa_build_us = t.elapsed().as_secs_f64() * 1e6;

    let t = Instant::now();
    let (matches, stats) = levenshtein::fuzzy_automaton(index, &dfa);
    let automaton_us = t.elapsed().as_secs_f64() * 1e6;

    let t = Instant::now();
    let slow = levenshtein::fuzzy_bruteforce(index, &term, max);
    let bruteforce_us = t.elapsed().as_secs_f64() * 1e6;
    debug_assert_eq!(matches, slow, "both strategies must agree");

    let mut matches: Vec<Expansion> = matches
        .into_iter()
        .map(|m| Expansion {
            term: index.term(m.term).to_string(),
            distance: m.distance,
            doc_freq: index.doc_freq(m.term),
        })
        .collect();
    matches.sort_by(|a, b| {
        a.distance
            .cmp(&b.distance)
            .then(b.doc_freq.cmp(&a.doc_freq))
    });

    FuzzyReport {
        term,
        max_distance: max,
        dictionary_size: index.num_terms(),
        dfa_states: dfa.state_count(),
        dfa_build_us,
        automaton_us,
        terms_visited: stats.terms_visited,
        transitions: stats.transitions,
        skips: stats.skips,
        bruteforce_us,
        matches,
    }
}

/// How to mark matched words in a snippet.
#[derive(Debug, Clone, Copy)]
pub enum Highlight {
    /// `<mark>…</mark>`, with the rest of the text HTML-escaped.
    Html,
    /// Bold yellow for terminals.
    Ansi,
}

const SNIPPET_BYTES: usize = 240;
const SNIPPET_LEAD: usize = 60;

/// Cuts a window of the document around the densest cluster of matches.
pub fn snippet(index: &Index, doc: u32, matched: &HashSet<String>, style: Highlight) -> String {
    let text = index.doc_text(doc);
    let tokens = tokenizer::tokenize(text);
    let hits: Vec<usize> = (0..tokens.len())
        .filter(|&i| matched.contains(&tokens[i].term))
        .collect();

    // Sliding window over the match positions: the start whose window covers
    // the most matches wins.
    let mut best_start = 0;
    if let Some(&first) = hits.first() {
        let mut best_count = 0;
        let mut j = 0;
        for i in 0..hits.len() {
            let window_end = tokens[hits[i]].start + SNIPPET_BYTES - SNIPPET_LEAD;
            while j < hits.len() && tokens[hits[j]].end <= window_end {
                j += 1;
            }
            if j - i > best_count {
                best_count = j - i;
                best_start = tokens[hits[i]].start;
            }
        }
        if best_count == 0 {
            best_start = tokens[first].start;
        }
    }

    let mut start = best_start.saturating_sub(SNIPPET_LEAD);
    while !text.is_char_boundary(start) {
        start -= 1;
    }
    // start on a word boundary
    if start > 0
        && let Some(space) = text[start..best_start].find(char::is_whitespace)
    {
        start += space;
    }
    let mut end = (start + SNIPPET_BYTES).min(text.len());
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    if end < text.len()
        && let Some(space) = text[start..end].rfind(char::is_whitespace)
    {
        end = start + space.max(1);
    }

    let (open, close) = match style {
        Highlight::Html => ("<mark>", "</mark>"),
        Highlight::Ansi => ("\x1b[1;33m", "\x1b[0m"),
    };
    let mut out = String::new();
    if start > 0 {
        out.push('…');
    }
    let mut cursor = start;
    for t in tokens.iter().filter(|t| t.start >= start && t.end <= end) {
        if matched.contains(&t.term) {
            push_plain(&mut out, &text[cursor..t.start], style);
            out.push_str(open);
            push_plain(&mut out, &text[t.start..t.end], style);
            out.push_str(close);
            cursor = t.end;
        }
    }
    push_plain(&mut out, &text[cursor..end], style);
    if end < text.len() {
        out.push('…');
    }
    out
}

/// Appends text with whitespace runs collapsed (and HTML-escaped if needed).
fn push_plain(out: &mut String, s: &str, style: Highlight) {
    let mut last_space = out.ends_with(' ');
    for c in s.chars() {
        if c.is_whitespace() {
            if !last_space {
                out.push(' ');
            }
            last_space = true;
            continue;
        }
        last_space = false;
        match (style, c) {
            (Highlight::Html, '<') => out.push_str("&lt;"),
            (Highlight::Html, '>') => out.push_str("&gt;"),
            (Highlight::Html, '&') => out.push_str("&amp;"),
            (Highlight::Html, '"') => out.push_str("&quot;"),
            _ => out.push(c),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::IndexBuilder;

    fn build(name: &str, docs: &[(&str, &str)]) -> (Index, std::path::PathBuf) {
        let mut b = IndexBuilder::new();
        for (i, (title, body)) in docs.iter().enumerate() {
            b.add_document(&format!("doc{i}"), title, body);
        }
        let dir =
            std::env::temp_dir().join(format!("mini_search_search_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        b.write(&dir).unwrap();
        (Index::open(&dir).unwrap(), dir)
    }

    const DOCS: &[(&str, &str)] = &[
        (
            "Search engines",
            "How a search engine builds an inverted index and ranks documents.",
        ),
        (
            "Cooking",
            "Bake the bread for forty minutes. Searching for the perfect crust.",
        ),
        (
            "Gardening",
            "Water the tomatoes every morning, and prune them in spring.",
        ),
        (
            "Ranking",
            "BM25 ranking: term frequency saturation and document length normalization.",
        ),
    ];

    #[test]
    fn exact_search_ranks_title_match_first() {
        let (index, dir) = build("exact", DOCS);
        let opts = SearchOptions {
            fuzzy: false,
            ..Default::default()
        };
        let r = search(&index, "searching", &opts);
        let urls: Vec<&str> = r.hits.iter().map(|h| h.url.as_str()).collect();
        assert_eq!(urls, ["doc0", "doc1"]);
        assert_eq!(
            search(&index, "serch", &opts).total,
            0,
            "no typo tolerance without fuzzy"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn fuzzy_search_fixes_typos() {
        let (index, dir) = build("fuzzy", DOCS);
        let opts = SearchOptions::default();
        let r = search(&index, "serch engne", &opts);
        assert_eq!(r.hits[0].url, "doc0");
        let r = search(&index, "tomtoes", &opts);
        assert_eq!(r.hits[0].url, "doc2");
        let r = search(&index, "rankng", &opts);
        assert_eq!(r.hits[0].url, "doc3");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn snippets_highlight_matches() {
        let (index, dir) = build("snippet", DOCS);
        let r = search(&index, "bread", &SearchOptions::default());
        let s = snippet(&index, r.hits[0].doc, &r.matched_terms(), Highlight::Html);
        assert!(s.contains("<mark>bread</mark>"), "{s}");
        std::fs::remove_dir_all(dir).ok();
    }
}
