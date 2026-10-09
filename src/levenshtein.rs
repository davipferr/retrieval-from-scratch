//! Typo tolerance, in two steps.
//!
//! **Step 1 — Levenshtein distance.** The minimum number of single-character
//! insertions, deletions and substitutions to turn one word into another,
//! computed with the classic dynamic-programming table. Finding all index terms
//! within distance 2 of a query word this way means computing the full table
//! against *every* term in the dictionary.
//!
//! **Step 2 — Levenshtein automaton.** Build a DFA that accepts exactly the
//! strings within distance `k` of the query word, then walk it in lock-step
//! with the *sorted* term dictionary. Two things make it fast:
//!
//! - terms sharing a prefix share the DFA walk for that prefix, and
//! - as soon as the DFA reaches its dead state after reading a prefix, no
//!   term starting with that prefix can match, so we binary-search past all of
//!   them at once instead of looking at each.
//!
//! This is the technique Lucene (and Tantivy) use for fuzzy queries; they
//! intersect the automaton with an FST rather than a sorted array, but the
//! idea — prune whole subtrees of the dictionary on dead states — is the same.
//!
//! How the DFA is built here: a row of the DP table *is* the automaton's state.
//! Reading one more character computes the next row. Values above `k` are
//! clamped to `k + 1` (they can never come back down to ≤ k), so there are
//! finitely many distinct rows, and we can enumerate them all up front. And
//! since every character that does not occur in the query behaves the same
//! way, the alphabet is just "the query's characters + one *other* symbol".
//! See Jules Jacobs, "Levenshtein automata can be simple and fast" (2015).

use std::collections::HashMap;

/// Plain edit distance with two rolling rows. O(|a|·|b|) time, O(|b|) memory.
pub fn distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0; b.len() + 1];
    for (i, ca) in a.chars().enumerate() {
        cur[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            let substitution = prev[j] + usize::from(ca != cb);
            let deletion = prev[j + 1] + 1;
            let insertion = cur[j] + 1;
            cur[j + 1] = substitution.min(deletion).min(insertion);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// The "how many typos do we allow" rule Elasticsearch calls `AUTO`:
/// exact for short words, 1 edit for medium ones, 2 for long ones.
pub fn auto_max_distance(word: &str) -> u8 {
    match word.chars().count() {
        0..=2 => 0,
        3..=5 => 1,
        _ => 2,
    }
}

/// Non-deterministic view of the automaton: states are DP rows, computed on demand.
struct RowAutomaton<'a> {
    query: &'a [char],
    max: u8,
}

impl RowAutomaton<'_> {
    /// Row 0 of the DP table: distance from the empty string to each query prefix.
    fn start(&self) -> Vec<u8> {
        (0..=self.query.len())
            .map(|i| (i as u8).min(self.max + 1))
            .collect()
    }

    /// Next DP row after reading `c` (or "some character not in the query" if `None`).
    fn step(&self, row: &[u8], c: Option<char>) -> Vec<u8> {
        let cap = self.max + 1;
        let mut next = Vec::with_capacity(row.len());
        next.push((row[0] + 1).min(cap));
        for (i, &qc) in self.query.iter().enumerate() {
            let cost = u8::from(Some(qc) != c);
            let value = (row[i] + cost).min(row[i + 1] + 1).min(next[i] + 1);
            next.push(value.min(cap));
        }
        next
    }
}

/// Sentinel meaning "this prefix can no longer match".
pub const DEAD: u32 = 0;

/// A Levenshtein automaton compiled to a DFA with a dense transition table.
pub struct LevenshteinDfa {
    /// Distinct characters of the query, sorted; symbol `i` is `alphabet[i]`,
    /// symbol `alphabet.len()` stands for every other character.
    alphabet: Vec<char>,
    /// `transitions[state * n_symbols + symbol]` -> next state.
    transitions: Vec<u32>,
    /// For each state, the edit distance if it is accepting.
    accept: Vec<Option<u8>>,
    pub max_distance: u8,
}

impl LevenshteinDfa {
    pub fn new(query: &str, max_distance: u8) -> Self {
        let query: Vec<char> = query.chars().collect();
        assert!(query.len() < 250, "query word too long for u8 rows");
        let nfa = RowAutomaton {
            query: &query,
            max: max_distance,
        };

        let mut alphabet = query.clone();
        alphabet.sort_unstable();
        alphabet.dedup();
        let symbols: Vec<Option<char>> = alphabet
            .iter()
            .copied()
            .map(Some)
            .chain(std::iter::once(None))
            .collect();

        // State 0 is the dead state: every cell above the limit.
        let dead_row = vec![max_distance + 1; query.len() + 1];
        let mut ids: HashMap<Vec<u8>, u32> = HashMap::new();
        let mut rows: Vec<Vec<u8>> = Vec::new();
        ids.insert(dead_row.clone(), DEAD);
        rows.push(dead_row);

        // State 1 is the start state (row 0 begins with a 0, so it is never dead).
        let start = nfa.start();
        ids.insert(start.clone(), 1);
        rows.push(start);

        // Breadth-first exploration of every reachable row.
        let mut transitions: Vec<u32> = Vec::new();
        let mut next_unprocessed = 0;
        while next_unprocessed < rows.len() {
            let row = rows[next_unprocessed].clone();
            for &symbol in &symbols {
                let next = nfa.step(&row, symbol);
                let id = match ids.get(&next) {
                    Some(&id) => id,
                    None => {
                        let id = rows.len() as u32;
                        ids.insert(next.clone(), id);
                        rows.push(next);
                        id
                    }
                };
                transitions.push(id);
            }
            next_unprocessed += 1;
        }

        let accept = rows
            .iter()
            .map(|row| {
                let last = *row.last().unwrap();
                (last <= max_distance).then_some(last)
            })
            .collect();

        Self {
            alphabet,
            transitions,
            accept,
            max_distance,
        }
    }

    pub fn start(&self) -> u32 {
        1
    }

    pub fn step(&self, state: u32, c: char) -> u32 {
        let n_symbols = self.alphabet.len() + 1;
        let symbol = self
            .alphabet
            .binary_search(&c)
            .unwrap_or(self.alphabet.len());
        self.transitions[state as usize * n_symbols + symbol]
    }

    /// `Some(distance)` if the input read so far is within the limit.
    pub fn accepts(&self, state: u32) -> Option<u8> {
        self.accept[state as usize]
    }

    pub fn state_count(&self) -> usize {
        self.accept.len()
    }

    /// Runs the whole word through the DFA.
    pub fn eval(&self, word: &str) -> Option<u8> {
        let mut state = self.start();
        for c in word.chars() {
            state = self.step(state, c);
            if state == DEAD {
                return None;
            }
        }
        self.accepts(state)
    }
}

/// Anything that exposes a sorted list of terms by index.
pub trait SortedTerms {
    fn len(&self) -> usize;
    fn term(&self, i: usize) -> &str;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Index of the first term that is greater than every term starting with
    /// `prefix`, searching forward from `from` (every term before `from` must
    /// sort before or inside the prefix block).
    ///
    /// Terms sort as `[< prefix] [starts with prefix] [> prefix]`, so "is this
    /// term past the block?" is false-then-true: a binary search works. Most
    /// blocks we skip are short, so we gallop first (probe from+1, +2, +4, …)
    /// to bracket the answer near `from`, then binary-search inside the bracket:
    /// O(log distance) instead of O(log len).
    fn skip_prefix(&self, prefix: &str, from: usize) -> usize {
        let past = |i: usize| {
            let t = self.term(i);
            !(t < prefix || t.starts_with(prefix))
        };
        let len = self.len();
        let (mut lo, mut hi, mut step) = (from, from, 1);
        loop {
            if hi >= len {
                hi = len;
                break;
            }
            if past(hi) {
                break;
            }
            lo = hi + 1;
            hi += step;
            step *= 2;
        }
        while lo < hi {
            let mid = (lo + hi) / 2;
            if past(mid) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        lo
    }
}

impl SortedTerms for [String] {
    fn len(&self) -> usize {
        <[String]>::len(self)
    }
    fn term(&self, i: usize) -> &str {
        &self[i]
    }
}

/// A fuzzy match: the term's position in the dictionary and its edit distance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FuzzyMatch {
    pub term: usize,
    pub distance: u8,
}

/// Counters that show how much work the automaton saved.
#[derive(Debug, Default, Clone, Copy)]
pub struct FuzzyStats {
    /// Terms whose characters were (at least partly) fed to the DFA.
    pub terms_visited: usize,
    /// DFA transitions taken.
    pub transitions: usize,
    /// Times a dead state let us jump over a block of terms.
    pub skips: usize,
}

/// Step 1: compare the query against every term with the DP algorithm.
pub fn fuzzy_bruteforce<T: SortedTerms + ?Sized>(
    terms: &T,
    query: &str,
    max: u8,
) -> Vec<FuzzyMatch> {
    (0..terms.len())
        .filter_map(|i| {
            let d = distance(query, terms.term(i));
            (d <= max as usize).then_some(FuzzyMatch {
                term: i,
                distance: d as u8,
            })
        })
        .collect()
}

/// Step 2: intersect the DFA with the sorted dictionary.
pub fn fuzzy_automaton<T: SortedTerms + ?Sized>(
    terms: &T,
    dfa: &LevenshteinDfa,
) -> (Vec<FuzzyMatch>, FuzzyStats) {
    let mut out = Vec::new();
    let mut stats = FuzzyStats::default();

    // `states[n]` is the DFA state after reading the first n chars of `prefix`.
    // Consecutive sorted terms share long prefixes, so most of this stack is reused.
    let mut prefix: Vec<char> = Vec::new();
    let mut states: Vec<u32> = vec![dfa.start()];
    let mut prefix_buf = String::new();

    let mut i = 0;
    while i < terms.len() {
        let term = terms.term(i);
        stats.terms_visited += 1;

        // Rewind to the longest prefix this term shares with the previous one.
        let common = prefix
            .iter()
            .zip(term.chars())
            .take_while(|(a, b)| *a == b)
            .count();
        prefix.truncate(common);
        states.truncate(common + 1);

        let mut dead = false;
        for c in term.chars().skip(common) {
            let next = dfa.step(*states.last().unwrap(), c);
            stats.transitions += 1;
            prefix.push(c);
            if next == DEAD {
                dead = true;
                break;
            }
            states.push(next);
        }

        if dead {
            // Nothing that starts with `prefix` can match: jump past all of it.
            prefix_buf.clear();
            prefix_buf.extend(prefix.iter());
            i = terms.skip_prefix(&prefix_buf, i + 1);
            prefix.pop(); // keep `prefix` in sync with `states`
            stats.skips += 1;
        } else {
            if let Some(distance) = dfa.accepts(*states.last().unwrap()) {
                out.push(FuzzyMatch { term: i, distance });
            }
            i += 1;
        }
    }
    (out, stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classic_distances() {
        assert_eq!(distance("kitten", "sitting"), 3);
        assert_eq!(distance("flaw", "lawn"), 2);
        assert_eq!(distance("", "abc"), 3);
        assert_eq!(distance("abc", ""), 3);
        assert_eq!(distance("same", "same"), 0);
        assert_eq!(distance("ação", "acao"), 2);
    }

    #[test]
    fn dfa_agrees_with_distance() {
        let words = [
            "",
            "a",
            "ab",
            "search",
            "serch",
            "seach",
            "saerch",
            "searches",
            "research",
            "starch",
            "march",
            "s",
            "sea",
            "searchengine",
            "xyz",
            "seárch",
        ];
        for query in ["search", "a", "", "abc", "seárch"] {
            for max in 0..=2 {
                let dfa = LevenshteinDfa::new(query, max);
                for w in words {
                    let d = distance(query, w);
                    let expected = (d <= max as usize).then_some(d as u8);
                    assert_eq!(
                        dfa.eval(w),
                        expected,
                        "query={query:?} word={w:?} max={max}"
                    );
                }
            }
        }
    }

    #[test]
    fn automaton_matches_bruteforce_on_dictionary() {
        let mut terms: Vec<String> = [
            "apple", "apply", "ape", "banana", "band", "bandana", "can", "cane", "candy", "search",
            "searched", "seared", "serch", "starch", "zebra", "", "sea", "sear",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        terms.sort();
        terms.dedup();
        for query in ["serch", "band", "ape", "zebr", "q"] {
            for max in 0..=2 {
                let dfa = LevenshteinDfa::new(query, max);
                let (fast, _) = fuzzy_automaton(terms.as_slice(), &dfa);
                let slow = fuzzy_bruteforce(terms.as_slice(), query, max);
                assert_eq!(fast, slow, "query={query} max={max}");
            }
        }
    }

    #[test]
    fn skip_prefix() {
        let terms: Vec<String> = ["a", "ba", "bb", "bba", "bbz", "bc", "c"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let t = terms.as_slice();
        assert_eq!(t.skip_prefix("bb", 0), 5);
        assert_eq!(t.skip_prefix("bb", 2), 5);
        assert_eq!(t.skip_prefix("bb", 4), 5);
        assert_eq!(t.skip_prefix("b", 1), 6);
        assert_eq!(t.skip_prefix("x", 0), 7);
        assert_eq!(t.skip_prefix("0", 0), 0);
        assert_eq!(t.skip_prefix("c", 6), 7);
    }
}
