//! Text analysis: turns raw text into the terms stored in the index.
//!
//! The pipeline, applied identically to documents and to queries:
//!
//! ```text
//! "Searching São Paulo's cafés!"
//!   split     -> ["Searching", "São", "Paulo", "s", "cafés"]
//!   lowercase -> ["searching", "são", "paulo", "s", "cafés"]
//!   fold      -> ["searching", "sao", "paulo", "s", "cafes"]
//!   stopwords -> ["searching", "sao", "paulo", "cafes"]
//!   stem      -> ["search", "sao", "paulo", "cafe"]
//! ```
//!
//! If indexing and querying did not apply exactly the same steps, "Café" in a
//! document would never match "cafe" in a query.

use crate::stemmer;

/// Tokens longer than this are almost always junk (hashes, base64, minified code).
const MAX_TOKEN_CHARS: usize = 40;

const STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it",
    "no", "not", "of", "on", "or", "s", "such", "that", "the", "their", "then", "there", "these",
    "they", "this", "to", "was", "will", "with",
];

/// A term together with where it came from in the original text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub term: String,
    /// Byte range of the original word in the input text.
    pub start: usize,
    pub end: usize,
}

/// Splits, normalizes, drops stopwords and stems.
pub fn tokenize(text: &str) -> Vec<Token> {
    words(text)
        .filter_map(|(start, end)| {
            let word = &text[start..end];
            if word.chars().count() > MAX_TOKEN_CHARS {
                return None;
            }
            let normalized = normalize(word);
            if is_stopword(&normalized) {
                return None;
            }
            Some(Token {
                term: stemmer::stem(&normalized),
                start,
                end,
            })
        })
        .collect()
}

/// Just the terms, for when positions are not needed.
pub fn analyze(text: &str) -> Vec<String> {
    tokenize(text).into_iter().map(|t| t.term).collect()
}

/// Analyzes a single query word (no stopword removal, so a lone "the" still
/// produces something to look up).
pub fn analyze_word(word: &str) -> String {
    stemmer::stem(&normalize(word))
}

pub fn is_stopword(normalized: &str) -> bool {
    STOPWORDS.binary_search(&normalized).is_ok()
}

/// Byte ranges of maximal runs of alphanumeric characters.
/// Everything else — spaces, punctuation, `_`, `-` — is a separator, which also
/// splits `snake_case` and `kebab-case` identifiers into their parts.
fn words(text: &str) -> impl Iterator<Item = (usize, usize)> + '_ {
    let mut chars = text.char_indices().peekable();
    std::iter::from_fn(move || {
        // skip separators
        while let Some(&(_, c)) = chars.peek() {
            if c.is_alphanumeric() {
                break;
            }
            chars.next();
        }
        let (start, _) = *chars.peek()?;
        let mut end = start;
        while let Some(&(i, c)) = chars.peek() {
            if !c.is_alphanumeric() {
                break;
            }
            end = i + c.len_utf8();
            chars.next();
        }
        Some((start, end))
    })
}

/// Lowercases and strips diacritics: `"Ação"` -> `"acao"`, `"Straße"` -> `"strasse"`.
pub fn normalize(word: &str) -> String {
    let mut out = String::with_capacity(word.len());
    for c in word.chars().flat_map(char::to_lowercase) {
        fold_char(c, &mut out);
    }
    out
}

/// Maps an accented Latin letter to its plain ASCII spelling.
///
/// Real engines use Unicode decomposition (NFD) and drop the combining marks;
/// a lookup table covering Latin-1 and Latin Extended-A is enough for
/// Portuguese, Spanish, French, German and most of Europe.
fn fold_char(c: char, out: &mut String) {
    let folded = match c {
        'à' | 'á' | 'â' | 'ã' | 'ä' | 'å' | 'ā' | 'ă' | 'ą' => "a",
        'ç' | 'ć' | 'ĉ' | 'ċ' | 'č' => "c",
        'ď' | 'đ' | 'ð' => "d",
        'è' | 'é' | 'ê' | 'ë' | 'ē' | 'ĕ' | 'ė' | 'ę' | 'ě' => "e",
        'ĝ' | 'ğ' | 'ġ' | 'ģ' => "g",
        'ĥ' | 'ħ' => "h",
        'ì' | 'í' | 'î' | 'ï' | 'ĩ' | 'ī' | 'ĭ' | 'į' | 'ı' => "i",
        'ĵ' => "j",
        'ķ' => "k",
        'ĺ' | 'ļ' | 'ľ' | 'ŀ' | 'ł' => "l",
        'ñ' | 'ń' | 'ņ' | 'ň' => "n",
        'ò' | 'ó' | 'ô' | 'õ' | 'ö' | 'ø' | 'ō' | 'ŏ' | 'ő' => "o",
        'ŕ' | 'ŗ' | 'ř' => "r",
        'ś' | 'ŝ' | 'ş' | 'š' => "s",
        'ţ' | 'ť' | 'ŧ' => "t",
        'ù' | 'ú' | 'û' | 'ü' | 'ũ' | 'ū' | 'ŭ' | 'ů' | 'ű' | 'ų' => "u",
        'ŵ' => "w",
        'ý' | 'ÿ' | 'ŷ' => "y",
        'ź' | 'ż' | 'ž' => "z",
        'ß' => "ss",
        'æ' => "ae",
        'œ' => "oe",
        'þ' => "th",
        _ => {
            out.push(c);
            return;
        }
    };
    out.push_str(folded);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stopwords_are_sorted() {
        assert!(
            STOPWORDS.windows(2).all(|w| w[0] < w[1]),
            "binary_search needs a sorted list"
        );
    }

    #[test]
    fn normalization() {
        assert_eq!(normalize("Ação"), "acao");
        assert_eq!(normalize("CAFÉ"), "cafe");
        assert_eq!(normalize("Straße"), "strasse");
        assert_eq!(normalize("Ünïcödé"), "unicode");
    }

    #[test]
    fn full_pipeline() {
        assert_eq!(
            analyze("Searching São Paulo's cafés!"),
            ["search", "sao", "paulo", "cafe"]
        );
        assert_eq!(
            analyze("the snake_case and kebab-case"),
            ["snake", "case", "kebab", "case"]
        );
    }

    #[test]
    fn offsets_point_at_original_text() {
        let text = "  Olá, mundo";
        let tokens = tokenize(text);
        assert_eq!(&text[tokens[0].start..tokens[0].end], "Olá");
        assert_eq!(&text[tokens[1].start..tokens[1].end], "mundo");
    }
}
