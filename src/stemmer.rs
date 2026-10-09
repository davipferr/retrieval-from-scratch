//! The Porter stemming algorithm (M. F. Porter, 1980), for English.
//!
//! Stemming maps inflected words onto a common root so that a query for
//! "searching" also finds "searched" and "searches" (all become `search`).
//! The stems are not always real words (`happy` -> `happi`), which is fine:
//! they only have to be consistent between indexing and querying.
//!
//! This follows Porter's reference C implementation, including its two
//! documented departures from the paper (`-bli` and `-logi` in step 2).
//!
//! Vocabulary used in the comments:
//! - a *consonant* is a letter other than a, e, i, o, u, and other than `y`
//!   when it follows a consonant;
//! - every word has the form `[C](VC)^m[V]`; `m` is the *measure*.
//!   `tree` m=0, `trouble` m=1, `troubles`/`private` m=2.

pub fn stem(word: &str) -> String {
    // Only ASCII lowercase words are stemmed; everything else (numbers,
    // other scripts, very short words) passes through unchanged.
    if word.len() <= 2 || !word.bytes().all(|b| b.is_ascii_lowercase()) {
        return word.to_string();
    }
    let mut s = Stemmer {
        b: word.as_bytes().to_vec(),
        k: word.len() as isize - 1,
        j: 0,
    };
    s.step1ab();
    if s.k > 0 {
        s.step1c();
        s.step2();
        s.step3();
        s.step4();
        s.step5();
    }
    s.b.truncate((s.k + 1) as usize);
    String::from_utf8(s.b).expect("stemmer only touches ASCII")
}

struct Stemmer {
    b: Vec<u8>,
    /// Index of the last character of the current word (it shrinks as suffixes are removed).
    k: isize,
    /// Set by `ends`: index of the last character before the matched suffix.
    j: isize,
}

impl Stemmer {
    fn at(&self, i: isize) -> u8 {
        self.b[i as usize]
    }

    fn is_consonant(&self, i: isize) -> bool {
        match self.at(i) {
            b'a' | b'e' | b'i' | b'o' | b'u' => false,
            b'y' => i == 0 || !self.is_consonant(i - 1),
            _ => true,
        }
    }

    /// The measure `m` of `b[0..=j]`.
    fn m(&self) -> usize {
        let mut n = 0;
        let mut i = 0;
        // skip the optional leading consonants
        loop {
            if i > self.j {
                return n;
            }
            if !self.is_consonant(i) {
                break;
            }
            i += 1;
        }
        i += 1;
        loop {
            // vowels
            loop {
                if i > self.j {
                    return n;
                }
                if self.is_consonant(i) {
                    break;
                }
                i += 1;
            }
            i += 1;
            n += 1;
            // consonants
            loop {
                if i > self.j {
                    return n;
                }
                if !self.is_consonant(i) {
                    break;
                }
                i += 1;
            }
            i += 1;
        }
    }

    /// Does `b[0..=j]` contain a vowel?
    fn vowel_in_stem(&self) -> bool {
        (0..=self.j).any(|i| !self.is_consonant(i))
    }

    /// Does `b[i-1..=i]` hold a double consonant?
    fn double_consonant(&self, i: isize) -> bool {
        i >= 1 && self.at(i) == self.at(i - 1) && self.is_consonant(i)
    }

    /// Is `b[i-2..=i]` consonant-vowel-consonant, where the last consonant is
    /// not w, x or y? Used to restore an `e` (`hop` vs `hope`).
    fn cvc(&self, i: isize) -> bool {
        if i < 2 || !self.is_consonant(i) || self.is_consonant(i - 1) || !self.is_consonant(i - 2) {
            return false;
        }
        !matches!(self.at(i), b'w' | b'x' | b'y')
    }

    /// Does the word end with `suffix`? If so, sets `j` to just before it.
    fn ends(&mut self, suffix: &str) -> bool {
        let len = suffix.len() as isize;
        if len > self.k + 1 {
            return false;
        }
        let start = (self.k - len + 1) as usize;
        if &self.b[start..=self.k as usize] != suffix.as_bytes() {
            return false;
        }
        self.j = self.k - len;
        true
    }

    /// Replaces `b[j+1..=k]` with `s`.
    fn set_to(&mut self, s: &str) {
        let start = (self.j + 1) as usize;
        self.b.truncate(start);
        self.b.extend_from_slice(s.as_bytes());
        self.k = self.j + s.len() as isize;
    }

    /// Replaces the matched suffix with `s`, but only if the stem has m > 0.
    fn r(&mut self, s: &str) {
        if self.m() > 0 {
            self.set_to(s);
        }
    }

    /// Plurals and -ed / -ing:
    /// caresses -> caress, ponies -> poni, cats -> cat,
    /// agreed -> agree, plastered -> plaster, motoring -> motor,
    /// conflated -> conflate, hopping -> hop, filing -> file.
    fn step1ab(&mut self) {
        if self.at(self.k) == b's' {
            if self.ends("sses") {
                self.k -= 2;
            } else if self.ends("ies") {
                self.set_to("i");
            } else if self.at(self.k - 1) != b's' {
                self.k -= 1;
            }
        }
        if self.ends("eed") {
            if self.m() > 0 {
                self.k -= 1;
            }
        } else if (self.ends("ed") || self.ends("ing")) && self.vowel_in_stem() {
            self.k = self.j;
            if self.ends("at") {
                self.set_to("ate");
            } else if self.ends("bl") {
                self.set_to("ble");
            } else if self.ends("iz") {
                self.set_to("ize");
            } else if self.double_consonant(self.k) {
                self.k -= 1;
                if matches!(self.at(self.k), b'l' | b's' | b'z') {
                    self.k += 1;
                }
            } else if self.m() == 1 && self.cvc(self.k) {
                self.set_to("e");
            }
        }
    }

    /// Terminal y -> i when there is another vowel in the stem: happy -> happi.
    fn step1c(&mut self) {
        if self.ends("y") && self.vowel_in_stem() {
            self.b[self.k as usize] = b'i';
        }
    }

    /// Double suffixes to single ones: relational -> relate, digitizer -> digitize.
    fn step2(&mut self) {
        const RULES: &[(&str, &str)] = &[
            ("ational", "ate"),
            ("tional", "tion"),
            ("enci", "ence"),
            ("anci", "ance"),
            ("izer", "ize"),
            ("bli", "ble"),
            ("alli", "al"),
            ("entli", "ent"),
            ("eli", "e"),
            ("ousli", "ous"),
            ("ization", "ize"),
            ("ation", "ate"),
            ("ator", "ate"),
            ("alism", "al"),
            ("iveness", "ive"),
            ("fulness", "ful"),
            ("ousness", "ous"),
            ("aliti", "al"),
            ("iviti", "ive"),
            ("biliti", "ble"),
            ("logi", "log"),
        ];
        self.apply_first(RULES);
    }

    /// -ic-, -full, -ness etc.: electrical -> electric, hopeful -> hope.
    fn step3(&mut self) {
        const RULES: &[(&str, &str)] = &[
            ("icate", "ic"),
            ("ative", ""),
            ("alize", "al"),
            ("iciti", "ic"),
            ("ical", "ic"),
            ("ful", ""),
            ("ness", ""),
        ];
        self.apply_first(RULES);
    }

    /// Applies the first rule whose suffix matches (the replacement still needs m > 0).
    fn apply_first(&mut self, rules: &[(&str, &str)]) {
        for (suffix, replacement) in rules {
            if self.ends(suffix) {
                self.r(replacement);
                return;
            }
        }
    }

    /// Removes -ant, -ence etc. when m > 1: adjustment -> adjust, revival -> reviv.
    fn step4(&mut self) {
        const SUFFIXES: &[&str] = &[
            "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment", "ent", "ion",
            "ou", "ism", "ate", "iti", "ous", "ive", "ize",
        ];
        for suffix in SUFFIXES {
            if self.ends(suffix) {
                // -ion is only removed after s or t (adoption -> adopt, but not onion)
                if *suffix == "ion" && !(self.j >= 0 && matches!(self.at(self.j), b's' | b't')) {
                    continue;
                }
                if self.m() > 1 {
                    self.k = self.j;
                }
                return;
            }
        }
    }

    /// Removes a final -e and reduces -ll: probate -> probat, controll -> control.
    fn step5(&mut self) {
        self.j = self.k;
        if self.at(self.k) == b'e' {
            let m = self.m();
            if m > 1 || (m == 1 && !self.cvc(self.k - 1)) {
                self.k -= 1;
            }
        }
        if self.at(self.k) == b'l' && self.double_consonant(self.k) && self.m() > 1 {
            self.k -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::stem;

    #[test]
    fn porter_examples() {
        let cases = [
            ("caresses", "caress"),
            ("ponies", "poni"),
            ("ties", "ti"),
            ("caress", "caress"),
            ("cats", "cat"),
            ("feed", "feed"),
            ("agreed", "agre"),
            ("plastered", "plaster"),
            ("motoring", "motor"),
            ("sing", "sing"),
            ("conflated", "conflat"),
            ("troubled", "troubl"),
            ("sized", "size"),
            ("hopping", "hop"),
            ("filing", "file"),
            ("happy", "happi"),
            ("relational", "relat"),
            ("generalization", "gener"),
            ("running", "run"),
            ("searching", "search"),
            ("searches", "search"),
            ("engine", "engin"),
            ("engines", "engin"),
            ("connection", "connect"),
            ("connected", "connect"),
            ("adjustment", "adjust"),
            ("hopeful", "hope"),
            ("electrical", "electr"),
            ("onion", "onion"),
            ("controlling", "control"),
        ];
        for (word, expected) in cases {
            assert_eq!(stem(word), expected, "stem({word})");
        }
    }

    #[test]
    fn leaves_non_ascii_and_short_words_alone() {
        assert_eq!(stem("is"), "is");
        assert_eq!(stem("2024"), "2024");
        assert_eq!(stem("日本語"), "日本語");
    }
}
