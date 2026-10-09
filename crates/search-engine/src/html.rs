//! Extracting indexable text, a title and outgoing links from an HTML page.

use reqwest::Url;
use scraper::{Html, Node, Selector};

pub struct ParsedPage {
    pub title: String,
    pub text: String,
    pub links: Vec<Url>,
}

/// Elements whose text is not page content.
const SKIPPED: &[&str] = &["script", "style", "noscript", "template", "svg", "head"];

pub fn parse(base: &Url, html: &str) -> ParsedPage {
    let doc = Html::parse_document(html);

    let title = first_text(&doc, "title")
        .or_else(|| first_text(&doc, "h1"))
        .unwrap_or_else(|| base.to_string());

    let mut text = String::new();
    for node in doc.root_element().descendants() {
        let Node::Text(t) = node.value() else {
            continue;
        };
        let hidden = node.ancestors().any(|a| {
            a.value()
                .as_element()
                .is_some_and(|e| SKIPPED.contains(&e.name()))
        });
        if !hidden {
            append_collapsed(&mut text, t);
        }
    }

    let anchors = Selector::parse("a[href]").unwrap();
    let links = doc
        .select(&anchors)
        .filter_map(|a| base.join(a.value().attr("href")?).ok())
        .map(|mut url| {
            url.set_fragment(None);
            url
        })
        .collect();

    ParsedPage {
        title,
        text: text.trim().to_string(),
        links,
    }
}

fn first_text(doc: &Html, selector: &str) -> Option<String> {
    let sel = Selector::parse(selector).unwrap();
    let mut out = String::new();
    for t in doc.select(&sel).next()?.text() {
        append_collapsed(&mut out, t);
    }
    let out = out.trim();
    (!out.is_empty()).then(|| out.to_string())
}

/// Appends `s` with runs of whitespace turned into single spaces. Adjacent
/// text nodes are separated by a space so `<li>a</li><li>b</li>` reads "a b".
fn append_collapsed(out: &mut String, s: &str) {
    if !out.is_empty() && !out.ends_with(' ') {
        out.push(' ');
    }
    let mut last_space = out.is_empty() || out.ends_with(' ');
    for c in s.chars() {
        if c.is_whitespace() {
            if !last_space {
                out.push(' ');
                last_space = true;
            }
        } else {
            out.push(c);
            last_space = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_title_text_and_links() {
        let base = Url::parse("https://example.com/docs/intro.html").unwrap();
        let page = parse(
            &base,
            r#"<html><head><title> My  Page </title><style>.x{}</style></head>
               <body><h1>Hello</h1><script>var x = 1;</script>
               <p>Some   <b>bold</b> text.</p>
               <a href="next.html#section">next</a><a href="/about">about</a></body></html>"#,
        );
        assert_eq!(page.title, "My Page");
        assert_eq!(page.text, "Hello Some bold text. next about");
        let links: Vec<&str> = page.links.iter().map(Url::as_str).collect();
        assert_eq!(
            links,
            [
                "https://example.com/docs/next.html",
                "https://example.com/about"
            ]
        );
    }
}
