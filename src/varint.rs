//! Variable-length integer encoding (unsigned LEB128).
//!
//! Each byte carries 7 bits of payload. The high bit says "more bytes follow".
//! Small numbers (which is what delta-encoded doc ids mostly are) take 1 byte
//! instead of 4:
//!
//! ```text
//!        5 -> 0000_0101
//!      300 -> 1010_1100 0000_0010   (low 7 bits first, continuation bit set)
//! ```

/// Appends `value` to `out` as a varint. Returns the number of bytes written.
pub fn encode(mut value: u64, out: &mut Vec<u8>) -> usize {
    let mut written = 0;
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        written += 1;
        if value == 0 {
            out.push(byte);
            return written;
        }
        out.push(byte | 0x80);
    }
}

/// Decodes one varint from the start of `buf`.
/// Returns the value and how many bytes it used, or `None` if `buf` is truncated
/// or the value overflows 64 bits.
pub fn decode(buf: &[u8]) -> Option<(u64, usize)> {
    let mut value: u64 = 0;
    for (i, &byte) in buf.iter().enumerate() {
        let shift = 7 * i as u32;
        if shift >= 64 {
            return None;
        }
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Some((value, i + 1));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_encodings() {
        let mut out = Vec::new();
        encode(5, &mut out);
        assert_eq!(out, [0x05]);
        out.clear();
        encode(300, &mut out);
        assert_eq!(out, [0xac, 0x02]);
        out.clear();
        encode(0, &mut out);
        assert_eq!(out, [0x00]);
    }

    #[test]
    fn roundtrip() {
        let values = [
            0,
            1,
            127,
            128,
            255,
            16_383,
            16_384,
            u32::MAX as u64,
            u64::MAX,
        ];
        for &v in &values {
            let mut out = Vec::new();
            let n = encode(v, &mut out);
            assert_eq!(n, out.len());
            assert_eq!(decode(&out), Some((v, n)), "value {v}");
        }
    }

    #[test]
    fn truncated_input() {
        assert_eq!(decode(&[0x80]), None);
        assert_eq!(decode(&[]), None);
    }
}
