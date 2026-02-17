use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use serde::{Deserialize, Serialize};

/// Simple character-level tokenizer with common word optimization
///
/// This tokenizer is designed for Chinese text:
/// - Treats each character as a token (Chinese-friendly)
/// - Adds common words/subwords as special tokens
/// - Uses UTF-8 bytes directly for compatibility
///
/// Vocabulary size: 4096 (4K) for the tiny_10m model
#[derive(Clone, Serialize, Deserialize)]
pub struct SimpleTokenizer {
    /// Token to ID mapping
    token_to_id: HashMap<String, u32>,
    /// ID to Token mapping
    id_to_token: Vec<String>,
    /// Special tokens
    pad_token: u32,
    unk_token: u32,
    bos_token: u32,
    eos_token: u32,
}

impl SimpleTokenizer {
    /// Special token IDs
    const PAD_ID: u32 = 0;
    const UNK_ID: u32 = 1;
    const BOS_ID: u32 = 2;
    const EOS_ID: u32 = 3;

    /// Create a new tokenizer from vocabulary
    pub fn new(vocab: Vec<String>) -> Self {
        let mut token_to_id = HashMap::new();
        let mut id_to_token = Vec::with_capacity(vocab.len());

        // Add special tokens first
        let special_tokens = vec!["[PAD]", "[UNK]", "[BOS]", "[EOS]"];
        for (idx, token) in special_tokens.iter().enumerate() {
            token_to_id.insert(token.to_string(), idx as u32);
            id_to_token.push(token.to_string());
        }

        // Add vocabulary tokens (start from ID 4)
        for token in vocab {
            if !token_to_id.contains_key(&token) {
                let id = id_to_token.len() as u32;
                token_to_id.insert(token.clone(), id);
                id_to_token.push(token);
            }
        }

        Self {
            token_to_id,
            id_to_token,
            pad_token: Self::PAD_ID,
            unk_token: Self::UNK_ID,
            bos_token: Self::BOS_ID,
            eos_token: Self::EOS_ID,
        }
    }

    /// Build vocabulary from text file
    ///
    /// This function:
    /// 1. Counts character frequencies
    /// 2. Takes the most common chars up to vocab_size
    /// 3. Returns the vocabulary
    pub fn build_vocab_from_file(
        path: &Path,
        vocab_size: usize,
    ) -> Result<Vec<String>, String> {
        let file = File::open(path)
            .map_err(|e| format!("Failed to open file: {}", e))?;
        let reader = BufReader::new(file);

        let mut char_freq: HashMap<char, usize> = HashMap::new();

        // Count character frequencies
        for line in reader.lines() {
            let line = line.map_err(|e| format!("Failed to read line: {}", e))?;

            // For Chinese: treat each character as a token
            for ch in line.chars() {
                // Skip control characters and whitespace
                if !ch.is_control() && !ch.is_whitespace() {
                    *char_freq.entry(ch).or_insert(0) += 1;
                }
            }
        }

        // Sort by frequency (descending)
        let mut char_vec: Vec<(char, usize)> = char_freq.into_iter().collect();
        char_vec.sort_by(|a, b| b.1.cmp(&a.1));

        // Reserve space for special tokens
        let num_regular_tokens = vocab_size.saturating_sub(4);

        // Build vocabulary
        let mut vocab = Vec::with_capacity(vocab_size);

        // Add common words/subwords (for Chinese)
        // These are frequently used bigrams/trigrams
        let common_chinese_subwords = vec![
            // Common particles
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
            "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
            "你", "会", "着", "没有", "看", "好", "自己", "这",
            // Numbers and digits
            "零", "一", "二", "三", "四", "五", "六", "七", "八", "九",
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            // Common punctuation
            "。", "，", "、", "：", "；", "！", "？", "（", "）", "「", "」",
        ];

        for word in common_chinese_subwords {
            if vocab.len() < num_regular_tokens {
                vocab.push(word.to_string());
            }
        }

        // Add most common characters
        for (ch, _freq) in char_vec {
            if vocab.len() >= num_regular_tokens {
                break;
            }
            vocab.push(ch.to_string());
        }

        // Fill remaining with bytes (for full coverage)
        while vocab.len() < num_regular_tokens {
            let byte_id = vocab.len() - num_regular_tokens;
            if byte_id < 256 {
                vocab.push(format!("<byte_{}>", byte_id));
            } else {
                break;
            }
        }

        Ok(vocab)
    }

    /// Build vocabulary from multiple files
    pub fn build_vocab_from_files(
        paths: &[&Path],
        vocab_size: usize,
    ) -> Result<Vec<String>, String> {
        let mut all_char_freq: HashMap<char, usize> = HashMap::new();

        // Process all files
        for path in paths {
            let file = File::open(path)
                .map_err(|e| format!("Failed to open {:?}: {}", path, e))?;
            let reader = BufReader::new(file);

            for line in reader.lines() {
                let line = line.map_err(|e| format!("Failed to read line: {}", e))?;

                for ch in line.chars() {
                    if !ch.is_control() && !ch.is_whitespace() {
                        *all_char_freq.entry(ch).or_insert(0) += 1;
                    }
                }
            }
        }

        // Sort by frequency
        let mut char_vec: Vec<(char, usize)> = all_char_freq.into_iter().collect();
        char_vec.sort_by(|a, b| b.1.cmp(&a.1));

        // Build vocabulary (same logic as single file)
        let num_regular_tokens = vocab_size.saturating_sub(4);
        let mut vocab = Vec::with_capacity(vocab_size);

        let common_chinese_subwords = vec![
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
            "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
            "你", "会", "着", "没有", "看", "好", "自己", "这",
            "零", "一", "二", "三", "四", "五", "六", "七", "八", "九",
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "。", "，", "、", "：", "；", "！", "？", "（", "）", "「", "」",
        ];

        for word in common_chinese_subwords {
            if vocab.len() < num_regular_tokens {
                vocab.push(word.to_string());
            }
        }

        for (ch, _freq) in char_vec {
            if vocab.len() >= num_regular_tokens {
                break;
            }
            vocab.push(ch.to_string());
        }

        while vocab.len() < num_regular_tokens {
            let byte_id = vocab.len() - num_regular_tokens;
            if byte_id < 256 {
                vocab.push(format!("<byte_{}>", byte_id));
            } else {
                break;
            }
        }

        Ok(vocab)
    }

    /// Encode text to token IDs
    ///
    /// Simplified greedy tokenization:
    /// 1. Try to match the longest possible subword first
    /// 2. Fall back to character-by-character if no match found
    /// 3. Skip control characters and whitespace
    pub fn encode(&self, text: &str, add_bos: bool, add_eos: bool) -> Vec<u32> {
        let mut tokens = Vec::new();

        if add_bos {
            tokens.push(self.bos_token);
        }

        // Process the text character by character with greedy matching
        let mut i = 0;
        let chars: Vec<char> = text.chars().collect();

        while i < chars.len() {
            let ch = chars[i];

            // Skip control characters and whitespace
            if ch.is_control() || ch.is_whitespace() {
                i += 1;
                continue;
            }

            // Try to find the longest matching token starting at position i
            let mut matched = false;
            let mut max_match_len = 0;
            let mut best_token: Option<u32> = None;

            // Try substrings of length 1 to 5 (max length for multi-char tokens)
            for look_ahead in 1..=5.min(chars.len() - i) {
                let substr: String = chars[i..i + look_ahead].iter().collect();
                if let Some(&token_id) = self.token_to_id.get(&substr) {
                    // Check if extending by one more character would also match
                    let next_look_ahead = look_ahead + 1;
                    let would_extend = if i + next_look_ahead <= chars.len() {
                        let extended: String = chars[i..i + next_look_ahead].iter().collect();
                        self.token_to_id.contains_key(&extended)
                    } else {
                        false
                    };

                    // Use this match only if extending wouldn't yield a longer match
                    if !would_extend {
                        max_match_len = look_ahead;
                        best_token = Some(token_id);
                    }
                }
            }

            if let Some(token_id) = best_token {
                tokens.push(token_id);
                i += max_match_len;
                matched = true;
            }

            // If no multi-char match, fall back to single character
            if !matched {
                let ch_str = ch.to_string();
                let token_id = self.token_to_id.get(&ch_str).copied().unwrap_or(self.unk_token);
                tokens.push(token_id);
                i += 1;
            }
        }

        if add_eos {
            tokens.push(self.eos_token);
        }

        tokens
    }

    /// Decode token IDs to text
    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut text = String::new();

        for &token_id in tokens {
            // Skip special tokens except UNK
            if token_id == self.pad_token || token_id == self.bos_token || token_id == self.eos_token {
                continue;
            }

            if let Some(token) = self.id_to_token.get(token_id as usize) {
                // Remove special angle bracket notation for bytes
                let token_str = if token.starts_with("<byte_") && token.ends_with('>') {
                    let byte_str = &token[6..token.len()-1];
                    if let Ok(byte_val) = byte_str.parse::<u8>() {
                        (byte_val as char).to_string()
                    } else {
                        token.clone()
                    }
                } else {
                    token.clone()
                };
                text.push_str(&token_str);
            } else {
                text.push('�'); // Replacement character
            }
        }

        text
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    /// Get special token IDs
    pub fn pad_id(&self) -> u32 { self.pad_token }
    pub fn unk_id(&self) -> u32 { self.unk_token }
    pub fn bos_id(&self) -> u32 { self.bos_token }
    pub fn eos_id(&self) -> u32 { self.eos_token }

    /// Save tokenizer to file
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize tokenizer: {}", e))?;

        let mut file = File::create(path)
            .map_err(|e| format!("Failed to create file: {}", e))?;

        file.write_all(json.as_bytes())
            .map_err(|e| format!("Failed to write: {}", e))?;

        Ok(())
    }

    /// Load tokenizer from file
    pub fn load(path: &Path) -> Result<Self, String> {
        let file = File::open(path)
            .map_err(|e| format!("Failed to open file: {}", e))?;

        let reader = BufReader::new(file);
        let tokenizer: Self = serde_json::from_reader(reader)
            .map_err(|e| format!("Failed to deserialize: {}", e))?;

        Ok(tokenizer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tokenization() {
        let vocab = vec![
            "[PAD]".to_string(),
            "[UNK]".to_string(),
            "[BOS]".to_string(),
            "[EOS]".to_string(),
            "我".to_string(),
            "爱".to_string(),
            "编程".to_string(),
        ];

        let tokenizer = SimpleTokenizer::new(vocab);

        // Test encoding
        let tokens = tokenizer.encode("我爱编程", true, true);
        assert_eq!(tokens, vec![2, 4, 5, 6, 3]); // BOS, 我, 爱, 编程, EOS

        // Test decoding
        let text = tokenizer.decode(&tokens);
        assert_eq!(text, "我爱编程");
    }

    #[test]
    fn test_unknown_tokens() {
        let vocab = vec![
            "[PAD]".to_string(),
            "[UNK]".to_string(),
            "[BOS]".to_string(),
            "[EOS]".to_string(),
            "我".to_string(),
        ];

        let tokenizer = SimpleTokenizer::new(vocab);

        let tokens = tokenizer.encode("我爱", false, false);
        // "爱" should be UNK
        assert_eq!(tokens, vec![4, 1]);
    }
}
