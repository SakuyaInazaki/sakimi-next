use crate::tokenizer::SimpleTokenizer;
use candle_core::{Device, Result, Tensor};
use rand::Rng;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;

/// Text dataset for language modeling
///
/// This dataset:
/// 1. Loads text from file(s)
/// 2. Tokenizes using the provided tokenizer
/// 3. Returns batches of (input_ids, targets)
///
/// Targets are input_ids shifted by 1 (next token prediction)
pub struct TextDataset {
    tokenizer: SimpleTokenizer,
    tokens: Vec<u32>,
    seq_len: usize,
    device: Device,
}

impl TextDataset {
    /// Create a new dataset from text file
    pub fn from_file(
        path: &Path,
        tokenizer: SimpleTokenizer,
        seq_len: usize,
        device: &Device,
    ) -> Result<Self> {
        // Read and tokenize the entire file
        let file = File::open(path)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to open file: {}", e)))?;

        let reader = BufReader::new(file);
        let mut all_tokens = Vec::new();

        println!("📖 Tokenizing dataset: {:?}", path);

        for line in reader.lines() {
            let line =
                line.map_err(|e| candle_core::Error::Msg(format!("Failed to read line: {}", e)))?;

            if line.trim().is_empty() {
                continue;
            }

            // Tokenize with BOS/EOS for each line
            let tokens = tokenizer.encode(&line, true, true);
            all_tokens.extend_from_slice(&tokens);
        }

        println!("   Total tokens: {}", all_tokens.len());
        println!("   Total sequences: {}", all_tokens.len() / seq_len);

        Ok(Self {
            tokenizer,
            tokens: all_tokens,
            seq_len,
            device: device.clone(),
        })
    }

    /// Create dataset from pre-tokenized binary file
    pub fn from_bin_file(
        path: &Path,
        tokenizer: SimpleTokenizer,
        seq_len: usize,
        device: &Device,
    ) -> Result<Self> {
        let mut file = File::open(path)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to open file: {}", e)))?;

        // Read token count
        let mut token_count_bytes = [0u8; 8];
        file.read_exact(&mut token_count_bytes)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to read token count: {}", e)))?;
        let token_count = u64::from_le_bytes(token_count_bytes) as usize;

        // Read tokens
        let mut tokens = vec![0u32; token_count];
        let mut bytes = vec![0u8; token_count * 4];
        file.read_exact(&mut bytes)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to read tokens: {}", e)))?;

        for (i, chunk) in bytes.chunks_exact(4).enumerate() {
            tokens[i] = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }

        println!("📖 Loaded {} tokens from {:?}", tokens.len(), path);

        Ok(Self {
            tokenizer,
            tokens,
            seq_len,
            device: device.clone(),
        })
    }

    /// Save tokenized data to binary file (for faster loading next time)
    pub fn save_to_bin(&self, path: &Path) -> Result<()> {
        let file = File::create(path)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to create file: {}", e)))?;

        let mut writer = BufWriter::new(file);

        // Write token count
        let token_count = self.tokens.len() as u64;
        writer
            .write_all(&token_count.to_le_bytes())
            .map_err(|e| candle_core::Error::Msg(format!("Failed to write token count: {}", e)))?;

        // Write tokens
        for &token in &self.tokens {
            writer
                .write_all(&token.to_le_bytes())
                .map_err(|e| candle_core::Error::Msg(format!("Failed to write token: {}", e)))?;
        }

        writer
            .flush()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to flush: {}", e)))?;

        println!("💾 Saved {} tokens to {:?}", self.tokens.len(), path);
        Ok(())
    }

    /// Get number of sequences in the dataset
    pub fn len(&self) -> usize {
        self.tokens.len() / self.seq_len
    }

    /// Check if dataset is empty
    pub fn is_empty(&self) -> bool {
        self.tokens.len() < self.seq_len
    }

    /// Get a batch of data
    ///
    /// Returns (input_ids, targets) where:
    /// - input_ids: [batch_size, seq_len]
    /// - targets: [batch_size, seq_len]
    ///
    /// Targets are input_ids shifted by 1 position
    pub fn get_batch(&self, batch_indices: &[usize]) -> Result<(Tensor, Tensor)> {
        let batch_size = batch_indices.len();
        let mut input_data = Vec::with_capacity(batch_size * self.seq_len);
        let mut target_data = Vec::with_capacity(batch_size * self.seq_len);

        for &idx in batch_indices {
            let start = idx * self.seq_len;

            // Make sure we have enough tokens
            if start + self.seq_len + 1 > self.tokens.len() {
                // Not enough tokens for this sequence, wrap around
                let wrap_start = 0;
                input_data.extend_from_slice(&self.tokens[wrap_start..wrap_start + self.seq_len]);
                target_data
                    .extend_from_slice(&self.tokens[wrap_start + 1..wrap_start + self.seq_len + 1]);
            } else {
                // Input: tokens[start..start+seq_len]
                input_data.extend_from_slice(&self.tokens[start..start + self.seq_len]);

                // Target: tokens[start+1..start+seq_len+1] (shifted by 1)
                target_data.extend_from_slice(&self.tokens[start + 1..start + self.seq_len + 1]);
            }
        }

        // Create tensors
        let input_tensor = Tensor::new(input_data.as_slice(), &self.device)?
            .reshape(&[batch_size, self.seq_len])?;

        let target_tensor = Tensor::new(target_data.as_slice(), &self.device)?
            .reshape(&[batch_size, self.seq_len])?;

        Ok((input_tensor, target_tensor))
    }

    /// Get random batch
    pub fn get_random_batch(&self, batch_size: usize) -> Result<(Tensor, Tensor)> {
        let num_seqs = self.len();
        if num_seqs == 0 {
            return Err(candle_core::Error::Msg("Dataset is empty".to_string()));
        }

        // Fast random sampling without shuffling the entire dataset each step.
        let mut rng = rand::thread_rng();
        let batch_indices: Vec<usize> = (0..batch_size)
            .map(|_| rng.gen_range(0..num_seqs))
            .collect();

        self.get_batch(&batch_indices)
    }

    /// Get sequential batch (for deterministic evaluation)
    pub fn get_sequential_batch(
        &self,
        start_idx: usize,
        batch_size: usize,
    ) -> Result<(Tensor, Tensor)> {
        let num_seqs = self.len();
        let end_idx = (start_idx + batch_size).min(num_seqs);

        if start_idx >= num_seqs {
            return Err(candle_core::Error::Msg(
                "Start index out of bounds".to_string(),
            ));
        }

        let batch_indices: Vec<usize> = (start_idx..end_idx).collect();
        self.get_batch(&batch_indices)
    }
}

/// Prepare training data from text file
///
/// This function:
/// 1. Loads/creates tokenizer
/// 2. Tokenizes the text
/// 3. Saves to binary format for faster loading
pub fn prepare_training_data(
    text_path: &Path,
    output_dir: &Path,
    vocab_size: usize,
    seq_len: usize,
) -> Result<SimpleTokenizer> {
    use std::fs;

    // Create output directory
    fs::create_dir_all(output_dir)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to create directory: {}", e)))?;

    let tokenizer_path = output_dir.join("tokenizer.json");
    let bin_path = output_dir.join("tokens.bin");

    // Check if already prepared
    if tokenizer_path.exists() && bin_path.exists() {
        println!("✅ Found existing prepared data in {:?}", output_dir);
        return SimpleTokenizer::load(&tokenizer_path)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to load tokenizer: {}", e)));
    }

    println!("🔨 Preparing training data...");
    println!("   Input: {:?}", text_path);
    println!("   Output: {:?}", output_dir);
    println!("   Vocab size: {}", vocab_size);
    println!("   Seq len: {}", seq_len);

    // Build vocabulary
    println!("\n📚 Building vocabulary...");
    let vocab = SimpleTokenizer::build_vocab_from_file(text_path, vocab_size)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to build vocab: {}", e)))?;
    println!(
        "   Base vocabulary size (without specials): {}",
        vocab.len()
    );

    // Create tokenizer
    let tokenizer = SimpleTokenizer::new(vocab.clone());
    println!("   Final tokenizer size: {}", tokenizer.vocab_size());

    // Save tokenizer
    tokenizer
        .save(&tokenizer_path)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to save tokenizer: {}", e)))?;
    println!("   Saved tokenizer to {:?}", tokenizer_path);

    // Load and tokenize data
    println!("\n🔤 Tokenizing data...");
    let dataset = TextDataset::from_file(text_path, tokenizer.clone(), seq_len, &Device::Cpu)?;

    // Save to binary
    dataset.save_to_bin(&bin_path)?;

    println!("\n✅ Data preparation complete!");
    println!("   Vocabulary: {} tokens", tokenizer.vocab_size());
    println!("   Dataset: {} sequences", dataset.len());

    Ok(tokenizer)
}

/// Create a dummy training text for testing
///
/// Generates some sample Chinese text for quick testing
pub fn create_dummy_training_data(path: &Path, num_lines: usize) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            candle_core::Error::Msg(format!("Failed to create parent directory: {}", e))
        })?;
    }

    let mut file = File::create(path)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to create file: {}", e)))?;

    let sample_texts = vec![
        "我爱编程，编程很有趣。",
        "深度学习是人工智能的未来。",
        "自然语言处理让计算机理解人类语言。",
        "机器学习可以从数据中学习规律。",
        "Rust是一门系统编程语言，安全且高效。",
        "量子计算将改变世界。",
        "人工智能正在改变我们的生活。",
        "深度学习模型需要大量数据进行训练。",
        "神经网络是受生物大脑启发的算法。",
        "语言模型可以生成流畅的文本。",
        "Transformer架构彻底改变了自然语言处理。",
        "注意力机制是Transformer的核心。",
        "GPT是生成式预训练变换器。",
        "BERT是双向编码器表示。",
        "知识图谱��以结构化地表示知识。",
        "强化学习让智能体通过奖励学习。",
        "计算机视觉让机器看懂图像。",
        "语音识别将语音转换为文本。",
        "机器翻译自动翻译不同语言。",
        "问答系统回答用户的问题。",
        "对话系统可以与人类自然交流。",
        "文本摘要提取文章的要点。",
        "情感分析判断文本的情感倾向。",
        "命名实体识别找出文本中的实体。",
        "文本分类将文本分配到预定义类别。",
    ];

    use rand::Rng;
    let mut rng = rand::thread_rng();

    for _ in 0..num_lines {
        let text = sample_texts[rng.gen_range(0..sample_texts.len())];
        writeln!(file, "{}", text)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to write: {}", e)))?;
    }

    println!(
        "📝 Created dummy training data: {:?} ({} lines)",
        path, num_lines
    );
    Ok(())
}
