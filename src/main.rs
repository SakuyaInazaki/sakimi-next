mod attention;
mod config;
mod conv;
mod data;
mod deltanet;
mod ffn;
mod mtp;
mod model;
mod rms_norm;
mod rope;
mod tokenizer;
mod training;

use candle_core::{Device, Result, DType, D, Tensor};
use clap::Parser;
use std::path::PathBuf;

use config::Config;
use model::MiniQwenNext;
use training::{Trainer, TrainingConfig};
use tokenizer::SimpleTokenizer;
use data::{TextDataset, prepare_training_data, create_dummy_training_data};

/// Sakimi-Next: Mini Qwen-Next implementation in Rust
///
/// A reimagined small model with balanced parameter distribution:
/// - 4K vocabulary (instead of 32K) saves ~7M parameters
/// - Parameters are used for deeper/wider model instead
/// - Result: A model that can actually learn!
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Configuration preset (tiny_10m, tiny_5m, small_50m)
    #[arg(short, long, default_value = "tiny_10m")]
    config: String,

    /// Device to use (cpu, cuda)
    #[arg(short, long, default_value = "cpu")]
    device: String,

    /// Batch size
    #[arg(short, long, default_value_t = 4)]
    batch_size: usize,

    /// Sequence length
    #[arg(short = 'l', long, default_value_t = 512)]
    seq_len: usize,

    /// Number of training steps
    #[arg(short = 'n', long, default_value_t = 1000)]
    steps: usize,

    /// Learning rate
    #[arg(long, default_value_t = 1e-4)]
    learning_rate: f64,

    /// Run mode (test, train, prepare_data, generate)
    #[arg(short, long, default_value = "test")]
    mode: String,

    /// Training data path
    #[arg(long, default_value = "data/train.txt")]
    train_data: String,

    /// Tokenizer path (will be created if not exists)
    #[arg(long, default_value = "data/tokenizer.json")]
    tokenizer: String,

    /// Number of lines for dummy data (only for prepare_data mode)
    #[arg(long, default_value_t = 1000)]
    dummy_lines: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Setup device
    let device = match args.device.as_str() {
        "cuda" => Device::cuda_if_available(0)?,
        "cpu" => Device::Cpu,
        _ => panic!("Unknown device: {}", args.device),
    };

    println!("🚀 Sakimi-Next: Mini Qwen-Next in Rust");
    println!("   Device: {:?}", device);
    println!("   Config: {}", args.config);
    println!();

    // Handle data preparation mode
    if args.mode == "prepare_data" {
        return run_prepare_data(&args);
    }

    // Load configuration
    let cfg = match args.config.as_str() {
        "tiny_10m" => Config::tiny_10m(),
        "tiny_5m" => Config::tiny_5m(),
        "small_50m" => Config::small_50m(),
        _ => panic!("Unknown config preset: {}", args.config),
    };

    println!("📋 Model Configuration:");
    println!("   Vocab size: {}", cfg.vocab_size);
    println!("   Layers: {} (~{} DeltaNet + {} Attention)",
        cfg.n_layers,
        (cfg.n_layers * 3 + 3) / 4,
        cfg.n_layers / 4
    );
    println!("   Hidden size: {}", cfg.d_model);
    println!("   Heads: {} (KV: {})", cfg.n_heads, cfg.kv_heads);
    println!("   State dim: {}", cfg.d_state);
    println!("   FFN size: {}", cfg.intermediate_size);
    println!();

    // Create model
    println!("🔨 Initializing model...");
    let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
    let model = MiniQwenNext::new(&vb, &device, cfg.clone())?;
    println!("   Model created successfully!");

    // Estimate parameters
    let params = model.count_params()?;
    println!("   Estimated parameters: {:.1}M", params as f32 / 1e6);

    // Calculate parameter distribution
    let embed_params = cfg.vocab_size * cfg.d_model * 2;
    let model_params = params - embed_params as usize;
    println!("   Embedding: {:.1}M ({:.1}%)", embed_params as f32 / 1e6, embed_params as f64 / params as f64 * 100.0);
    println!("   Model body: {:.1}M ({:.1}%)", model_params as f32 / 1e6, model_params as f64 / params as f64 * 100.0);
    println!();

    match args.mode.as_str() {
        "test" => run_test(&model, &device, &cfg, args.batch_size, args.seq_len),
        "train" => run_training(&model, args, device, cfg),
        "generate" => run_generation(&model, &device, &args),
        _ => panic!("Unknown mode: {}", args.mode),
    }
}

/// Prepare training data: build tokenizer and tokenize
fn run_prepare_data(args: &Args) -> Result<()> {
    println!("📚 Preparing training data...");
    println!("   Input: {}", args.train_data);
    println!("   Output directory: data/");
    println!();

    let train_path = PathBuf::from(&args.train_data);

    // Create dummy data if file doesn't exist
    if !train_path.exists() {
        println!("⚠️  Training file not found, creating dummy data...");
        let dummy_path = PathBuf::from("data/train.txt");
        create_dummy_training_data(&dummy_path, args.dummy_lines)?;
    }

    // Prepare data (build tokenizer, tokenize, save to binary)
    let output_dir = PathBuf::from("data");
    let vocab_size = match args.config.as_str() {
        "tiny_10m" => 4096,
        "tiny_5m" => 2048,
        "small_50m" => 50257,
        _ => 4096,
    };

    let _tokenizer = prepare_training_data(
        &train_path,
        &output_dir,
        vocab_size,
        args.seq_len,
    ).map_err(|e| candle_core::Error::Msg(format!("Failed to prepare data: {}", e)))?;

    println!();
    println!("✅ Data preparation complete!");
    println!("   Next step: Run with --mode train");

    Ok(())
}

/// Test forward pass
fn run_test(model: &MiniQwenNext, device: &Device, cfg: &Config, batch_size: usize, seq_len: usize) -> Result<()> {
    println!("🧪 Running test forward pass...");

    // Use actual vocab size from config
    let vocab_size = cfg.vocab_size;

    // Create random input (within vocab range)
    let mut input_data = Vec::new();
    use rand::Rng;
    let mut rng = rand::thread_rng();
    for _ in 0..(batch_size * seq_len) {
        input_data.push(rng.gen_range(0..vocab_size) as u32);
    }

    let input_ids = Tensor::new(input_data.as_slice(), device)?
        .reshape(&[batch_size, seq_len])?;

    let start = std::time::Instant::now();
    let output = model.forward(&input_ids);
    let output = output?;
    let elapsed = start.elapsed();

    println!("   Input shape: {:?}", input_ids.dims());
    println!("   Output shape: {:?}", output.dims());
    println!("   Forward pass time: {:?}", elapsed);
    println!();

    // Test with actual tokenizer if available
    let tokenizer_path = PathBuf::from("data/tokenizer.json");
    if tokenizer_path.exists() {
        println!("🧪 Testing with real tokenizer...");

        let tokenizer = SimpleTokenizer::load(&tokenizer_path)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to load tokenizer: {}", e)))?;

        let test_text = "我爱编程，编程很有趣。";
        let tokens = tokenizer.encode(test_text, false, false);

        println!("   Test text: {}", test_text);
        println!("   Tokens: {:?}", tokens);
        println!("   Decoded: {}", tokenizer.decode(&tokens));

        if tokens.len() <= seq_len {
            let mut batch_tokens = vec![tokenizer.pad_id(); batch_size * seq_len];
            for (i, &token) in tokens.iter().enumerate() {
                if i < batch_size * seq_len {
                    batch_tokens[i] = token;
                }
            }

            let test_input = Tensor::new(batch_tokens.as_slice(), device)?
                .reshape(&[batch_size, seq_len])?;

            let output = model.forward(&test_input)?;
            println!("   Output shape: {:?}", output.dims());

            // Get predicted token for first position
            let first_output = output.narrow(0, 0, 1)?.narrow(1, 0, 1)?.squeeze(0)?.squeeze(0)?;
            let max_val = first_output.max(D::Minus1)?;
            let max_idx = first_output.argmax(D::Minus1)?;
            println!("   Max logit: {:.2}, Predicted token: {}", max_val.to_scalar::<f32>()?, max_idx.to_scalar::<u32>()?);
        }

        println!();
    }

    println!("✅ Test completed successfully!");

    Ok(())
}

/// Run training loop with real data
fn run_training(model: &MiniQwenNext, args: Args, device: Device, cfg: Config) -> Result<()> {
    println!("🏋️  Starting training...");

    // Load tokenizer
    let tokenizer_path = PathBuf::from(&args.tokenizer);
    let tokenizer = if tokenizer_path.exists() {
        SimpleTokenizer::load(&tokenizer_path)
            .map_err(|e| candle_core::Error::Msg(e))?
    } else {
        println!("⚠️  Tokenizer not found at {:?}, creating dummy tokenizer...", tokenizer_path);
        // Create a simple dummy tokenizer
        let mut vocab = vec![
            "[PAD]".to_string(),
            "[UNK]".to_string(),
            "[BOS]".to_string(),
            "[EOS]".to_string(),
        ];
        // Add common Chinese characters
        for ch in "我爱编程人工智能深度学习机器语言模型".chars() {
            vocab.push(ch.to_string());
        }
        // Fill to vocab_size
        while vocab.len() < cfg.vocab_size {
            vocab.push(format!("<tok_{}>", vocab.len()));
        }
        SimpleTokenizer::new(vocab)
    };

    println!("   Tokenizer loaded: {} tokens", tokenizer.vocab_size());
    if tokenizer.vocab_size() != cfg.vocab_size {
        println!("   ⚠️  Warning: Tokenizer vocab ({}) != config vocab ({})",
            tokenizer.vocab_size(), cfg.vocab_size);
        println!("   This is OK for small datasets, but may cause issues.");
    }

    // Load dataset
    let bin_path = PathBuf::from("data/tokens.bin");

    let dataset = if bin_path.exists() {
        println!("   Loading pre-tokenized data from {:?}", bin_path);
        TextDataset::from_bin_file(&bin_path, tokenizer, args.seq_len, &device)?
    } else {
        println!("   Loading text data from {}", args.train_data);
        let train_path = PathBuf::from(&args.train_data);
        if !train_path.exists() {
            return Err(candle_core::Error::Msg(
                format!("Training file not found: {}. Run with --mode prepare_data first.", args.train_data)
            ));
        }
        TextDataset::from_file(&train_path, tokenizer, args.seq_len, &device)?
    };

    println!("   Dataset loaded: {} sequences", dataset.len());

    if dataset.is_empty() {
        return Err(candle_core::Error::Msg("Dataset is empty!".to_string()));
    }

    // Create trainer
    let train_cfg = TrainingConfig {
        batch_size: args.batch_size,
        seq_len: args.seq_len,
        learning_rate: args.learning_rate,
        max_steps: args.steps,
        ..Default::default()
    };

    let mut trainer = Trainer::new(model.clone(), train_cfg, device.clone())?;

    println!();
    println!("   Batch size: {}", args.batch_size);
    println!("   Sequence length: {}", args.seq_len);
    println!("   Learning rate: {}", args.learning_rate);
    println!("   Total steps: {}", args.steps);
    println!("   Epochs: {}", args.steps / dataset.len());
    println!();

    // Training loop
    let mut step = 0;
    let total_batches = args.steps.min(dataset.len() * 10); // Cap at 10 epochs

    while step < total_batches {
        // Get random batch
        let (input_ids, targets) = dataset.get_random_batch(args.batch_size)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to get batch: {}", e)))?;

        // Create training batch
        let batch = training::TrainingBatch::new(input_ids, targets);

        // Training step
        let output = trainer.step(&batch)?;

        step += 1;

        // Print progress
        if step % 100 == 0 {
            println!(
                "Step {:>5}/{:<5} | Loss: {:.4} | Time: {:?}",
                step,
                total_batches,
                output.loss,
                output.elapsed
            );
        }
    }

    println!();
    println!("✅ Training completed!");

    Ok(())
}

/// Run text generation
fn run_generation(model: &MiniQwenNext, device: &Device, args: &Args) -> Result<()> {
    println!("📝 Running text generation...");

    // Load tokenizer
    let tokenizer_path = PathBuf::from(&args.tokenizer);
    let tokenizer = if tokenizer_path.exists() {
        SimpleTokenizer::load(&tokenizer_path)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to load tokenizer: {}", e)))?
    } else {
        return Err(candle_core::Error::Msg(
            format!("Tokenizer not found at {:?}. Run with --mode prepare_data first.", tokenizer_path)
        ));
    };

    let prompt = "我爱";

    println!("   Prompt: {}", prompt);

    // Tokenize prompt
    let tokens = tokenizer.encode(prompt, false, false);
    println!("   Tokens: {:?}", tokens);

    // For now, just show the tokenization
    println!();
    println!("   Generation not fully implemented yet!");
    println!("   TODO: Add sampling loop, temperature, top-k/p");

    Ok(())
}
