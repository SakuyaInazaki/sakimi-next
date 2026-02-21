mod activation;
mod attention;
mod cache;
mod config;
mod conv;
mod data;
mod deltanet;
mod ffn;
mod model;
mod mtp;
mod rms_norm;
mod rope;
mod tokenizer;
mod trainable;
mod training;

use candle_core::{DType, Device, Result, Tensor, Var, D};
use clap::Parser;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::panic::{self, AssertUnwindSafe};
use std::time::Instant;

use config::Config;
use data::{create_dummy_training_data, prepare_training_data, TextDataset};
use model::MiniQwenNext;
use tokenizer::SimpleTokenizer;
use training::{cross_entropy_loss, Trainer, TrainingConfig};

/// Sakimi-Next: Mini Qwen-Next implementation in Rust
///
/// A reimagined small model with balanced parameter distribution:
/// - 8K vocabulary (instead of 32K) balances compression and parameter budget
/// - Parameters are used for deeper/wider model instead
/// - Result: A model that can actually learn!
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Configuration preset (tiny_10m, tiny_5m, small_50m). Training is locked to tiny_10m.
    #[arg(short, long, default_value = "tiny_10m")]
    config: String,

    /// Device to use (auto, cpu, cuda, metal)
    #[arg(short, long, default_value = "auto")]
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

    /// Rebuild tokenizer from train/val corpus before training
    #[arg(long, default_value_t = false)]
    rebuild_tokenizer: bool,

    /// Allow tokenizer vocab to differ from model vocab (not recommended)
    #[arg(long, default_value_t = false)]
    allow_vocab_mismatch: bool,

    /// Number of lines for dummy data (only for prepare_data mode)
    #[arg(long, default_value_t = 1000)]
    dummy_lines: usize,

    /// Disable optimized kernel path (for debugging/reference checks)
    #[arg(long, default_value_t = false)]
    disable_fast_kernels: bool,

    /// Print training log every N steps
    #[arg(long, default_value_t = 100)]
    print_every: usize,

    /// Save checkpoint every N steps
    #[arg(long, default_value_t = 5000)]
    save_every: usize,

    /// Resume/load from checkpoint (.safetensors) for train/generate
    #[arg(long, default_value = "", alias = "checkpoint")]
    resume_from: String,

    /// Prompt text for generation mode
    #[arg(long, default_value = "我爱")]
    prompt: String,

    /// Max new tokens for generation mode
    #[arg(long, default_value_t = 64)]
    max_new_tokens: usize,

    /// Optional validation text file path (enables validation loss)
    #[arg(long, default_value = "")]
    val_data: String,

    /// Run validation every N steps (when --val-data is set)
    #[arg(long, default_value_t = 200)]
    eval_every: usize,

    /// Number of validation batches per evaluation
    #[arg(long, default_value_t = 16)]
    eval_batches: usize,

    /// Early-stop patience on validation loss (0 disables early stopping)
    #[arg(long, default_value_t = 0)]
    early_stop_patience: usize,

    /// Minimum val-loss improvement to reset patience
    #[arg(long, default_value_t = 1e-4)]
    early_stop_min_delta: f32,
}

const TRAIN_TARGET_CONFIG: &str = "tiny_10m";
const TOKEN_CACHE_LAYOUT: &str = "contiguous_v1";

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for b in bytes {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn file_fingerprint(path: &Path) -> Result<String> {
    let data = std::fs::read(path).map_err(|e| {
        candle_core::Error::Msg(format!("Failed to read file for fingerprint {:?}: {}", path, e))
    })?;
    Ok(format!("{:016x}", fnv1a64(&data)))
}

fn validate_resume_checkpoint_metadata(
    resume_path: &str,
    cfg: &Config,
    config_name: &str,
    tokenizer_vocab: usize,
    tokenizer_fingerprint: &str,
) -> Result<()> {
    let metadata_path = std::path::Path::new(resume_path).with_extension("json");
    if !metadata_path.exists() {
        println!(
            "   ⚠️  Resume metadata not found at {:?}, only tensor-shape checks will be applied.",
            metadata_path
        );
        return Ok(());
    }

    let raw = std::fs::read_to_string(&metadata_path).map_err(|e| {
        candle_core::Error::Msg(format!(
            "Failed to read resume metadata {:?}: {}",
            metadata_path, e
        ))
    })?;
    let meta: serde_json::Value = serde_json::from_str(&raw).map_err(|e| {
        candle_core::Error::Msg(format!(
            "Failed to parse resume metadata {:?}: {}",
            metadata_path, e
        ))
    })?;

    if let Some(saved_cfg) = meta.get("config_name").and_then(|v| v.as_str()) {
        if saved_cfg != config_name {
            return Err(candle_core::Error::Msg(format!(
                "Resume config mismatch: checkpoint has '{}', current is '{}'",
                saved_cfg, config_name
            )));
        }
    }

    if let Some(saved_vocab) = meta.get("model_vocab_size").and_then(|v| v.as_u64()) {
        if saved_vocab as usize != cfg.vocab_size {
            return Err(candle_core::Error::Msg(format!(
                "Resume model vocab mismatch: checkpoint has {}, current is {}",
                saved_vocab, cfg.vocab_size
            )));
        }
    }

    if let Some(saved_tok_vocab) = meta.get("tokenizer_vocab_size").and_then(|v| v.as_u64()) {
        if saved_tok_vocab as usize != tokenizer_vocab {
            return Err(candle_core::Error::Msg(format!(
                "Resume tokenizer vocab mismatch: checkpoint has {}, current is {}",
                saved_tok_vocab, tokenizer_vocab
            )));
        }
    }

    if let Some(saved_fp) = meta.get("tokenizer_fingerprint").and_then(|v| v.as_str()) {
        if saved_fp != tokenizer_fingerprint {
            return Err(candle_core::Error::Msg(format!(
                "Resume tokenizer fingerprint mismatch: checkpoint has {}, current is {}",
                saved_fp, tokenizer_fingerprint
            )));
        }
    }

    Ok(())
}

fn write_checkpoint_training_metadata(
    ckpt_path: &str,
    cfg: &Config,
    config_name: &str,
    tokenizer_path: &Path,
    tokenizer_vocab: usize,
    tokenizer_fingerprint: &str,
    param_count: usize,
) -> Result<()> {
    let metadata_path = std::path::Path::new(ckpt_path).with_extension("json");
    if !metadata_path.exists() {
        return Ok(());
    }

    let raw = std::fs::read_to_string(&metadata_path).map_err(|e| {
        candle_core::Error::Msg(format!(
            "Failed to read checkpoint metadata {:?}: {}",
            metadata_path, e
        ))
    })?;

    let mut meta: serde_json::Value = serde_json::from_str(&raw).unwrap_or_else(|_| serde_json::json!({}));
    if !meta.is_object() {
        meta = serde_json::json!({});
    }

    let obj = meta
        .as_object_mut()
        .ok_or_else(|| candle_core::Error::Msg("Checkpoint metadata root is not an object".to_string()))?;

    obj.insert("checkpoint_format_version".to_string(), serde_json::Value::from(2));
    obj.insert("config_name".to_string(), serde_json::Value::String(config_name.to_string()));
    obj.insert("model_vocab_size".to_string(), serde_json::Value::from(cfg.vocab_size as u64));
    obj.insert("model_d_model".to_string(), serde_json::Value::from(cfg.d_model as u64));
    obj.insert("model_n_layers".to_string(), serde_json::Value::from(cfg.n_layers as u64));
    obj.insert("model_n_heads".to_string(), serde_json::Value::from(cfg.n_heads as u64));
    obj.insert("model_param_count".to_string(), serde_json::Value::from(param_count as u64));
    obj.insert("tied_lm_head".to_string(), serde_json::Value::Bool(true));
    obj.insert(
        "tokenizer_path".to_string(),
        serde_json::Value::String(tokenizer_path.display().to_string()),
    );
    obj.insert(
        "tokenizer_vocab_size".to_string(),
        serde_json::Value::from(tokenizer_vocab as u64),
    );
    obj.insert(
        "tokenizer_fingerprint".to_string(),
        serde_json::Value::String(tokenizer_fingerprint.to_string()),
    );

    let out = serde_json::to_string_pretty(&meta).map_err(|e| {
        candle_core::Error::Msg(format!("Failed to serialize checkpoint metadata: {}", e))
    })?;

    std::fs::write(&metadata_path, out).map_err(|e| {
        candle_core::Error::Msg(format!(
            "Failed to write checkpoint metadata {:?}: {}",
            metadata_path, e
        ))
    })?;

    Ok(())
}

fn safe_metal_device() -> Option<Device> {
    let prev_hook = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    let attempt = panic::catch_unwind(AssertUnwindSafe(|| Device::new_metal(0)));
    panic::set_hook(prev_hook);
    match attempt {
        Ok(Ok(device)) => Some(device),
        Ok(Err(err)) => {
            eprintln!("⚠️  Metal device init failed, fallback to CPU: {}", err);
            None
        }
        Err(_) => {
            eprintln!("⚠️  Metal device init panicked, fallback to CPU.");
            None
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Setup device
    let device = match args.device.as_str() {
        "auto" => {
            #[cfg(target_os = "macos")]
            {
                if let Some(device) = safe_metal_device() {
                    device
                } else {
                    Device::Cpu
                }
            }
            #[cfg(not(target_os = "macos"))]
            {
                Device::cuda_if_available(0)?
            }
        }
        "metal" => safe_metal_device().unwrap_or(Device::Cpu),
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
    let mut cfg = match args.config.as_str() {
        "tiny_10m" => Config::tiny_10m(),
        "tiny_5m" => Config::tiny_5m(),
        "small_50m" => Config::small_50m(),
        _ => panic!("Unknown config preset: {}", args.config),
    };
    if args.disable_fast_kernels {
        cfg.use_fast_kernels = false;
    }

    println!("📋 Model Configuration:");
    println!("   Vocab size: {}", cfg.vocab_size);
    println!(
        "   Layers: {} ({} linear + {} full attention)",
        cfg.n_layers,
        cfg.n_linear_layers(),
        cfg.n_full_attention_layers()
    );
    println!("   Hidden size: {}", cfg.d_model);
    println!(
        "   Heads: {} (KV: {}, head_dim: {})",
        cfg.n_heads, cfg.kv_heads, cfg.head_dim
    );
    println!("   FFN size: {}", cfg.intermediate_size);
    println!(
        "   Act: {}, init_std: {:.4}, rms_eps: {:.1e}",
        cfg.hidden_act, cfg.initializer_range, cfg.rms_norm_eps
    );
    println!(
        "   RoPE: theta={}, partial={:.2}",
        cfg.rope_theta, cfg.partial_rotary_factor
    );
    println!(
        "   Linear heads: K={}x{}, V={}x{}, ConvK={}",
        cfg.linear_num_key_heads,
        cfg.linear_key_head_dim,
        cfg.linear_num_value_heads,
        cfg.linear_value_head_dim,
        cfg.linear_conv_kernel_dim
    );
    println!("   Fast kernels: {}", cfg.use_fast_kernels);
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
    // LM head is tied with embedding, so embedding parameters are counted once.
    let embed_params = cfg.vocab_size * cfg.d_model;
    let model_params = params - embed_params as usize;
    println!(
        "   Embedding: {:.1}M ({:.1}%)",
        embed_params as f32 / 1e6,
        embed_params as f64 / params as f64 * 100.0
    );
    println!(
        "   Model body: {:.1}M ({:.1}%)",
        model_params as f32 / 1e6,
        model_params as f64 / params as f64 * 100.0
    );
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
        "tiny_5m" => 8192,
        "small_50m" => 50257,
        _ => 4096,
    };

    let _tokenizer = prepare_training_data(&train_path, &output_dir, vocab_size, args.seq_len)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to prepare data: {}", e)))?;

    println!();
    println!("✅ Data preparation complete!");
    println!("   Next step: Run with --mode train");

    Ok(())
}

/// Test forward pass
fn run_test(
    model: &MiniQwenNext,
    device: &Device,
    cfg: &Config,
    batch_size: usize,
    seq_len: usize,
) -> Result<()> {
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

    let input_ids = Tensor::new(input_data.as_slice(), device)?.reshape(&[batch_size, seq_len])?;

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

            let test_input =
                Tensor::new(batch_tokens.as_slice(), device)?.reshape(&[batch_size, seq_len])?;

            let output = model.forward(&test_input)?;
            println!("   Output shape: {:?}", output.dims());

            // Get predicted token for first position
            let first_output = output
                .narrow(0, 0, 1)?
                .narrow(1, 0, 1)?
                .squeeze(0)?
                .squeeze(0)?;
            let max_val = first_output.max(D::Minus1)?;
            let max_idx = first_output.argmax(D::Minus1)?;
            println!(
                "   Max logit: {:.2}, Predicted token: {}",
                max_val.to_scalar::<f32>()?,
                max_idx.to_scalar::<u32>()?
            );
        }

        println!();
    }

    println!("✅ Test completed successfully!");

    Ok(())
}

/// Run training loop with real data
fn run_training(model: &MiniQwenNext, args: Args, device: Device, cfg: Config) -> Result<()> {
    println!("🏋️  Starting training...");

    if args.config != TRAIN_TARGET_CONFIG {
        return Err(candle_core::Error::Msg(format!(
            "Training is locked to '{}'. Received config='{}'.",
            TRAIN_TARGET_CONFIG, args.config
        )));
    }

    let train_path = PathBuf::from(&args.train_data);
    if !train_path.exists() {
        return Err(candle_core::Error::Msg(format!(
            "Training file not found: {}. Run with --mode prepare_data first.",
            args.train_data
        )));
    }

    let val_path = if args.val_data.trim().is_empty() {
        None
    } else {
        let p = PathBuf::from(&args.val_data);
        if !p.exists() {
            return Err(candle_core::Error::Msg(format!(
                "Validation file not found: {}",
                args.val_data
            )));
        }
        Some(p)
    };

    // Load tokenizer (or rebuild from corpus when requested).
    let tokenizer_path = PathBuf::from(&args.tokenizer);
    let rebuild_tokenizer = args.rebuild_tokenizer || !tokenizer_path.exists();
    if !args.resume_from.trim().is_empty() && rebuild_tokenizer {
        return Err(candle_core::Error::Msg(
            "Refusing to resume while rebuilding tokenizer. Keep tokenizer fixed for resume.".to_string(),
        ));
    }
    let tokenizer = if rebuild_tokenizer {
        println!("   Building tokenizer from training corpus...");
        let mut sources = vec![train_path.as_path()];
        if let Some(ref p) = val_path {
            sources.push(p.as_path());
        }
        let vocab = SimpleTokenizer::build_vocab_from_files(&sources, cfg.vocab_size)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to build tokenizer: {}", e)))?;
        let tokenizer = SimpleTokenizer::new(vocab);

        if let Some(parent) = tokenizer_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                candle_core::Error::Msg(format!(
                    "Failed to create tokenizer directory {:?}: {}",
                    parent, e
                ))
            })?;
        }
        tokenizer
            .save(&tokenizer_path)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to save tokenizer: {}", e)))?;
        println!(
            "   Tokenizer rebuilt and saved: {:?} (size={})",
            tokenizer_path,
            tokenizer.vocab_size()
        );
        tokenizer
    } else {
        SimpleTokenizer::load(&tokenizer_path).map_err(|e| {
            candle_core::Error::Msg(format!("Failed to load tokenizer: {}", e))
        })?
    };

    println!("   Tokenizer loaded: {} tokens", tokenizer.vocab_size());
    if tokenizer.vocab_size() != cfg.vocab_size {
        if args.allow_vocab_mismatch {
            println!(
                "   Warning: Tokenizer vocab ({}) != config vocab ({})",
                tokenizer.vocab_size(),
                cfg.vocab_size
            );
            println!("   Proceeding because --allow-vocab-mismatch was set.");
        } else {
            return Err(candle_core::Error::Msg(format!(
                "Tokenizer vocab ({}) != config vocab ({}). Re-run with --rebuild-tokenizer (recommended) or --allow-vocab-mismatch.",
                tokenizer.vocab_size(),
                cfg.vocab_size
            )));
        }
    }

    let tokenizer_fingerprint = file_fingerprint(&tokenizer_path)?;
    let param_count = model.count_params()?;

    println!("\n🔒 Training Readiness Check:");
    println!("   Target config: {}", TRAIN_TARGET_CONFIG);
    println!("   Effective config: {}", args.config);
    println!("   Param count: {} ({:.2}M)", param_count, param_count as f64 / 1e6);
    println!("   Tied LM head: true");
    println!("   Model vocab size: {}", cfg.vocab_size);
    println!("   Tokenizer vocab size: {}", tokenizer.vocab_size());
    println!("   Tokenizer file: {}", tokenizer_path.display());
    println!("   Tokenizer fingerprint: {}", tokenizer_fingerprint);

    // Load train dataset (with tokenizer-fingerprint cache to avoid re-tokenizing on restart).
    let fp_short = if tokenizer_fingerprint.len() >= 12 {
        &tokenizer_fingerprint[..12]
    } else {
        &tokenizer_fingerprint
    };
    let train_bin_path = PathBuf::from(format!("{}.{}.s{}.{}.tokens.bin", args.train_data, fp_short, args.seq_len, TOKEN_CACHE_LAYOUT));
    let train_dataset = if train_bin_path.exists() {
        println!("   Loading pre-tokenized train data from {:?}", train_bin_path);
        TextDataset::from_bin_file(&train_bin_path, tokenizer.clone(), args.seq_len, &device)?
    } else {
        println!("   Loading text data from {}", args.train_data);
        let ds = TextDataset::from_file(&train_path, tokenizer.clone(), args.seq_len, &device)?;
        ds.save_to_bin(&train_bin_path)?;
        println!("   Saved pre-tokenized train cache to {:?}", train_bin_path);
        ds
    };

    println!("   Train dataset loaded: {} sequences", train_dataset.len());
    if train_dataset.is_empty() {
        return Err(candle_core::Error::Msg("Train dataset is empty!".to_string()));
    }

    // Optional validation dataset (also cached by tokenizer fingerprint).
    let val_dataset = if let Some(ref val_path) = val_path {
        let val_bin_path = PathBuf::from(format!("{}.{}.s{}.{}.tokens.bin", args.val_data, fp_short, args.seq_len, TOKEN_CACHE_LAYOUT));
        let ds = if val_bin_path.exists() {
            println!(
                "   Loading pre-tokenized validation data from {:?}",
                val_bin_path
            );
            TextDataset::from_bin_file(&val_bin_path, tokenizer.clone(), args.seq_len, &device)?
        } else {
            println!("   Loading validation text data from {}", args.val_data);
            let ds = TextDataset::from_file(val_path, tokenizer.clone(), args.seq_len, &device)?;
            ds.save_to_bin(&val_bin_path)?;
            println!("   Saved pre-tokenized validation cache to {:?}", val_bin_path);
            ds
        };

        if ds.is_empty() {
            return Err(candle_core::Error::Msg(
                "Validation dataset is empty!".to_string(),
            ));
        }
        println!("   Validation dataset loaded: {} sequences", ds.len());
        Some(ds)
    } else {
        None
    };

    // Create trainer
    let train_cfg = TrainingConfig {
        batch_size: args.batch_size,
        seq_len: args.seq_len,
        learning_rate: args.learning_rate,
        max_steps: args.steps,
        print_every: args.print_every,
        save_every: args.save_every,
        ..Default::default()
    };
    let print_every = train_cfg.print_every.max(1);
    let save_every = train_cfg.save_every.max(1);
    let checkpoint_dir = train_cfg.checkpoint_dir.clone();

    let mut trainer = Trainer::new(model.clone(), train_cfg, device.clone())?;

    if !args.resume_from.trim().is_empty() {
        println!("   🔁 Resuming from checkpoint: {}", args.resume_from);
        validate_resume_checkpoint_metadata(
            &args.resume_from,
            &cfg,
            &args.config,
            tokenizer.vocab_size(),
            &tokenizer_fingerprint,
        )?;
        trainer.load_checkpoint(&args.resume_from)?;
        println!("   ✅ Resume loaded (step={})", trainer.current_step());
    }

    let resumed_step = trainer.current_step();

    println!();
    println!("   Batch size: {}", args.batch_size);
    println!("   Sequence length: {}", args.seq_len);
    println!("   Learning rate: {}", args.learning_rate);
    println!("   Total steps: {}", args.steps);
    if resumed_step > 0 {
        println!("   Resume step: {}", resumed_step);
        println!("   Remaining steps: {}", args.steps.saturating_sub(resumed_step));
    }
    println!("   Approx train epochs: {:.2}", args.steps as f64 / train_dataset.len() as f64);
    if val_dataset.is_some() {
        println!(
            "   Validation: every {} steps, {} batches/eval, early-stop patience={} min-delta={}",
            args.eval_every.max(1),
            args.eval_batches.max(1),
            args.early_stop_patience,
            args.early_stop_min_delta
        );
    } else {
        println!("   Validation: disabled (no --val-data)");
    }
    println!();

    // Training loop
    let mut step = trainer.current_step();
    let total_batches = args.steps;
    let first_step_after_resume = resumed_step.saturating_add(1);
    let eval_every = args.eval_every.max(1);
    let eval_batches = args.eval_batches.max(1);
    let mut best_val_loss = f32::INFINITY;
    let mut no_improve_rounds = 0usize;

    while step < total_batches {
        let (input_ids, targets) = train_dataset
            .get_random_batch(args.batch_size)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to get train batch: {}", e)))?;

        let batch = training::TrainingBatch::new(input_ids, targets);
        let output = trainer.step(&batch)?;
        step = output.step;

        if step % print_every == 0 || step == first_step_after_resume {
            println!(
                "Step {:>5}/{:<5} | Loss: {:.4} | LR: {:.2e} | Time: {:?}",
                step, total_batches, output.loss, output.learning_rate, output.elapsed
            );
        }

        if step % save_every == 0 {
            let ckpt_path = format!("{}/step_{:06}.safetensors", checkpoint_dir, step);
            trainer.save_checkpoint(&ckpt_path)?;
            write_checkpoint_training_metadata(
                &ckpt_path,
                &cfg,
                &args.config,
                &tokenizer_path,
                tokenizer.vocab_size(),
                &tokenizer_fingerprint,
                param_count,
            )?;
            println!("   💾 Saved checkpoint: {}", ckpt_path);
        }

        if let Some(ref val_ds) = val_dataset {
            if step % eval_every == 0 || step == total_batches {
                let val_loss = evaluate_validation_loss(trainer.model(), val_ds, args.batch_size, eval_batches)?;
                println!("   📊 Validation @ step {:>5}: loss={:.4}", step, val_loss);

                if val_loss + args.early_stop_min_delta < best_val_loss {
                    best_val_loss = val_loss;
                    no_improve_rounds = 0;
                    let best_ckpt = format!("{}/best_val.safetensors", checkpoint_dir);
                    trainer.save_checkpoint(&best_ckpt)?;
                    write_checkpoint_training_metadata(
                        &best_ckpt,
                        &cfg,
                        &args.config,
                        &tokenizer_path,
                        tokenizer.vocab_size(),
                        &tokenizer_fingerprint,
                        param_count,
                    )?;
                    println!(
                        "   🏅 New best val loss {:.4}, saved: {}",
                        best_val_loss, best_ckpt
                    );
                } else if args.early_stop_patience > 0 {
                    no_improve_rounds += 1;
                    println!(
                        "   ⏳ No val improvement ({}/{} rounds)",
                        no_improve_rounds, args.early_stop_patience
                    );
                    if no_improve_rounds >= args.early_stop_patience {
                        println!(
                            "🛑 Early stopping at step {} (best val loss {:.4})",
                            step, best_val_loss
                        );
                        break;
                    }
                }
            }
        }
    }

    let final_ckpt = format!("{}/final_step_{:06}.safetensors", checkpoint_dir, step);
    trainer.save_checkpoint(&final_ckpt)?;
    write_checkpoint_training_metadata(
        &final_ckpt,
        &cfg,
        &args.config,
        &tokenizer_path,
        tokenizer.vocab_size(),
        &tokenizer_fingerprint,
        param_count,
    )?;
    println!("   💾 Saved final checkpoint: {}", final_ckpt);

    println!();
    println!("✅ Training completed!");

    Ok(())
}

fn evaluate_validation_loss(
    model: &MiniQwenNext,
    dataset: &TextDataset,
    batch_size: usize,
    max_batches: usize,
) -> Result<f32> {
    if dataset.is_empty() {
        return Err(candle_core::Error::Msg(
            "Validation dataset is empty".to_string(),
        ));
    }

    let batch_size = batch_size.max(1);
    let max_batches = max_batches.max(1);

    model.set_training(false);
    let eval_res = (|| -> Result<f32> {
        let mut total = 0f64;
        let mut n = 0usize;
        let mut start_idx = 0usize;

        while n < max_batches && start_idx < dataset.len() {
            let (input_ids, targets) = dataset
                .get_sequential_batch(start_idx, batch_size)
                .map_err(|e| {
                    candle_core::Error::Msg(format!("Failed to get validation batch: {}", e))
                })?;

            let logits = model.forward(&input_ids)?;
            let loss = cross_entropy_loss(&logits, &targets)?;
            total += loss.to_scalar::<f32>()? as f64;

            n += 1;
            start_idx += batch_size;
        }

        if n == 0 {
            return Err(candle_core::Error::Msg(
                "No validation batches were evaluated".to_string(),
            ));
        }

        Ok((total / n as f64) as f32)
    })();

    model.set_training(true);
    eval_res
}

fn load_model_weights_from_checkpoint(model: &MiniQwenNext, path: &str, device: &Device) -> Result<()> {
    let ckpt_path = std::path::Path::new(path);
    if !ckpt_path.exists() {
        return Err(candle_core::Error::Msg(format!(
            "Checkpoint not found: {}",
            ckpt_path.display()
        )));
    }

    let tensors = candle_core::safetensors::load(ckpt_path, device)?;
    let model_tensors = model.get_tensors();

    for (idx, dst) in model_tensors.iter().enumerate() {
        let name = format!("param_{idx:05}");
        let src = tensors.get(&name).ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "Missing tensor '{}' in checkpoint {}",
                name,
                ckpt_path.display()
            ))
        })?;

        if src.dims() != dst.dims() {
            return Err(candle_core::Error::Msg(format!(
                "Shape mismatch for {}: checkpoint {:?} vs model {:?}",
                name,
                src.dims(),
                dst.dims()
            )));
        }

        let src = if src.dtype() != dst.dtype() {
            src.to_dtype(dst.dtype())?
        } else {
            src.clone()
        };

        let dst_var = Var::from_tensor(dst)?;
        dst_var.set(&src)?;
    }

    Ok(())
}

/// Run text generation
fn run_generation(model: &MiniQwenNext, device: &Device, args: &Args) -> Result<()> {
    println!("📝 Running text generation...");

    if !args.resume_from.trim().is_empty() {
        println!("   Loading checkpoint: {}", args.resume_from);
        load_model_weights_from_checkpoint(model, &args.resume_from, device)?;
    } else {
        println!("   No checkpoint provided, using current model weights.");
    }

    model.set_training(false);

    // Load tokenizer
    let tokenizer_path = PathBuf::from(&args.tokenizer);
    let tokenizer = if tokenizer_path.exists() {
        SimpleTokenizer::load(&tokenizer_path)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to load tokenizer: {}", e)))?
    } else {
        return Err(candle_core::Error::Msg(format!(
            "Tokenizer not found at {:?}. Run with --mode prepare_data first.",
            tokenizer_path
        )));
    };

    let prompt = args.prompt.trim();
    if prompt.is_empty() {
        return Err(candle_core::Error::Msg("Prompt is empty. Use --prompt to set a non-empty prompt.".to_string()));
    }

    println!("   Prompt: {}", prompt);

    // Tokenize prompt
    let tokens = tokenizer.encode(prompt, false, false);
    println!("   Tokens: {:?}", tokens);
    if tokens.is_empty() {
        return Err(candle_core::Error::Msg(
            "Prompt produced no tokens after tokenization".to_string(),
        ));
    }

    let mut generated = tokens.clone();
    let mut cache = model.create_cache();

    // Prefill cache with full prompt.
    let prompt_input = Tensor::new(tokens.as_slice(), device)?.reshape(&[1, tokens.len()])?;
    let mut logits = model.forward_with_cache(&prompt_input, &mut cache)?;

    let max_new_tokens = args.max_new_tokens.max(1);
    println!("   Max new tokens: {}", max_new_tokens);
    println!("   Streaming output:");
    print!("   ");
    io::stdout().flush().ok();
    let gen_start = Instant::now();
    let mut produced = 0usize;

    for _ in 0..max_new_tokens {
        let (_, l, _) = logits.dims3()?;
        let last_logits = logits
            .narrow(0, 0, 1)?
            .narrow(1, l - 1, 1)?
            .squeeze(0)?
            .squeeze(0)?;
        let next_token = last_logits.argmax(D::Minus1)?.to_scalar::<u32>()?;
        generated.push(next_token);
        produced += 1;

        if next_token == tokenizer.eos_id() {
            break;
        }

        // Stream one token chunk immediately so generation progress is visible.
        let piece = tokenizer.decode(&[next_token]);
        print!("{}", piece);
        if produced % 20 == 0 {
            let secs = gen_start.elapsed().as_secs_f32().max(1e-6);
            let tps = produced as f32 / secs;
            print!("\n   [{} / {} | {:.2} tok/s] ", produced, max_new_tokens, tps);
        }
        io::stdout().flush().ok();

        let step_input = Tensor::new([next_token].as_slice(), device)?.reshape(&[1, 1])?;
        logits = model.forward_with_cache(&step_input, &mut cache)?;
    }
    println!();

    let generated_text = tokenizer.decode(&generated);
    let elapsed = gen_start.elapsed().as_secs_f32().max(1e-6);
    let tps = produced as f32 / elapsed;
    println!(
        "   Generation speed: {:.2} tok/s ({:.2}s for {} new tokens)",
        tps, elapsed, produced
    );
    println!("   Generated tokens: {:?}", generated);
    println!("   Generated text: {}", generated_text);

    Ok(())
}
