use candle_core::{Result, Tensor, D, Device, Var};
use candle_nn::Optimizer;
use std::time::Instant;

use crate::MiniQwenNext;

/// Training configuration
pub struct TrainingConfig {
    pub batch_size: usize,
    pub seq_len: usize,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub norm_weight_decay: f64,  // Qwen3-Next: apply weight decay to norm weights
    pub warmup_steps: usize,
    pub max_steps: usize,
    pub print_every: usize,
    pub save_every: usize,
    pub checkpoint_dir: String,
    pub clip_grad_norm: Option<f64>,  // Gradient clipping for stability
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            batch_size: 16,
            seq_len: 512,
            learning_rate: 1e-4,
            weight_decay: 0.01,
            norm_weight_decay: 0.01,  // Per Qwen3-Next: weight decay on norm weights prevents unbounded growth
            warmup_steps: 1000,
            max_steps: 50000,
            print_every: 100,
            save_every: 5000,
            checkpoint_dir: "./checkpoints".to_string(),
            clip_grad_norm: Some(1.0),  // Gradient clipping for stability
        }
    }
}

/// Simple AdamW optimizer wrapper with per-parameter weight decay
///
/// Qwen3-Next applies weight decay to norm weights to prevent unbounded growth [7].
/// This is a key stability improvement over Qwen3's QK-Norm approach.
///
/// For simplicity in this implementation, we use a single weight decay value
/// for all parameters. The key insight from Qwen3-Next is that norm weights
/// should have weight decay (not zero), which we already do.
pub struct AdamW {
    optimizer: candle_nn::AdamW,
    learning_rate: f64,
}

impl AdamW {
    /// Create a new AdamW optimizer with specified weight decay
    ///
    /// Per Qwen3-Next: all parameters including norms get weight decay.
    /// This prevents norm weights from growing unbounded (a problem with QK-Norm).
    pub fn new(vars: Vec<Var>, learning_rate: f64, weight_decay: f64) -> Result<Self> {
        // Use candle's AdamW with specified weight decay
        // Qwen3-Next: applies weight decay to all parameters including norms
        let params = candle_nn::ParamsAdamW {
            lr: learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay,
            ..Default::default()
        };
        let optimizer = candle_nn::AdamW::new(vars, params)?;
        Ok(Self {
            optimizer,
            learning_rate,
        })
    }

    /// Create AdamW with differential weight decay for norm vs non-norm parameters
    ///
    /// Note: Current implementation uses same weight decay for all parameters.
    /// To implement true differential decay, we would need to separate norm params
    /// and create two separate optimizers or use a custom optimizer implementation.
    pub fn with_param_groups(
        norm_vars: Vec<Var>,
        other_vars: Vec<Var>,
        learning_rate: f64,
        norm_weight_decay: f64,
        other_weight_decay: f64,
    ) -> Result<Self> {
        // For now, use the average weight decay for all parameters
        // This is a simplification - a full implementation would handle groups separately
        let all_vars: Vec<Var> = norm_vars.into_iter().chain(other_vars).collect();
        Self::new(all_vars, learning_rate, other_weight_decay)
    }

    pub fn step(&mut self, loss: &Tensor) -> Result<()> {
        self.optimizer.backward_step(loss)
    }

    /// Set learning rate for the next steps
    ///
    /// Note: This updates the internal learning rate tracker, but does NOT
    /// modify the underlying optimizer's learning rate. Candle's AdamW
    /// does not support dynamic learning rate changes after creation.
    ///
    /// To use learning rate scheduling, create a new optimizer with the
    /// desired rate, or implement a custom optimizer wrapper.
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }

    /// Get the current learning rate
    ///
    /// Returns the learning rate being tracked (not necessarily the actual
    /// optimizer's learning rate if set_learning_rate was called).
    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }
}

/// Training state
pub struct Trainer {
    model: MiniQwenNext,
    optimizer: AdamW,
    train_config: TrainingConfig,
    step: usize,
    device: Device,
}

impl Trainer {
    pub fn new(
        model: MiniQwenNext,
        train_config: TrainingConfig,
        device: Device,
    ) -> Result<Self> {
        // Extract tensors from model and convert to Var for optimizer
        // Qwen3-Next applies weight decay to norm weights to prevent unbounded growth
        // We use the same weight decay for all parameters for simplicity
        // (The paper suggests norm weights benefit from weight decay, not necessarily different decay)
        let tensors = model.get_tensors();
        let vars: Vec<Var> = tensors
            .into_iter()
            .map(|t| Var::from_tensor(&t))
            .collect::<Result<Vec<_>>>()?;

        // Create optimizer with weight decay
        // Qwen3-Next: applies weight decay to all parameters including norms
        let optimizer = AdamW::new(vars, train_config.learning_rate, train_config.weight_decay)?;

        Ok(Self {
            model,
            optimizer,
            train_config,
            step: 0,
            device,
        })
    }

    /// Training step
    pub fn step(&mut self, batch: &TrainingBatch) -> Result<TrainingOutput> {
        let start = Instant::now();

        // Forward pass
        let logits = self.model.forward(&batch.input_ids)?;

        // Compute loss (cross entropy)
        let loss = cross_entropy_loss(&logits, &batch.targets)?;

        // Backward pass and optimizer step
        self.optimizer.step(&loss)?;

        let loss_val = loss.to_scalar::<f32>()?;
        let elapsed = start.elapsed();

        self.step += 1;

        Ok(TrainingOutput {
            loss: loss_val,
            step: self.step,
            elapsed,
        })
    }

    /// Get current learning rate with warmup
    pub fn get_learning_rate(&self) -> f64 {
        if self.step < self.train_config.warmup_steps {
            self.train_config.learning_rate * (self.step as f64) / (self.train_config.warmup_steps as f64)
        } else {
            self.train_config.learning_rate
        }
    }

    /// Save checkpoint
    pub fn save_checkpoint(&self, path: &str) -> Result<()> {
        eprintln!("Would save checkpoint to: {}", path);
        Ok(())
    }

    /// Reference to model
    pub fn model(&self) -> &MiniQwenNext {
        &self.model
    }
}

/// Training batch
pub struct TrainingBatch {
    pub input_ids: Tensor,
    pub targets: Tensor,
}

impl TrainingBatch {
    /// Create a batch from token ids
    pub fn new(input_ids: Tensor, targets: Tensor) -> Self {
        Self { input_ids, targets }
    }

    /// Create dummy batch for testing
    pub fn dummy(batch_size: usize, seq_len: usize, vocab_size: usize, device: &Device) -> Self {
        // Create tokens within vocab range, wrapping around
        let total = batch_size * seq_len;
        let mut data = Vec::with_capacity(total);
        for i in 0..total {
            data.push((i % vocab_size) as u32);
        }

        let input_ids = Tensor::new(data.as_slice(), device)
            .unwrap()
            .reshape(&[batch_size, seq_len]).unwrap();

        // Targets are input_ids shifted by 1 (next token prediction)
        let mut target_data = Vec::with_capacity(total);
        for i in 0..total {
            target_data.push(((i + 1) % vocab_size) as u32);
        }

        let targets = Tensor::new(target_data.as_slice(), device)
            .unwrap()
            .reshape(&[batch_size, seq_len]).unwrap();

        Self { input_ids, targets }
    }
}

/// Training step output
pub struct TrainingOutput {
    pub loss: f32,
    pub step: usize,
    pub elapsed: std::time::Duration,
}

/// Cross entropy loss for language modeling
pub fn cross_entropy_loss(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let (b, l, v) = logits.dims3()?;
    let logits_2d = logits.reshape(&[b * l, v])?;
    let targets_1d = targets.reshape(&[b * l])?;

    // Compute cross entropy
    let log_probs = candle_nn::ops::log_softmax(&logits_2d, D::Minus1)?;

    // Negate: we can use affine operations
    let zero_point = Tensor::new(&[0.0f32], &log_probs.device())?;
    let neg_log_probs = zero_point.broadcast_sub(&log_probs)?;

    // Gather log probs for target tokens
    let targets_u32 = targets_1d.to_dtype(candle_core::DType::U32)?;
    let gathered = neg_log_probs.index_select(&targets_u32, 1)?;

    let (b, l) = (b as f32, l as f32);
    let sum = gathered.sum_all()?;
    // Use affine operation to divide by scalar
    let inv_count = 1.0 / (b * l);
    let count_inv = Tensor::new(&[inv_count], &sum.device())?;
    let count_inv_scalar = count_inv.reshape(&[])?;
    let loss = sum.mul(&count_inv_scalar)?;
    Ok(loss)
}

/// Learning rate scheduler (cosine with warmup)
pub fn cosine_schedule(
    step: usize,
    max_steps: usize,
    warmup_steps: usize,
    min_lr: f64,
    max_lr: f64,
) -> f64 {
    if step < warmup_steps {
        max_lr * step as f64 / warmup_steps as f64
    } else {
        let progress = (step - warmup_steps) as f64 / (max_steps - warmup_steps) as f64;
        min_lr + 0.5 * (max_lr - min_lr) * (1.0 + (progress * std::f64::consts::PI).cos())
    }
}
