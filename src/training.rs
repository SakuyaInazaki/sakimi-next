use candle_core::{Device, Result, Tensor, Var};
use candle_nn::Optimizer;
use std::collections::HashMap;
use std::time::Instant;

use crate::MiniQwenNext;

/// Training configuration
pub struct TrainingConfig {
    pub batch_size: usize,
    pub seq_len: usize,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub norm_weight_decay: f64, // Qwen3-Next: apply weight decay to norm weights
    pub warmup_steps: usize,
    pub max_steps: usize,
    pub print_every: usize,
    pub save_every: usize,
    pub checkpoint_dir: String,
    pub clip_grad_norm: Option<f64>, // Gradient clipping for stability
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            batch_size: 16,
            seq_len: 512,
            learning_rate: 1e-4,
            weight_decay: 0.01,
            norm_weight_decay: 0.01, // Per Qwen3-Next: weight decay on norm weights prevents unbounded growth
            warmup_steps: 1000,
            max_steps: 50000,
            print_every: 100,
            save_every: 5000,
            checkpoint_dir: "./checkpoints".to_string(),
            clip_grad_norm: Some(1.0), // Gradient clipping for stability
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
        // Candle AdamW has a single decay value, so we blend both groups by count.
        let norm_count = norm_vars.len();
        let other_count = other_vars.len();
        let total_count = norm_count + other_count;
        let blended_weight_decay = if total_count == 0 {
            other_weight_decay
        } else {
            (norm_weight_decay * norm_count as f64 + other_weight_decay * other_count as f64)
                / total_count as f64
        };
        let all_vars: Vec<Var> = norm_vars.into_iter().chain(other_vars).collect();
        Self::new(all_vars, learning_rate, blended_weight_decay)
    }

    pub fn step(&mut self, loss: &Tensor) -> Result<()> {
        self.optimizer.backward_step(loss)
    }

    /// Set learning rate for the next steps
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
        self.optimizer.set_learning_rate(lr);
    }

    /// Get the current learning rate
    ///
    /// Returns the learning rate currently used by the optimizer.
    pub fn learning_rate(&self) -> f64 {
        self.optimizer.learning_rate()
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
    pub fn new(model: MiniQwenNext, train_config: TrainingConfig, device: Device) -> Result<Self> {
        model.set_training(true);

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

        std::fs::create_dir_all(&train_config.checkpoint_dir).map_err(|e| {
            candle_core::Error::Msg(format!(
                "Failed to create checkpoint directory {}: {}",
                train_config.checkpoint_dir, e
            ))
        })?;

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
        let lr = self.get_learning_rate();
        self.optimizer.set_learning_rate(lr);

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
            learning_rate: lr,
        })
    }

    /// Get current learning rate with warmup
    pub fn get_learning_rate(&self) -> f64 {
        if self.step < self.train_config.warmup_steps {
            self.train_config.learning_rate * ((self.step + 1) as f64)
                / (self.train_config.warmup_steps as f64)
        } else {
            self.train_config.learning_rate
        }
    }

    /// Save checkpoint
    pub fn save_checkpoint(&self, path: &str) -> Result<()> {
        let path = std::path::Path::new(path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                candle_core::Error::Msg(format!(
                    "Failed to create checkpoint parent directory {:?}: {}",
                    parent, e
                ))
            })?;
        }

        let tensors = self.model.get_tensors();
        let mut named_tensors: HashMap<String, Tensor> = HashMap::with_capacity(tensors.len());
        for (idx, tensor) in tensors.into_iter().enumerate() {
            named_tensors.insert(format!("param_{idx:05}"), tensor);
        }
        candle_core::safetensors::save(&named_tensors, path)?;

        let metadata_path = path.with_extension("json");
        let metadata = serde_json::json!({
            "step": self.step,
            "learning_rate": self.optimizer.learning_rate(),
            "num_tensors": named_tensors.len(),
        });
        let metadata_str = serde_json::to_string_pretty(&metadata)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to serialize metadata: {}", e)))?;
        std::fs::write(&metadata_path, metadata_str).map_err(|e| {
            candle_core::Error::Msg(format!(
                "Failed to write checkpoint metadata {:?}: {}",
                metadata_path, e
            ))
        })?;

        Ok(())
    }

    /// Load model weights from a safetensors checkpoint.
    ///
    /// Checkpoint format follows save_checkpoint: param_00000, param_00001, ...
    /// If sidecar metadata json exists, trainer step is restored from step.
    pub fn load_checkpoint(&mut self, path: &str) -> Result<()> {
        let ckpt_path = std::path::Path::new(path);
        if !ckpt_path.exists() {
            return Err(candle_core::Error::Msg(format!(
                "Checkpoint not found: {}",
                ckpt_path.display()
            )));
        }

        let tensors = candle_core::safetensors::load(ckpt_path, &self.device)?;
        let model_tensors = self.model.get_tensors();

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

        let metadata_path = ckpt_path.with_extension("json");
        if metadata_path.exists() {
            let metadata_raw = std::fs::read_to_string(&metadata_path).map_err(|e| {
                candle_core::Error::Msg(format!(
                    "Failed to read checkpoint metadata {:?}: {}",
                    metadata_path, e
                ))
            })?;
            let metadata: serde_json::Value = serde_json::from_str(&metadata_raw).map_err(|e| {
                candle_core::Error::Msg(format!(
                    "Failed to parse checkpoint metadata {:?}: {}",
                    metadata_path, e
                ))
            })?;
            if let Some(step) = metadata.get("step").and_then(|v| v.as_u64()) {
                self.step = step as usize;
            }
        }

        Ok(())
    }

    /// Current global optimization step.
    pub fn current_step(&self) -> usize {
        self.step
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
            .reshape(&[batch_size, seq_len])
            .unwrap();

        // Targets are input_ids shifted by 1 (next token prediction)
        let mut target_data = Vec::with_capacity(total);
        for i in 0..total {
            target_data.push(((i + 1) % vocab_size) as u32);
        }

        let targets = Tensor::new(target_data.as_slice(), device)
            .unwrap()
            .reshape(&[batch_size, seq_len])
            .unwrap();

        Self { input_ids, targets }
    }
}

/// Training step output
pub struct TrainingOutput {
    pub loss: f32,
    pub step: usize,
    pub elapsed: std::time::Duration,
    pub learning_rate: f64,
}

/// Cross entropy loss for language modeling
pub fn cross_entropy_loss(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let (b, l, v) = logits.dims3()?;
    let logits_2d = logits.reshape(&[b * l, v])?;
    let targets_1d = targets.reshape(&[b * l])?.to_dtype(candle_core::DType::U32)?;

    // Use Candle's optimized cross-entropy implementation.
    candle_nn::loss::cross_entropy(&logits_2d, &targets_1d)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Config;
    use candle_core::{DType, Device};
    use candle_nn::VarBuilder;

    #[test]
    fn test_training_step_updates_model_weights() -> Result<()> {
        let device = Device::Cpu;
        let cfg = Config::tiny_5m();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MiniQwenNext::new(&vb, &device, cfg.clone())?;

        let train_cfg = TrainingConfig {
            batch_size: 1,
            seq_len: 4,
            learning_rate: 1e-3,
            warmup_steps: 1,
            max_steps: 1,
            ..Default::default()
        };
        let mut trainer = Trainer::new(model, train_cfg, device.clone())?;

        let before = trainer
            .model()
            .get_tensors()
            .into_iter()
            .map(|t| t.force_contiguous())
            .collect::<Result<Vec<_>>>()?;
        let batch = TrainingBatch::dummy(1, 4, cfg.vocab_size, &device);
        let logits = trainer.model().forward(&batch.input_ids)?;
        let loss = cross_entropy_loss(&logits, &batch.targets)?;
        let grads = loss.backward()?;
        let grad_count = trainer
            .model()
            .get_tensors()
            .iter()
            .filter(|t| grads.get(t).is_some())
            .count();
        assert!(grad_count > 0, "no gradient found for model tensors");

        trainer.step(&batch)?;
        let after = trainer.model().get_tensors();

        let mut changed = false;
        for (before_t, after_t) in before.iter().zip(after.iter()) {
            let diff = before_t
                .broadcast_sub(after_t)?
                .abs()?
                .sum_all()?
                .to_scalar::<f32>()?;
            if diff > 0.0 {
                changed = true;
                break;
            }
        }

        assert!(changed, "no trainable parameter was updated");
        Ok(())
    }

    #[test]
    fn test_warmup_learning_rate_is_nonzero_on_first_step() {
        let cfg = TrainingConfig {
            learning_rate: 1e-4,
            warmup_steps: 1000,
            ..Default::default()
        };
        let trainer = Trainer {
            model: panic_model(),
            optimizer: panic_optimizer(),
            train_config: cfg,
            step: 0,
            device: Device::Cpu,
        };
        assert!(trainer.get_learning_rate() > 0.0);
    }

    fn panic_model() -> MiniQwenNext {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        MiniQwenNext::new(&vb, &device, Config::tiny_5m()).unwrap()
    }

    fn panic_optimizer() -> AdamW {
        let vars: Vec<Var> = vec![];
        AdamW::new(vars, 1e-4, 0.01).unwrap()
    }
}
