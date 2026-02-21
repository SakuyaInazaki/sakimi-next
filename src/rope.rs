use candle_core::{Result, Tensor};
use std::sync::{Arc, RwLock};

/// Cached RoPE frequencies
#[derive(Clone)]
struct RopeCache {
    sin: Tensor,
    cos: Tensor,
    seq_len: usize,
}

/// Rotary Position Encoding (RoPE)
///
/// Qwen3-Next applies RoPE only to the first 25% of position dimensions,
/// which improves extrapolation to longer sequences.
///
/// This implementation caches sin/cos values using RwLock for thread safety
/// with better read performance than Mutex.
#[derive(Clone)]
pub struct RoPE {
    dim: usize,
    max_seq_len: usize,
    base: f64,
    // Use RwLock for better read performance: multiple readers can access cache simultaneously
    cache: Arc<RwLock<Option<RopeCache>>>,
}

impl RoPE {
    pub fn new(head_dim: usize, max_seq_len: usize, rope_ratio: f64, rope_theta: f64) -> Self {
        // Ensure at least 2 dimensions for RoPE (need at least 1 pair)
        let dim = ((head_dim as f64 * rope_ratio) as usize).max(2);
        // Also ensure dim is even
        let dim = if dim % 2 == 0 { dim } else { dim - 1 };
        Self {
            dim,
            max_seq_len,
            base: rope_theta,
            cache: Arc::new(RwLock::new(None)),
        }
    }

    /// Build cached sin/cos values for the given sequence length
    fn build_cache(&self, seq_len: usize, device: &candle_core::Device) -> Result<RopeCache> {
        // Create inv_freq: theta_i = base^(-2i/dim) for i in [0, dim/2)
        let inv_freq: Vec<f64> = (0..(self.dim / 2))
            .map(|i| 1.0 / (self.base.powf((2 * i) as f64 / self.dim as f64)))
            .collect();

        // Compute freqs: positions * inv_freq
        let mut freqs = vec![0.0; seq_len * (self.dim / 2)];
        for pos in 0..seq_len {
            for (dim_idx, &inv_f) in inv_freq.iter().enumerate() {
                freqs[pos * (self.dim / 2) + dim_idx] = (pos as f64) * inv_f;
            }
        }

        // Compute sin and cos
        let mut sin_vals = vec![0.0f32; freqs.len()];
        let mut cos_vals = vec![0.0f32; freqs.len()];
        for (i, &f) in freqs.iter().enumerate() {
            sin_vals[i] = f.sin() as f32;
            cos_vals[i] = f.cos() as f32;
        }

        // Create sin/cos tensors: (seq_len, dim/2)
        let sin = Tensor::new(sin_vals.as_slice(), device)?.reshape(&[seq_len, self.dim / 2])?;
        let cos = Tensor::new(cos_vals.as_slice(), device)?.reshape(&[seq_len, self.dim / 2])?;

        Ok(RopeCache { sin, cos, seq_len })
    }

    /// Get or build cached sin/cos values
    ///
    /// This method uses RwLock for better read performance:
    /// - Multiple threads can read the cache simultaneously
    /// - Only one thread can write (when building new cache)
    fn get_cache(&self, seq_len: usize, device: &candle_core::Device) -> Result<RopeCache> {
        let cache_len_needed = seq_len.max(self.max_seq_len);

        // Fast path: try read lock first (multiple readers can access simultaneously)
        if let Ok(read_guard) = self.cache.try_read() {
            if let Some(ref cache) = *read_guard {
                if cache.seq_len >= seq_len {
                    // Cache is valid, clone and return
                    // (drop read guard before doing expensive tensor operations)
                    drop(read_guard);

                    // Re-acquire for the actual operation
                    let read_guard = self.cache.read().map_err(|e| {
                        candle_core::Error::Msg(format!("RoPE cache lock poisoned: {}", e))
                    })?;
                    if let Some(ref cache) = *read_guard {
                        if cache.seq_len == seq_len {
                            return Ok(RopeCache {
                                sin: cache.sin.clone(),
                                cos: cache.cos.clone(),
                                seq_len,
                            });
                        }
                        let sin = cache.sin.narrow(0, 0, seq_len)?;
                        let cos = cache.cos.narrow(0, 0, seq_len)?;
                        return Ok(RopeCache { sin, cos, seq_len });
                    }
                }
            }
        }

        // Slow path: need to build or extend cache
        let mut write_guard = self
            .cache
            .write()
            .map_err(|e| candle_core::Error::Msg(format!("RoPE cache lock poisoned: {}", e)))?;

        // Check again in case another thread already built it
        if let Some(ref cache) = *write_guard {
            if cache.seq_len >= seq_len {
                if cache.seq_len == seq_len {
                    return Ok(RopeCache {
                        sin: cache.sin.clone(),
                        cos: cache.cos.clone(),
                        seq_len,
                    });
                }
                let sin = cache.sin.narrow(0, 0, seq_len)?;
                let cos = cache.cos.narrow(0, 0, seq_len)?;
                return Ok(RopeCache { sin, cos, seq_len });
            }
        }

        // Build new cache
        let new_cache = self.build_cache(cache_len_needed, device)?;

        // Store the cache
        *write_guard = Some(RopeCache {
            sin: new_cache.sin.clone(),
            cos: new_cache.cos.clone(),
            seq_len: cache_len_needed,
        });

        // Return appropriately sized cache
        if cache_len_needed == seq_len {
            Ok(new_cache)
        } else {
            let sin = new_cache.sin.narrow(0, 0, seq_len)?;
            let cos = new_cache.cos.narrow(0, 0, seq_len)?;
            Ok(RopeCache { sin, cos, seq_len })
        }
    }

    /// Apply RoPE to Q and K tensors
    ///
    /// Input shapes:
    /// - q: (batch_size, seq_len, n_heads, head_dim)
    /// - k: (batch_size, seq_len, n_kv_heads, head_dim)
    ///
    /// Only first 25% of head_dim is rotated.
    pub fn forward(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (b, l, _n_heads, _head_dim) = q.dims4()?;
        let device = q.device();

        // Get cached sin/cos values
        let total_len = offset + l;
        let cache = self.get_cache(total_len, device)?;

        // Extract the relevant slice [offset:offset+l]
        let (sin, cos) = if cache.seq_len == l && offset == 0 {
            (cache.sin.clone(), cache.cos.clone())
        } else {
            let sin = cache.sin.narrow(0, offset, l)?;
            let cos = cache.cos.narrow(0, offset, l)?;
            (sin, cos)
        };

        // sin, cos have shape (l, dim/2), need to expand to (b, l, 1, dim/2)
        let sin = sin
            .reshape(&[l, self.dim / 2])?
            .reshape(&[1, l, 1, self.dim / 2])?
            .broadcast_as(&[b, l, 1, self.dim / 2])?;
        let cos = cos
            .reshape(&[l, self.dim / 2])?
            .reshape(&[1, l, 1, self.dim / 2])?
            .broadcast_as(&[b, l, 1, self.dim / 2])?;

        // Apply rotary embedding to Q
        let q_rotated = self.apply_rotary(q, &cos, &sin, self.dim)?;

        // Apply rotary embedding to K
        let k_rotated = self.apply_rotary(k, &cos, &sin, self.dim)?;

        Ok((q_rotated, k_rotated))
    }

    /// Apply rotary transformation
    ///
    /// Only the first `rot_dim` dimensions are rotated.
    /// Formula:
    ///   x_rot[:rot_dim/2] = x[:rot_dim/2] * cos - x[rot_dim/2:rot_dim] * sin
    ///   x_rot[rot_dim/2:rot_dim] = x[:rot_dim/2] * sin + x[rot_dim/2:rot_dim] * cos
    ///   x_rot[rot_dim:] = x[rot_dim:] (unchanged)
    fn apply_rotary(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        rot_dim: usize,
    ) -> Result<Tensor> {
        let (b, l, n_heads, head_dim) = x.dims4()?;
        let half_dim = rot_dim / 2;

        // Split into left and right halves
        let x_left = x.narrow(3, 0, half_dim)?; // (B, L, H, half_dim)
        let x_right = x.narrow(3, half_dim, half_dim)?; // (B, L, H, half_dim)
        let x_rest = if rot_dim < head_dim {
            Some(x.narrow(3, rot_dim, head_dim - rot_dim)?)
        } else {
            None
        };

        // Broadcast cos and sin to match head count
        let cos = cos.broadcast_as(&[b, l, n_heads, half_dim])?;
        let sin = sin.broadcast_as(&[b, l, n_heads, half_dim])?;

        // Apply rotation
        // left_rot = left * cos - right * sin
        let left_rot = x_left
            .broadcast_mul(&cos)?
            .broadcast_sub(&x_right.broadcast_mul(&sin)?)?;

        // right_rot = left * sin + right * cos
        let right_rot = x_left
            .broadcast_mul(&sin)?
            .broadcast_add(&x_right.broadcast_mul(&cos)?)?;

        // Concatenate: [left_rot, right_rot, rest]
        let rotated = Tensor::cat(&[&left_rot, &right_rot], 3)?;

        match x_rest {
            Some(rest) => Tensor::cat(&[&rotated, &rest], 3),
            None => Ok(rotated),
        }
    }

    /// Get the rotary dimension
    pub fn dim(&self) -> usize {
        self.dim
    }
}
