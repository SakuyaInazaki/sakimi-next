use candle_core::{Error, Result, Tensor};

use crate::Config;

/// Dynamic cache aligned with Qwen3-Next hybrid decoder needs.
///
/// For full-attention layers, stores KV cache.
/// For linear-attention layers, stores convolution and recurrent states.
#[derive(Clone)]
pub struct Qwen3NextDynamicCache {
    layer_types: Vec<String>,
    key_cache: Vec<Option<Tensor>>,
    value_cache: Vec<Option<Tensor>>,
    conv_states: Vec<Option<Tensor>>,
    recurrent_states: Vec<Option<Tensor>>,
    last_linear_layer: Option<usize>,
    seen_tokens: usize,
}

impl Qwen3NextDynamicCache {
    pub fn new(cfg: &Config) -> Self {
        let last_linear_layer = cfg
            .layer_types
            .iter()
            .rposition(|t| t.as_str() == "linear_attention");

        let n_layers = cfg.n_layers;
        Self {
            layer_types: cfg.layer_types.clone(),
            key_cache: vec![None; n_layers],
            value_cache: vec![None; n_layers],
            conv_states: vec![None; n_layers],
            recurrent_states: vec![None; n_layers],
            last_linear_layer,
            seen_tokens: 0,
        }
    }

    pub fn seen_tokens(&self) -> usize {
        self.seen_tokens
    }

    pub fn advance(&mut self, n_tokens: usize) {
        self.seen_tokens += n_tokens;
    }

    pub fn len(&self) -> usize {
        self.layer_types.len()
    }

    pub fn has_previous_state(&self) -> bool {
        self.last_linear_layer
            .and_then(|idx| self.conv_states.get(idx))
            .and_then(|x| x.as_ref())
            .is_some()
    }

    fn check_layer_idx(&self, layer_idx: usize) -> Result<()> {
        if layer_idx >= self.len() {
            return Err(Error::Msg(format!(
                "layer_idx {} out of range for {} layers",
                layer_idx,
                self.len()
            )));
        }
        Ok(())
    }

    pub fn attention_seq_len(&self, layer_idx: usize) -> Result<usize> {
        self.check_layer_idx(layer_idx)?;
        if let Some(ref k) = self.key_cache[layer_idx] {
            let (_, l, _, _) = k.dims4()?;
            Ok(l)
        } else {
            Ok(0)
        }
    }

    pub fn update_attention(
        &mut self,
        layer_idx: usize,
        key_states: &Tensor,
        value_states: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        self.check_layer_idx(layer_idx)?;

        let key_out = if let Some(ref key_cache) = self.key_cache[layer_idx] {
            Tensor::cat(&[key_cache, key_states], 1)?
        } else {
            key_states.clone()
        };

        let value_out = if let Some(ref value_cache) = self.value_cache[layer_idx] {
            Tensor::cat(&[value_cache, value_states], 1)?
        } else {
            value_states.clone()
        };

        self.key_cache[layer_idx] = Some(key_out.clone());
        self.value_cache[layer_idx] = Some(value_out.clone());

        Ok((key_out, value_out))
    }

    /// Beam-search style cache reordering by batch index.
    pub fn reorder_cache(&mut self, beam_idx: &Tensor) -> Result<()> {
        for layer_idx in 0..self.len() {
            if let Some(ref key_cache) = self.key_cache[layer_idx] {
                let beam_idx = beam_idx.to_device(key_cache.device())?;
                self.key_cache[layer_idx] = Some(key_cache.index_select(&beam_idx, 0)?);
                if let Some(ref value_cache) = self.value_cache[layer_idx] {
                    self.value_cache[layer_idx] = Some(value_cache.index_select(&beam_idx, 0)?);
                }
            }

            if let Some(ref conv_state) = self.conv_states[layer_idx] {
                let beam_idx = beam_idx.to_device(conv_state.device())?;
                self.conv_states[layer_idx] = Some(conv_state.index_select(&beam_idx, 0)?);
                if let Some(ref recurrent_state) = self.recurrent_states[layer_idx] {
                    self.recurrent_states[layer_idx] =
                        Some(recurrent_state.index_select(&beam_idx, 0)?);
                }
            }
        }
        Ok(())
    }

    /// Sequence length currently stored in attention KV cache.
    pub fn get_seq_length(&self, layer_idx: Option<usize>) -> Result<usize> {
        let idx = if let Some(idx) = layer_idx {
            idx
        } else if let Some(first_attn_idx) = self
            .layer_types
            .iter()
            .position(|t| t.as_str() == "full_attention")
        {
            first_attn_idx
        } else {
            return Ok(0);
        };
        self.attention_seq_len(idx)
    }

    pub fn conv_state(&self, layer_idx: usize) -> Option<Tensor> {
        self.conv_states
            .get(layer_idx)
            .and_then(|x| x.as_ref())
            .cloned()
    }

    pub fn set_conv_state(&mut self, layer_idx: usize, state: Tensor) -> Result<()> {
        self.check_layer_idx(layer_idx)?;
        self.conv_states[layer_idx] = Some(state);
        Ok(())
    }

    pub fn recurrent_state(&self, layer_idx: usize) -> Option<Tensor> {
        self.recurrent_states
            .get(layer_idx)
            .and_then(|x| x.as_ref())
            .cloned()
    }

    pub fn set_recurrent_state(&mut self, layer_idx: usize, state: Tensor) -> Result<()> {
        self.check_layer_idx(layer_idx)?;
        self.recurrent_states[layer_idx] = Some(state);
        Ok(())
    }
}
