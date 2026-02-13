use anyhow::Result;
use candle_core::Tensor;
use candle_nn::VarBuilder;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct LoRAConfig {
    pub rank: usize,
    pub alpha: f64,
    pub dropout: f64,
}

/// LoRA (Low-Rank Adaptation) layer
/// Implements: h = W_0 * x + (W_a @ W_b) * x
pub struct LoRALinear {
    // Original weight (frozen during training)
    _original_dim: (usize, usize),

    // LoRA parameters (trainable)
    w_a: Arc<Tensor>, // (hidden_dim, rank)
    w_b: Arc<Tensor>, // (rank, hidden_dim)

    lora_alpha: f64,
    dropout_rate: f64,
}

impl LoRALinear {
    pub fn new(
        vb: &VarBuilder,
        in_dim: usize,
        out_dim: usize,
        config: &LoRAConfig,
    ) -> Result<Self> {
        // Initialize LoRA matrices with small random values
        let w_a = vb.get((in_dim, config.rank), "w_a")?;
        let w_b = vb.get((config.rank, out_dim), "w_b")?;

        Ok(Self {
            _original_dim: (in_dim, out_dim),
            w_a: Arc::new(w_a),
            w_b: Arc::new(w_b),
            lora_alpha: config.alpha,
            dropout_rate: config.dropout,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Compute LoRA contribution: (W_a @ W_b) @ x
        // x shape: (batch, in_dim)

        let lora_out = x.matmul(&self.w_a)?; // (batch, rank)
        let lora_out = lora_out.matmul(&self.w_b)?; // (batch, out_dim)

        // Scale by alpha/rank
        let scale = self.lora_alpha / (self.w_a.shape().dims()[1] as f64);
        let lora_out = (&lora_out * scale)?;

        Ok(lora_out)
    }
}

pub struct LoRA {
    layers: Vec<LoRALinear>,
    config: LoRAConfig,
}

impl LoRA {
    pub fn new(vb: &VarBuilder, config: LoRAConfig) -> Result<Self> {
        let layer_dims = [
            (256, 512), // Input embedding to hidden
            (512, 512), // Hidden to hidden
            (512, 256), // Hidden to output
        ];

        let mut layers = Vec::new();
        for (i, (in_dim, out_dim)) in layer_dims.iter().enumerate() {
            let layer_vb = vb.pp(format!("lora_layer_{}", i));
            let layer = LoRALinear::new(&layer_vb, *in_dim, *out_dim, &config)?;
            layers.push(layer);
        }

        Ok(Self { layers, config })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut output = x.clone();

        for (i, layer) in self.layers.iter().enumerate() {
            let layer_out = layer.forward(&output)?;
            output = if i < self.layers.len() - 1 {
                // Apply ReLU activation to hidden layers
                layer_out.relu()?
            } else {
                layer_out
            };
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarMap;

    #[test]
    fn test_lora_forward() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let config = LoRAConfig {
            rank: 8,
            alpha: 16.0,
            dropout: 0.1,
        };

        let lora = LoRA::new(&vb, config)?;

        // Create dummy input (batch_size=2, input_dim=256)
        let input_data: Vec<f32> = (0..512).map(|i| (i as f32) / 1000.0).collect();
        let input = Tensor::new(&input_data, &device)?.reshape((2, 256))?;

        let output = lora.forward(&input)?;

        // Check output shape (batch_size=2, output_dim=256)
        assert_eq!(output.shape().dims(), &[2, 256]);

        Ok(())
    }
}
