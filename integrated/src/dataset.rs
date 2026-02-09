// Dataset Module - Load and process JSSP instances

use serde::{Deserialize, Serialize};
use anyhow::{Result, Context};
use std::fs;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JSPInstance {
    pub num_jobs: usize,
    pub num_machines: usize,
    pub input: String,
    pub output: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RawDataset {
    pub data: Vec<JSPInstance>,
}

pub struct StarjobDataset {
    pub instances: Vec<JSPInstance>,
    pub train_size: usize,
    pub val_size: usize,
}

impl StarjobDataset {
    pub fn load(path: &str, sample_size: usize) -> Result<Self> {
        let content = fs::read_to_string(path)
            .context("Failed to read dataset file")?;
        
        let raw_dataset: RawDataset = serde_json::from_str(&content)
            .context("Failed to parse JSON")?;
        
        let mut instances = raw_dataset.data;
        
        // Limit to sample size
        if instances.len() > sample_size {
            instances.truncate(sample_size);
        }
        
        let train_size = (instances.len() as f32 * 0.8) as usize;
        let val_size = instances.len() - train_size;
        
        Ok(StarjobDataset {
            instances,
            train_size,
            val_size,
        })
    }
    
    pub fn train_set(&self) -> Vec<&JSPInstance> {
        self.instances.iter().take(self.train_size).collect()
    }
    
    pub fn val_set(&self) -> Vec<&JSPInstance> {
        self.instances.iter().skip(self.train_size).collect()
    }
}
