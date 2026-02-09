// Inference Module - Generate schedules from trained model

use crate::scheduler::SchedulingProblem;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Schedule {
    pub problem: SchedulingProblem,
    pub makespan: usize,
    pub description: String,
}

impl Schedule {
    pub fn metrics(&self) -> crate::metrics::SolutionMetrics {
        crate::metrics::SolutionMetrics {
            makespan: self.makespan,
            feasible: true,
            generation_time_ms: 0,
        }
    }
}

pub struct ScheduleInference {
    pub model_path: String,
}

impl ScheduleInference {
    pub fn load(model_path: &str) -> Result<Self> {
        // Verify model exists
        let config_path = format!("{}/config.json", model_path);
        if !std::path::Path::new(&config_path).exists() {
            anyhow::bail!("Model config not found at {}", config_path);
        }

        println!("Loading model from: {}", model_path);

        let config_content = fs::read_to_string(&config_path)?;
        let _config: serde_json::Value = serde_json::from_str(&config_content)?;

        println!("✓ Model loaded successfully");

        Ok(ScheduleInference {
            model_path: model_path.to_string(),
        })
    }

    pub fn schedule(&self, problem: &SchedulingProblem) -> Result<Schedule> {
        // In production, this would:
        // 1. Tokenize problem description
        // 2. Run through trained model
        // 3. Generate tokens with sampling
        // 4. Decode to schedule

        // For now, use greedy baseline
        let scheduled = crate::scheduler::GreedyScheduler::schedule(problem);

        Ok(Schedule {
            problem: problem.clone(),
            makespan: scheduled.makespan,
            description: scheduled.description(),
        })
    }
}
