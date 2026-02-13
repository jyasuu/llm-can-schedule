/// Starjob Dataset Loader
/// 
/// This module provides functionality to load the Starjob130k dataset
/// from the JSON file format provided by the Starjob project.
///
/// Dataset source: https://github.com/starjob42/Starjob
/// Hugging Face: https://huggingface.co/datasets/henri24/Starjob

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs;
use crate::jssp::{JSSPInstance, Schedule};

/// Raw Starjob entry from JSON
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StarjobEntry {
    pub num_jobs: u64,
    pub num_machines: u64,
    pub instruction: String,
    pub input: String,
    pub output: String,
    pub matrix: Vec<Vec<u64>>,
}

/// Processed Starjob dataset
#[derive(Debug, Clone)]
pub struct StarjobDataset {
    pub instances: Vec<JSSPInstance>,
    pub entries: Vec<StarjobEntry>,
}

/// Load Starjob dataset from JSON file
/// 
/// # Arguments
/// * `path` - Path to starjob130k.json file
/// * `limit` - Optional limit on number of instances to load (None = load all)
/// 
/// # Returns
/// * `StarjobDataset` - Loaded instances and raw entries
/// 
/// # Example
/// ```rust
/// let dataset = load_starjob_dataset("data/starjob130k.json", Some(100))?;
/// println!("Loaded {} instances", dataset.instances.len());
/// ```
pub fn load_starjob_dataset(path: &str, limit: Option<usize>) -> Result<StarjobDataset> {
    println!("Loading Starjob dataset from: {}", path);
    
    // Read file
    let file_content = fs::read_to_string(path)
        .map_err(|e| anyhow!("Failed to read file {}: {}", path, e))?;

    // Parse JSON
    let entries = parse_starjob_json(&file_content, limit)?;
    println!("Parsed {} entries from JSON", entries.len());

    // Convert to JSSP instances
    let mut instances = Vec::new();
    let mut failed_count = 0;

    for (idx, entry) in entries.iter().enumerate() {
        match convert_entry_to_instance(entry) {
            Ok(instance) => {
                instances.push(instance);
            }
            Err(e) => {
                if idx < 5 {
                    eprintln!("Warning: Failed to convert entry {}: {}", idx, e);
                }
                failed_count += 1;
            }
        }
    }

    println!("Successfully converted {} instances", instances.len());
    if failed_count > 0 {
        println!("Failed to convert {} entries", failed_count);
    }

    Ok(StarjobDataset {
        instances,
        entries,
    })
}

/// Parse JSON file containing Starjob entries
fn parse_starjob_json(content: &str, limit: Option<usize>) -> Result<Vec<StarjobEntry>> {
    let mut entries = Vec::new();

    // Try parsing as JSON array
    if let Ok(Value::Array(arr)) = serde_json::from_str::<Value>(content) {
        for (idx, value) in arr.iter().enumerate() {
            if let Some(lim) = limit {
                if entries.len() >= lim {
                    break;
                }
            }

            match serde_json::from_value::<StarjobEntry>(value.clone()) {
                Ok(entry) => entries.push(entry),
                Err(e) => {
                    if idx < 3 {
                        eprintln!("Warning: Failed to parse entry {}: {}", idx, e);
                    }
                }
            }
        }
    } else if let Ok(Value::Object(_)) = serde_json::from_str::<Value>(content) {
        // Try parsing as single object (for testing)
        let entry: StarjobEntry = serde_json::from_str(content)?;
        entries.push(entry);
    } else {
        // Try line-by-line JSONL format
        for (idx, line) in content.lines().enumerate() {
            if let Some(lim) = limit {
                if entries.len() >= lim {
                    break;
                }
            }

            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            match serde_json::from_str::<StarjobEntry>(trimmed) {
                Ok(entry) => entries.push(entry),
                Err(e) => {
                    if idx < 3 {
                        eprintln!("Warning: Failed to parse line {}: {}", idx, e);
                    }
                }
            }
        }
    }

    if entries.is_empty() {
        return Err(anyhow!("No valid entries found in JSON file"));
    }

    Ok(entries)
}

/// Convert Starjob entry to JSSP instance
/// 
/// Starjob matrix format:
/// [job0_m0, job0_m1, ..., job0_mn, job1_m0, job1_m1, ..., job1_mn, ...]
fn convert_entry_to_instance(entry: &StarjobEntry) -> Result<JSSPInstance> {
    let num_jobs = entry.num_jobs as usize;
    let num_machines = entry.num_machines as usize;

    // Extract processing times from matrix
    let processing_times: Vec<u32> = entry.matrix.iter()
        .flat_map(|row| row.iter())
        .map(|&t| t as u32)
        .collect();

    // Validate matrix size
    if processing_times.len() != num_jobs * num_machines {
        return Err(anyhow!(
            "Matrix size mismatch: expected {}, got {}",
            num_jobs * num_machines,
            processing_times.len()
        ));
    }

    // Generate sequential machine ordering
    // (Each job visits machines in order 0, 1, 2, ...)
    let machine_sequences = (0..num_jobs)
        .map(|_| (0..num_machines).collect())
        .collect();

    JSSPInstance::new(num_jobs, num_machines, processing_times, machine_sequences)
}

/// Convert Starjob entry with custom machine ordering
/// 
/// Useful when you have specific machine sequences in the data
pub fn convert_entry_with_sequences(
    entry: &StarjobEntry,
    machine_sequences: Vec<Vec<usize>>,
) -> Result<JSSPInstance> {
    let num_jobs = entry.num_jobs as usize;
    let num_machines = entry.num_machines as usize;

    let processing_times: Vec<u32> = entry.matrix.iter()
        .flat_map(|row| row.iter())
        .map(|&t| t as u32)
        .collect();

    if processing_times.len() != num_jobs * num_machines {
        return Err(anyhow!(
            "Matrix size mismatch: expected {}, got {}",
            num_jobs * num_machines,
            processing_times.len()
        ));
    }

    if machine_sequences.len() != num_jobs {
        return Err(anyhow!(
            "Machine sequences count mismatch: expected {}, got {}",
            num_jobs,
            machine_sequences.len()
        ));
    }

    JSSPInstance::new(num_jobs, num_machines, processing_times, machine_sequences)
}

/// Load a subset of Starjob dataset by size (number of jobs/machines)
/// 
/// # Arguments
/// * `path` - Path to starjob130k.json
/// * `min_jobs` - Minimum number of jobs
/// * `max_jobs` - Maximum number of jobs
/// * `min_machines` - Minimum number of machines
/// * `max_machines` - Maximum number of machines
/// * `limit` - Maximum number of instances to load
pub fn load_starjob_by_size(
    path: &str,
    min_jobs: usize,
    max_jobs: usize,
    min_machines: usize,
    max_machines: usize,
    limit: Option<usize>,
) -> Result<StarjobDataset> {
    let full_dataset = load_starjob_dataset(path, None)?;

    let filtered_instances: Vec<JSSPInstance> = full_dataset.instances
        .iter()
        .filter(|inst| {
            inst.num_jobs >= min_jobs
                && inst.num_jobs <= max_jobs
                && inst.num_machines >= min_machines
                && inst.num_machines <= max_machines
        })
        .take(limit.unwrap_or(usize::MAX))
        .cloned()
        .collect();

    let filtered_entries: Vec<StarjobEntry> = full_dataset.entries
        .iter()
        .zip(full_dataset.instances.iter())
        .filter_map(|(entry, inst)| {
            if inst.num_jobs >= min_jobs
                && inst.num_jobs <= max_jobs
                && inst.num_machines >= min_machines
                && inst.num_machines <= max_machines
            {
                Some(entry.clone())
            } else {
                None
            }
        })
        .take(limit.unwrap_or(usize::MAX))
        .collect();

    println!(
        "Filtered to {} instances with {} <= jobs <= {} and {} <= machines <= {}",
        filtered_instances.len(),
        min_jobs,
        max_jobs,
        min_machines,
        max_machines
    );

    Ok(StarjobDataset {
        instances: filtered_instances,
        entries: filtered_entries,
    })
}

/// Load Starjob dataset and compute statistics
pub fn load_and_analyze_starjob(path: &str, sample_size: Option<usize>) -> Result<StarjobStatistics> {
    let dataset = load_starjob_dataset(path, sample_size)?;

    let mut job_counts = std::collections::HashMap::new();
    let mut machine_counts = std::collections::HashMap::new();
    let mut processing_times = Vec::new();

    for instance in &dataset.instances {
        *job_counts.entry(instance.num_jobs).or_insert(0) += 1;
        *machine_counts
            .entry(instance.num_machines)
            .or_insert(0) += 1;
        processing_times.extend(&instance.processing_times);
    }

    let total_instances = dataset.instances.len();
    let avg_time = if !processing_times.is_empty() {
        processing_times.iter().map(|&t| t as f64).sum::<f64>() / processing_times.len() as f64
    } else {
        0.0
    };

    let min_time = processing_times.iter().min().copied().unwrap_or(0);
    let max_time = processing_times.iter().max().copied().unwrap_or(0);

    Ok(StarjobStatistics {
        total_instances,
        job_distribution: job_counts,
        machine_distribution: machine_counts,
        avg_processing_time: avg_time,
        min_processing_time: min_time,
        max_processing_time: max_time,
    })
}

/// Statistics about Starjob dataset
#[derive(Debug, Clone)]
pub struct StarjobStatistics {
    pub total_instances: usize,
    pub job_distribution: std::collections::HashMap<usize, usize>,
    pub machine_distribution: std::collections::HashMap<usize, usize>,
    pub avg_processing_time: f64,
    pub min_processing_time: u32,
    pub max_processing_time: u32,
}

impl StarjobStatistics {
    pub fn print_summary(&self) {
        println!("\n=== Starjob Dataset Statistics ===\n");
        println!("Total instances: {}", self.total_instances);

        println!("\nJob distribution:");
        let mut jobs: Vec<_> = self.job_distribution.iter().collect();
        jobs.sort_by_key(|&(j, _)| j);
        for (num_jobs, count) in jobs {
            println!("  {} jobs: {} instances", num_jobs, count);
        }

        println!("\nMachine distribution:");
        let mut machines: Vec<_> = self.machine_distribution.iter().collect();
        machines.sort_by_key(|&(m, _)| m);
        for (num_machines, count) in machines {
            println!("  {} machines: {} instances", num_machines, count);
        }

        println!("\nProcessing time statistics:");
        println!("  Min: {}", self.min_processing_time);
        println!("  Max: {}", self.max_processing_time);
        println!("  Average: {:.2}", self.avg_processing_time);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_starjob_entry_parsing() {
        let json = r#"{
            "num_jobs": 3,
            "num_machines": 2,
            "instruction": "Test",
            "input": "test input",
            "output": "test output",
            "matrix": [[5, 4], [4, 5], [3, 6]]
        }"#;

        let entry: StarjobEntry = serde_json::from_str(json).unwrap();
        assert_eq!(entry.num_jobs, 3);
        assert_eq!(entry.num_machines, 2);
        assert_eq!(entry.matrix.len(), 3);
    }

    #[test]
    fn test_convert_entry_to_instance() -> Result<()> {
        let entry = StarjobEntry {
            num_jobs: 3,
            num_machines: 2,
            instruction: "Test".to_string(),
            input: "input".to_string(),
            output: "output".to_string(),
            matrix: vec![vec![5, 4], vec![4, 5], vec![3, 6]],
        };

        let instance = convert_entry_to_instance(&entry)?;
        assert_eq!(instance.num_jobs, 3);
        assert_eq!(instance.num_machines, 2);
        assert_eq!(instance.processing_times.len(), 6);

        Ok(())
    }

    #[test]
    fn test_parse_jsonl_format() -> Result<()> {
        let content = r#"{"num_jobs": 2, "num_machines": 2, "instruction": "Test", "input": "in", "output": "out", "matrix": [[1, 2], [3, 4]]}
{"num_jobs": 3, "num_machines": 2, "instruction": "Test2", "input": "in2", "output": "out2", "matrix": [[5, 4], [4, 5], [3, 6]]}"#;

        let entries = parse_starjob_json(content, None)?;
        assert_eq!(entries.len(), 2);

        Ok(())
    }
}
