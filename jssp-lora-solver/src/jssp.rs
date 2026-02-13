use anyhow::{anyhow, Result};
use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JSSPInstance {
    pub num_jobs: usize,
    pub num_machines: usize,
    /// Processing times: job_id -> machine_id -> time
    pub processing_times: Vec<u32>,
    /// Machine sequence: job_id -> [machine_id, machine_id, ...]
    pub machine_sequences: Vec<Vec<usize>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Schedule {
    pub job_sequence: Vec<usize>,
    pub makespan: u32,
    pub completion_times: Vec<u32>,
}

pub struct JSSPSolver;

impl JSSPInstance {
    pub fn new(
        num_jobs: usize,
        num_machines: usize,
        processing_times: Vec<u32>,
        machine_sequences: Vec<Vec<usize>>,
    ) -> Result<Self> {
        if processing_times.len() != num_jobs * num_machines {
            return Err(anyhow!(
                "Processing times length mismatch: expected {}, got {}",
                num_jobs * num_machines,
                processing_times.len()
            ));
        }

        Ok(Self {
            num_jobs,
            num_machines,
            processing_times,
            machine_sequences,
        })
    }

    /// Encode JSSP instance as feature vector
    pub fn encode(&self) -> Result<Vec<f32>> {
        let mut features = Vec::new();

        // Add problem size information
        features.push(self.num_jobs as f32);
        features.push(self.num_machines as f32);

        // Normalize and add processing times
        let max_time = *self.processing_times.iter().max().unwrap_or(&1) as f32;

        for &time in &self.processing_times {
            features.push((time as f32) / max_time);
        }

        // Pad to fixed size (256 dimensions)
        while features.len() < 256 {
            features.push(0.0);
        }

        Ok(features[..256].to_vec())
    }

    /// Get processing time for job j on machine m
    pub fn get_processing_time(&self, job: usize, machine: usize) -> u32 {
        if job < self.num_jobs && machine < self.num_machines {
            self.processing_times[job * self.num_machines + machine]
        } else {
            0
        }
    }
}

impl Schedule {
    /// Create a schedule from job sequence
    pub fn from_sequence(instance: &JSSPInstance, sequence: Vec<usize>) -> Result<Self> {
        let mut completion_times = vec![0u32; instance.num_jobs];
        let mut machine_available = vec![0u32; instance.num_machines];

        for &job in &sequence {
            if job >= instance.num_jobs {
                return Err(anyhow!("Invalid job id: {}", job));
            }

            for (step, &machine) in instance.machine_sequences[job].iter().enumerate() {
                let proc_time = instance.get_processing_time(job, step);
                let start_time = machine_available[machine].max(completion_times[job]);
                let end_time = start_time + proc_time as u32;

                completion_times[job] = end_time;
                machine_available[machine] = end_time;
            }
        }

        let makespan = *completion_times.iter().max().unwrap_or(&0);

        Ok(Schedule {
            job_sequence: sequence,
            makespan,
            completion_times,
        })
    }

    /// Convert schedule to tensor
    pub fn to_tensor(&self, device: &Device) -> Result<Tensor> {
        let mut tensor_data = Vec::new();

        // Add makespan
        tensor_data.push(self.makespan as f32);

        // Add completion times (normalized)
        for &time in &self.completion_times {
            tensor_data.push(time as f32);
        }

        // Pad to 256 dimensions
        while tensor_data.len() < 256 {
            tensor_data.push(0.0);
        }

        Tensor::new(&tensor_data[..256], device)
    }

    /// Create schedule from model output tensor
    pub fn from_tensor(tensor: &Tensor) -> Result<Self> {
        let data = tensor.to_vec1::<f32>()?;

        // Extract makespan (first element)
        let makespan = data[0].max(1.0) as u32;

        // Extract completion times from remaining elements
        let completion_times: Vec<u32> = data[1..std::cmp::min(256, data.len())]
            .iter()
            .map(|&x| (x.max(0.0) as u32))
            .filter(|&x| x > 0)
            .collect();

        // Generate job sequence (simplified: order by completion time)
        let mut sequence: Vec<usize> = (0..completion_times.len()).collect();
        sequence.sort_by_key(|&i| completion_times[i]);

        Ok(Schedule {
            job_sequence: sequence,
            makespan,
            completion_times,
        })
    }
}

impl JSSPSolver {
    /// Simple greedy solver for validation
    pub fn solve_greedy(instance: &JSSPInstance) -> Result<Schedule> {
        // Simple job order: 0, 1, 2, ...
        let sequence = (0..instance.num_jobs).collect();
        Schedule::from_sequence(instance, sequence)
    }

    /// Nearest Neighbor solver
    pub fn solve_nearest_neighbor(instance: &JSSPInstance) -> Result<Schedule> {
        let mut sequence = vec![0];
        let mut used = vec![false; instance.num_jobs];
        used[0] = true;

        while sequence.len() < instance.num_jobs {
            let last = *sequence.last().unwrap();
            let mut nearest = None;
            let mut min_distance = f32::MAX;

            for job in 0..instance.num_jobs {
                if used[job] {
                    continue;
                }

                // Simple distance metric
                let distance = (last as f32 - job as f32).abs();
                if distance < min_distance {
                    min_distance = distance;
                    nearest = Some(job);
                }
            }

            if let Some(job) = nearest {
                sequence.push(job);
                used[job] = true;
            }
        }

        Schedule::from_sequence(instance, sequence)
    }

    /// Random solution generator
    pub fn solve_random(instance: &JSSPInstance) -> Result<Schedule> {
        use rand::seq::SliceRandom;

        let mut sequence: Vec<usize> = (0..instance.num_jobs).collect();
        let mut rng = rand::rng();
        sequence.shuffle(&mut rng);

        Schedule::from_sequence(instance, sequence)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_instance() -> JSSPInstance {
        // 3 jobs, 2 machines
        let processing_times = vec![
            5, 4, // Job 0: 5 on M0, 4 on M1
            4, 5, // Job 1: 4 on M0, 5 on M1
            3, 6, // Job 2: 3 on M0, 6 on M1
        ];

        let machine_sequences = vec![
            vec![0, 1], // Job 0: M0 -> M1
            vec![0, 1], // Job 1: M0 -> M1
            vec![0, 1], // Job 2: M0 -> M1
        ];

        JSSPInstance::new(3, 2, processing_times, machine_sequences).unwrap()
    }

    #[test]
    fn test_instance_creation() {
        let instance = create_test_instance();
        assert_eq!(instance.num_jobs, 3);
        assert_eq!(instance.num_machines, 2);
    }

    #[test]
    fn test_schedule_generation() -> Result<()> {
        let instance = create_test_instance();
        let sequence = vec![0, 1, 2];
        let schedule = Schedule::from_sequence(&instance, sequence)?;

        assert!(schedule.makespan > 0);
        assert_eq!(schedule.completion_times.len(), instance.num_jobs);

        Ok(())
    }

    #[test]
    fn test_greedy_solver() -> Result<()> {
        let instance = create_test_instance();
        let schedule = JSSPSolver::solve_greedy(&instance)?;

        assert!(schedule.makespan > 0);
        assert_eq!(schedule.job_sequence.len(), instance.num_jobs);

        Ok(())
    }

    #[test]
    fn test_nn_solver() -> Result<()> {
        let instance = create_test_instance();
        let schedule = JSSPSolver::solve_nearest_neighbor(&instance)?;

        assert!(schedule.makespan > 0);
        assert_eq!(schedule.job_sequence.len(), instance.num_jobs);

        Ok(())
    }
}
