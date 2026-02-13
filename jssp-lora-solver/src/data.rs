use crate::jssp::{JSSPInstance, JSSPSolver, Schedule};
use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JSSPDataset {
    pub instances: Vec<JSSPInstance>,
    pub optimal_schedules: Vec<Schedule>,
}

/// Load sample JSSP instances for training
pub fn load_sample_data() -> Result<JSSPDataset> {
    let instances = vec![
        // 3x2 JSSP (3 jobs, 2 machines)
        create_3x2_instance()?,
        // 4x3 JSSP (4 jobs, 3 machines)
        create_4x3_instance()?,
        // 5x3 JSSP (5 jobs, 3 machines)
        create_5x3_instance()?,
        // Another 3x2 variant
        create_3x2_variant()?,
        // Another 4x3 variant
        create_4x3_variant()?,
    ];

    // Generate optimal-like schedules using heuristics
    let optimal_schedules = instances
        .iter()
        .map(|instance| {
            // Try multiple approaches and pick the best
            let greedy = JSSPSolver::solve_greedy(instance).ok();
            let nn = JSSPSolver::solve_nearest_neighbor(instance).ok();

            

            match (greedy, nn) {
                (Some(g), Some(n)) => {
                    if g.makespan <= n.makespan {
                        g
                    } else {
                        n
                    }
                }
                (Some(g), None) => g,
                (None, Some(n)) => n,
                _ => Schedule {
                    job_sequence: (0..instance.num_jobs).collect(),
                    makespan: 0,
                    completion_times: vec![0; instance.num_jobs],
                },
            }
        })
        .collect();

    Ok(JSSPDataset {
        instances,
        optimal_schedules,
    })
}

/// 3 jobs, 2 machines benchmark
fn create_3x2_instance() -> Result<JSSPInstance> {
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

    JSSPInstance::new(3, 2, processing_times, machine_sequences)
}

/// 3 jobs, 2 machines variant
fn create_3x2_variant() -> Result<JSSPInstance> {
    let processing_times = vec![
        6, 3, // Job 0
        2, 7, // Job 1
        5, 4, // Job 2
    ];

    let machine_sequences = vec![vec![0, 1], vec![0, 1], vec![0, 1]];

    JSSPInstance::new(3, 2, processing_times, machine_sequences)
}

/// 4 jobs, 3 machines benchmark
fn create_4x3_instance() -> Result<JSSPInstance> {
    let processing_times = vec![
        // Job 0
        4, 3, 2, // Job 1
        3, 4, 1, // Job 2
        2, 5, 3, // Job 3
        3, 2, 4,
    ];

    let machine_sequences = vec![
        vec![0, 1, 2], // Job 0: M0 -> M1 -> M2
        vec![1, 2, 0], // Job 1: M1 -> M2 -> M0
        vec![2, 0, 1], // Job 2: M2 -> M0 -> M1
        vec![0, 2, 1], // Job 3: M0 -> M2 -> M1
    ];

    JSSPInstance::new(4, 3, processing_times, machine_sequences)
}

/// 4 jobs, 3 machines variant
fn create_4x3_variant() -> Result<JSSPInstance> {
    let processing_times = vec![
        // Job 0
        5, 2, 3, // Job 1
        2, 4, 2, // Job 2
        3, 3, 5, // Job 3
        4, 3, 2,
    ];

    let machine_sequences = vec![vec![0, 1, 2], vec![1, 0, 2], vec![2, 1, 0], vec![0, 1, 2]];

    JSSPInstance::new(4, 3, processing_times, machine_sequences)
}

/// 5 jobs, 3 machines benchmark (larger instance)
fn create_5x3_instance() -> Result<JSSPInstance> {
    let processing_times = vec![
        // Job 0
        4, 3, 2, // Job 1
        3, 4, 1, // Job 2
        2, 5, 3, // Job 3
        3, 2, 4, // Job 4
        5, 1, 3,
    ];

    let machine_sequences = vec![
        vec![0, 1, 2],
        vec![1, 2, 0],
        vec![2, 0, 1],
        vec![0, 2, 1],
        vec![1, 0, 2],
    ];

    JSSPInstance::new(5, 3, processing_times, machine_sequences)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_sample_data() -> Result<()> {
        let dataset = load_sample_data()?;

        assert_eq!(dataset.instances.len(), 5);
        assert_eq!(dataset.optimal_schedules.len(), 5);

        for (instance, schedule) in dataset
            .instances
            .iter()
            .zip(dataset.optimal_schedules.iter())
        {
            assert_eq!(schedule.job_sequence.len(), instance.num_jobs);
            assert!(schedule.makespan > 0);
        }

        Ok(())
    }

    #[test]
    fn test_3x2_instance() -> Result<()> {
        let instance = create_3x2_instance()?;
        assert_eq!(instance.num_jobs, 3);
        assert_eq!(instance.num_machines, 2);
        Ok(())
    }

    #[test]
    fn test_4x3_instance() -> Result<()> {
        let instance = create_4x3_instance()?;
        assert_eq!(instance.num_jobs, 4);
        assert_eq!(instance.num_machines, 3);
        Ok(())
    }
}
