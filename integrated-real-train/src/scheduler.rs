// Scheduler Module - Problem representation and scheduling algorithms

use anyhow::Result;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Job {
    pub id: usize,
    pub operations: Vec<Operation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Operation {
    pub machine_id: usize,
    pub duration: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulingProblem {
    pub num_jobs: usize,
    pub num_machines: usize,
    pub jobs: Vec<Job>,
}

impl SchedulingProblem {
    /// Generate random problem
    pub fn random(num_jobs: usize, num_machines: usize, seed: u64) -> Self {
        let mut rng = rand::thread_rng();

        let mut jobs = Vec::new();
        for job_id in 0..num_jobs {
            let num_ops = rng.gen_range(2..std::cmp::min(5, num_machines + 1));
            let mut operations = Vec::new();

            for _ in 0..num_ops {
                operations.push(Operation {
                    machine_id: rng.gen_range(0..num_machines),
                    duration: rng.gen_range(1..20),
                });
            }

            jobs.push(Job {
                id: job_id,
                operations,
            });
        }

        SchedulingProblem {
            num_jobs,
            num_machines,
            jobs,
        }
    }

    /// Load from JSON file
    pub fn from_json(path: &str) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        let problem = serde_json::from_str(&content)?;
        Ok(problem)
    }

    /// Save to JSON file
    pub fn to_json(&self, path: &str) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        Ok(())
    }

    /// Get problem description as text
    pub fn description(&self) -> String {
        let mut desc = format!(
            "Job Shop Scheduling Problem\nJobs: {}, Machines: {}\n\nJob Details:\n",
            self.num_jobs, self.num_machines
        );

        for job in &self.jobs {
            let ops = job
                .operations
                .iter()
                .map(|op| format!("M{}({})", op.machine_id, op.duration))
                .collect::<Vec<_>>()
                .join(" -> ");
            desc.push_str(&format!("Job {}: {}\n", job.id, ops));
        }

        desc
    }
}

pub struct ScheduledSolution {
    pub num_jobs: usize,
    pub num_machines: usize,
    pub makespan: usize,
    pub schedule: Vec<ScheduledOperation>,
}

#[derive(Debug, Clone)]
pub struct ScheduledOperation {
    pub job_id: usize,
    pub op_id: usize,
    pub machine_id: usize,
    pub start_time: usize,
    pub end_time: usize,
}

impl ScheduledSolution {
    pub fn description(&self) -> String {
        let mut desc = format!("Solution (Makespan: {})\n\nSchedule:\n", self.makespan);

        for op in &self.schedule {
            desc.push_str(&format!(
                "  Job {}.Op{}: Machine {} [{}, {})\n",
                op.job_id, op.op_id, op.machine_id, op.start_time, op.end_time
            ));
        }

        desc
    }
}

pub struct GreedyScheduler;

impl GreedyScheduler {
    /// Simple greedy scheduling
    pub fn schedule(problem: &SchedulingProblem) -> ScheduledSolution {
        let mut machine_free_at = vec![0; problem.num_machines];
        let mut job_end_time = vec![0; problem.num_jobs];
        let mut schedule = Vec::new();

        for (job_id, job) in problem.jobs.iter().enumerate() {
            let mut current_time = job_end_time[job_id];

            for (op_idx, op) in job.operations.iter().enumerate() {
                let start_time = std::cmp::max(current_time, machine_free_at[op.machine_id]);
                let end_time = start_time + op.duration;

                schedule.push(ScheduledOperation {
                    job_id,
                    op_id: op_idx,
                    machine_id: op.machine_id,
                    start_time,
                    end_time,
                });

                machine_free_at[op.machine_id] = end_time;
                current_time = end_time;
            }

            job_end_time[job_id] = current_time;
        }

        let makespan = *machine_free_at.iter().max().unwrap_or(&0);

        ScheduledSolution {
            num_jobs: problem.num_jobs,
            num_machines: problem.num_machines,
            makespan,
            schedule,
        }
    }
}
