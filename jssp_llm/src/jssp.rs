// src/jssp.rs
//
// Core JSSP types, input parsing, constraint validation, makespan calculation.
// Mirrors the Python helpers: format_job_centric, validate, etc.

use anyhow::{bail, Result};

/// One operation: (machine_id, processing_duration)
pub type Op = (u32, u32);

/// A job is an ordered sequence of operations.
pub type Job = Vec<Op>;

/// A complete problem instance.
pub type Jobs = Vec<Job>;

/// A start-time schedule: schedule[job][op_index] = start_time
pub type Schedule = Vec<Vec<u32>>;

// ── Parsing ───────────────────────────────────────────────────────────────────

/// Parse a CLI jobs string like `"0,3,1,5,2,2;2,4,0,2,1,3"`.
/// Each segment separated by `;` is one job; pairs within a job are `m,d`.
pub fn parse_jobs_str(s: &str) -> Result<Jobs> {
    let mut jobs = Vec::new();
    for job_str in s.split(';') {
        let nums: Vec<u32> = job_str
            .split(',')
            .map(|t| t.trim().parse::<u32>())
            .collect::<std::result::Result<_, _>>()
            .map_err(|e| anyhow::anyhow!("Parse error in jobs string: {e}"))?;
        if nums.len() % 2 != 0 {
            bail!("Each job must have an even number of values (m,d pairs): {:?}", nums);
        }
        let ops = nums.chunks(2).map(|c| (c[0], c[1])).collect();
        jobs.push(ops);
    }
    Ok(jobs)
}

// ── Formatting (mirrors Python's format_job_centric) ─────────────────────────

/// Returns the problem description in job-centric text format.
/// Example output (3×3):
///   Jobs: 3  Machines: 3
///   Job 0: (machine 0, duration 3) -> (machine 1, duration 5) -> (machine 2, duration 2)
///   ...
pub fn format_job_centric(jobs: &Jobs) -> String {
    let n_machines = count_machines(jobs);
    let mut out = format!("Jobs: {}  Machines: {}\n", jobs.len(), n_machines);
    for (j, job) in jobs.iter().enumerate() {
        let ops: Vec<String> = job
            .iter()
            .map(|(m, d)| format!("(machine {}, duration {})", m, d))
            .collect();
        out.push_str(&format!("Job {}: {}\n", j, ops.join(" -> ")));
    }
    out
}

/// Returns the problem description in machine-centric text format.
/// Lists, for each machine, the sequence of jobs that visit it.
pub fn format_machine_centric(jobs: &Jobs) -> String {
    let n_machines = count_machines(jobs);
    let mut machine_ops: Vec<Vec<(usize, u32)>> = vec![Vec::new(); n_machines as usize];
    for (j, job) in jobs.iter().enumerate() {
        for &(m, d) in job {
            machine_ops[m as usize].push((j, d));
        }
    }
    let mut out = format!("Jobs: {}  Machines: {}\n", jobs.len(), n_machines);
    for (m, ops) in machine_ops.iter().enumerate() {
        let ops_str: Vec<String> = ops
            .iter()
            .map(|(j, d)| format!("(job {}, duration {})", j, d))
            .collect();
        out.push_str(&format!("Machine {}: {}\n", m, ops_str.join(", ")));
    }
    out
}

// ── Validation (mirrors Python's validate) ────────────────────────────────────

/// Validate a proposed schedule against JSSP constraints.
/// Returns `(true, makespan)` on success, or `(false, 0)` on violation.
///
/// Constraints:
///   1. Precedence: each operation must start after the previous operation in
///      the same job completes.
///   2. No-overlap: on each machine, the time intervals of all operations must
///      be disjoint.
///   3. Non-negativity: all start times >= 0.
pub fn validate(jobs: &Jobs, schedule: &Schedule) -> (bool, u32) {
    // --- Basic shape check ---
    if schedule.len() != jobs.len() {
        return (false, 0);
    }
    for (j, job) in jobs.iter().enumerate() {
        if schedule[j].len() != job.len() {
            return (false, 0);
        }
    }

    // --- Precedence constraints ---
    for (j, job) in jobs.iter().enumerate() {
        let mut earliest = 0u32;
        for (op_idx, &(_m, d)) in job.iter().enumerate() {
            let st = schedule[j][op_idx];
            if st < earliest {
                return (false, 0);
            }
            earliest = st + d;
        }
    }

    // --- Machine no-overlap constraints ---
    let n_machines = count_machines(jobs) as usize;
    let mut machine_intervals: Vec<Vec<(u32, u32)>> = vec![Vec::new(); n_machines];

    for (j, job) in jobs.iter().enumerate() {
        for (op_idx, &(m, d)) in job.iter().enumerate() {
            let st = schedule[j][op_idx];
            machine_intervals[m as usize].push((st, st + d));
        }
    }

    for intervals in &mut machine_intervals {
        intervals.sort_unstable();
        for w in intervals.windows(2) {
            if w[0].1 > w[1].0 {
                // Overlap detected
                return (false, 0);
            }
        }
    }

    // --- Makespan ---
    let makespan = jobs
        .iter()
        .enumerate()
        .map(|(j, job)| {
            let last = job.len() - 1;
            schedule[j][last] + job[last].1
        })
        .max()
        .unwrap_or(0);

    (true, makespan)
}

// ── Utilities ─────────────────────────────────────────────────────────────────

pub fn count_machines(jobs: &Jobs) -> u32 {
    jobs.iter()
        .flat_map(|j| j.iter())
        .map(|&(m, _)| m + 1)
        .max()
        .unwrap_or(0)
}
