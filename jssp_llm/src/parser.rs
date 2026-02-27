// src/parser.rs
//
// Mirrors the Python parse_output() function EXACTLY.
//
// The fine-tuned model was trained on this output format (format_solution in Python):
//
//   Job 0 Operation 0 on Machine 2 : 0 + 1 -> 1
//   Job 0 Operation 1 on Machine 0 : 1 + 3 -> 4
//   ...
//   Makespan: 55, as it is the maximum end completion time of Operation N
//
// Python regex: r"Job\s+(\d+)\s+Operation\s+(\d+)\s+on\s+Machine\s+(\d+)\s*:\s*(\d+)\s*\+\s*(\d+)\s*->\s*(\d+)"
// Validation:   jobs[j][k] == (machine, dur) must match
//
// NOTE: The old Rust parser matched "Job N: t0 t1 ..." which is WRONG —
// the model never produces that format because it wasn't trained on it.

use regex::Regex;
use std::collections::HashMap;
use crate::jssp::{Jobs, Schedule};

/// Parse the LLM's raw text output into a Schedule.
///
/// Expected format (produced by the fine-tuned model, matching Python's format_solution):
///   Job 0 Operation 0 on Machine 2 : 0 + 1 -> 1
///   Job 1 Operation 0 on Machine 1 : 0 + 8 -> 8
///   ...
///   Makespan: 55, as it is the maximum end completion time of Operation 5
///
/// Returns `None` if the text is malformed or inconsistent with `jobs`.
pub fn parse_output(text: &str, jobs: &Jobs) -> Option<Schedule> {
    // Mirrors Python: r"Job\s+(\d+)\s+Operation\s+(\d+)\s+on\s+Machine\s+(\d+)\s*:\s*(\d+)\s*\+\s*(\d+)\s*->\s*(\d+)"
    let re = Regex::new(
        r"(?i)Job\s+(\d+)\s+Operation\s+(\d+)\s+on\s+Machine\s+(\d+)\s*:\s*(\d+)\s*\+\s*(\d+)\s*->\s*(\d+)"
    ).ok()?;

    let n_jobs = jobs.len();
    // Collect (job_idx, op_idx) -> start_time
    let mut st: HashMap<(usize, usize), u32> = HashMap::new();

    for cap in re.captures_iter(text) {
        let j:       usize = cap[1].parse().ok()?;
        let k:       usize = cap[2].parse().ok()?;
        let machine: u32   = cap[3].parse().ok()?;
        let start:   u32   = cap[4].parse().ok()?;
        let dur:     u32   = cap[5].parse().ok()?;
        // cap[6] is end time — we don't need it

        if j >= n_jobs || k >= jobs[j].len() {
            return None; // hallucinated job/op index
        }

        // Validate machine and duration match the problem definition (mirrors Python)
        if jobs[j][k] != (machine, dur) {
            return None;
        }

        st.insert((j, k), start);
    }

    // Every (j, k) operation must have been parsed
    let expected: std::collections::HashSet<(usize, usize)> = jobs
        .iter()
        .enumerate()
        .flat_map(|(j, job)| (0..job.len()).map(move |k| (j, k)))
        .collect();

    if st.len() != expected.len() || st.keys().any(|key| !expected.contains(key)) {
        return None;
    }

    // Build Schedule: schedule[j][k] = start_time
    let schedule: Schedule = (0..n_jobs)
        .map(|j| (0..jobs[j].len()).map(|k| *st.get(&(j, k)).unwrap()).collect())
        .collect();

    Some(schedule)
}