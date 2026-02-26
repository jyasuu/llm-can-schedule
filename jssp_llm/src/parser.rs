// src/parser.rs
//
// Mirrors the Python parse_output() function.
// Extracts a Schedule (Vec<Vec<u32>>) from raw LLM text using regex.

use regex::Regex;
use crate::jssp::Schedule;

/// Parse the LLM's raw text output into a Schedule.
///
/// Expected format produced by the fine-tuned model:
///   Job 0: 0 3 8
///   Job 1: 3 5 0
///   ...
///
/// Returns `None` if the text is malformed or inconsistent with `jobs`.
pub fn parse_output(text: &str, jobs: &[Vec<(u32, u32)>]) -> Option<Schedule> {
    // Regex: "Job <N>:" followed by a sequence of non-negative integers.
    let re = Regex::new(r"(?i)job\s+(\d+)\s*:\s*([\d\s]+)").ok()?;

    let n_jobs = jobs.len();
    let mut schedule: Schedule = vec![Vec::new(); n_jobs];
    let mut found = vec![false; n_jobs];

    for cap in re.captures_iter(text) {
        let job_idx: usize = cap[1].parse().ok()?;
        if job_idx >= n_jobs {
            return None; // model hallucinated an extra job
        }

        let times: Vec<u32> = cap[2]
            .split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();

        // Must have exactly one start time per operation
        if times.len() != jobs[job_idx].len() {
            return None;
        }

        schedule[job_idx] = times;
        found[job_idx] = true;
    }

    // Every job must have been parsed
    if found.iter().any(|&f| !f) {
        return None;
    }

    Some(schedule)
}
