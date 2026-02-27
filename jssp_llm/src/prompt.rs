// src/prompt.rs
//
// Mirrors Python's build_prompt(), format_job_centric(), and format_solution()
// EXACTLY as used during fine-tuning. The format MUST match training or the
// model will produce garbage output.
//
// Python training prompt template (build_prompt with no solution):
//   <|user|>
//   {problem}<|end|>
//   <|assistant|>
//
// Python format_job_centric() output (Listing 2 from the paper):
//   Optimize schedule for N Jobs across M Machines to minimize makespan. ...
//
//   Problem:
//
//    Job 0 consists of the following Operations:
//     Operation 0 on Machine 2 duration 1 mins.
//     Operation 1 on Machine 0 duration 3 mins.
//   ...
//
// NOTE: The old Rust prompt used a different system/user template and a
// simplified format that the model was NEVER trained on.

use crate::jssp::{count_machines, Jobs};

// The paper uses 3 instruction variants (chosen randomly during training).
// At inference time we always use variant 0 (same as Python with rng=None).
const INSTRUCTION: &str =
    "Optimize schedule for {nj} Jobs across {nm} Machines to minimize makespan. \
Each job involves a series of Operations needing specific machines and times. \
Operations are processed in order, without interruption, on a single Machine at a time.";

/// Mirrors Python's format_job_centric(jobs, rng=None) — Paper Listing 2.
pub fn format_job_centric(jobs: &Jobs) -> String {
    let nj = jobs.len();
    let nm = count_machines(jobs) as usize;

    let instr = INSTRUCTION
        .replace("{nj}", &nj.to_string())
        .replace("{nm}", &nm.to_string());

    let mut lines = vec![instr, String::new(), "Problem:".to_string()];

    for (j, job) in jobs.iter().enumerate() {
        lines.push(String::new());
        lines.push(format!(" Job {} consists of the following Operations:", j));
        for (k, &(m, d)) in job.iter().enumerate() {
            lines.push(format!(
                "  Operation {} on Machine {} duration {} mins.",
                k, m, d
            ));
        }
    }

    lines.join("\n")
}

/// Mirrors Python's build_prompt(problem, solution=None) — inference template.
/// Uses Phi-3 chat template without <|system|> (matches training exactly).
pub fn build_prompt(problem_text: &str) -> String {
    format!(
        "<|user|>\n{}<|end|>\n<|assistant|>\n",
        problem_text.trim()
    )
}

/// Convenience wrapper: format jobs and build the inference prompt in one call.
pub fn build_prompt_for_jobs(jobs: &Jobs) -> String {
    let problem = format_job_centric(jobs);
    build_prompt(&problem)
}