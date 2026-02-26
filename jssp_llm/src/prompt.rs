// src/prompt.rs
//
// Mirrors Python's build_prompt() and the alpaca-style template used during
// Phi-3 fine-tuning. The exact format must match what the model was trained on.

use crate::jssp::{Jobs, format_job_centric};

/// System message used during fine-tuning.
const SYSTEM: &str = "\
You are an expert scheduler. Given a Job Shop Scheduling Problem (JSSP), \
output a valid schedule that minimises the makespan (total completion time). \
For each job list the start time of each operation in order. \
Output ONLY the schedule, no explanation.";

/// Expected output format hint appended to the user message.
const FORMAT_HINT: &str = "\
Output format (one line per job, space-separated start times):
Job 0: <t0> <t1> ... <tk>
Job 1: <t0> <t1> ... <tk>
...";

/// Build the full prompt string for a problem encoded as `problem_text`.
///
/// Uses the Phi-3 instruct chat template:
///   <|system|>\n{system}<|end|>\n<|user|>\n{user}<|end|>\n<|assistant|>\n
///
/// Adjust the template delimiters if you switch to a different base model.
pub fn build_prompt(problem_text: &str) -> String {
    let user = format!(
        "Solve the following Job Shop Scheduling Problem:\n\n{}\n\n{}",
        problem_text.trim(),
        FORMAT_HINT,
    );

    format!(
        "<|system|>\n{}<|end|>\n<|user|>\n{}<|end|>\n<|assistant|>\n",
        SYSTEM, user
    )
}

/// Convenience wrapper: format jobs and build the prompt in one call.
pub fn build_prompt_for_jobs(jobs: &Jobs) -> String {
    let problem = format_job_centric(jobs);
    build_prompt(&problem)
}
