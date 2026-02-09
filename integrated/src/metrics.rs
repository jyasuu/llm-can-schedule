// Metrics Module - Evaluate solution quality

#[derive(Debug, Clone)]
pub struct SolutionMetrics {
    pub makespan: usize,
    pub feasible: bool,
    pub generation_time_ms: u128,
}

impl SolutionMetrics {
    pub fn display(&self) {
        println!("Metrics:");
        println!("  Makespan: {}", self.makespan);
        println!("  Feasible: {}", self.feasible);
        println!("  Generation Time: {}ms", self.generation_time_ms);
    }
}

pub struct Comparator;

impl Comparator {
    pub fn compare(llm: &SolutionMetrics, baseline: &SolutionMetrics) -> ComparisonResult {
        let improvement = if baseline.makespan > 0 {
            1.0 - (llm.makespan as f32 / baseline.makespan as f32)
        } else {
            0.0
        };
        
        ComparisonResult {
            llm_makespan: llm.makespan,
            baseline_makespan: baseline.makespan,
            improvement_percent: improvement * 100.0,
            llm_time_ms: llm.generation_time_ms,
        }
    }
}

pub struct ComparisonResult {
    pub llm_makespan: usize,
    pub baseline_makespan: usize,
    pub improvement_percent: f32,
    pub llm_time_ms: u128,
}

impl ComparisonResult {
    pub fn display(&self) {
        println!("Comparison:");
        println!("  LLM Makespan: {}", self.llm_makespan);
        println!("  Baseline Makespan: {}", self.baseline_makespan);
        println!("  Improvement: {:.2}%", self.improvement_percent);
        println!("  Generation Time: {}ms", self.llm_time_ms);
    }
}
