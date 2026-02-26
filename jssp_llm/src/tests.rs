// src/tests.rs  (included via `#[cfg(test)] mod tests;` in each module, or run directly)
//
// Run with: cargo test

#[cfg(test)]
mod jssp_tests {
    use crate::jssp::*;

    fn simple_jobs() -> Jobs {
        vec![
            vec![(0, 3), (1, 5), (2, 2)],
            vec![(2, 4), (0, 2), (1, 3)],
            vec![(1, 6), (2, 1), (0, 4)],
        ]
    }

    #[test]
    fn test_parse_jobs_str() {
        let jobs = parse_jobs_str("0,3,1,5,2,2;2,4,0,2,1,3").unwrap();
        assert_eq!(jobs.len(), 2);
        assert_eq!(jobs[0][0], (0, 3));
        assert_eq!(jobs[1][2], (1, 3));
    }

    #[test]
    fn test_validate_valid_schedule() {
        let jobs = simple_jobs();
        // Hand-crafted feasible schedule
        let schedule = vec![
            vec![0, 3, 11],
            vec![0, 4, 9],
            vec![0, 6, 7],
        ];
        let (ok, ms) = validate(&jobs, &schedule);
        assert!(ok, "Expected valid schedule");
        assert!(ms > 0);
    }

    #[test]
    fn test_validate_precedence_violation() {
        let jobs = simple_jobs();
        // Op 1 starts before op 0 finishes → precedence violation
        let schedule = vec![
            vec![0, 1, 11], // job 0: op1 starts at 1, but op0 duration=3 → violation
            vec![0, 4, 9],
            vec![0, 6, 7],
        ];
        let (ok, _) = validate(&jobs, &schedule);
        assert!(!ok, "Expected precedence violation");
    }

    #[test]
    fn test_validate_machine_overlap() {
        let jobs = simple_jobs();
        // Both job 0 op 0 (machine 0, start 0, dur 3) and job 1 op 1 (machine 0, start 1, dur 2)
        // overlap on machine 0
        let schedule = vec![
            vec![0, 3, 8],
            vec![4, 1, 9], // job 1 op1 on machine 0 at t=1 overlaps with job 0 op0 (0..3)
            vec![0, 6, 7],
        ];
        let (ok, _) = validate(&jobs, &schedule);
        assert!(!ok, "Expected machine overlap violation");
    }

    #[test]
    fn test_format_job_centric() {
        let jobs = simple_jobs();
        let s = format_job_centric(&jobs);
        assert!(s.contains("Jobs: 3"));
        assert!(s.contains("Machines: 3"));
        assert!(s.contains("Job 0:"));
        assert!(s.contains("machine 0, duration 3"));
    }

    #[test]
    fn test_count_machines() {
        let jobs = simple_jobs();
        assert_eq!(count_machines(&jobs), 3);
    }
}

#[cfg(test)]
mod parser_tests {
    use crate::parser::parse_output;
    use crate::jssp::Jobs;

    fn jobs_3x3() -> Jobs {
        vec![
            vec![(0, 3), (1, 5), (2, 2)],
            vec![(2, 4), (0, 2), (1, 3)],
            vec![(1, 6), (2, 1), (0, 4)],
        ]
    }

    #[test]
    fn test_parse_valid_output() {
        let text = "Job 0: 0 3 8\nJob 1: 4 0 9\nJob 2: 0 6 7\n";
        let jobs = jobs_3x3();
        let schedule = parse_output(text, &jobs);
        assert!(schedule.is_some());
        let s = schedule.unwrap();
        assert_eq!(s[0], vec![0, 3, 8]);
        assert_eq!(s[1], vec![4, 0, 9]);
        assert_eq!(s[2], vec![0, 6, 7]);
    }

    #[test]
    fn test_parse_missing_job() {
        let text = "Job 0: 0 3 8\nJob 1: 4 0 9\n"; // missing Job 2
        let jobs = jobs_3x3();
        assert!(parse_output(text, &jobs).is_none());
    }

    #[test]
    fn test_parse_wrong_op_count() {
        let text = "Job 0: 0 3\nJob 1: 4 0 9\nJob 2: 0 6 7\n"; // Job 0 has 2 times, needs 3
        let jobs = jobs_3x3();
        assert!(parse_output(text, &jobs).is_none());
    }

    #[test]
    fn test_parse_case_insensitive() {
        let text = "JOB 0: 0 3 8\njob 1: 4 0 9\nJob 2: 0 6 7\n";
        let jobs = jobs_3x3();
        assert!(parse_output(text, &jobs).is_some());
    }
}

#[cfg(test)]
mod solver_tests {
    use crate::solver::solve_greedy;
    use crate::jssp::validate;

    #[test]
    fn test_greedy_3x3() {
        let jobs = vec![
            vec![(0u32, 3u32), (1, 5), (2, 2)],
            vec![(2, 4), (0, 2), (1, 3)],
            vec![(1, 6), (2, 1), (0, 4)],
        ];
        let ms = solve_greedy(&jobs);
        assert!(ms >= 10, "Makespan {} seems too small", ms);
        assert!(ms <= 50, "Makespan {} seems too large", ms);
    }

    #[test]
    fn test_greedy_schedule_is_valid() {
        use crate::solver::greedy_schedule;
        let jobs = vec![
            vec![(0u32, 3u32), (1, 5), (2, 2)],
            vec![(2, 4), (0, 2), (1, 3)],
        ];
        let schedule = greedy_schedule(&jobs);
        let (ok, _) = validate(&jobs, &schedule);
        assert!(ok, "Greedy schedule should be valid");
    }

    #[test]
    fn test_greedy_single_job() {
        let jobs = vec![vec![(0u32, 5u32), (1, 3)]];
        assert_eq!(solve_greedy(&jobs), 8);
    }
}

#[cfg(test)]
mod prompt_tests {
    use crate::prompt::build_prompt_for_jobs;

    #[test]
    fn test_prompt_contains_system() {
        let jobs = vec![vec![(0u32, 3u32), (1, 5)]];
        let p = build_prompt_for_jobs(&jobs);
        assert!(p.contains("<|system|>"));
        assert!(p.contains("<|user|>"));
        assert!(p.contains("<|assistant|>"));
        assert!(p.contains("Job 0:"));
    }
}
