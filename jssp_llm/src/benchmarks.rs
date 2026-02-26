// src/benchmarks.rs
//
// Standard JSSP benchmark instances used in the paper.
// Each instance stores:
//   name    – identifier string
//   optimal – known optimal makespan from the literature
//   jobs    – Vec<Vec<(machine_id, duration)>>
//
// Sources:
//   ft06  : Fisher & Thompson (1963) — 6×6, optimal = 55
//   la01  : Lawrence (1984)          — 10×5, optimal = 666
//   la02  : Lawrence (1984)          — 10×5, optimal = 655
//   la05  : Lawrence (1984)          — 10×5, optimal = 593
//   la10  : Lawrence (1984)          — 15×5, optimal = 958

use crate::jssp::{Job, Jobs};

pub struct Benchmark {
    pub name:    String,
    pub optimal: u32,
    pub jobs:    Jobs,
}

impl Benchmark {
    pub fn n_machines(&self) -> usize {
        self.jobs
            .iter()
            .flat_map(|j| j.iter())
            .map(|&(m, _)| m + 1)
            .max()
            .unwrap_or(0) as usize
    }
}

fn b(name: &str, optimal: u32, raw: &[(u32, u32, u32, u32, u32, u32, u32, u32, u32, u32)]) -> Benchmark {
    // Each row is one job: (m0,d0, m1,d1, m2,d2, m3,d3, m4,d4)
    let jobs: Jobs = raw
        .iter()
        .map(|&(m0,d0, m1,d1, m2,d2, m3,d3, m4,d4)| {
            let ops: Job = vec![(m0,d0),(m1,d1),(m2,d2),(m3,d3),(m4,d4)];
            // Drop trailing (0,0) placeholders (used for 6-machine instances below)
            ops.into_iter().filter(|&(_,d)| d > 0).collect()
        })
        .collect();
    Benchmark { name: name.to_string(), optimal, jobs }
}

/// Return the standard benchmark suite.
pub fn standard_suite() -> Vec<Benchmark> {
    vec![
        ft06(),
        la01(),
        la02(),
        la05(),
        la10(),
    ]
}

// ── ft06  (6×6) ───────────────────────────────────────────────────────────────

fn ft06() -> Benchmark {
    // Fisher & Thompson 6×6, optimal makespan = 55
    // Format: job rows, each cell = (machine, duration)
    let jobs: Jobs = vec![
        vec![(2,1),(0,3),(1,6),(3,7),(5,3),(4,6)],
        vec![(1,8),(2,5),(4,10),(5,10),(0,10),(3,4)],
        vec![(2,5),(3,4),(5,8),(0,9),(1,1),(4,7)],
        vec![(1,5),(0,5),(2,5),(3,3),(4,8),(5,9)],
        vec![(2,9),(1,3),(4,5),(5,4),(0,3),(3,1)],
        vec![(1,3),(3,3),(5,9),(0,10),(4,4),(2,1)],
    ];
    Benchmark { name: "ft06".into(), optimal: 55, jobs }
}

// ── la01  (10×5) ─────────────────────────────────────────────────────────────

fn la01() -> Benchmark {
    // Lawrence la01, 10 jobs × 5 machines, optimal = 666
    let jobs: Jobs = vec![
        vec![(1,21),(0,53),(4,95),(3,55),(2,34)],
        vec![(0,21),(3,52),(4,16),(2,26),(1,71)],
        vec![(3,39),(4,98),(1,42),(2,31),(0,12)],
        vec![(1,77),(0,55),(4,79),(2,66),(3,77)],
        vec![(0,83),(3,34),(2,64),(1,19),(4,37)],
        vec![(1,54),(2,43),(4,79),(0,92),(3,62)],
        vec![(3,69),(4,77),(1,87),(2,87),(0,93)],
        vec![(2,38),(0,60),(1,41),(3,24),(4,83)],
        vec![(3,17),(1,49),(4,25),(0,44),(2,98)],
        vec![(4,77),(3,79),(2,43),(1,75),(0,96)],
    ];
    Benchmark { name: "la01".into(), optimal: 666, jobs }
}

// ── la02  (10×5) ─────────────────────────────────────────────────────────────

fn la02() -> Benchmark {
    // Lawrence la02, 10 jobs × 5 machines, optimal = 655
    let jobs: Jobs = vec![
        vec![(0,20),(3,87),(1,31),(4,76),(2,17)],
        vec![(4,25),(2,32),(0,24),(1,18),(3,81)],
        vec![(1,72),(2,23),(4,28),(0,58),(3,99)],
        vec![(2,86),(1,76),(4,97),(0,45),(3,90)],
        vec![(3,27),(2,42),(0,48),(4,17),(1,46)],
        vec![(1,67),(0,98),(4,48),(3,27),(2,62)],
        vec![(4,28),(1,12),(3,19),(0,80),(2,50)],
        vec![(1,63),(0,87),(2,26),(3,30),(4,38)],
        vec![(4,60),(1,39),(0,54),(3,91),(2,16)],
        vec![(4,98),(0,28),(3,46),(1,49),(2,36)],
    ];
    Benchmark { name: "la02".into(), optimal: 655, jobs }
}

// ── la05  (10×5) ─────────────────────────────────────────────────────────────

fn la05() -> Benchmark {
    // Lawrence la05, 10 jobs × 5 machines, optimal = 593
    let jobs: Jobs = vec![
        vec![(1,72),(0,87),(3,95),(4,66),(2,60)],
        vec![(4,5),(2,73),(3,22),(0,9),(1,58)],
        vec![(4,14),(0,67),(2,98),(3,91),(1,50)],
        vec![(1,27),(3,8),(2,74),(0,16),(4,34)],
        vec![(0,84),(3,38),(1,51),(4,50),(2,57)],
        vec![(1,88),(2,64),(3,73),(4,60),(0,7)],
        vec![(2,97),(4,38),(0,75),(1,5),(3,25)],
        vec![(3,66),(2,88),(1,51),(4,60),(0,6)],
        vec![(1,55),(4,31),(3,39),(2,46),(0,55)],
        vec![(4,43),(0,97),(2,61),(1,66),(3,16)],
    ];
    Benchmark { name: "la05".into(), optimal: 593, jobs }
}

// ── la10  (15×5) ─────────────────────────────────────────────────────────────

fn la10() -> Benchmark {
    // Lawrence la10, 15 jobs × 5 machines, optimal = 958
    let jobs: Jobs = vec![
        vec![(0,58),(1,44),(2,22),(4,69),(3,33)],
        vec![(4,56),(2,6),(0,44),(3,82),(1,77)],
        vec![(2,44),(3,38),(4,83),(1,49),(0,60)],
        vec![(4,76),(0,24),(3,47),(1,73),(2,52)],
        vec![(3,97),(2,77),(1,23),(0,25),(4,54)],
        vec![(2,27),(0,57),(4,65),(3,94),(1,5)],
        vec![(3,11),(2,72),(4,46),(0,34),(1,55)],
        vec![(4,47),(0,96),(1,35),(3,31),(2,97)],
        vec![(0,56),(4,53),(1,18),(3,33),(2,81)],
        vec![(3,64),(0,42),(1,84),(4,61),(2,77)],
        vec![(1,88),(3,6),(0,19),(2,89),(4,98)],
        vec![(3,60),(0,49),(4,58),(2,90),(1,49)],
        vec![(0,27),(2,70),(3,34),(4,64),(1,65)],
        vec![(4,72),(1,46),(0,32),(3,77),(2,11)],
        vec![(4,14),(2,55),(0,31),(3,20),(1,49)],
    ];
    Benchmark { name: "la10".into(), optimal: 958, jobs }
}
