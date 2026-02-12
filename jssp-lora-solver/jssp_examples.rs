/// Example 1: Simple JSSP Instance and Solving
/// 
/// This demonstrates how to create a JSSP instance and solve it
/// using different strategies.

use anyhow::Result;
use jssp_lora::jssp::{JSSPInstance, JSSPSolver, Schedule};

fn example_1_basic_jssp() -> Result<()> {
    println!("\n=== Example 1: Basic JSSP Instance ===\n");

    // Create a 3x2 problem (3 jobs, 2 machines)
    let processing_times = vec![
        5, 4, // Job 0: 5 time units on Machine 0, 4 on Machine 1
        4, 5, // Job 1: 4 time units on Machine 0, 5 on Machine 1
        3, 6, // Job 2: 3 time units on Machine 0, 6 on Machine 1
    ];

    let machine_sequences = vec![
        vec![0, 1], // Job 0: Execute on Machine 0, then Machine 1
        vec![0, 1], // Job 1: Execute on Machine 0, then Machine 1
        vec![0, 1], // Job 2: Execute on Machine 0, then Machine 1
    ];

    let instance = JSSPInstance::new(3, 2, processing_times, machine_sequences)?;

    println!("Problem: {} jobs, {} machines", instance.num_jobs, instance.num_machines);
    println!("Processing times per job and machine:");
    for job in 0..instance.num_jobs {
        print!("  Job {}: ", job);
        for machine in 0..instance.num_machines {
            let time = instance.get_processing_time(job, machine);
            print!("[M{}:{}] ", machine, time);
        }
        println!();
    }

    // Solve using different approaches
    println!("\n--- Greedy Solver ---");
    let greedy = JSSPSolver::solve_greedy(&instance)?;
    println!("Greedy schedule: {:?}", greedy.job_sequence);
    println!("Makespan (total time): {}", greedy.makespan);
    println!("Completion times: {:?}", greedy.completion_times);

    println!("\n--- Nearest Neighbor Solver ---");
    let nn = JSSPSolver::solve_nearest_neighbor(&instance)?;
    println!("NN schedule: {:?}", nn.job_sequence);
    println!("Makespan: {}", nn.makespan);

    println!("\n--- Random Solver ---");
    let random = JSSPSolver::solve_random(&instance)?;
    println!("Random schedule: {:?}", random.job_sequence);
    println!("Makespan: {}", random.makespan);

    // Compare results
    println!("\n--- Comparison ---");
    println!("Best solution: {}", 
             if greedy.makespan <= nn.makespan { greedy.makespan } else { nn.makespan });

    Ok(())
}

/// Example 2: Complex JSSP with Multiple Machines
///
/// This shows a larger problem with varied machine sequences.

fn example_2_complex_jssp() -> Result<()> {
    println!("\n=== Example 2: Complex JSSP (4x3) ===\n");

    // 4 jobs, 3 machines with different machine orderings
    let processing_times = vec![
        // Job 0: M0 -> M1 -> M2
        4, 3, 2,
        // Job 1: M1 -> M2 -> M0 (different order)
        3, 4, 1,
        // Job 2: M2 -> M0 -> M1 (different order)
        2, 5, 3,
        // Job 3: M0 -> M2 -> M1 (different order)
        3, 2, 4,
    ];

    let machine_sequences = vec![
        vec![0, 1, 2], // Job 0: M0 -> M1 -> M2
        vec![1, 2, 0], // Job 1: M1 -> M2 -> M0
        vec![2, 0, 1], // Job 2: M2 -> M0 -> M1
        vec![0, 2, 1], // Job 3: M0 -> M2 -> M1
    ];

    let instance = JSSPInstance::new(4, 3, processing_times, machine_sequences)?;

    println!("Problem: {} jobs, {} machines", instance.num_jobs, instance.num_machines);
    println!("Machine sequences (order of operations):");
    for (job, sequence) in machine_sequences.iter().enumerate() {
        println!("  Job {}: {:?}", job, sequence);
    }

    // Try multiple solvers
    println!("\n--- Multiple Solvers ---");
    
    let greedy = JSSPSolver::solve_greedy(&instance)?;
    println!("Greedy makespan:        {}", greedy.makespan);
    
    let nn = JSSPSolver::solve_nearest_neighbor(&instance)?;
    println!("Nearest Neighbor:       {}", nn.makespan);
    
    let random1 = JSSPSolver::solve_random(&instance)?;
    println!("Random attempt 1:       {}", random1.makespan);
    
    let random2 = JSSPSolver::solve_random(&instance)?;
    println!("Random attempt 2:       {}", random2.makespan);

    let best = [greedy.makespan, nn.makespan, random1.makespan, random2.makespan]
        .iter()
        .min()
        .unwrap();
    println!("\nBest found: {}", best);

    Ok(())
}

/// Example 3: Custom Schedule Creation
///
/// This demonstrates creating custom schedules and computing their makespan.

fn example_3_custom_schedules() -> Result<()> {
    println!("\n=== Example 3: Custom Schedule Creation ===\n");

    let processing_times = vec![
        6, 3,
        2, 7,
        5, 4,
    ];

    let machine_sequences = vec![
        vec![0, 1],
        vec![0, 1],
        vec![0, 1],
    ];

    let instance = JSSPInstance::new(3, 2, processing_times, machine_sequences)?;

    // Try different job sequences
    let sequences = vec![
        vec![0, 1, 2], // Job order: 0, 1, 2
        vec![0, 2, 1], // Job order: 0, 2, 1
        vec![1, 0, 2], // Job order: 1, 0, 2
        vec![2, 1, 0], // Job order: 2, 1, 0
    ];

    println!("Testing different job sequences:\n");

    let mut best_schedule = None;
    let mut best_makespan = u32::MAX;

    for sequence in sequences {
        let schedule = Schedule::from_sequence(&instance, sequence.clone())?;
        println!("Sequence {:?}: makespan = {}", sequence, schedule.makespan);

        if schedule.makespan < best_makespan {
            best_makespan = schedule.makespan;
            best_schedule = Some(schedule);
        }
    }

    if let Some(best) = best_schedule {
        println!("\nBest schedule: {:?}", best.job_sequence);
        println!("Optimal makespan: {}", best.makespan);
        println!("Job completion times: {:?}", best.completion_times);
    }

    Ok(())
}

/// Example 4: Schedule Analysis
///
/// This shows how to analyze and understand a schedule in detail.

fn example_4_schedule_analysis() -> Result<()> {
    println!("\n=== Example 4: Schedule Analysis ===\n");

    let processing_times = vec![
        5, 4,
        4, 5,
        3, 6,
    ];

    let machine_sequences = vec![
        vec![0, 1],
        vec![0, 1],
        vec![0, 1],
    ];

    let instance = JSSPInstance::new(3, 2, processing_times, machine_sequences)?;
    let schedule = JSSPSolver::solve_greedy(&instance)?;

    println!("Schedule Analysis");
    println!("=================\n");

    println!("Job Execution Order: {:?}\n", schedule.job_sequence);

    println!("Completion Times (when each job finishes):");
    for (job, &completion_time) in schedule.completion_times.iter().enumerate() {
        println!("  Job {}: {} time units", job, completion_time);
    }

    println!("\nCritical Path Analysis:");
    println!("Total makespan (project completion time): {}", schedule.makespan);

    // Calculate utilization
    let total_work: u32 = processing_times.iter().sum();
    let total_capacity = (instance.num_machines as u32) * schedule.makespan;
    let utilization = (total_work as f32 / total_capacity as f32) * 100.0;

    println!("Total work units:      {}", total_work);
    println!("Total capacity:        {}", total_capacity);
    println!("Machine utilization:   {:.2}%", utilization);

    Ok(())
}

/// Example 5: Batch Processing Multiple Instances
///
/// This shows how to process multiple JSSP instances efficiently.

fn example_5_batch_processing() -> Result<()> {
    println!("\n=== Example 5: Batch Processing ===\n");

    // Create multiple instances
    let instances = vec![
        (3, 2, vec![5,4, 4,5, 3,6]),
        (3, 2, vec![6,3, 2,7, 5,4]),
        (4, 3, vec![4,3,2, 3,4,1, 2,5,3, 3,2,4]),
    ];

    let machine_sequences = vec![
        vec![vec![0,1], vec![0,1], vec![0,1]],
        vec![vec![0,1], vec![0,1], vec![0,1]],
        vec![vec![0,1,2], vec![1,2,0], vec![2,0,1], vec![0,2,1]],
    ];

    println!("Processing {} instances:\n", instances.len());

    let mut results = Vec::new();

    for (idx, ((jobs, machines, times), sequences)) in instances.iter().zip(machine_sequences.iter()).enumerate() {
        let instance = JSSPInstance::new(*jobs, *machines, times.clone(), sequences.clone())?;
        let schedule = JSSPSolver::solve_nearest_neighbor(&instance)?;
        
        results.push((idx + 1, *jobs, *machines, schedule.makespan));
        println!("Instance {}: {} jobs × {} machines = makespan {}", 
                 idx + 1, jobs, machines, schedule.makespan);
    }

    println!("\n--- Summary ---");
    let avg_makespan = results.iter().map(|r| r.3).sum::<u32>() as f32 / results.len() as f32;
    println!("Average makespan: {:.2}", avg_makespan);

    let largest = results.iter().max_by_key(|r| r.3).unwrap();
    println!("Largest problem: {} jobs × {} machines (makespan: {})", 
             largest.1, largest.2, largest.3);

    Ok(())
}

/// Main entry point to run all examples
fn main() -> Result<()> {
    println!("\n╔═══════════════════════════════════════════════╗");
    println!("║  JSSP LoRA Solver - Practical Examples        ║");
    println!("╚═══════════════════════════════════════════════╝");

    example_1_basic_jssp()?;
    example_2_complex_jssp()?;
    example_3_custom_schedules()?;
    example_4_schedule_analysis()?;
    example_5_batch_processing()?;

    println!("\n╔═══════════════════════════════════════════════╗");
    println!("║  All examples completed successfully!         ║");
    println!("╚═══════════════════════════════════════════════╝\n");

    Ok(())
}
