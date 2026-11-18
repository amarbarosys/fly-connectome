//! Stage 2 correctness oracle: Rust reference vs Mojo kernel, differential.
//!
//! Python is out of the new screening path, so the independent reference for
//! the FlyBrain batch dynamics lives here, in Rust, on the CPU. The kernel
//! under test is `fly-spmv/src/toy_batch_gpu.mojo` on the GPU. The two are
//! written from the numerical contract separately, so agreement is evidence
//! and not a shared bug.
//!
//! Scope limit, stated on purpose: the toy kernel uses raw `MutPointer` buffers
//! rather than the production `TileTensor` plumbing. This oracle pins the
//! NUMERICS and the shared-memory tree reduction. The TileTensor layer is only
//! exercised by the production binary.
//!
//! Run:
//!   cargo run --release --bin stage2_oracle            # full gate, GPU
//!   cargo run --release --bin stage2_oracle -- --dry   # CPU preflight only
//!
//! Exit code is non-zero unless every case matches. The last line is always
//! `STAGE2_PARITY_PASS=<passed>/<total>`.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

// ---------------------------------------------------------------------------
// TOYG01 wire format. Must stay byte-compatible with toy_batch_gpu.mojo.
// ---------------------------------------------------------------------------

const MAGIC: &[u8; 6] = b"TOYG01";
const MAGIC_LEN: usize = 8;
const HEADER_U32: usize = 6;
const HEADER_BYTES: usize = MAGIC_LEN + HEADER_U32 * 4;
const GAIN_SLOTS: usize = 8;
const BLOCK_SIZE: usize = 32;

// ---------------------------------------------------------------------------
// The numerical contract. Pinned deliberately: these are the constants the
// production kernels use, and the oracle exists to catch a silent change.
// ---------------------------------------------------------------------------

const DECAY: f32 = 0.85;
const COUPLING: f32 = 0.08;
const THRESHOLD: f32 = 1.0;
const V_CLAMP: f32 = -5.0;
const SPIKE_PREDICATE: f32 = 0.0;
const COUNT_PREDICATE: f32 = 0.5;
const REFRESH_EVERY: usize = 10;

/// Voltages are compared with a tolerance rather than bit-equality: the Rust
/// and Mojo compilers may each contract `a*b + c` into an FMA, and that is a
/// legitimate difference. Counts are integers and are compared exactly.
const V_TOLERANCE: f32 = 1.0e-6;

/// Synchrony is computed by the same one-pass formula on both sides
/// (E[x^2] - E[x]^2), which is cancellation-prone, so it is compared with a
/// tolerance rather than exactly.
const SYNCHRONY_TOLERANCE: f64 = 1.0e-9;

// ---------------------------------------------------------------------------
// Case specification
// ---------------------------------------------------------------------------

struct Case {
    name: &'static str,
    row_ptr: Vec<i32>,
    col_idx: Vec<i32>,
    weights: Vec<f32>,
    nt_types: Vec<u8>,
    gains: Vec<f32>,
    descending: Vec<i32>,
    drive_neurons: usize,
    num_steps: usize,
}

impl Case {
    fn num_neurons(&self) -> usize {
        self.nt_types.len()
    }

    /// Hand-computed values, independent of the reference implementation.
    /// Without these the reference is only checking itself.
    fn pinned(&self) -> Option<Vec<(&'static str, Vec<(u32, u32)>)>> {
        match self.name {
            // n=1, self-loop w=12.5, gain 1, drive 1.
            //   step0: acc=12.5, new_v=0.85*0+0.08*12.5 = 1.0 exactly
            //          1.0 >= 1.0 -> spike, v=0
            //   every later step re-reads the previous spike and repeats.
            "boundary_exact_threshold" => Some(vec![(
                "all four steps spike, none anywhere else",
                vec![(1, 1), (1, 1), (1, 1), (1, 1)],
            )]),
            // n=1, self-loop w=-100, gain 1, drive 1, no descending.
            //   step0: acc=-100, new_v=0.08*(-100)=-8.0 -> clamped to -5.0
            //   step1: y is all zero so acc=0 -> v = 0.85*(-5)   = -4.25
            //   step2: v = 0.85*(-4.25)                        = -3.6125
            //   step3: v = 0.85*(-3.6125)                      = -3.070625
            //   Nothing ever reaches the threshold.
            "clamp_only_never_spikes" => Some(vec![(
                "no step spikes, the clamp holds",
                vec![(0, 0), (0, 0), (0, 0), (0, 0)],
            )]),
            _ => None,
        }
    }

    fn pinned_final_v(&self) -> Option<Vec<f32>> {
        match self.name {
            "boundary_exact_threshold" => Some(vec![0.0]),
            "clamp_only_never_spikes" => Some(vec![-3.070625]),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Wire format writer
// ---------------------------------------------------------------------------

fn pack(case: &Case) -> Result<Vec<u8>, String> {
    let n = case.num_neurons();
    let nnz = case.col_idx.len();

    if case.row_ptr.len() != n + 1 {
        return Err(format!("row_ptr must have {} entries", n + 1));
    }
    if case.weights.len() != nnz {
        return Err(format!("weights must have {} entries", nnz));
    }
    if case.gains.len() != GAIN_SLOTS {
        return Err(format!("gains must have {GAIN_SLOTS} entries"));
    }
    if case.row_ptr[0] != 0 || case.row_ptr[n] as usize != nnz {
        return Err("row_ptr must span exactly [0, nnz)".to_string());
    }
    for (k, &c) in case.col_idx.iter().enumerate() {
        if c < 0 || c as usize >= n {
            return Err(format!("col_idx[{k}] = {c} is out of range"));
        }
    }
    for (i, &t) in case.nt_types.iter().enumerate() {
        if t as usize >= GAIN_SLOTS {
            return Err(format!("nt_types[{i}] = {t} would read gains out of range"));
        }
    }
    for (i, &d) in case.descending.iter().enumerate() {
        if d < 0 || d as usize >= n {
            return Err(format!("descending[{i}] = {d} is out of range"));
        }
    }
    if case.drive_neurons > n {
        return Err("drive_neurons exceeds num_neurons".to_string());
    }

    let mut out = Vec::with_capacity(HEADER_BYTES + 8 * n);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&[0u8, 0u8]); // pad MAGIC to 8 bytes
    debug_assert_eq!(out.len(), MAGIC_LEN);
    for value in [
        n as u32,
        nnz as u32,
        case.descending.len() as u32,
        case.drive_neurons as u32,
        BLOCK_SIZE as u32,
        GAIN_SLOTS as u32,
    ] {
        out.extend_from_slice(&value.to_le_bytes());
    }
    debug_assert_eq!(out.len(), HEADER_BYTES);

    for &v in &case.row_ptr {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for &v in &case.col_idx {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for &v in &case.weights {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&case.nt_types);
    while out.len() % 4 != 0 {
        out.push(0);
    }
    for &v in &case.descending {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for &v in &case.gains {
        out.extend_from_slice(&v.to_le_bytes());
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Independent CPU reference
// ---------------------------------------------------------------------------

struct Reference {
    counts: Vec<(u32, u32)>,
    final_v: Vec<f32>,
    synchrony: f64,
    /// How many extra counts a list-walk of `descending` would report versus
    /// the unique mask this binary emits. pharma_gpu.mojo:236-241 walks a list
    /// and counts with multiplicity; fly_batch_gpu.mojo uses a mask. That is a
    /// real semantic difference, so the oracle pins it instead of hiding it.
    multiplicity_excess: usize,
}

fn reference(case: &Case) -> Result<Reference, String> {
    pack(case)?; // fail closed on the same validation the binary does

    let n = case.num_neurons();
    let mut x = vec![0.0f32; n];
    let mut y = vec![0.0f32; n];
    let mut v = vec![0.0f32; n];
    for i in 0..n {
        x[i] = if i < case.drive_neurons { 1.0 } else { 0.0 };
    }

    let mut counts = Vec::with_capacity(case.num_steps);
    let mut spiking_descending: BTreeSet<i32> = BTreeSet::new();
    for step in 0..case.num_steps {
        let even = step % 2 == 0;
        let (read, write) = if even {
            (&mut x, &mut y)
        } else {
            (&mut y, &mut x)
        };
        if step % REFRESH_EVERY == 0 {
            // The read buffer is fully overwritten, never blended, so any
            // spike carried over from the previous step is discarded.
            for i in 0..n {
                read[i] = if i < case.drive_neurons { 1.0 } else { 0.0 };
            }
        }

        for r in 0..n {
            let lo = case.row_ptr[r] as usize;
            let hi = case.row_ptr[r + 1] as usize;
            let mut acc = 0.0f32;
            for k in lo..hi {
                let source = case.col_idx[k] as usize;
                let spike = read[source];
                if spike > SPIKE_PREDICATE {
                    let gain = case.gains[case.nt_types[source] as usize];
                    acc += case.weights[k] * gain * spike;
                }
            }
            let new_v = DECAY * v[r] + COUPLING * acc;
            if new_v >= THRESHOLD {
                write[r] = 1.0;
                v[r] = 0.0;
            } else {
                write[r] = 0.0;
                v[r] = if new_v >= V_CLAMP { new_v } else { V_CLAMP };
            }
        }

        // Three different populations, kept apart on purpose:
        //   active_total  every spiking neuron          -> kernel's first field
        //   desc_mask     DISTINCT descending, spiking  -> kernel's second field
        //   desc_list     the same list walked with multiplicity, not compared
        // pharma_gpu.mojo:236-241 counts desc_list; fly_batch_gpu.mojo counts
        // desc_mask. Both are correct for their own kernel, so the oracle pins
        // the difference instead of hiding it.
        let mut active_total = 0u32;
        for i in 0..n {
            if write[i] > COUNT_PREDICATE {
                active_total += 1;
            }
        }
        let mut desc_mask = 0u32;
        let mut desc_list = 0u32;
        for &d in &case.descending {
            if write[d as usize] > COUNT_PREDICATE {
                desc_list += 1;
                if !spiking_descending.contains(&d) {
                    spiking_descending.insert(d);
                    desc_mask += 1;
                }
            }
        }
        // The list walk differs from the mask only by duplicate entries that
        // happen to point at a neuron which spiked.
        let static_excess = (case.descending.len()
            - case.descending.iter().copied().collect::<BTreeSet<_>>().len())
            as u32;
        let extra = desc_list - desc_mask;
        if extra > static_excess {
            return Err(format!(
                "step {step}: list walk exceeds the mask by {extra}, more than the {static_excess} duplicate entries in the list"
            ));
        }
        spiking_descending.clear();
        counts.push((active_total, desc_mask));
    }

    // One-pass synchrony, matching the kernel: sd / (mean + 1).
    let steps = case.num_steps as f64;
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    for &(a, _) in &counts {
        let a = a as f64;
        sum += a;
        sum_sq += a * a;
    }
    let mean = sum / steps;
    let mut variance = sum_sq / steps - mean * mean;
    if variance < 0.0 {
        variance = 0.0;
    }
    let synchrony = variance.sqrt() / (mean + 1.0);

    let unique_descending: BTreeSet<i32> = case.descending.iter().copied().collect();
    Ok(Reference {
        counts,
        final_v: v,
        synchrony,
        multiplicity_excess: case.descending.len() - unique_descending.len(),
    })
}

// ---------------------------------------------------------------------------
// Mojo output parsing
// ---------------------------------------------------------------------------

struct MojoOutput {
    config: Option<(usize, usize, usize, usize)>,
    counts: Vec<(u32, u32)>,
    final_v: Option<Vec<f32>>,
    synchrony: Option<f64>,
}

fn parse_mojo(stdout: &str) -> Result<MojoOutput, String> {
    let mut out = MojoOutput {
        config: None,
        counts: Vec::new(),
        final_v: None,
        synchrony: None,
    };
    for line in stdout.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("TOY_CONFIG ") {
            let mut fields = std::collections::BTreeMap::new();
            for token in rest.split_whitespace() {
                if let Some((k, v)) = token.split_once('=') {
                    fields.insert(
                        k.to_string(),
                        v.parse::<usize>().map_err(|e| format!("{k}={v}: {e}"))?,
                    );
                }
            }
            let get = |k: &str| fields.get(k).copied().ok_or_else(|| format!("missing {k}"));
            out.config = Some((get("n")?, get("nnz")?, get("steps")?, get("drive")?));
        } else if let Some(rest) = line.strip_prefix("TOY_STEP ") {
            let parts: Vec<&str> = rest.split_whitespace().collect();
            if parts.len() != 3 {
                return Err(format!("TOY_STEP needs 3 fields, got {parts:?}"));
            }
            let step: usize = parts[0].parse().map_err(|e| format!("step: {e}"))?;
            let a: u32 = parts[1].parse().map_err(|e| format!("active: {e}"))?;
            let d: u32 = parts[2].parse().map_err(|e| format!("descending: {e}"))?;
            if step != out.counts.len() {
                return Err(format!("TOY_STEP {step} arrived out of order"));
            }
            out.counts.push((a, d));
        } else if let Some(rest) = line.strip_prefix("TOY_FINAL_V") {
            let mut v = Vec::new();
            for token in rest.split_whitespace() {
                v.push(token.parse::<f32>().map_err(|e| format!("v={token}: {e}"))?);
            }
            out.final_v = Some(v);
        } else if let Some(rest) = line.strip_prefix("TOY_SYNCHRONY=") {
            out.synchrony = Some(rest.trim().parse().map_err(|e| format!("sync: {e}"))?);
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Running the kernel
// ---------------------------------------------------------------------------

fn toy_binary() -> PathBuf {
    PathBuf::from(std::env::var("TOY_BATCH_GPU")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../../mojo/fly-spmv/.work/toy_batch_gpu").to_string()))
}

fn work_dir() -> PathBuf {
    PathBuf::from(std::env::var("STAGE2_WORK")
        .unwrap_or_else(|_| ".work/stage2".to_string()))
}

/// Run every case, twice, inside a SINGLE waiting-room admission. Sixteen
/// admissions for a microsecond kernel would be queue overhead masquerading as
/// measurement; one admission is also what makes the determinism check free.
///
/// `dry` swaps the waiting room for a plain `bash` with `GPU_DRY=1`, so the
/// preflight proves the graphs and the binary without touching the GPU.
///
/// The script is written to disk rather than passed with `bash -c` so the
/// harness and the shell agree on the exact command that ran.
fn run_all(binary: &Path, work: &Path, all: &[Case], dry: bool) -> Result<String, String> {
    let script = work.join("run_cases.sh");
    let mut body = String::from("#!/usr/bin/env bash\nset -euo pipefail\n");
    for case in all {
        body.push_str(&format!(
            "for pass in 1 2; do\n  echo \"TOY_CASE {} pass=$pass\"\n  {} \"{}\" {}\ndone\n",
            case.name,
            binary.display(),
            work.join(format!("{}.bin", case.name)).display(),
            case.num_steps
        ));
    }
    fs::write(&script, &body).map_err(|e| format!("writing {}: {e}", script.display()))?;

    let mut command = if dry {
        let mut c = Command::new("bash");
        c.env("GPU_DRY", "1");
        c
    } else {
        let mut c = Command::new("gpu-wait");
        c.args(["run", "--vram", "1", "--timeout", "300", "--", "bash"]);
        c
    };
    let output = command
        .arg(&script)
        .output()
        .map_err(|e| format!("spawning {:?}: {e}", binary.display()))?;
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    if !output.status.success() {
        return Err(format!(
            "kernel run exited with {:?}\nstdout:\n{stdout}\nstderr:\n{}",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(stdout)
}

/// Split the combined output back into one `MojoOutput` per (case, pass).
/// The lines that must be identical between two runs. `TOY_ELAPSED_MS` is
/// excluded on purpose: it is a timing measurement, not a result.
fn deterministic_lines(block: &str) -> Vec<&str> {
    block
        .lines()
        .filter(|l| {
            l.starts_with("TOY_STEP") || l.starts_with("TOY_FINAL_V") || l.starts_with("TOY_SYNCHRONY")
        })
        .collect()
}

fn split_by_case(stdout: &str) -> std::collections::BTreeMap<(String, u8), String> {
    let mut out = std::collections::BTreeMap::new();
    let mut current: Option<(String, u8)> = None;
    let mut buf = String::new();
    for line in stdout.lines() {
        if let Some(rest) = line.strip_prefix("TOY_CASE ") {
            if let Some(key) = current.take() {
                out.insert(key, std::mem::take(&mut buf));
            }
            let mut parts = rest.split_whitespace();
            let name = parts.next().unwrap_or_default().to_string();
            let pass = parts
                .find_map(|p| p.strip_prefix("pass=").map(|v| v.parse().unwrap_or(0)))
                .unwrap_or(0);
            current = Some((name, pass));
            continue;
        }
        if current.is_some() {
            buf.push_str(line);
            buf.push('\n');
        }
    }
    if let Some(key) = current {
        out.insert(key, buf);
    }
    out
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

fn compare(
    name: &str,
    case: &Case,
    want: &Reference,
    got: &MojoOutput,
    quiet: bool,
) -> Vec<String> {
    let mut failures = Vec::new();

    match got.config {
        Some((n, nnz, steps, drive)) => {
            if n != case.num_neurons() || nnz != case.col_idx.len() {
                failures.push(format!("kernel reports n={n} nnz={nnz}, graph says n={} nnz={}", case.num_neurons(), case.col_idx.len()));
            }
            if steps != case.num_steps || drive != case.drive_neurons {
                failures.push(format!("kernel reports steps={steps} drive={drive}, expected {} and {}", case.num_steps, case.drive_neurons));
            }
        }
        None => failures.push("kernel printed no TOY_CONFIG line".to_string()),
    }

    if got.counts.len() != want.counts.len() {
        failures.push(format!("kernel produced {} steps, expected {}", got.counts.len(), want.counts.len()));
    } else {
        for (step, (expected, actual)) in want.counts.iter().zip(&got.counts).enumerate() {
            if *expected != *actual {
                failures.push(format!(
                    "step {step}: kernel active={} descending={}, reference active={} descending={}",
                    actual.0, actual.1, expected.0, expected.1
                ));
            }
        }
    }

    match got.synchrony {
        None => failures.push("kernel printed no TOY_SYNCHRONY".to_string()),
        Some(s) if (s - want.synchrony).abs() > SYNCHRONY_TOLERANCE => failures.push(
            format!("synchrony kernel={s} reference={} delta={:e}", want.synchrony, (s - want.synchrony).abs()),
        ),
        Some(_) => {}
    }

    match &got.final_v {
        None => failures.push("kernel printed no TOY_FINAL_V".to_string()),
        Some(v) => {
            if v.len() != want.final_v.len() {
                failures.push(format!("kernel returned {} voltages, expected {}", v.len(), want.final_v.len()));
            } else {
                for (i, (a, b)) in v.iter().zip(&want.final_v).enumerate() {
                    if (a - b).abs() > V_TOLERANCE {
                        failures.push(format!("final v[{i}] kernel={a} reference={b}"));
                    }
                }
            }
        }
    }

    if !failures.is_empty() && !quiet {
        println!("FAIL {name}");
        for f in &failures {
            println!("     {f}");
        }
    }
    failures
}

/// Negative arm. A gate that has never rejected anything is not evidence, so
/// this feeds the comparator four deliberately corrupted references and
/// requires every one to be caught. If a perturbation slips through, the
/// comparator is blind to that channel and the gate says so.
fn negative_arm(case: &Case, want: &Reference, got: &MojoOutput) -> Result<usize, String> {
    let mut detected = 0usize;
    let mut checks: Vec<(&str, Box<dyn Fn(&mut Reference)>)> = vec![
        ("one active count off by one", Box::new(|r: &mut Reference| {
            r.counts[0].0 += 1;
        })),
        ("one descending count off by one", Box::new(|r: &mut Reference| {
            r.counts[0].1 += 1;
        })),
        ("one final voltage nudged past tolerance", Box::new(|r: &mut Reference| {
            r.final_v[0] += V_TOLERANCE * 10.0;
        })),
        ("synchrony nudged past tolerance", Box::new(|r: &mut Reference| {
            r.synchrony += SYNCHRONY_TOLERANCE * 10.0;
        })),
    ];
    for (what, perturb) in &mut checks {
        let mut corrupted = Reference {
            counts: want.counts.clone(),
            final_v: want.final_v.clone(),
            synchrony: want.synchrony,
            multiplicity_excess: want.multiplicity_excess,
        };
        perturb(&mut corrupted);
        if compare(case.name, case, &corrupted, got, true).is_empty() {
            println!("FAIL negative arm: '{what}' was NOT detected, the comparator is blind to that channel");
        } else {
            detected += 1;
        }
    }
    println!(
        "STAGE2_NEGATIVE_ARM={detected}/{} on {}",
        checks.len(),
        case.name
    );
    Ok(detected)
}

// ---------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------

/// Unit gain on every real transmitter channel, 0.0 on the unused spare slot 7
/// as a deliberate poison: a case that ever declares nt 7 would zero its edge
/// instead of quietly working.
fn neutral_gains() -> Vec<f32> {
    vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
}

fn cases() -> Vec<Case> {
    vec![
        // Exact inclusive threshold: 0.08 * 12.5 == 1.0 in f32, so this pins
        // the `>=` and the threshold constant.
        Case {
            name: "boundary_exact_threshold",
            row_ptr: vec![0, 1],
            col_idx: vec![0],
            weights: vec![12.5],
            nt_types: vec![0],
            gains: neutral_gains(),
            descending: vec![0],
            drive_neurons: 1,
            num_steps: 4,
        },
        // Clamp path, and the observation that v is never positive: the kernel
        // resets v to 0 on a spike and otherwise only ever stores <= 0.
        Case {
            name: "clamp_only_never_spikes",
            row_ptr: vec![0, 1],
            col_idx: vec![0],
            weights: vec![-100.0],
            nt_types: vec![0],
            gains: neutral_gains(),
            descending: vec![],
            drive_neurons: 1,
            num_steps: 4,
        },
        // Mixed transmitters and a signed inhibitory edge. nt 1 carries a
        // gain of 2.0, nt 3 a gain of 0.5.
        Case {
            name: "signed_mixed_transmitters",
            row_ptr: vec![0, 2, 3, 3, 3],
            col_idx: vec![1, 0, 2],
            weights: vec![3.0, -2.0, 1.0],
            nt_types: vec![0, 1, 3, 1],
            gains: vec![1.0, 2.0, 1.0, 0.5, 1.0, 1.0, 1.0, 0.0],
            descending: vec![0, 2],
            drive_neurons: 3,
            num_steps: 6,
        },
        // 7 neurons with BLOCK_SIZE 32: 25 of 32 lanes idle in the single
        // block, and 1 of 2 in the reduce, so the partial-block path and the
        // per-block partial summing are both covered.
        Case {
            name: "partial_block_idle_lanes",
            row_ptr: vec![0, 1, 1, 2, 3, 3, 4, 6, 6],
            col_idx: vec![0, 1, 4, 5, 0, 3],
            weights: vec![12.5, 4.0, 6.0, 9.0, 2.0, 5.0],
            nt_types: vec![0, 0, 1, 0, 0, 2, 0, 1],
            gains: vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            descending: vec![1, 3, 5],
            drive_neurons: 5,
            num_steps: 12,
        },
        // No edges at all, and no descending neurons: everything stays zero and
        // synchrony must be exactly 0/(0+1) = 0.
        Case {
            name: "empty_edges_all_quiet",
            row_ptr: vec![0, 0, 0, 0, 0],
            col_idx: vec![],
            weights: vec![],
            nt_types: vec![0, 0, 0, 0],
            gains: neutral_gains(),
            descending: vec![],
            drive_neurons: 4,
            num_steps: 5,
        },
        // Odd step count on a graph that is still spiking, so the final
        // voltage is read from the write buffer of the last step.
        Case {
            name: "odd_step_count_pingpong",
            row_ptr: vec![0, 1, 2],
            col_idx: vec![0, 1],
            weights: vec![12.5, 12.5],
            nt_types: vec![0, 0],
            gains: neutral_gains(),
            descending: vec![0, 1],
            drive_neurons: 2,
            num_steps: 7,
        },
        // Duplicate descending index. The mask counts it once, a list walk
        // would count it twice; the oracle pins both numbers.
        Case {
            name: "descending_duplicate_mask_vs_list",
            row_ptr: vec![0, 2, 2],
            col_idx: vec![0, 1],
            weights: vec![12.5, 12.5],
            nt_types: vec![0, 0],
            gains: neutral_gains(),
            descending: vec![0, 0, 1],
            drive_neurons: 2,
            num_steps: 4,
        },
        // 24 neurons, one block, with a refresh landing mid-run at step 10.
        Case {
            name: "refresh_at_step_ten",
            row_ptr: vec![0, 1, 1, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8, 9, 9, 10, 11, 11, 12, 13, 13, 14, 15, 15, 16],
            col_idx: vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            weights: vec![12.5, -3.0, 7.0, 2.0, 12.5, 1.0, 4.0, 12.5, 0.5, 2.5, 12.5, 3.5, 1.5, 12.5, 2.0, 1.0],
            nt_types: (0..24).map(|i| (i % 4) as u8).collect(),
            gains: vec![1.0, 2.0, 0.5, 1.5, 1.0, 1.0, 1.0, 0.0],
            descending: vec![2, 5, 9, 14, 20.min(23)],
            drive_neurons: 20,
            num_steps: 15,
        },
    ]
}

// ---------------------------------------------------------------------------
// Gate
// ---------------------------------------------------------------------------

fn main() {
    let dry = std::env::args().any(|a| a == "--dry");
    let all = cases();
    let work = work_dir();
    if let Err(e) = fs::create_dir_all(&work) {
        eprintln!("FAIL creating {}: {e}", work.display());
        std::process::exit(1);
    }

    let binary = toy_binary();
    if !binary.exists() {
        println!(
            "FAIL stage2: {} is missing. Build it with:\n  mojo build $FLY_SPMV_DIR/src/toy_batch_gpu.mojo -o {}",
            binary.display(),
            binary.display()
        );
        println!("STAGE2_PARITY_PASS=0/{}", all.len());
        std::process::exit(1);
    }

    // A wrong pin would make the whole gate agree on a wrong answer, so the
    // hand-computed values are checked against the reference before either is
    // compared to the kernel.
    let mut failures = 0usize;
    for case in &all {
        let Some(pins) = case.pinned() else { continue };
        let want = match reference(case) {
            Ok(r) => r,
            Err(e) => {
                println!("FAIL {} reference does not build: {e}", case.name);
                failures += 1;
                continue;
            }
        };
        for (why, expected) in pins {
            if want.counts != *expected {
                println!("FAIL {} reference disagrees with its own pin ({why})", case.name);
                println!("     reference: {:?}", want.counts);
                println!("     pinned:    {expected:?}");
                failures += 1;
            }
        }
        if let Some(pinned_v) = case.pinned_final_v() {
            for (i, (a, b)) in want.final_v.iter().zip(&pinned_v).enumerate() {
                if (a - b).abs() > V_TOLERANCE {
                    println!("FAIL {} reference v[{i}]={a} disagrees with the pin {b}", case.name);
                    failures += 1;
                }
            }
        }
    }
    if failures > 0 {
        println!("STAGE2_PARITY_PASS=0/{}", all.len());
        std::process::exit(1);
    }

    // One admission for every case, both passes.
    let combined = match run_all(&binary, &work, &all, dry) {
        Ok(s) => s,
        Err(e) => {
            println!("FAIL stage2: {e}");
            println!("STAGE2_PARITY_PASS=0/{}", all.len());
            std::process::exit(1);
        }
    };
    let by_case = split_by_case(&combined);

    let mut passed = 0usize;
    let mut negative_ran = false;
    for case in &all {
        let first = by_case
            .get(&(case.name.to_string(), 1))
            .ok_or_else(|| format!("no output for {}", case.name));
        let second = by_case
            .get(&(case.name.to_string(), 2))
            .ok_or_else(|| format!("no repeat output for {}", case.name));
        let (first, second) = match (first, second) {
            (Ok(a), Ok(b)) => (a.clone(), b.clone()),
            (Err(e), _) | (_, Err(e)) => {
                println!("FAIL {e}");
                continue;
            }
        };
        let (a, b) = (deterministic_lines(&first), deterministic_lines(&second));
        if a != b {
            println!("FAIL {} is not deterministic across two identical runs", case.name);
            for (x, y) in a.iter().zip(&b) {
                if x != y {
                    println!("     pass 1: {x}");
                    println!("     pass 2: {y}");
                }
            }
            if a.len() != b.len() {
                println!("     line counts differ: {} vs {}", a.len(), b.len());
            }
            continue;
        }

        if dry {
            // The dry run stops before the GPU on purpose, so it validates the
            // graphs and the binary, not the arithmetic.
            println!("PASS {} (dry, graph validated)", case.name);
            passed += 1;
            continue;
        }

        let want = match reference(case) {
            Ok(r) => r,
            Err(e) => {
                println!("FAIL {} reference does not build: {e}", case.name);
                continue;
            }
        };
        let got = match parse_mojo(&first) {
            Ok(g) => g,
            Err(e) => {
                println!("FAIL {} unparsable output: {e}", case.name);
                continue;
            }
        };
        // The negative arm runs on the first case only: it proves the
        // comparator rejects a corrupted reference, which is a property of the
        // comparator, not of the case.
        if !negative_ran {
            negative_ran = true;
            if negative_arm(case, &want, &got).unwrap_or(0) < 4 {
                failures += 1;
            }
        }

        let before = failures;
        failures += compare(case.name, case, &want, &got, false).len();
        if failures == before {
            passed += 1;
            println!(
                "PASS {} n={} nnz={} steps={} sync={:.6} mult_excess={}",
                case.name,
                case.num_neurons(),
                case.col_idx.len(),
                case.num_steps,
                want.synchrony,
                want.multiplicity_excess
            );
        }
    }

    if failures > 0 {
        println!("FAIL {failures} mismatch(es) across {} case(s)", all.len());
    }
    println!("STAGE2_PARITY_PASS={passed}/{}", all.len());
    if passed != all.len() {
        std::process::exit(1);
    }
}
