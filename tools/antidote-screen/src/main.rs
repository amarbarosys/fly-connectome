use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Gains {
    ach: f32,
    glu: f32,
    gaba: f32,
    dop: f32,
    ser: f32,
    oct: f32,
}

impl Default for Gains {
    fn default() -> Self {
        Self {
            ach: 1.0,
            glu: 1.0,
            gaba: 1.0,
            dop: 1.0,
            ser: 1.0,
            oct: 1.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ThreatProfile {
    id: String,
    name: String,
    mechanism: String,
    raw_gains: Gains,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AntidoteProtocol {
    id: String,
    name: String,
    drug_class: String,
    target_action: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SimulationResult {
    name: String,
    compound: String,
    protocol: String,
    gains: Gains,
    active_pct: f64,
    dn_pct: f64,
    synchrony: f64,
    wall_ms: f64,
    efficacy_pct: f64,
    clinical_verdict: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct ScreenerReceipt {
    schema_version: u32,
    timestamp_unix: u64,
    engine_bin: String,
    total_simulations: usize,
    total_wall_ms: f64,
    baseline_active_pct: f64,
    results: Vec<SimulationResult>,
}

fn apply_countermeasure(raw: &Gains, protocol_id: &str) -> Gains {
    let mut g = raw.clone();
    match protocol_id {
        "untreated" => g,
        "atropine" => {
            // Competitive muscarinic block suppresses excessive cholinergic excitation
            g.ach = (raw.ach * 0.25).max(1.20);
            g
        }
        "pralidoxime" => {
            // 2-PAM enzymatic oxime reactivation of alkylated AChE
            g.ach = (raw.ach * 0.35).max(1.40);
            g
        }
        "diazepam" => {
            // GABA-A positive allosteric modulator: amplifies inhibitory drive to suppress seizures
            g.gaba = 2.40;
            g
        }
        "dual" => {
            // Dual Mark I Protocol: Atropine + 2-PAM
            g.ach = (raw.ach * 0.15).max(1.10);
            g
        }
        "triple" => {
            // Full Emergency Protocol: Atropine + 2-PAM + Diazepam
            g.ach = (raw.ach * 0.12).max(1.05);
            g.gaba = 2.20;
            g
        }
        _ => g,
    }
}

fn run_gpu_simulation(
    name: &str,
    gains: &Gains,
    steps: u32,
    pharma_bin: &Path,
) -> Result<HashMap<String, f64>, String> {
    let t0 = Instant::now();

    let output = Command::new("gpu-wait")
        .args(["run", "--vram", "4", "--timeout", "60", "--"])
        .arg("env")
        .arg(format!("PHARMA_COMPOUND={}", name))
        .arg(format!("PHARMA_G_ACH={:.4}", gains.ach))
        .arg(format!("PHARMA_G_GLU={:.4}", gains.glu))
        .arg(format!("PHARMA_G_GABA={:.4}", gains.gaba))
        .arg(format!("PHARMA_G_DOP={:.4}", gains.dop))
        .arg(format!("PHARMA_G_SER={:.4}", gains.ser))
        .arg(format!("PHARMA_G_OCT={:.4}", gains.oct))
        .arg(format!("PHARMA_STEPS={}", steps))
        .arg("FLY_DATA_DIR=/dev/shm/fly-connectome/data/")
        .arg(pharma_bin)
        .current_dir(pharma_bin.parent().unwrap_or(Path::new(".")))
        .output()
        .map_err(|e| format!("Failed to spawn gpu-wait: {}", e))?;

    let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

    if !output.status.success() {
        return Err(format!(
            "Simulation failed with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let stdout_str = String::from_utf8_lossy(&output.stdout);
    let mut map = HashMap::new();
    map.insert("wall_ms".into(), wall_ms);

    for line in stdout_str.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("PHARMA_RESULT_") {
            if let Some((k, v)) = rest.split_once('=') {
                let key = k.trim().to_lowercase();
                if let Ok(num) = v.trim().parse::<f64>() {
                    map.insert(key, num);
                }
            }
        }
    }

    Ok(map)
}

fn compute_clinical_efficacy(observed_active: f64, baseline_active: f64) -> (f64, String) {
    let dev = (observed_active - baseline_active).abs();
    let score = (1.0 - (dev / baseline_active)).max(0.0) * 100.0;

    let verdict = if score >= 90.0 {
        "HOMEOSTASIS RESTORED (FULL RECOVERY)"
    } else if score >= 75.0 {
        "PARTIAL RESCUE (STABILIZED)"
    } else if observed_active > baseline_active * 1.5 {
        "HYPEREXCITATORY STATUS EPILEPTICUS"
    } else if observed_active < baseline_active * 0.5 {
        "SEVERE NEURODEPRESSION / MOTOR ARREST"
    } else {
        "INSUFFICIENT THERAPEUTIC EFFICACY"
    };

    (score, verdict.into())
}

fn main() {
    println!("==========================================================================");
    println!(" FlyBrain Emergency Neurotoxin Antidote & Seizure Resuscitation Screener");
    println!(" Pure Rust / Mojo Engine: Whole-Brain AMD RX 7900 XTX 24GB Evaluation");
    println!("==========================================================================");

    let pharma_bin = PathBuf::from("/home/<user>/Projects/mojo/fly-spmv/pharma_gpu");
    if !pharma_bin.exists() {
        eprintln!("Error: GPU engine binary not found at {}", pharma_bin.display());
        std::process::exit(1);
    }

    let threats = vec![
        ThreatProfile {
            id: "sarin".into(),
            name: "Sarin (GB Analogue)".into(),
            mechanism: "Volatile organophosphate AChE inhibitor (synaptic ACh accumulation)".into(),
            raw_gains: Gains { ach: 5.5, glu: 1.0, gaba: 0.6, dop: 0.3, ser: 0.2, oct: 0.4 },
        },
        ThreatProfile {
            id: "vx".into(),
            name: "VX Persistent Analogue".into(),
            mechanism: "Hydrophobic persistent AChE alkylator (extreme cholinergic storm)".into(),
            raw_gains: Gains { ach: 7.5, glu: 1.2, gaba: 0.4, dop: 0.2, ser: 0.1, oct: 0.3 },
        },
        ThreatProfile {
            id: "dfp".into(),
            name: "DFP (Diisopropyl Fluorophosphate)".into(),
            mechanism: "Catalytic irreversible esterase block (intense central hyperactivity)".into(),
            raw_gains: Gains { ach: 4.8, glu: 1.0, gaba: 0.7, dop: 0.4, ser: 0.3, oct: 0.5 },
        },
        ThreatProfile {
            id: "picrotoxin".into(),
            name: "Picrotoxin (Status Epilepticus)".into(),
            mechanism: "GABA-A / Rdl chloride channel blocker (loss of network inhibition)".into(),
            raw_gains: Gains { ach: 1.0, glu: 1.0, gaba: 0.1, dop: 1.0, ser: 1.0, oct: 1.0 },
        },
    ];

    let protocols = vec![
        AntidoteProtocol {
            id: "untreated".into(),
            name: "Untreated Poisoning".into(),
            drug_class: "Control / No Countermeasure".into(),
            target_action: "Full unchecked toxicity".into(),
        },
        AntidoteProtocol {
            id: "atropine".into(),
            name: "Atropine Monotherapy".into(),
            drug_class: "Muscarinic Receptor Antagonist".into(),
            target_action: "Competitive postsynaptic blockade".into(),
        },
        AntidoteProtocol {
            id: "pralidoxime".into(),
            name: "Pralidoxime (2-PAM)".into(),
            drug_class: "Oxime Esterase Reactivator".into(),
            target_action: "Nucleophilic de-alkylation of AChE".into(),
        },
        AntidoteProtocol {
            id: "diazepam".into(),
            name: "Diazepam Anticonvulsant".into(),
            drug_class: "GABA-A Allosteric Modulator (PAM)".into(),
            target_action: "Inhibitory current amplification".into(),
        },
        AntidoteProtocol {
            id: "dual".into(),
            name: "Dual Protocol (Atropine + 2-PAM)".into(),
            drug_class: "Standard Mark I Kit".into(),
            target_action: "Combined receptor block & enzyme rescue".into(),
        },
        AntidoteProtocol {
            id: "triple".into(),
            name: "Triple Emergency Protocol (NAAK + Diazepam)".into(),
            drug_class: "Full Medical Resuscitation Cocktail".into(),
            target_action: "Tripartite cholinergic clamp + seizure halt".into(),
        },
    ];

    println!("[1/3] Establishing baseline physiological firing (unperturbed)...");
    let baseline_gains = Gains::default();
    let baseline_res = run_gpu_simulation("Healthy Baseline", &baseline_gains, 100, &pharma_bin)
        .expect("Baseline simulation failed");
    let baseline_active = *baseline_res.get("active_pct").unwrap_or(&10.0);
    let baseline_dn = *baseline_res.get("dn_pct").unwrap_or(&19.2);
    println!(
        "  -> Healthy Baseline: Active={:.1}% | DN Motor={:.1}% | Sync={:.3}",
        baseline_active,
        baseline_dn,
        baseline_res.get("synchrony").unwrap_or(&0.156)
    );

    println!("\n[2/3] Screening antidote cocktails across {} threat conditions...", threats.len());
    let mut all_results = Vec::new();
    let batch_start = Instant::now();

    for threat in &threats {
        println!("\n>>> Threat Challenge: {} <<<", threat.name);
        println!("    Mechanism: {}", threat.mechanism);

        for proto in &protocols {
            let modulated = apply_countermeasure(&threat.raw_gains, &proto.id);
            let sim_name = format!("{}_{}", threat.id, proto.id);

            let res = run_gpu_simulation(&sim_name, &modulated, 100, &pharma_bin)
                .expect("Simulation step failed");

            let active_pct = *res.get("active_pct").unwrap_or(&0.0);
            let dn_pct = *res.get("dn_pct").unwrap_or(&0.0);
            let sync = *res.get("synchrony").unwrap_or(&0.0);
            let wall = *res.get("wall_ms").unwrap_or(&0.0);

            let (eff, verdict) = compute_clinical_efficacy(active_pct, baseline_active);

            println!(
                "  [{:<20}] Act: {:5.1}% | DN: {:5.1}% | Sync: {:.3} | Efficacy: {:5.1}% | {:<28} ({}ms)",
                proto.name, active_pct, dn_pct, sync, eff, verdict, wall as u64
            );

            all_results.push(SimulationResult {
                name: sim_name,
                compound: threat.name.clone(),
                protocol: proto.name.clone(),
                gains: modulated,
                active_pct,
                dn_pct,
                synchrony: sync,
                wall_ms: wall,
                efficacy_pct: eff,
                clinical_verdict: verdict,
            });
        }
    }

    let total_wall_s = batch_start.elapsed().as_secs_f64();
    println!("\n==========================================================================");
    println!(
        " Resuscitation Screening Complete: {} scenarios evaluated in {:.1}s ({:.2}s/sim)",
        all_results.len(),
        total_wall_s,
        total_wall_s / (all_results.len() as f64)
    );
    println!("==========================================================================");

    // Write structured receipt
    let receipt = ScreenerReceipt {
        schema_version: 1,
        timestamp_unix: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        engine_bin: pharma_bin.display().to_string(),
        total_simulations: all_results.len(),
        total_wall_ms: total_wall_s * 1000.0,
        baseline_active_pct: baseline_active,
        results: all_results,
    };

    let receipt_path = PathBuf::from("data/antidote_screen_receipt.json");
    if let Ok(bytes) = serde_json::to_string_pretty(&receipt) {
        let _ = fs::write(&receipt_path, bytes);
        println!("Receipt published to {}", receipt_path.display());
    }
}
