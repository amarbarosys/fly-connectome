use memmap2::MmapOptions;
use rayon::prelude::*;
use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

struct ScaffoldFilter {
    name: &'static str,
    target_role: &'static str,
    min_len: usize,
    max_len: usize,
    required_substrings: &'static [&'static str],
    excluded_substrings: &'static [&'static str],
}

const PRESETS: &[ScaffoldFilter] = &[
    ScaffoldFilter {
        name: "diazepam",
        target_role: "GABA-A Anticonvulsant / Status Epilepticus Rescue",
        min_len: 25,
        max_len: 75,
        // 1,4-benzodiazepine core motif
        required_substrings: &["C(=O)CN=C(", "Cl"],
        excluded_substrings: &["[Si]", "[Sn]", "[Pb]", "[Hg]"],
    },
    ScaffoldFilter {
        name: "atropine",
        target_role: "Muscarinic Acetylcholine Receptor Blocker",
        min_len: 25,
        max_len: 75,
        // Tropane ester + phenyl motif
        required_substrings: &["OC(=O)C(", "C=CC=C"],
        excluded_substrings: &["[Si]", "[Sn]"],
    },
    ScaffoldFilter {
        name: "oxime",
        target_role: "AChE Oxime Reactivator (2-PAM Analogue)",
        min_len: 10,
        max_len: 60,
        // Aldoxime + aromatic ring motif
        required_substrings: &["=NO", "C=CC=C"],
        excluded_substrings: &["[Si]"],
    },
];


fn matches_filter(smiles: &str, filter: &ScaffoldFilter) -> bool {
    let len = smiles.len();
    if len < filter.min_len || len > filter.max_len {
        return false;
    }
    for req in filter.required_substrings {
        if !smiles.contains(req) {
            return false;
        }
    }
    for exc in filter.excluded_substrings {
        if smiles.contains(exc) {
            return false;
        }
    }
    true
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let query_name = args.get(1).map(|s| s.as_str()).unwrap_or("diazepam");

    let filter = PRESETS
        .iter()
        .find(|p| p.name.eq_ignore_ascii_case(query_name))
        .unwrap_or(&PRESETS[0]);

    println!("==========================================================================");
    println!(" PubChem 118M Analog Miner (Parallel Zero-Copy Rust Stream Engine)");
    println!(" Target Scaffold: {} ({})", filter.name.to_uppercase(), filter.target_role);
    println!("==========================================================================");

    let smiles_path = PathBuf::from("/home/<user>/Projects/science/fly-connectome/data/chemical_universe/CID-SMILES");
    if !smiles_path.exists() {
        eprintln!("Error: PubChem SMILES file not found at {}", smiles_path.display());
        std::process::exit(1);
    }

    println!("[1/3] Memory-mapping 118M PubChem universe ({:.2} GB)...", smiles_path.metadata().unwrap().len() as f64 / 1024.0 / 1024.0 / 1024.0);
    let t_mmap = Instant::now();
    let file = File::open(&smiles_path).expect("Failed to open CID-SMILES");
    let mmap = unsafe { MmapOptions::new().map(&file).expect("Failed to mmap file") };
    println!("  -> Mapped in {:.2} ms", t_mmap.elapsed().as_secs_f64() * 1000.0);

    println!("[2/3] Parallel scanning 118,000,000 molecules with Rayon (16 threads)...");
    let t_scan = Instant::now();

    // Chunk by line boundaries for parallel processing
    let bytes = &mmap[..];
    let num_threads = rayon::current_num_threads();
    let chunk_size = bytes.len() / num_threads;

    let mut chunk_boundaries = Vec::new();
    let mut current_offset = 0;

    for i in 0..num_threads {
        let start = current_offset;
        let mut end = if i == num_threads - 1 {
            bytes.len()
        } else {
            let rough_end = (start + chunk_size).min(bytes.len());
            // Advance to next newline
            let mut pos = rough_end;
            while pos < bytes.len() && bytes[pos] != b'\n' {
                pos += 1;
            }
            if pos < bytes.len() { pos + 1 } else { bytes.len() }
        };
        chunk_boundaries.push((start, end));
        current_offset = end;
    }

    let candidates: Vec<(u64, String)> = chunk_boundaries
        .into_par_iter()
        .flat_map(|(start, end)| {
            let slice = &bytes[start..end];
            let mut matches = Vec::new();

            for line in slice.split(|&b| b == b'\n') {
                if line.is_empty() {
                    continue;
                }
                // Format: CID\tSMILES
                if let Some(tab_pos) = line.iter().position(|&b| b == b'\t') {
                    if let Ok(cid_str) = std::str::from_utf8(&line[..tab_pos]) {
                        if let Ok(cid) = cid_str.parse::<u64>() {
                            if let Ok(smiles) = std::str::from_utf8(&line[tab_pos + 1..]) {
                                if matches_filter(smiles, filter) {
                                    matches.push((cid, smiles.to_string()));
                                }
                            }
                        }
                    }
                }
            }
            matches
        })
        .collect();

    let scan_duration = t_scan.elapsed().as_secs_f64();
    println!("  -> Scanned 118M molecules in {:.2}s ({:.1}M molecules/sec)", scan_duration, 118.0 / scan_duration);
    println!("  -> Extracted {} candidate analog structures", candidates.len());

    println!("[3/3] Publishing top candidate analogs...");
    let out_path = PathBuf::from(format!("data/pubchem_analogs_{}.tsv", filter.name));
    if let Some(parent) = out_path.parent() {
        let _ = fs::create_dir_all(parent);
    }

    let mut out_file = File::create(&out_path).expect("Failed to create output TSV");
    writeln!(out_file, "pubchem_cid\tcanonical_smiles\ttarget_scaffold").unwrap();
    for (cid, smiles) in &candidates {
        writeln!(out_file, "{}\t{}\t{}", cid, smiles, filter.name).unwrap();
    }

    println!("Published {} analogs to {}", candidates.len(), out_path.display());
    println!("\nTop 10 Sample Candidates:");
    for (i, (cid, smiles)) in candidates.iter().take(10).enumerate() {
        println!("  [{:02}] CID: {:<10} | SMILES: {}", i + 1, cid, smiles);
    }
}
