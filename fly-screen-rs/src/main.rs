use duckdb::{Connection, params};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::env;
use std::ffi::CString;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const MAX_CONDITIONS: usize = 50;
const GAINS_PER_CONDITION: usize = 8;
const DEFAULT_STEPS: u32 = 100;
const EXPECTED_NEURONS: usize = 139_255;
const EXPECTED_EDGES: usize = 10_572_269;

#[derive(Debug, Clone, Serialize)]
struct Condition {
    condition_id: String,
    compound_id: String,
    name: String,
    dose_level: String,
    dose_nm: f64,
    gains: [f32; GAINS_PER_CONDITION],
}

#[derive(Clone, Serialize)]
struct InputHash {
    path: String,
    size_bytes: u64,
    sha256: String,
}

#[derive(Serialize)]
struct StageManifest {
    schema_version: u32,
    files: Vec<StageFile>,
}

#[derive(Serialize)]
struct StageFile {
    filename: String,
    size_bytes: u64,
    sha256: String,
}

#[derive(Serialize)]
struct StageReceipt {
    mode: String,
    directory: String,
    reused: bool,
    operation_ms: f64,
    manifest_sha256: Option<String>,
}

#[derive(Serialize)]
struct ResultRow {
    condition_id: String,
    compound_id: String,
    name: String,
    dose_level: String,
    dose_nm: f64,
    steps: u32,
    mean_active: f64,
    active_pct: f64,
    mean_descending: f64,
    descending_pct: f64,
    synchrony: f64,
}

#[derive(Serialize)]
struct Receipt {
    schema_version: u32,
    run_id: String,
    created_unix_seconds: u64,
    conditions: usize,
    conditions_requested: usize,
    dose_ladder_policy: String,
    descending_neurons: usize,
    steps: u32,
    condition_manifest: Vec<Condition>,
    engine: String,
    engine_sha256: String,
    driver_sha256: String,
    engine_reported_wall_ms: f64,
    gpujob_wall_ms: f64,
    host_wall_ms: f64,
    peak_rss_bytes: Option<u64>,
    compact_readback_bytes: usize,
    data_stage: StageReceipt,
    input_hashes: Vec<InputHash>,
    manifest_sha256: String,
    results_sha256: String,
    worker_stdout_sha256: String,
    results: Vec<ResultRow>,
}

fn fail<T>(message: impl Into<String>) -> Result<T, String> {
    Err(message.into())
}

fn sha256_file(path: &Path) -> io::Result<InputHash> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    let mut size = 0_u64;
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        size += read as u64;
        hasher.update(&buffer[..read]);
    }
    Ok(InputHash {
        path: path.display().to_string(),
        size_bytes: size,
        sha256: format!("{:x}", hasher.finalize()),
    })
}

fn available_bytes(path: &Path) -> Option<u64> {
    let path = CString::new(path.as_os_str().as_bytes()).ok()?;
    let mut stats = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    let rc = unsafe { libc::statvfs(path.as_ptr(), stats.as_mut_ptr()) };
    if rc != 0 {
        return None;
    }
    let stats = unsafe { stats.assume_init() };
    (stats.f_bavail as u64).checked_mul(stats.f_frsize as u64)
}

fn copy_and_hash(source: &Path, destination: &Path) -> io::Result<InputHash> {
    let mut input = File::open(source)?;
    let mut output = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    let mut size = 0_u64;
    loop {
        let read = input.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        output.write_all(&buffer[..read])?;
        size += read as u64;
    }
    output.sync_all()?;
    Ok(InputHash {
        path: destination.display().to_string(),
        size_bytes: size,
        sha256: format!("{:x}", hasher.finalize()),
    })
}

fn prepare_ram_stage(
    data_sources: &[PathBuf],
    source_hashes: &[InputHash],
) -> Result<(PathBuf, StageReceipt), String> {
    if data_sources.len() != source_hashes.len() {
        return fail("RAM stage source/hash list length mismatch");
    }
    let started = Instant::now();
    let root = PathBuf::from(
        env::var("FLY_RAM_STAGE_ROOT")
            .unwrap_or_else(|_| "/dev/shm/fly-connectome/fly-screen-stages/v1".into()),
    );
    fs::create_dir_all(&root).map_err(|e| format!("create RAM stage root: {e}"))?;
    let mut key_hasher = Sha256::new();
    let mut files = Vec::with_capacity(data_sources.len());
    for (source, source_hash) in data_sources.iter().zip(source_hashes) {
        let filename = source
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or_else(|| format!("invalid data filename: {}", source.display()))?
            .to_string();
        key_hasher.update(filename.as_bytes());
        key_hasher.update(source_hash.sha256.as_bytes());
        files.push(StageFile {
            filename,
            size_bytes: source_hash.size_bytes,
            sha256: source_hash.sha256.clone(),
        });
    }
    let key = format!("{:x}", key_hasher.finalize());
    let final_dir = root.join(&key);
    let stage_manifest = StageManifest {
        schema_version: 1,
        files,
    };

    if final_dir.exists() {
        let manifest_path = final_dir.join("stage-manifest.json");
        let existing: serde_json::Value = serde_json::from_slice(
            &fs::read(&manifest_path)
                .map_err(|e| format!("read immutable RAM stage manifest: {e}"))?,
        )
        .map_err(|e| format!("parse immutable RAM stage manifest: {e}"))?;
        let expected = serde_json::to_value(&stage_manifest)
            .map_err(|e| format!("encode stage manifest: {e}"))?;
        if existing != expected {
            return fail("content-addressed RAM stage manifest does not match its key");
        }
        for (source, expected) in data_sources.iter().zip(source_hashes) {
            let staged = sha256_file(&final_dir.join(source.file_name().unwrap()))
                .map_err(|e| format!("verify RAM stage file: {e}"))?;
            if staged.sha256 != expected.sha256 || staged.size_bytes != expected.size_bytes {
                return fail(format!(
                    "immutable RAM stage failed hash verification: {}",
                    source.display()
                ));
            }
        }
        let manifest_hash = sha256_file(&manifest_path)
            .map_err(|e| format!("hash stage manifest: {e}"))?
            .sha256;
        return Ok((
            final_dir.clone(),
            StageReceipt {
                mode: "ram_reused".into(),
                directory: final_dir.display().to_string(),
                reused: true,
                operation_ms: started.elapsed().as_secs_f64() * 1000.0,
                manifest_sha256: Some(manifest_hash),
            },
        ));
    }

    let required = source_hashes
        .iter()
        .try_fold(0_u64, |sum, f| sum.checked_add(f.size_bytes))
        .ok_or_else(|| "RAM stage size overflow".to_string())?;
    let available = available_bytes(&root)
        .ok_or_else(|| format!("cannot determine free tmpfs space at {}", root.display()))?;
    let reserve = 64 * 1024 * 1024_u64;
    if available < required.saturating_add(reserve) {
        return fail(format!(
            "insufficient RAM stage space: need {} bytes plus {} reserve, have {available}",
            required, reserve
        ));
    }

    let temp_dir = root.join(format!(".{key}.{}.tmp", std::process::id()));
    if temp_dir.exists() {
        fs::remove_dir_all(&temp_dir).map_err(|e| format!("remove stale stage temp: {e}"))?;
    }
    fs::create_dir(&temp_dir).map_err(|e| format!("create temporary RAM stage: {e}"))?;
    for (source, expected) in data_sources.iter().zip(source_hashes) {
        let name = source.file_name().unwrap();
        let copied = copy_and_hash(source, &temp_dir.join(name))
            .map_err(|e| format!("copy/hash {}: {e}", source.display()))?;
        if copied.sha256 != expected.sha256 || copied.size_bytes != expected.size_bytes {
            let _ = fs::remove_dir_all(&temp_dir);
            return fail(format!(
                "source changed while staging: {}",
                source.display()
            ));
        }
    }
    let manifest_bytes = serde_json::to_vec_pretty(&stage_manifest)
        .map_err(|e| format!("encode stage manifest: {e}"))?;
    let manifest_tmp = temp_dir.join("stage-manifest.json.tmp");
    let mut mf = File::create(&manifest_tmp).map_err(|e| format!("create stage manifest: {e}"))?;
    mf.write_all(&manifest_bytes)
        .and_then(|_| mf.sync_all())
        .map_err(|e| format!("write stage manifest: {e}"))?;
    fs::rename(&manifest_tmp, temp_dir.join("stage-manifest.json"))
        .map_err(|e| format!("publish stage manifest: {e}"))?;
    File::open(&temp_dir)
        .and_then(|f| f.sync_all())
        .map_err(|e| format!("sync temporary RAM stage: {e}"))?;
    if let Err(error) = fs::rename(&temp_dir, &final_dir) {
        if final_dir.exists() {
            fs::remove_dir_all(&temp_dir)
                .map_err(|e| format!("remove duplicate concurrent stage: {e}"))?;
            return prepare_ram_stage(data_sources, source_hashes);
        }
        let _ = fs::remove_dir_all(&temp_dir);
        return fail(format!("publish immutable RAM stage: {error}"));
    }
    File::open(&root)
        .and_then(|f| f.sync_all())
        .map_err(|e| format!("sync RAM stage directory: {e}"))?;
    let manifest_hash = sha256_file(&final_dir.join("stage-manifest.json"))
        .map_err(|e| format!("hash stage manifest: {e}"))?
        .sha256;
    Ok((
        final_dir.clone(),
        StageReceipt {
            mode: "ram_created".into(),
            directory: final_dir.display().to_string(),
            reused: false,
            operation_ms: started.elapsed().as_secs_f64() * 1000.0,
            manifest_sha256: Some(manifest_hash),
        },
    ))
}

fn resolve_channel(target: &str) -> Option<usize> {
    let target = target.to_lowercase();
    // Preserve biophysical_kinetics.py target-map insertion order.
    const TARGETS: [(&str, usize); 29] = [
        ("rdl", 2),
        ("gaba_a", 2),
        ("gaba_b", 2),
        ("gaba receptor", 2),
        ("gaba-a", 2),
        ("chloride channel", 2),
        ("nachr", 0),
        ("machr", 0),
        ("ache", 0),
        ("acetylcholinesterase", 0),
        ("nicotinic", 0),
        ("muscarinic", 0),
        ("glucl", 1),
        ("mglur", 1),
        ("nmdar", 1),
        ("ampar", 1),
        ("vglut", 1),
        ("dop1r", 3),
        ("dop2r", 3),
        ("dat", 3),
        ("dopamine receptor", 3),
        ("5-ht1", 4),
        ("5-ht2", 4),
        ("sert", 4),
        ("5-ht", 4),
        ("serotonin receptor", 4),
        ("oamb", 5),
        ("octr", 5),
        ("octopamine receptor", 5),
    ];
    // Python also checks octbeta last.
    TARGETS
        .iter()
        .find_map(|(needle, channel)| target.contains(needle).then_some(*channel))
        .or_else(|| target.contains("octbeta").then_some(5))
}

fn gain_for(action: &str, dose_nm: f64, affinity_nm: f64) -> f64 {
    if dose_nm <= 0.0 || affinity_nm <= 0.0 || !dose_nm.is_finite() || !affinity_nm.is_finite() {
        return 1.0;
    }
    let ratio = dose_nm / affinity_nm; // Hill coefficient is 1 in the oracle.
    let mode = action.to_lowercase();
    if mode.contains("antagonist") || mode.contains("blocker") || mode.contains("channel_blocker") {
        (1.0 / (1.0 + ratio)).clamp(0.0, 1.0)
    } else if mode.contains("agonist")
        || mode.contains("inhibitor")
        || mode.contains("ache")
    {
        1.0 + (2.0 * ratio) / (1.0 + ratio)
    } else if mode.contains("positive_allosteric_modulator") || mode.contains("pam") {
        1.0 + (1.6 * ratio) / (1.0 + ratio)
    } else {
        1.0
    }
}

fn round_gain_like_python(value: f64) -> f64 {
    // Rust's `round` uses half-away-from-zero; Python's round(value, 4)
    // rounds the decimal representation to nearest/even. Formatting at the
    // target decimal precision matches Python's result for finite gains.
    format!("{value:.4}")
        .parse::<f64>()
        .expect("formatting a finite gain always produces a float")
}

/// Perturbable gain slots. `resolve_channel` returns 0..=5 and writes `gains[channel + 1]`, so
/// slot 0 is never perturbed and slot 7 is the reserved flag every real condition sets to 0.
const PERTURBABLE_SLOTS: [usize; 6] = [1, 2, 3, 4, 5, 6];
/// Neutral baseline: every slot passthrough, reserved flag cleared, as the controls use.
const NEUTRAL_GAINS: [f32; GAINS_PER_CONDITION] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0];

/// The three no-op conditions every set carries, so a run always contains its own reference.
fn push_controls(conditions: &mut Vec<Condition>, target_count: usize) {
    for (i, name) in ["Water", "Saline", "D-Glucose"].iter().enumerate() {
        if conditions.len() >= target_count {
            return;
        }
        conditions.push(Condition {
            condition_id: format!("control-{i:02}"),
            compound_id: (*name).into(),
            name: format!("{name} (Control)"),
            dose_level: "Control (0 nM)".into(),
            dose_nm: 0.0,
            gains: NEUTRAL_GAINS,
        });
    }
}

/// Silence one gain slot, then each pair of slots. Six singles plus fifteen pairs plus three
/// controls is 24 conditions. Slots are named by index, not by receptor: the index-to-receptor
/// mapping lives in `resolve_channel` and this generator makes no biological claim about it.
fn build_lesion_conditions(target_count: usize) -> Vec<Condition> {
    let mut conditions = Vec::with_capacity(target_count);
    push_controls(&mut conditions, target_count);
    for slot in PERTURBABLE_SLOTS {
        if conditions.len() >= target_count {
            return conditions;
        }
        let mut gains = NEUTRAL_GAINS;
        gains[slot] = 0.0;
        conditions.push(Condition {
            condition_id: format!("lesion-s{slot}"),
            compound_id: format!("slot-{slot}"),
            name: format!("Silence gain slot {slot}"),
            dose_level: "Lesion (gain 0.0)".into(),
            dose_nm: 0.0,
            gains,
        });
    }
    for (index, &first) in PERTURBABLE_SLOTS.iter().enumerate() {
        for &second in &PERTURBABLE_SLOTS[index + 1..] {
            if conditions.len() >= target_count {
                return conditions;
            }
            let mut gains = NEUTRAL_GAINS;
            gains[first] = 0.0;
            gains[second] = 0.0;
            conditions.push(Condition {
                condition_id: format!("lesion-s{first}-s{second}"),
                compound_id: format!("slot-{first}+{second}"),
                name: format!("Silence gain slots {first} and {second}"),
                dose_level: "Lesion pair (gain 0.0)".into(),
                dose_nm: 0.0,
                gains,
            });
        }
    }
    conditions
}

/// Gain-response curve per slot. Pure multipliers, no affinity and no dose: 1.0 is omitted because
/// a single slot at 1.0 is bit-identical to a control and would waste a condition slot.
fn build_sweep_conditions(target_count: usize) -> Vec<Condition> {
    const LEVELS: [(f32, &str); 5] = [
        (0.0, "0.00x"),
        (0.25, "0.25x"),
        (0.5, "0.50x"),
        (2.0, "2.00x"),
        (4.0, "4.00x"),
    ];
    let mut conditions = Vec::with_capacity(target_count);
    push_controls(&mut conditions, target_count);
    for slot in PERTURBABLE_SLOTS {
        for (level, label) in LEVELS {
            if conditions.len() >= target_count {
                return conditions;
            }
            let mut gains = NEUTRAL_GAINS;
            gains[slot] = level;
            conditions.push(Condition {
                condition_id: format!("sweep-s{slot}-{label}"),
                compound_id: format!("slot-{slot}"),
                name: format!("Gain slot {slot} at {label}"),
                dose_level: format!("Sweep {label}"),
                dose_nm: 0.0,
                gains,
            });
        }
    }
    conditions
}

/// Pick the condition generator. `compound` needs `compounds/registry.parquet`, which moved to the
/// fly-pharma repository, so it fails here by design rather than silently producing nothing.
fn build_condition_set(set: &str, registry: &Path, target_count: usize) -> Result<Vec<Condition>, String> {
    match set {
        "compound" => build_conditions(registry, target_count),
        "lesion" => Ok(build_lesion_conditions(target_count)),
        "sweep" => Ok(build_sweep_conditions(target_count)),
        other => fail(format!(
            "FLY_CONDITION_SET must be compound, lesion or sweep, got {other}"
        )),
    }
}

fn build_conditions(registry: &Path, target_count: usize) -> Result<Vec<Condition>, String> {
    let connection = Connection::open_in_memory().map_err(|e| format!("open DuckDB: {e}"))?;
    let path = registry.to_string_lossy().replace('\'', "''");
    let sql = format!(
        "SELECT compound_id, name, primary_target, bioassay_target, action_mode, affinity_val_nm \
         FROM read_parquet('{path}') ORDER BY name"
    );
    let mut statement = connection
        .prepare(&sql)
        .map_err(|e| format!("prepare catalog query: {e}"))?;
    let mut rows = statement
        .query([])
        .map_err(|e| format!("query catalog: {e}"))?;
    let mut conditions = Vec::with_capacity(target_count);
    for (i, name) in ["Water", "Saline", "D-Glucose"].iter().enumerate() {
        if conditions.len() >= target_count {
            break;
        }
        conditions.push(Condition {
            condition_id: format!("control-{i:02}"),
            compound_id: (*name).into(),
            name: format!("{name} (Control)"),
            dose_level: "Control (0 nM)".into(),
            dose_nm: 0.0,
            gains: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        });
    }
    let doses = [
        (0.2, "Low (0.2x Ki)"),
        (1.0, "Standard (1.0x Ki)"),
        (5.0, "High (5.0x Ki)"),
    ];
    while let Some(row) = rows.next().map_err(|e| format!("read catalog row: {e}"))? {
        let compound_id: Option<String> = row.get(0).map_err(|e| format!("compound_id: {e}"))?;
        let name: Option<String> = row.get(1).map_err(|e| format!("name: {e}"))?;
        let primary: Option<String> = row.get(2).map_err(|e| format!("primary_target: {e}"))?;
        let bioassay: Option<String> = row.get(3).map_err(|e| format!("bioassay_target: {e}"))?;
        let action: Option<String> = row.get(4).map_err(|e| format!("action_mode: {e}"))?;
        let affinity: Option<f64> = row.get(5).map_err(|e| format!("affinity_val_nm: {e}"))?;
        let Some(affinity) = affinity.filter(|value| value.is_finite() && *value > 0.0) else {
            continue;
        };
        let Some(name) = name else { continue };
        let compound_id = compound_id.unwrap_or_else(|| name.clone());
        let target = primary.filter(|s| !s.is_empty()).or(bioassay);
        let action = action.unwrap_or_else(|| "antagonist".into());
        // Treat FLY_CONDITIONS as a cap. Never publish a partial dose ladder.
        if conditions.len() + doses.len() > target_count {
            break;
        }
        for (multiplier, label) in doses {
            let dose_nm = affinity * multiplier;
            let mut gains = [1.0_f32; GAINS_PER_CONDITION];
            gains[7] = 0.0;
            if let Some(channel) = target.as_deref().and_then(resolve_channel) {
                let oracle_gain = gain_for(&action, dose_nm, affinity);
                gains[channel + 1] = round_gain_like_python(oracle_gain) as f32;
            }
            conditions.push(Condition {
                condition_id: format!("{}-{label}", compound_id),
                compound_id: compound_id.clone(),
                name: format!("{name} [{label}]"),
                dose_level: label.into(),
                dose_nm,
                gains,
            });
        }
        if conditions.len() >= MAX_CONDITIONS {
            break;
        }
    }
    Ok(conditions)
}

fn write_binary_manifest(path: &Path, conditions: &[Condition], steps: u32) -> io::Result<()> {
    let mut file = File::create(path)?;
    file.write_all(b"FLYBAT01")?;
    file.write_all(&(conditions.len() as u32).to_le_bytes())?;
    file.write_all(&steps.to_le_bytes())?;
    file.write_all(&(GAINS_PER_CONDITION as u32).to_le_bytes())?;
    for condition in conditions {
        for gain in condition.gains {
            file.write_all(&gain.to_le_bytes())?;
        }
    }
    file.sync_all()
}

fn parse_output(
    output: &str,
    conditions: &[Condition],
    steps: u32,
) -> Result<(f64, usize, usize, Vec<ResultRow>), String> {
    let mut wall_ms = None;
    let mut count = None;
    let mut descending_neurons = None;
    let mut reported_conditions = None;
    let mut reported_steps = None;
    let mut version = None;
    let mut rows = Vec::with_capacity(conditions.len());
    let mut current: Option<usize> = None;
    let mut metrics: [Option<f64>; 5] = [None; 5];
    let mut seen = vec![false; conditions.len()];
    let mut complete = false;
    for line in output.lines() {
        if line.trim() == "FLY_BATCH_COMPLETE=1" {
            complete = true;
        }
        if let Some(value) = line.strip_prefix("FLY_BATCH_RESULT_VERSION=") {
            version = value.trim().parse::<u32>().ok();
        }
        if let Some(value) = line.strip_prefix("FLY_BATCH_RESULT_CONDITIONS= ") {
            reported_conditions = value.trim().parse::<usize>().ok();
        }
        if let Some(value) = line.strip_prefix("FLY_BATCH_RESULT_STEPS= ") {
            reported_steps = value.trim().parse::<u32>().ok();
        }
        if let Some(value) = line.strip_prefix("FLY_BATCH_RESULT_WALL_MS= ") {
            wall_ms = value.trim().parse().ok();
        }
        if let Some(value) = line.strip_prefix("FLY_BATCH_RESULT_STEP_COUNT_READBACK_BYTES= ") {
            count = value.trim().parse().ok();
        }
        if let Some(value) = line.strip_prefix("FLY_BATCH_DESCENDING_UNIQUE_COUNT= ") {
            descending_neurons = value.trim().parse::<usize>().ok();
        }
        if let Some(value) = line.strip_prefix("FLY_CONDITION_BEGIN= ") {
            let i = value
                .trim()
                .parse::<usize>()
                .map_err(|_| "worker emitted an invalid condition index")?;
            if i >= conditions.len() || seen[i] || current.is_some() {
                return fail(format!(
                    "worker emitted duplicate or invalid condition index {i}"
                ));
            }
            current = Some(i);
            metrics = [None; 5];
        }
        if let Some(i) = current {
            for (key, slot) in [
                ("PHARMA_RESULT_MEAN_ACTIVE= ", 0),
                ("PHARMA_RESULT_ACTIVE_PCT= ", 1),
                ("PHARMA_RESULT_MEAN_DN= ", 2),
                ("PHARMA_RESULT_DN_PCT= ", 3),
                ("PHARMA_RESULT_SYNCHRONY= ", 4),
            ] {
                if let Some(value) = line.strip_prefix(key) {
                    if metrics[slot].is_some() {
                        return fail(format!("worker condition {i} duplicated metric {key}"));
                    }
                    metrics[slot] = Some(value.trim().parse().unwrap_or(f64::NAN));
                }
            }
            if line.starts_with("FLY_CONDITION_END= ") {
                if !metrics
                    .iter()
                    .all(|value| value.is_some_and(f64::is_finite))
                {
                    return fail(format!("worker condition {i} omitted finite metrics"));
                }
                let values = metrics.map(|value| value.expect("validated metric"));
                let c = conditions
                    .get(i)
                    .ok_or_else(|| format!("worker returned unknown condition index {i}"))?;
                rows.push(ResultRow {
                    condition_id: c.condition_id.clone(),
                    compound_id: c.compound_id.clone(),
                    name: c.name.clone(),
                    dose_level: c.dose_level.clone(),
                    dose_nm: c.dose_nm,
                    steps,
                    mean_active: values[0],
                    active_pct: values[1],
                    mean_descending: values[2],
                    descending_pct: values[3],
                    synchrony: values[4],
                });
                seen[i] = true;
                current = None;
            }
        }
    }
    if !complete {
        return fail("worker omitted FLY_BATCH_COMPLETE marker");
    }
    if version != Some(1)
        || reported_conditions != Some(conditions.len())
        || reported_steps != Some(steps)
    {
        return fail("worker version, condition count, or step count did not match the manifest");
    }
    if rows.len() != conditions.len() || seen.iter().any(|value| !value) {
        return fail(format!(
            "worker returned {} of {} condition rows",
            rows.len(),
            conditions.len()
        ));
    }
    if rows.iter().any(|r| {
        !r.mean_active.is_finite() || !r.mean_descending.is_finite() || !r.synchrony.is_finite()
    }) {
        return fail("worker emitted missing or non-finite metrics");
    }
    Ok((
        wall_ms.ok_or("worker omitted batch wall time")?,
        count.ok_or("worker omitted readback size")?,
        descending_neurons
            .filter(|count| *count > 0)
            .ok_or("worker omitted or reported zero unique descending neurons")?,
        rows,
    ))
}

fn validate_dry_output(output: &str, conditions: usize, steps: u32) -> Result<(), String> {
    let expected = [
        ("FLY_BATCH_DRY_PASS=", "1".to_string()),
        ("FLY_BATCH_DRY_CONDITIONS=", conditions.to_string()),
        ("FLY_BATCH_DRY_STEPS=", steps.to_string()),
        ("FLY_BATCH_DRY_NEURONS=", EXPECTED_NEURONS.to_string()),
        ("FLY_BATCH_DRY_EDGES=", EXPECTED_EDGES.to_string()),
    ];
    for (key, expected_value) in expected {
        let value = output
            .lines()
            .find_map(|line| line.trim().strip_prefix(key))
            .ok_or_else(|| format!("CPU preflight omitted {key}"))?;
        if value.trim() != expected_value {
            return fail(format!(
                "CPU preflight {key} expected {expected_value}, got {}",
                value.trim()
            ));
        }
    }
    output
        .lines()
        .find_map(|line| {
            line.trim()
                .strip_prefix("FLY_BATCH_DRY_DESCENDING_UNIQUE_COUNT=")
        })
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|count| *count > 0)
        .ok_or("CPU preflight omitted a positive unique descending-neuron count")?;
    Ok(())
}

fn sql_quote(value: &str) -> String {
    value.replace('\'', "''")
}

fn peak_rss_bytes() -> Option<u64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    let kib = status.lines().find_map(|line| {
        line.strip_prefix("VmHWM:")?
            .split_whitespace()
            .next()?
            .parse::<u64>()
            .ok()
    })?;
    Some(kib * 1024)
}

fn write_parquet(path: &Path, rows: &[ResultRow]) -> Result<(), String> {
    let connection =
        Connection::open_in_memory().map_err(|e| format!("open result DuckDB: {e}"))?;
    connection.execute_batch("CREATE TABLE results(condition_id VARCHAR, compound_id VARCHAR, name VARCHAR, dose_level VARCHAR, dose_nm DOUBLE, steps UINTEGER, mean_active DOUBLE, active_pct DOUBLE, mean_descending DOUBLE, descending_pct DOUBLE, synchrony DOUBLE);").map_err(|e| format!("create results table: {e}"))?;
    let mut insert = connection
        .prepare("INSERT INTO results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)")
        .map_err(|e| format!("prepare results insert: {e}"))?;
    for r in rows {
        insert
            .execute(params![
                r.condition_id,
                r.compound_id,
                r.name,
                r.dose_level,
                r.dose_nm,
                r.steps,
                r.mean_active,
                r.active_pct,
                r.mean_descending,
                r.descending_pct,
                r.synchrony
            ])
            .map_err(|e| format!("insert result: {e}"))?;
    }
    let dest = sql_quote(&path.to_string_lossy());
    connection
        .execute_batch(&format!(
            "COPY results TO '{dest}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        ))
        .map_err(|e| format!("write Parquet: {e}"))
}

fn run() -> Result<(), String> {
    let host_started = Instant::now();
    let project = PathBuf::from(
        env::var("FLY_CONNECTOME_DIR")
            .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/..").into()),
    );
    let engine_dir = PathBuf::from(
        env::var("FLY_SPMV_DIR").unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../../mojo/fly-spmv").into()),
    );
    let engine = PathBuf::from(
        env::var("FLY_GPU_ENGINE")
            .unwrap_or_else(|_| engine_dir.join(".work/fly_batch_gpu").display().to_string()),
    );
    let registry = project.join("compounds/registry.parquet");
    let data_dir = PathBuf::from(
        env::var("FLY_DATA_DIR").unwrap_or_else(|_| project.join("data").display().to_string()),
    );
    let steps: u32 = env::var("FLY_STEPS")
        .ok()
        .map(|v| v.parse())
        .transpose()
        .map_err(|e| format!("invalid FLY_STEPS: {e}"))?
        .unwrap_or(DEFAULT_STEPS);
    if !(1..=128).contains(&steps) {
        return fail("FLY_STEPS must be in [1,128]");
    }
    let target_count: usize = env::var("FLY_CONDITIONS")
        .ok()
        .map(|v| v.parse())
        .transpose()
        .map_err(|e| format!("invalid FLY_CONDITIONS: {e}"))?
        .unwrap_or(MAX_CONDITIONS);
    if !(1..=MAX_CONDITIONS).contains(&target_count) {
        return fail("FLY_CONDITIONS must be in [1,50]");
    }
    let since_epoch = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| e.to_string())?;
    let now = since_epoch.as_secs();
    let run_id = format!("{}-{}", std::process::id(), since_epoch.as_nanos());
    let out_dir = project.join(".work/runs").join(&run_id);
    fs::create_dir_all(&out_dir).map_err(|e| format!("create run directory: {e}"))?;
    let manifest_path = out_dir.join("manifest.bin");
    let condition_set = env::var("FLY_CONDITION_SET").unwrap_or_else(|_| "compound".into());
    let conditions = build_condition_set(&condition_set, &registry, target_count)?;
    if conditions.is_empty() {
        return fail(format!("condition set {condition_set} produced no conditions"));
    }
    let manifest_tmp = out_dir.join("manifest.bin.tmp");
    write_binary_manifest(&manifest_tmp, &conditions, steps)
        .map_err(|e| format!("write batch manifest: {e}"))?;
    fs::rename(&manifest_tmp, &manifest_path)
        .map_err(|e| format!("publish batch manifest: {e}"))?;

    let data_sources = [
        data_dir.join("row_ptr.bin"),
        data_dir.join("col_idx.bin"),
        data_dir.join("weights.bin"),
        data_dir.join("neuron_nt_types.bin"),
        data_dir.join("descending_indices.bin"),
    ];
    let mut input_hashes =
        vec![sha256_file(&registry).map_err(|e| format!("hash {}: {e}", registry.display()))?];
    let mut data_hashes = Vec::with_capacity(data_sources.len());
    for path in &data_sources {
        data_hashes.push(sha256_file(path).map_err(|e| format!("hash {}: {e}", path.display()))?);
    }
    input_hashes.extend(data_hashes.iter().cloned());
    let ram_stage_enabled = match env::var("FLY_RAM_STAGE").as_deref() {
        Ok("1") | Ok("true") => true,
        Ok("0") | Ok("false") | Err(_) => false,
        _ => return fail("FLY_RAM_STAGE must be 0 or 1"),
    };
    let (worker_data_dir, data_stage) = if ram_stage_enabled {
        prepare_ram_stage(&data_sources, &data_hashes)?
    } else {
        (
            data_dir.clone(),
            StageReceipt {
                mode: "direct".into(),
                directory: data_dir.display().to_string(),
                reused: false,
                operation_ms: 0.0,
                manifest_sha256: None,
            },
        )
    };
    let engine_hash = sha256_file(&engine)
        .map_err(|e| format!("hash engine {}: {e}", engine.display()))?
        .sha256;
    let driver_path = env::current_exe().map_err(|e| format!("locate fly-screen binary: {e}"))?;
    let driver_hash = sha256_file(&driver_path)
        .map_err(|e| format!("hash driver {}: {e}", driver_path.display()))?
        .sha256;
    let manifest_hash = sha256_file(&manifest_path)
        .map_err(|e| format!("hash manifest: {e}"))?
        .sha256;

    if env::var("FLY_DRY_ONLY").as_deref() == Ok("1") {
        let dry = Command::new(&engine)
            .arg(&manifest_path)
            .env("GPU_DRY", "1")
            .env("FLY_DATA_DIR", &worker_data_dir)
            .output()
            .map_err(|e| format!("launch CPU preflight: {e}"))?;
        if !dry.status.success() {
            return fail(format!(
                "CPU preflight failed with {}: {}",
                dry.status,
                String::from_utf8_lossy(&dry.stderr)
            ));
        }
        let dry_stdout = String::from_utf8_lossy(&dry.stdout);
        validate_dry_output(&dry_stdout, conditions.len(), steps)?;
        println!("FLY_SCREEN_PREFLIGHT_PASS=1");
        println!("FLY_SCREEN_PREFLIGHT_CONDITIONS={}", conditions.len());
        println!("FLY_SCREEN_PREFLIGHT_CONDITIONS_REQUESTED={target_count}");
        println!("FLY_SCREEN_PREFLIGHT_STEPS={steps}");
        println!("FLY_SCREEN_PREFLIGHT_NEURONS={EXPECTED_NEURONS}");
        println!("FLY_SCREEN_PREFLIGHT_EDGES={EXPECTED_EDGES}");
        return Ok(());
    }

    let gpu_job = env::var("FLY_GPUJOB")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../../gpu-media/Blender/kit/gpujob").into());
    let worker_output_path = out_dir.join("worker.stdout");
    let compile_only = env::var("FLY_GPU_COMPILE_ONLY").as_deref() == Ok("1");
    let worker_command = r#"env FLY_DATA_DIR="$4" FLY_COMPILE_ONLY="$5" "$1" "$2" >"$3" 2>&1; rc=$?; cat "$3"; exit "$rc""#;
    let child = Command::new(gpu_job)
        .arg(format!("fly-screen-{run_id}"))
        .args([
            "--vram",
            if compile_only { "1" } else { "22" },
            "--timeout",
            "180",
            "--",
        ])
        .arg("sh")
        .arg("-c")
        .arg(worker_command)
        .arg("fly-screen-worker")
        .arg(&engine)
        .arg(&manifest_path)
        .arg(&worker_output_path)
        .arg(&worker_data_dir)
        .arg(if compile_only { "1" } else { "0" })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("launch GPU worker through kit/gpujob: {e}"))?;
    let gpujob_started = Instant::now();
    let output = child
        .wait_with_output()
        .map_err(|e| format!("wait for GPU worker: {e}"))?;
    let gpujob_wall_ms = gpujob_started.elapsed().as_secs_f64() * 1000.0;
    if !output.status.success() {
        return fail(format!(
            "gpujob worker exited {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let stdout = String::from_utf8(output.stdout)
        .map_err(|e| format!("gpujob output was not UTF-8: {e}"))?;
    let worker_stdout = fs::read_to_string(&worker_output_path)
        .map_err(|e| format!("read worker output {}: {e}", worker_output_path.display()))?;
    if compile_only {
        if !worker_stdout
            .lines()
            .any(|line| line.trim() == "FLY_BATCH_KERNEL_COMPILE_PASS=1")
        {
            return fail("GPU compile check omitted FLY_BATCH_KERNEL_COMPILE_PASS=1");
        }
        println!("FLY_SCREEN_GPU_KERNEL_COMPILE_PASS=1");
        println!("FLY_SCREEN_GPU_JOB_MS={gpujob_wall_ms:.3}");
        return Ok(());
    }
    for expected in &input_hashes {
        let observed = sha256_file(Path::new(&expected.path))
            .map_err(|e| format!("re-hash {}: {e}", expected.path))?;
        if observed.sha256 != expected.sha256 || observed.size_bytes != expected.size_bytes {
            return fail(format!("input changed during run: {}", expected.path));
        }
    }
    if ram_stage_enabled {
        for (source, expected) in data_sources.iter().zip(&data_hashes) {
            let staged = sha256_file(&worker_data_dir.join(source.file_name().unwrap()))
                .map_err(|e| format!("re-hash staged input: {e}"))?;
            if staged.sha256 != expected.sha256 || staged.size_bytes != expected.size_bytes {
                return fail(format!(
                    "RAM stage changed during run: {}",
                    source.display()
                ));
            }
        }
    }
    let (wall_ms, readback_bytes, descending_neurons, rows) =
        parse_output(&worker_stdout, &conditions, steps)
            .map_err(|e| format!("parse worker output: {e}; gpujob summary: {stdout}"))?;
    let results_path = out_dir.join("results.parquet");
    let results_tmp = out_dir.join("results.parquet.tmp");
    write_parquet(&results_tmp, &rows)?;
    fs::rename(&results_tmp, &results_path).map_err(|e| format!("publish Parquet results: {e}"))?;
    let results_hash = sha256_file(&results_path)
        .map_err(|e| format!("hash results: {e}"))?
        .sha256;
    let receipt = Receipt {
        schema_version: 1,
        run_id,
        created_unix_seconds: now,
        conditions: conditions.len(),
        conditions_requested: target_count,
        dose_ladder_policy: "complete_triplets_only".into(),
        descending_neurons,
        steps,
        condition_manifest: conditions.clone(),
        engine: engine.display().to_string(),
        engine_sha256: engine_hash,
        driver_sha256: driver_hash,
        engine_reported_wall_ms: wall_ms,
        gpujob_wall_ms,
        host_wall_ms: host_started.elapsed().as_secs_f64() * 1000.0,
        peak_rss_bytes: peak_rss_bytes(),
        compact_readback_bytes: readback_bytes,
        data_stage,
        input_hashes,
        manifest_sha256: manifest_hash,
        results_sha256: results_hash,
        worker_stdout_sha256: sha256_file(&worker_output_path)
            .map_err(|e| format!("hash worker output: {e}"))?
            .sha256,
        results: rows,
    };
    let receipt_path = out_dir.join("receipt.json");
    let receipt_bytes =
        serde_json::to_vec_pretty(&receipt).map_err(|e| format!("serialize receipt: {e}"))?;
    let receipt_tmp = out_dir.join("receipt.json.tmp");
    fs::write(&receipt_tmp, receipt_bytes).map_err(|e| format!("write receipt: {e}"))?;
    fs::rename(&receipt_tmp, &receipt_path).map_err(|e| format!("publish receipt: {e}"))?;
    println!("FLY_SCREEN_RUN={}", receipt_path.display());
    println!("FLY_SCREEN_RESULTS={}", results_path.display());
    println!("FLY_SCREEN_CONDITIONS={}", conditions.len());
    println!("FLY_SCREEN_CONDITIONS_REQUESTED={target_count}");
    println!("FLY_SCREEN_STEPS={steps}");
    println!("FLY_SCREEN_GPU_BATCH_MS={wall_ms:.3}");
    println!("FLY_SCREEN_READBACK_BYTES={readback_bytes}");
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("fly-screen: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lesion_set_silences_every_slot_singly_and_in_pairs() {
        let conditions = build_lesion_conditions(50);
        assert_eq!(conditions.len(), 3 + 6 + 15, "3 controls, 6 singles, 15 pairs");
        let controls = &conditions[..3];
        assert!(controls.iter().all(|c| c.gains == NEUTRAL_GAINS));
        for condition in &conditions[3..] {
            // slot 0 is never perturbed and slot 7 is the reserved flag
            assert_eq!(condition.gains[0], 1.0, "{}", condition.condition_id);
            assert_eq!(condition.gains[7], 0.0, "{}", condition.condition_id);
            let silenced = PERTURBABLE_SLOTS
                .iter()
                .filter(|&&slot| condition.gains[slot] == 0.0)
                .count();
            assert!(matches!(silenced, 1 | 2), "{} silenced {}", condition.condition_id, silenced);
            assert_eq!(condition.dose_nm, 0.0, "a lesion carries no dose");
        }
        let ids: std::collections::HashSet<_> = conditions.iter().map(|c| &c.condition_id).collect();
        assert_eq!(ids.len(), conditions.len(), "condition ids must be unique");
    }

    #[test]
    fn lesion_set_respects_the_condition_cap() {
        for cap in [1, 4, 10, 24] {
            assert_eq!(build_lesion_conditions(cap).len(), cap, "cap {cap}");
        }
    }

    #[test]
    fn sweep_set_perturbs_one_slot_per_condition_and_omits_unity() {
        let conditions = build_sweep_conditions(50);
        assert_eq!(conditions.len(), 3 + 6 * 5, "3 controls, 6 slots, 5 levels");
        for condition in &conditions[3..] {
            let changed: Vec<_> = PERTURBABLE_SLOTS
                .iter()
                .filter(|&&slot| condition.gains[slot] != 1.0)
                .collect();
            assert_eq!(changed.len(), 1, "{} changed {:?}", condition.condition_id, changed);
            let level = condition.gains[*changed[0]];
            assert_ne!(level, 1.0, "unity would duplicate a control");
            assert!(level >= 0.0 && level <= 4.0, "{level} out of swept range");
            assert_eq!(condition.gains[7], 0.0);
        }
    }

    #[test]
    fn unknown_condition_set_is_rejected_by_name() {
        let error = build_condition_set("doses", Path::new("/nonexistent"), 10)
            .expect_err("an unknown set must not silently fall back");
        assert!(error.contains("doses"), "{error}");
        assert!(error.contains("lesion"), "the error must name the valid sets: {error}");
    }

    #[test]
    fn target_resolution_matches_oracle_channel_order() {
        assert_eq!(resolve_channel("Drosophila nAChR alpha"), Some(0));
        assert_eq!(resolve_channel("GABA-A chloride channel"), Some(2));
        assert_eq!(resolve_channel("octbeta receptor"), Some(5));
        assert_eq!(resolve_channel("unmapped receptor"), None);
    }

    #[test]
    fn hill_modes_match_oracle_values() {
        assert!((gain_for("antagonist", 20.0, 100.0) - (1.0 / 1.2)).abs() < 1e-12);
        assert_eq!(gain_for("agonist", 100.0, 100.0), 2.0);
        assert_eq!(gain_for("enzyme inhibitor", 100.0, 100.0), 2.0);
        assert_eq!(gain_for("PAM", 100.0, 100.0), 1.8);
        assert_eq!(gain_for("unknown", 100.0, 100.0), 1.0);
        assert_eq!(gain_for("agonist", 0.0, 100.0), 1.0);
    }

    #[test]
    fn gain_rounding_matches_python_decimal_rounding() {
        for (input, expected) in [
            (1.23445, 1.2345),
            (1.23455, 1.2346),
            (1.23465, 1.2347),
            (0.00005, 0.0001),
            (0.00015, 0.0001),
        ] {
            assert_eq!(round_gain_like_python(input), expected);
        }
    }

    #[test]
    fn binary_manifest_has_versioned_little_endian_header() {
        let path = env::temp_dir().join(format!("flybat-test-{}.bin", std::process::id()));
        let condition = Condition {
            condition_id: "test".into(),
            compound_id: "test".into(),
            name: "test".into(),
            dose_level: "control".into(),
            dose_nm: 0.0,
            gains: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        };
        write_binary_manifest(&path, &[condition], 7).unwrap();
        let bytes = fs::read(&path).unwrap();
        assert_eq!(&bytes[..8], b"FLYBAT01");
        assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), 1);
        assert_eq!(u32::from_le_bytes(bytes[12..16].try_into().unwrap()), 7);
        assert_eq!(u32::from_le_bytes(bytes[16..20].try_into().unwrap()), 8);
        assert_eq!(bytes.len(), 20 + 8 * 4);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn parser_rejects_missing_metrics_and_rows_and_accepts_complete_batch() {
        let condition = Condition {
            condition_id: "c0".into(),
            compound_id: "cmp".into(),
            name: "sample".into(),
            dose_level: "control".into(),
            dose_nm: 0.0,
            gains: [1.0; 8],
        };
        let output = "FLY_BATCH_RESULT_VERSION=1\nFLY_BATCH_RESULT_CONDITIONS= 1\nFLY_BATCH_RESULT_STEPS= 1\nFLY_BATCH_RESULT_WALL_MS= 2.5\nFLY_BATCH_RESULT_STEP_COUNT_READBACK_BYTES= 8\nFLY_BATCH_DESCENDING_UNIQUE_COUNT= 1303\nFLY_CONDITION_BEGIN= 0\nPHARMA_RESULT_MEAN_ACTIVE= 2\nPHARMA_RESULT_ACTIVE_PCT= 0.1\nPHARMA_RESULT_MEAN_DN= 1\nPHARMA_RESULT_DN_PCT= 0.2\nPHARMA_RESULT_SYNCHRONY= 0.3\nFLY_CONDITION_END= 0\nFLY_BATCH_COMPLETE=1\n";
        let parsed = parse_output(output, &[condition.clone()], 1).unwrap();
        assert_eq!(parsed.0, 2.5);
        assert_eq!(parsed.1, 8);
        assert_eq!(parsed.3[0].mean_active, 2.0);
        for metric in [
            "PHARMA_RESULT_MEAN_ACTIVE= 2\n",
            "PHARMA_RESULT_ACTIVE_PCT= 0.1\n",
            "PHARMA_RESULT_MEAN_DN= 1\n",
            "PHARMA_RESULT_DN_PCT= 0.2\n",
            "PHARMA_RESULT_SYNCHRONY= 0.3\n",
        ] {
            assert!(
                parse_output(&output.replace(metric, ""), &[condition.clone()], 1).is_err(),
                "accepted missing metric {metric}"
            );
        }
        assert!(
            parse_output(
                &output.replace(
                    "PHARMA_RESULT_SYNCHRONY= 0.3",
                    "PHARMA_RESULT_SYNCHRONY= NaN"
                ),
                &[condition.clone()],
                1
            )
            .is_err()
        );
        assert!(parse_output("", &[condition], 1).is_err());
    }

    #[test]
    fn cpu_preflight_requires_reported_dimensions() {
        let valid = "FLY_BATCH_DRY_PASS=1\nFLY_BATCH_DRY_CONDITIONS= 48\nFLY_BATCH_DRY_STEPS= 100\nFLY_BATCH_DRY_NEURONS= 139255\nFLY_BATCH_DRY_EDGES= 10572269\nFLY_BATCH_DRY_DESCENDING_UNIQUE_COUNT= 1303\n";
        assert!(validate_dry_output(valid, 48, 100).is_ok());
        assert!(validate_dry_output(&valid.replace("139255", "1"), 48, 100).is_err());
        assert!(validate_dry_output(&valid.replace("10572269", "1"), 48, 100).is_err());
        assert!(
            validate_dry_output(&valid.replace("CONDITIONS= 48", "CONDITIONS= 50"), 48, 100)
                .is_err()
        );
        assert!(
            validate_dry_output(&valid.replace("FLY_BATCH_DRY_PASS=1\n", ""), 48, 100).is_err()
        );
    }
}
