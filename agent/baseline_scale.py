"""Scale the matched graph baseline across 64 layouts and five graph seeds.

Each arm keeps the frozen odor-memory training protocol. Evaluation is fanned out
over independent scene/control trials, with a fixed 16-layout sanity subset for the
straight and odor-absent controls.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np

from agent import neural_connectome
from agent import odor_memory_gate


ROOT = neural_connectome.ROOT
BASE = ROOT / ".work/agent/matched-baseline"
ARMS = (
    "real",
    "shuffled-s20261002", "shuffled-s7", "shuffled-s1009",
    "shuffled-s424242", "shuffled-s31337",
    "random-s20261002", "random-s7", "random-s1009",
    "random-s424242", "random-s31337",
)
ALL_CONTROLS = ("trained", "untrained_straight", "shuffled_reward", "odor_absent")
FULL_CONTROLS = ("trained", "shuffled_reward")
SUBSET_CONTROLS = ("untrained_straight", "odor_absent")


class _UnusedRenderer:
    """The gate passes eyes=None. Fail if a future path starts requesting frames."""

    def update_scene(self, *args, **kwargs):
        raise RuntimeError("odor-memory gate unexpectedly requested an eye frame")

    def render(self, *args, **kwargs):
        raise RuntimeError("odor-memory gate unexpectedly requested an eye frame")

    def close(self):
        pass


def graph_directory(arm: str) -> Path:
    return BASE / arm


def check_control_graph_provenance(graphs_path: Path, source_col_idx: Path) -> None:
    """Fail before arm execution if control graph provenance is stale."""
    graphs = json.loads(Path(graphs_path).read_text())
    expected = graphs.get("sources_sha256", {}).get("data/col_idx.bin")
    actual = neural_connectome.digest(source_col_idx)
    if expected != actual:
        raise RuntimeError(
            f"FAIL provenance: graphs.json col_idx.bin SHA-256 {expected!r} "
            f"does not match {source_col_idx} ({actual})"
        )


def output_directory(arm: str) -> Path:
    # Keep the first, invalid attempt as an immutable audit trail.
    return BASE / f"scale-corrected-{arm}"


def _patch_runtime(arm: str):
    original = neural_connectome.graph_arrays
    arm_directory = graph_directory(arm)

    def arm_graph_arrays(*_args, **_kwargs):
        return original(directory=arm_directory)

    return (
        patch("agent.neural_connectome.graph_arrays", side_effect=arm_graph_arrays),
        patch("mujoco.Renderer", side_effect=lambda *args, **kwargs: _UnusedRenderer()),
    )


def _assert_loaded_graph(arm: str) -> str:
    """Verify the graph_arrays call visible to CPUBrain resolves to this arm."""
    _row, col, _weights = neural_connectome.graph_arrays()
    loaded_hash = hashlib.sha256(np.ascontiguousarray(col, dtype="<i4").tobytes()).hexdigest()
    expected_hash = neural_connectome.digest(graph_directory(arm) / "col_idx.bin")
    assert loaded_hash == expected_hash, (
        f"{arm}: runtime col_idx SHA-256 {loaded_hash} != arm directory {expected_hash}"
    )
    return loaded_hash


def _assert_loaded_graph_for_arm(arm: str) -> str:
    graph_patch, _renderer_patch = _patch_runtime(arm)
    with graph_patch:
        return _assert_loaded_graph(arm)


def _assert_frozen_sources(protocol: dict, protocol_path: Path) -> None:
    for source, expected in protocol["source_sha256"].items():
        assert neural_connectome.digest(Path(source)) == expected, f"source differs from freeze: {source}"
    assert neural_connectome.digest(protocol_path)


def run_stage(stage: str, arm: str) -> int:
    output = output_directory(arm)
    protocol_path = output / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    _assert_frozen_sources(protocol, protocol_path)
    graph_patch, renderer_patch = _patch_runtime(arm)
    with graph_patch, renderer_patch:
        col_idx_sha256 = _assert_loaded_graph(arm)
        if stage == "train":
            report = odor_memory_gate.train(output, protocol)
            report["graph_col_idx_sha256"] = col_idx_sha256
            report["graph_directory"] = str(graph_directory(arm).relative_to(ROOT))
            (output / "training.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"stage": stage, "arm": arm, "status": report["status"],
                              "training_pid": report.get("training_pid"),
                              "graph_col_idx_sha256": col_idx_sha256,
                              "trained_model_sha256": report["models"]["trained"]["sha256"]}), flush=True)
            return 0 if report["status"] == "PASS" else 1
    return 2


def _run_trial(task: dict, protocol: dict, models: dict, brain) -> dict:
    from agent.neural_single import EthologyArena, Environment, Organism

    scene = task["scene"]
    control = task["control"]
    arena = EthologyArena(seed=scene["seed"], food_patches=[scene["food"]])
    try:
        initial = arena.get_state()
        arena.set_state(initial)
        environment = Environment([scene["food"]], seed=scene["seed"])
        organism = Organism({})
        features, inputs = odor_memory_gate.sensory(
            arena, environment, organism, brain, odor=control != "odor_absent"
        )
        model = models["trained"] if control == "odor_absent" else models.get(control)
        gait = model.predict(features[None])[0, :2] if model else odor_memory_gate.GAITS["straight"]
        if model and np.max(np.abs(features)) > 1e-8:
            gait = odor_memory_gate.GAITS["left"] if gait[1] > gait[0] else odor_memory_gate.GAITS["right"]
        else:
            gait = odor_memory_gate.GAITS["straight"]
        result = odor_memory_gate.execute(arena, environment, organism, gait)
        return {"scene": scene, "control": control, "inputs": inputs, "gait": list(gait), **result}
    finally:
        arena.close()


def run_trial_worker(jobs_path: Path, worker_id: int) -> int:
    import os

    groups = json.loads(jobs_path.read_text())
    worker_receipts = []
    for group in groups:
        arm = group["arm"]
        jobs = group["tasks"]
        output = output_directory(arm)
        protocol = json.loads((output / "protocol.json").read_text())
        training = json.loads((output / "training.json").read_text())
        assert training["status"] == "PASS"
        _assert_frozen_sources(protocol, output / "protocol.json")
        graph_patch, renderer_patch = _patch_runtime(arm)
        with graph_patch, renderer_patch:
            col_idx_sha256 = _assert_loaded_graph(arm)
            assert training.get("graph_col_idx_sha256") == col_idx_sha256
            models = {name: neural_connectome.Readout(output / f"{name}.npz")
                      for name in ("trained", "shuffled_reward")}
            brain = neural_connectome.CPUBrain()
            for task in jobs:
                row = _run_trial(task, protocol, models, brain)
                path = output / f"eval-{row['scene']['id']}-{row['control']}.json"
                path.write_text(json.dumps(row) + "\n")
        del brain, models, protocol, training, graph_patch, renderer_patch
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform.startswith("linux"):
        peak_rss *= 1024
    for group in groups:
        arm = group["arm"]
        output = output_directory(arm)
        receipt = {"arm": arm, "worker_id": worker_id, "pid": os.getpid(),
                   "trials": len(group["tasks"]), "peak_rss_bytes": int(peak_rss),
                   "protocol_sha256": neural_connectome.digest(output / "protocol.json")}
        (output / f"trial-batch-{worker_id:016d}.json").write_text(json.dumps(receipt, indent=2) + "\n")
        worker_receipts.append(receipt)
    print(json.dumps({"worker_id": worker_id, "pid": os.getpid(),
                      "peak_rss_bytes": int(peak_rss), "groups": worker_receipts}), flush=True)
    return 0


def _trial_tasks(protocol: dict) -> tuple[list[dict], list[dict]]:
    scenes = protocol["evaluation"]
    subset = scenes[:16]
    full = [{"scene": scene, "control": control}
            for control in FULL_CONTROLS for scene in scenes]
    sanity = [{"scene": scene, "control": control}
              for control in SUBSET_CONTROLS for scene in subset]
    return full + sanity, subset


def _partition(items: list[dict], count: int) -> list[list[dict]]:
    quotient, remainder = divmod(len(items), count)
    partitions = []
    start = 0
    for index in range(count):
        size = quotient + (1 if index < remainder else 0)
        partitions.append(items[start:start + size])
        start += size
    return partitions


def _aggregate_arm(arm: str, subset: list[dict]) -> dict:
    output = output_directory(arm)
    protocol_path = output / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    training = json.loads((output / "training.json").read_text())
    criteria = protocol["criteria"]
    models = {name: neural_connectome.Readout(output / f"{name}.npz")
              for name in ("trained", "shuffled_reward")}
    fixture = np.load(output / "restart-fixture.npz")["features"]
    restart_errors = {}
    for name, model in models.items():
        assert neural_connectome.digest(output / f"{name}.npz") == training["models"][name]["sha256"]
        restart_errors[name] = float(np.max(np.abs(
            model.predict(fixture) - training["models"][name]["prediction"]
        )))

    summaries = []
    by_control: dict[str, list[dict]] = {name: [] for name in ALL_CONTROLS}
    evaluation_pids = set()
    for control in ALL_CONTROLS:
        expected_scenes = protocol["evaluation"] if control in FULL_CONTROLS else subset
        for scene in expected_scenes:
            path = output / f"eval-{scene['id']}-{control}.json"
            row = json.loads(path.read_text())
            assert row["scene"] == scene and row["control"] == control
            by_control[control].append(row)
            summaries.append({key: row[key] for key in ("control", "ingested", "minimum_upright")}
                             | {"scene_id": row["scene"]["id"]})
    receipt_paths = list(output.glob("trial-worker-*.json")) + list(output.glob("trial-batch-*.json"))
    worker_receipts = [json.loads(p.read_text()) for p in sorted(receipt_paths)]
    recorded_worker_trials = sum(receipt["trials"] for receipt in worker_receipts)
    total_trial_rows = sum(len(rows) for rows in by_control.values())
    assert recorded_worker_trials <= total_trial_rows
    evaluation_pids.update(row["pid"] for row in worker_receipts)

    success = lambda row: row["ingested"] >= criteria["success_ingested_food_u"]
    metrics = {}
    for control in ALL_CONTROLS:
        rows = by_control[control]
        metrics[control] = {
            "success_fraction": float(np.mean([success(row) for row in rows])),
            "mean_ingested": float(np.mean([row["ingested"] for row in rows])),
            "minimum_upright": min(row["minimum_upright"] for row in rows),
            "trials": len(rows),
        }
    train_subset_scenes = {json.dumps(scene, sort_keys=True) for scene in subset}
    trained_subset = [row for row in by_control["trained"]
                      if json.dumps(row["scene"], sort_keys=True) in train_subset_scenes]
    trained_subset_fraction = float(np.mean([success(row) for row in trained_subset]))
    trained = metrics["trained"]["success_fraction"]
    checks = {
        "trained_success": trained >= criteria["trained_success_fraction"],
        "finite_upright_body": all(value["minimum_upright"] >= criteria["minimum_upright"]
                                    for value in metrics.values()),
        "restart_predictions": max(restart_errors.values()) <= criteria["restart_max_abs_prediction_error"],
        "evaluation_weights_unchanged": all(
            neural_connectome.digest(output / f"{name}.npz") == training["models"][name]["sha256"]
            for name in models
        ),
        "shuffled_labels_changed": training["changed_shuffle_labels"] >= training["informative_pairs"] // 2,
        "beats_untrained_straight_on_sanity_subset": (
            trained_subset_fraction - metrics["untrained_straight"]["success_fraction"]
            >= criteria["minimum_success_improvement_over_each_control"]
        ),
        "beats_shuffled_reward_on_all_layouts": (
            trained - metrics["shuffled_reward"]["success_fraction"]
            >= criteria["minimum_success_improvement_over_each_control"]
        ),
        "beats_odor_absent_on_sanity_subset": (
            trained_subset_fraction - metrics["odor_absent"]["success_fraction"]
            >= criteria["minimum_success_improvement_over_each_control"]
        ),
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "metrics": metrics,
        "trained_success_fraction_on_sanity_subset": trained_subset_fraction,
        "sanity_subset": {"size": len(subset), "scene_ids": [scene["id"] for scene in subset]},
        "full_layout_controls": list(FULL_CONTROLS),
        "subset_controls": list(SUBSET_CONTROLS),
        "restart_prediction_max_abs_error": restart_errors,
        "training_pid": training["training_pid"],
        "evaluation_pids": sorted(evaluation_pids),
        "worker_receipt_coverage": {
            "trials_with_final_worker_receipt": recorded_worker_trials,
            "total_trial_rows": total_trial_rows,
            "trials_without_final_worker_receipt": total_trial_rows - recorded_worker_trials,
        },
        "learning_enabled_during_evaluation": False,
        "protocol_sha256": neural_connectome.digest(protocol_path),
        "graph_col_idx_sha256": training["graph_col_idx_sha256"],
        "trained_model_sha256": training["models"]["trained"]["sha256"],
        "shuffled_reward_model_sha256": training["models"]["shuffled_reward"]["sha256"],
        "trials": summaries,
        "scope": "fixed connectome with learned odor steering readout; controls use the disclosed 16-layout subset",
        "workers": worker_receipts,
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _run_logged(command: list[str], log_path: Path) -> int:
    with log_path.open("w") as stream:
        return subprocess.run(command, cwd=ROOT, stdout=stream,
                              stderr=subprocess.STDOUT, check=False).returncode


def execute(evaluation_seeds: int, workers: int) -> dict:
    check_control_graph_provenance(BASE / "graphs.json", ROOT / "data/col_idx.bin")
    if evaluation_seeds != 32:
        raise ValueError("the scale brief requires --evaluation-seeds 32 (64 layouts)")
    if not 1 <= workers <= 16:
        raise ValueError("workers must be in [1,16]")
    if len(odor_memory_gate.scenes(range(100, 100 + evaluation_seeds))) != 64:
        raise AssertionError("32 evaluation seeds must produce 64 held-out layouts")
    expected_evaluation = odor_memory_gate.scenes(range(100, 100 + evaluation_seeds))
    protocols = {}
    for arm in ARMS:
        graph = graph_directory(arm)
        for name in ("row_ptr.bin", "col_idx.bin", "weights.bin", "proofread_root_ids_783.npy"):
            if not (graph / name).is_file():
                raise FileNotFoundError(graph / name)
        output = output_directory(arm)
        if output.exists():
            protocol_path = output / "protocol.json"
            if not protocol_path.is_file():
                raise FileExistsError(f"existing scale output lacks frozen protocol: {output}")
            protocol = json.loads(protocol_path.read_text())
            assert protocol["evaluation"] == expected_evaluation, f"frozen evaluation differs for {arm}"
            assert protocol["controls"] == list(ALL_CONTROLS)
            _assert_frozen_sources(protocol, protocol_path)
            if (output / "metrics.json").exists():
                raise FileExistsError(f"scale metrics receipt already exists: {output / 'metrics.json'}")
        else:
            protocol = odor_memory_gate.freeze(output, evaluation_seeds)
        protocols[arm] = protocol

    started = time.perf_counter()
    protocol_hashes = {arm: neural_connectome.digest(output_directory(arm) / "protocol.json") for arm in ARMS}
    if len(set(protocol_hashes.values())) != 1:
        raise AssertionError(f"frozen protocol content differs across arms: {protocol_hashes}")

    python = ROOT / ".venv/bin/python"
    if not python.is_file():
        python = Path(sys.executable)
    train_futures = {}
    # Training holds a full CPUBrain per process; keep this low-memory. The
    # trial stage below is the work fanned out across the requested workers.
    with ThreadPoolExecutor(max_workers=min(2, workers, len(ARMS))) as pool:
        for arm in ARMS:
            training_path = output_directory(arm) / "training.json"
            if training_path.is_file():
                saved_training = json.loads(training_path.read_text())
                if saved_training.get("status") == "PASS":
                    for name in ("trained.npz", "shuffled_reward.npz", "restart-fixture.npz"):
                        assert (output_directory(arm) / name).is_file(), f"missing frozen training output {arm}/{name}"
                    continue
            command = [str(python), "-m", "agent.baseline_scale", "--stage", "train", "--arm", arm]
            train_futures[pool.submit(_run_logged, command,
                                      output_directory(arm) / "train.log")] = arm
        for future in as_completed(train_futures):
            arm = train_futures[future]
            code = future.result()
            if code:
                raise RuntimeError(f"training failed for {arm}; inspect {output_directory(arm) / 'train.log'}")

    # A missing graph patch produces identical readouts even though every trial
    # receipt looks internally consistent. Refuse to evaluate such a run.
    training_hashes = {}
    graph_hashes = {}
    for arm in ARMS:
        training = json.loads((output_directory(arm) / "training.json").read_text())
        graph_hashes[arm] = _assert_loaded_graph_for_arm(arm)
        assert training.get("graph_col_idx_sha256") == graph_hashes[arm], (
            f"training graph hash mismatch for {arm}"
        )
        training_hashes[arm] = {
            name: training["models"][name]["sha256"]
            for name in ("trained", "shuffled_reward")
        }
    for model_name in ("trained", "shuffled_reward"):
        by_hash = {}
        for arm, hashes in training_hashes.items():
            by_hash.setdefault(hashes[model_name], []).append(arm)
        duplicates = {sha: names for sha, names in by_hash.items() if len(names) > 1}
        assert not duplicates, f"{model_name} fitted model hashes must be pairwise distinct: {duplicates}"

    tasks_by_arm = {}
    subset_by_arm = {}
    worker_assignments = []
    for arm in ARMS:
        tasks, subset = _trial_tasks(protocols[arm])
        if len(tasks) != 160:
            raise AssertionError(f"expected 160 trials per arm, got {len(tasks)} for {arm}")
        tasks_by_arm[arm], subset_by_arm[arm] = tasks, subset
    flat_jobs = []
    for arm in ARMS:
        for task in tasks_by_arm[arm]:
            existing = output_directory(arm) / f"eval-{task['scene']['id']}-{task['control']}.json"
            if existing.is_file():
                prior = json.loads(existing.read_text())
                if prior.get("scene") != task["scene"] or prior.get("control") != task["control"]:
                    raise RuntimeError(f"existing trial receipt does not match frozen task: {existing}")
                continue
            flat_jobs.append({"arm": arm, "task": task})
    batch_size = 2
    partitions = [flat_jobs[start:start + batch_size]
                  for start in range(0, len(flat_jobs), batch_size)]
    run_token = time.time_ns() % 10**12
    jobs_root = BASE / f"scale-trial-batches-{run_token:012d}"
    jobs_root.mkdir(parents=True, exist_ok=False)
    for local_id, partition in enumerate(partitions):
        worker_id = run_token * 10000 + local_id
        groups = []
        for entry in partition:
            if not groups or groups[-1]["arm"] != entry["arm"]:
                groups.append({"arm": entry["arm"], "tasks": []})
            groups[-1]["tasks"].append(entry["task"])
        jobs_path = jobs_root / f"batch-{local_id:04d}.json"
        jobs_path.write_text(json.dumps(groups))
        worker_assignments.append((worker_id, jobs_path, len(partition)))
    worker_codes = {}

    def run_batch(worker_id: int, jobs_path: Path) -> tuple[int, int, Path]:
        command = [str(python), "-m", "agent.baseline_scale", "--stage", "trial-worker",
                   "--jobs-file", str(jobs_path), "--worker-id", str(worker_id)]
        log_path = BASE / f"scale-batch-{worker_id:016d}.log"
        code = _run_logged(command, log_path)
        return worker_id, code, log_path

    with ThreadPoolExecutor(max_workers=workers) as pool:
        batch_futures = {
            pool.submit(run_batch, worker_id, jobs_path): worker_id
            for worker_id, jobs_path, _task_count in worker_assignments
        }
        for future in as_completed(batch_futures):
            worker_id, code, log_path = future.result()
            worker_codes[str(worker_id)] = code
            if code:
                raise RuntimeError(f"trial batch {worker_id} failed; inspect {log_path}")

    metrics = {arm: _aggregate_arm(arm, subset_by_arm[arm]) for arm in ARMS}
    scale_comparison = {
        "status": "COMPLETE",
        "evaluation_seeds": evaluation_seeds,
        "evaluation_layouts": 64,
        "full_layout_controls": {name: 64 for name in FULL_CONTROLS},
        "sanity_subset_controls": {name: 16 for name in SUBSET_CONTROLS},
        "sanity_subset_scene_ids": [scene["id"] for scene in subset_by_arm[ARMS[0]]],
        "arms": {arm: {
            "gate_status": metrics[arm]["status"],
            "protocol_sha256": metrics[arm]["protocol_sha256"],
            "graph_col_idx_sha256": graph_hashes[arm],
            "trained_model_sha256": training_hashes[arm]["trained"],
            "shuffled_reward_model_sha256": training_hashes[arm]["shuffled_reward"],
            "trained_successes": int(round(metrics[arm]["metrics"]["trained"]["success_fraction"] * 64)),
            "trained_trials": metrics[arm]["metrics"]["trained"]["trials"],
            "shuffled_reward_successes": int(round(metrics[arm]["metrics"]["shuffled_reward"]["success_fraction"] * 64)),
            "shuffled_reward_trials": metrics[arm]["metrics"]["shuffled_reward"]["trials"],
            "untrained_straight_success_fraction_subset": metrics[arm]["metrics"]["untrained_straight"]["success_fraction"],
            "odor_absent_success_fraction_subset": metrics[arm]["metrics"]["odor_absent"]["success_fraction"],
            "trained_success_fraction_on_sanity_subset": metrics[arm]["trained_success_fraction_on_sanity_subset"],
            "minimum_upright": min(v["minimum_upright"] for v in metrics[arm]["metrics"].values()),
        } for arm in ARMS},
        "protocols_identical": len(set(protocol_hashes.values())) == 1,
        "loaded_graph_column_hashes_match_arm_directories": True,
        "fitted_model_hashes_pairwise_distinct": {
            name: len({training_hashes[arm][name] for arm in ARMS}) == len(ARMS)
            for name in ("trained", "shuffled_reward")
        },
        "worker_count": workers,
        "trial_batch_size": batch_size,
        "trial_batch_count_this_resume": len(worker_assignments),
        "worker_exit_codes": worker_codes,
        "source_sha256": {"agent/baseline_scale.py": neural_connectome.digest(Path(__file__))},
        "graph_preservation_receipt": ".work/agent/matched-baseline/graphs.json",
        "scope": "one of four G1 comparisons; topology comparison on frozen odor task, with two sanity controls sampled on the fixed 16-layout subset",
    }
    (BASE / "scale-comparison-corrected.json").write_text(json.dumps(scale_comparison, indent=2) + "\n")
    execution = {"status": "COMPLETE", "arms": len(ARMS), "trials_per_arm": 160,
                 "total_trials": len(ARMS) * 160, "workers": workers,
                 "trial_batch_size": batch_size,
                 "trials_reused_from_prior_completed_workers": 1760 - len(flat_jobs),
                 "wall_seconds": time.perf_counter() - started,
                 "gate_statuses": {arm: metrics[arm]["status"] for arm in ARMS}}
    (BASE / "scale-execution-corrected.json").write_text(json.dumps(execution, indent=2) + "\n")
    print(json.dumps(execution, indent=2), flush=True)
    return execution


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-seeds", type=int, default=32)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--stage", choices=("train", "trial-worker"))
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--jobs-file", type=Path)
    parser.add_argument("--worker-id", type=int)
    args = parser.parse_args()
    if args.stage == "train":
        if not args.arm:
            parser.error("--arm is required for train stage")
        raise SystemExit(run_stage("train", args.arm))
    if args.stage == "trial-worker":
        if not args.jobs_file or args.worker_id is None:
            parser.error("trial-worker requires --jobs-file and --worker-id")
        raise SystemExit(run_trial_worker(args.jobs_file, args.worker_id))
    execute(args.evaluation_seeds, args.workers)


if __name__ == "__main__":
    main()
