"""Paired significance for frozen-layout gates with mirrored-seed inference.

The 64 held-out layouts are 32 mirrored pairs generated from one scene per seed.
Significance therefore collapses mirror outcomes to seed-level differences.
"""
import json
import argparse
from math import comb
from pathlib import Path
import time

from agent.neural_connectome import ROOT, digest


BASELINE = ROOT / ".work/agent/matched-baseline"
SEQUENTIAL = ROOT / ".work/agent/odor-memory/calibration-v2"
OUT = ROOT / ".work/agent/paired-significance"
SHUFFLE_ARMS = tuple(f"shuffled-s{seed}" for seed in
                     (20261002, 7, 1009, 424242, 31337))
RANDOM_ARMS = tuple(f"random-s{seed}" for seed in
                    (20261002, 7, 1009, 424242, 31337))
SCALE_ARMS = ("real",) + SHUFFLE_ARMS + RANDOM_ARMS


def exact_mcnemar(first, second):
    """Discordant-pair counts and the exact two-sided binomial test."""
    assert set(first) == set(second), "paired McNemar requires identical layout sets"
    layouts = sorted(first)
    wins = sum(1 for key in layouts if first[key] and not second[key])
    losses = sum(1 for key in layouts if second[key] and not first[key])
    discordant = wins + losses
    if not discordant:
        return wins, losses, 0, 1.0
    tail = sum(comb(discordant, index) for index in range(min(wins, losses) + 1))
    return wins, losses, discordant, min(1.0, 2.0 * tail / 2**discordant)


def outcomes(directory, control, decide):
    results = {}
    for path in sorted(directory.glob(f"eval-*-{control}.json")):
        row = json.loads(path.read_text())
        if row.get("control") != control:
            continue
        results[json.dumps(row["scene"], sort_keys=True)] = bool(decide(row))
    return results


def compare(label, first, second, pair):
    assert set(first) == set(second), f"{label}: compared arms do not share identical layout sets"
    wins, losses, discordant, p_value = exact_mcnemar(first, second)
    return {
        "comparison": label,
        "arm_successes": sum(first.values()),
        "control_successes": sum(second.values()),
        "layouts": len(first),
        "arm_only_wins": wins,
        "control_only_wins": losses,
        "discordant_pairs": discordant,
        "exact_two_sided_p": p_value,
        "significant_at_0_05": p_value < 0.05,
        "pair": pair,
    }


def _seed_cluster_outcomes(layouts):
    """Collapse mirrored layout outcomes to one fractional mean per seed."""
    clustered = {}
    for key, value in layouts.items():
        scene = json.loads(key)
        seed = str(scene["seed"])
        clustered.setdefault(seed, []).append(int(value))
    return {seed: sum(values) / len(values) for seed, values in clustered.items()}


def _seed_layout_counts(layouts):
    counts = {}
    for key in layouts:
        seed = str(json.loads(key)["seed"])
        counts[seed] = counts.get(seed, 0) + 1
    return counts


def _exact_sign_flip(differences, scale=1):
    """Exact two-sided sign-flip over independent seed-level paired differences."""
    values = [int(round(value * scale)) for value in differences if value]
    observed = abs(sum(int(round(value * scale)) for value in differences))
    distribution = {0: 1}
    for value in values:
        updated = {}
        for subtotal, count in distribution.items():
            updated[subtotal + value] = updated.get(subtotal + value, 0) + count
            updated[subtotal - value] = updated.get(subtotal - value, 0) + count
        distribution = updated
    extreme = sum(count for total, count in distribution.items()
                  if abs(total) >= observed - 1e-12)
    return extreme / (2 ** len(values)) if values else 1.0, len(values)


def _cluster_sign_flip(real, controls):
    """Exact sign-flip over seed clusters for a pooled ensemble effect.

    Each seed contributes its mean real outcome minus the mean of the five
    graph-realization outcomes. Half-success mirror pairs are scaled to integers
    before the dynamic program counts all sign assignments exactly.
    """
    assert controls
    keys = set(real)
    for label, arm in controls.items():
        assert set(arm) == keys, f"pooled {label}: compared arm does not share identical layout set"
    realization_count = len(controls)
    real_by_seed = _seed_cluster_outcomes(real)
    control_by_seed = [_seed_cluster_outcomes(arm) for arm in controls.values()]
    seeds = sorted(real_by_seed)
    assert all(set(arm) == set(seeds) for arm in control_by_seed)
    contributions = [realization_count * real_by_seed[seed]
                     - sum(arm[seed] for arm in control_by_seed) for seed in seeds]
    observed = abs(sum(contributions))
    p_value, nonzero_count = _exact_sign_flip(contributions, scale=2)
    total_pairs = len(keys) * realization_count
    wins = sum(sum(1 for key in keys if real[key] and not arm[key]) for arm in controls.values())
    losses = sum(sum(1 for key in keys if arm[key] and not real[key]) for arm in controls.values())
    real_repeated_successes = sum(real.values()) * realization_count
    control_successes = sum(sum(arm.values()) for arm in controls.values())
    return {
        "test": "exact two-sided seed-cluster sign-flip; each seed averages its mirrored layouts and real outcome is compared with mean across graph realizations",
        "seed_clusters": len(seeds),
        "mirrored_layouts_per_seed": sorted(set(_seed_layout_counts(real).values())),
        "layout_count": len(keys),
        "graph_realizations": realization_count,
        "paired_arm_layouts": total_pairs,
        "real_successes_repeated_for_descriptive_pair_total": real_repeated_successes,
        "control_successes_across_realizations": control_successes,
        "pooled_real_success_fraction": float(sum(real.values()) / len(keys)),
        "pooled_control_success_fraction": float(control_successes / total_pairs),
        "mean_success_fraction_difference": float(sum(contributions) / (len(seeds) * realization_count)),
        "combined_discordant_pairs": wins + losses,
        "combined_real_only_wins": wins,
        "combined_control_only_wins": losses,
        "nonzero_seed_clusters": nonzero_count,
        "exact_two_sided_p": p_value,
        "assumption": "seed clusters are independent; mirrored layouts within seed are dependent; paired seed-difference signs are exchangeable under the null",
    }


def _seed_summary(real, controls, label):
    rows = []
    differences = []
    for arm_name, outcomes_for_arm in controls.items():
        row = compare(f"real vs {arm_name}", real, outcomes_for_arm,
                      "trained readouts on the same 64 frozen layouts")
        row["graph_seed_arm"] = arm_name
        row["success_count_difference"] = row["arm_successes"] - row["control_successes"]
        seed_real = _seed_cluster_outcomes(real)
        seed_arm = _seed_cluster_outcomes(outcomes_for_arm)
        seed_p, seed_nonzero = _exact_sign_flip(
            [seed_real[seed] - seed_arm[seed] for seed in sorted(seed_real)], scale=2
        )
        row["independent_seed_clusters"] = len(seed_real)
        row["seed_cluster_arm_only_fractional_wins"] = sum(
            max(0.0, seed_real[seed] - seed_arm[seed]) for seed in seed_real
        )
        row["seed_cluster_control_only_fractional_wins"] = sum(
            max(0.0, seed_arm[seed] - seed_real[seed]) for seed in seed_real
        )
        row["exact_two_sided_seed_cluster_sign_flip_p"] = seed_p
        row["nonzero_seed_clusters"] = seed_nonzero
        differences.append(row["success_count_difference"])
        rows.append(row)
    return {
        "control_family": label,
        "per_seed": rows,
        "pooled": _cluster_sign_flip(real, controls),
        "spread_across_seeds": {
            "success_count_differences": differences,
            "minimum_difference": min(differences),
            "maximum_difference": max(differences),
            "mean_difference": sum(differences) / len(differences),
            "all_directions_favour_real": all(value > 0 for value in differences),
            "significant_seed_cluster_comparisons_at_0_05": sum(
                row["exact_two_sided_seed_cluster_sign_flip_p"] < 0.05 for row in rows
            ),
        },
    }


def report(scale_prefix="scale-", output_name="significance.json"):
    protocol = json.loads((BASELINE / f"{scale_prefix}real/protocol.json").read_text())
    success_threshold = protocol["criteria"]["success_ingested_food_u"]
    ingested = lambda row: row["ingested"] >= success_threshold
    avoided = lambda row: row["contact_exchanges"] <= 1
    real = outcomes(BASELINE / f"{scale_prefix}real", "trained", ingested)
    shuffled = {arm: outcomes(BASELINE / f"{scale_prefix}{arm}", "trained", ingested)
                for arm in SHUFFLE_ARMS}
    random = {arm: outcomes(BASELINE / f"{scale_prefix}{arm}", "trained", ingested)
              for arm in RANDOM_ARMS}
    expected_layouts = len(protocol["evaluation"])
    assert expected_layouts > 0
    for arm_name, arm_outcomes in {"real": real, **shuffled, **random}.items():
        assert set(arm_outcomes) == set(real), f"{arm_name}: layout set differs from real arm"
        assert len(arm_outcomes) == expected_layouts, (
            f"{arm_name}: expected {expected_layouts} paired layouts, got {len(arm_outcomes)}"
        )

    scale_comparisons = {
        "real_vs_five_shuffled_seeds": _seed_summary(real, shuffled, "degree-preserving shuffled"),
        "real_vs_five_random_seeds": _seed_summary(real, random, "random-column controls"),
    }

    seq = {control: outcomes(SEQUENTIAL, control, avoided)
           for control in ("trained", "untrained_straight", "shuffled_reward")}
    sequential_rows = [
        compare(f"sequential avoidance: trained vs {control}", seq["trained"], seq[control],
                "calibration-v2 frozen held-out layouts")
        for control in ("untrained_straight", "shuffled_reward")
    ]
    out = {
        "status": "PASS",
        "test": "exact two-sided McNemar retained as descriptive layout comparison; exact paired seed-level sign-flip for individual and pooled inference",
        "why": "each evaluation seed generates mirrored layouts with shared distance, bearing and odor; mirror outcomes are collapsed within seed before exact inference",
        "success_ingested_food_u_threshold": success_threshold,
        "independent_seed_clusters": len(_seed_cluster_outcomes(real)),
        "evaluation_layouts": expected_layouts,
        "graph_realizations_per_control_family": 5,
        "scale_comparisons": scale_comparisons,
        "significant_seed_cluster_comparisons_at_0_05": sum(
            row["exact_two_sided_seed_cluster_sign_flip_p"] < 0.05
            for family in scale_comparisons.values() for row in family["per_seed"]
        ),
        "total_seed_comparisons": 10,
        "sequential_comparisons": sequential_rows,
        "scope": "paired per-layout outcome evidence for these frozen tasks and graph realizations; it does not establish effects on other tasks or biological learning",
        "single_shuffle_realisation": False,
        "sources_sha256": {str(Path(__file__).relative_to(ROOT)): digest(Path(__file__))},
        "recorded_unix": time.time(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / output_name).write_text(json.dumps(out, indent=2) + "\n")
    for family in scale_comparisons.values():
        for row in family["per_seed"]:
            print(f"{row['comparison']:52s} {row['arm_successes']:2d}/{row['layouts']} vs "
                  f"{row['control_successes']:2d}/{row['layouts']}  "
                  f"discordant {row['arm_only_wins']}/{row['control_only_wins']}  "
                  f"layout McNemar p={row['exact_two_sided_p']:.4f} (descriptive)  "
                  f"seed-cluster p={row['exact_two_sided_seed_cluster_sign_flip_p']:.4f}  "
                  f"{'SIGNIFICANT' if row['exact_two_sided_seed_cluster_sign_flip_p'] < 0.05 else 'not significant'}")
    for name, family in scale_comparisons.items():
        pooled = family["pooled"]
        print(f"pooled {name}: difference {pooled['mean_success_fraction_difference']:+.4f}  "
              f"seed-cluster p={pooled['exact_two_sided_p']:.6f}  ({pooled['seed_clusters']} seeds)")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale-prefix", default="scale-")
    parser.add_argument("--output-name", default="significance.json")
    args = parser.parse_args()
    report(args.scale_prefix, args.output_name)
