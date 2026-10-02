"""Paired significance for frozen-layout gates using exact McNemar tests.

Scaled graph-arm comparisons use one shared set of held-out layouts. For pooled
graph-seed summaries, the exact sign-flip test treats layout as the independent
cluster so the one real outcome per layout is not counted five times as if it
were a new observation.
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


def _cluster_sign_flip(real, controls):
    """Exact two-sided sign-flip test over layout clusters for a pooled ensemble effect.

    Each layout contributes real_success minus the mean of the five independent
    graph-realization outcomes. Integer contributions are scaled by the number
    of realizations; the dynamic program counts all sign assignments exactly.
    """
    assert controls
    keys = set(real)
    for label, arm in controls.items():
        assert set(arm) == keys, f"pooled {label}: compared arm does not share identical layout set"
    realization_count = len(controls)
    contributions = [
        realization_count * int(real[key]) - sum(int(arm[key]) for arm in controls.values())
        for key in sorted(keys)
    ]
    observed = abs(sum(contributions))
    nonzero = [value for value in contributions if value]
    distribution = {0: 1}
    for value in nonzero:
        updated = {}
        for subtotal, count in distribution.items():
            updated[subtotal + value] = updated.get(subtotal + value, 0) + count
            updated[subtotal - value] = updated.get(subtotal - value, 0) + count
        distribution = updated
    extreme = sum(count for total, count in distribution.items() if abs(total) >= observed)
    p_value = extreme / (2 ** len(nonzero)) if nonzero else 1.0
    total_pairs = len(keys) * realization_count
    wins = sum(sum(1 for key in keys if real[key] and not arm[key]) for arm in controls.values())
    losses = sum(sum(1 for key in keys if arm[key] and not real[key]) for arm in controls.values())
    real_repeated_successes = sum(real.values()) * realization_count
    control_successes = sum(sum(arm.values()) for arm in controls.values())
    return {
        "test": "exact two-sided layout-cluster sign-flip; real outcome compared with mean across graph realizations",
        "layout_clusters": len(keys),
        "graph_realizations": realization_count,
        "paired_arm_layouts": total_pairs,
        "real_successes_repeated_for_descriptive_pair_total": real_repeated_successes,
        "control_successes_across_realizations": control_successes,
        "pooled_real_success_fraction": float(sum(real.values()) / len(keys)),
        "pooled_control_success_fraction": float(control_successes / total_pairs),
        "mean_success_fraction_difference": float(sum(contributions) / total_pairs),
        "combined_discordant_pairs": wins + losses,
        "combined_real_only_wins": wins,
        "combined_control_only_wins": losses,
        "nonzero_layout_clusters": len(nonzero),
        "exact_two_sided_p": p_value,
        "assumption": "layout clusters are independent and the paired cluster-difference signs are exchangeable under the null",
    }


def _seed_summary(real, controls, label):
    rows = []
    differences = []
    for arm_name, outcomes_for_arm in controls.items():
        row = compare(f"real vs {arm_name}", real, outcomes_for_arm,
                      "trained readouts on the same 64 frozen layouts")
        row["graph_seed_arm"] = arm_name
        row["success_count_difference"] = row["arm_successes"] - row["control_successes"]
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
            "significant_comparisons_at_0_05": sum(row["significant_at_0_05"] for row in rows),
        },
    }


def report(scale_prefix="scale-", output_name="significance.json"):
    ingested = lambda row: row["ingested"] >= 0.05
    avoided = lambda row: row["contact_exchanges"] <= 1
    real = outcomes(BASELINE / f"{scale_prefix}real", "trained", ingested)
    shuffled = {arm: outcomes(BASELINE / f"{scale_prefix}{arm}", "trained", ingested)
                for arm in SHUFFLE_ARMS}
    random = {arm: outcomes(BASELINE / f"{scale_prefix}{arm}", "trained", ingested)
              for arm in RANDOM_ARMS}
    protocol = json.loads((BASELINE / f"{scale_prefix}real/protocol.json").read_text())
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
        "test": "exact two-sided McNemar for each paired graph-seed comparison; exact layout-cluster sign-flip for pooled graph-family summaries",
        "why": "all compared graph arms share identical frozen layout sets; pooled inference clusters repeated comparisons by layout",
        "evaluation_layouts": expected_layouts,
        "graph_realizations_per_control_family": 5,
        "scale_comparisons": scale_comparisons,
        "significant_seed_comparisons_at_0_05": sum(
            row["significant_at_0_05"]
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
                  f"p={row['exact_two_sided_p']:.4f}  "
                  f"{'SIGNIFICANT' if row['significant_at_0_05'] else 'not significant'}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale-prefix", default="scale-")
    parser.add_argument("--output-name", default="significance.json")
    args = parser.parse_args()
    report(args.scale_prefix, args.output_name)
