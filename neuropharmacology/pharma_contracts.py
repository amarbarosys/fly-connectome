"""Typed contracts for the current transmitter-gain simulation interface.

This model can scale signed edges by presynaptic transmitter label. It does not
model compound/receptor binding, dose, or pharmacokinetics; the type names keep
that boundary explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


TRANSMITTERS = ("ach", "glu", "gaba", "dop", "ser", "oct")
PROFILE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TransmitterGainEffect:
    transmitter: str
    value: float
    evidence_status: str
    evidence_source: str
    rationale: str

    @property
    def sign_behavior(self) -> str:
        if self.value < 0:
            return "inverts_signed_edge_sign"
        if self.value == 0:
            return "removes_labeled_edge_contribution"
        return "preserves_signed_edge_sign"

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": {"kind": "presynaptic_transmitter", "id": self.transmitter},
            "action": "scale_signed_synaptic_weight",
            "value": self.value,
            "unit": "dimensionless_multiplier",
            "sign_behavior": self.sign_behavior,
            "evidence": {
                "status": self.evidence_status,
                "source": self.evidence_source,
                "rationale": self.rationale,
            },
        }


@dataclass(frozen=True)
class PerturbationProfile:
    name: str
    profile_type: str
    effects: tuple[TransmitterGainEffect, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "name": self.name,
            "profile_type": self.profile_type,
            "effects": [effect.to_dict() for effect in self.effects],
        }


def profile_from_dict(data: Mapping[str, Any]) -> PerturbationProfile:
    if not isinstance(data, Mapping):
        raise ValueError("profile must be a JSON object")
    if "gains" in data:
        raise ValueError("anonymous gains maps are unsupported; use typed effects")
    if data.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise ValueError(f"profile schema_version must be {PROFILE_SCHEMA_VERSION}")
    name = data.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("profile name is required")
    profile_type = data.get("profile_type")
    if profile_type not in {"null_model_arm", "transmitter_gain_hypothesis", "unmodeled"}:
        raise ValueError("profile_type must be null_model_arm, transmitter_gain_hypothesis, or unmodeled")
    raw_effects = data.get("effects")
    if not isinstance(raw_effects, list):
        raise ValueError("profile effects must be a list")
    if profile_type in {"null_model_arm", "unmodeled"} and raw_effects:
        raise ValueError(f"{profile_type} profile must have no effects")
    if profile_type == "transmitter_gain_hypothesis" and not raw_effects:
        raise ValueError("transmitter_gain_hypothesis requires at least one typed effect")

    effects: list[TransmitterGainEffect] = []
    seen: set[str] = set()
    for index, effect in enumerate(raw_effects):
        where = f"effects[{index}]"
        if not isinstance(effect, Mapping):
            raise ValueError(f"{where} must be an object")
        target = effect.get("target")
        if not isinstance(target, Mapping) or target.get("kind") != "presynaptic_transmitter":
            raise ValueError(f"{where}.target.kind must be presynaptic_transmitter")
        transmitter = target.get("id")
        if transmitter not in TRANSMITTERS:
            raise ValueError(f"{where}.target.id must be one of {', '.join(TRANSMITTERS)}")
        if transmitter in seen:
            raise ValueError(f"duplicate transmitter effect: {transmitter}")
        seen.add(transmitter)
        if effect.get("action") != "scale_signed_synaptic_weight":
            raise ValueError(f"{where}.action must be scale_signed_synaptic_weight")
        if effect.get("unit") != "dimensionless_multiplier":
            raise ValueError(f"{where}.unit must be dimensionless_multiplier")
        try:
            value = float(effect.get("value"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{where}.value must be numeric") from exc
        if not math.isfinite(value):
            raise ValueError(f"{where}.value must be finite")
        sign_behavior = (
            "inverts_signed_edge_sign" if value < 0 else
            "removes_labeled_edge_contribution" if value == 0 else
            "preserves_signed_edge_sign"
        )
        if effect.get("sign_behavior") != sign_behavior:
            raise ValueError(f"{where}.sign_behavior must be {sign_behavior} for value {value}")
        evidence = effect.get("evidence")
        if not isinstance(evidence, Mapping):
            raise ValueError(f"{where}.evidence is required")
        status = evidence.get("status")
        if status not in {"hypothesis", "supported"}:
            raise ValueError(f"{where}.evidence.status must be hypothesis or supported")
        source = evidence.get("source")
        rationale = evidence.get("rationale")
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"{where}.evidence.source is required")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError(f"{where}.evidence.rationale is required")
        if status == "supported" and not source.startswith(("https://", "doi:")):
            raise ValueError(f"{where} supported evidence needs a URL or DOI source")
        effects.append(TransmitterGainEffect(transmitter, value, status, source, rationale))
    return PerturbationProfile(name, profile_type, tuple(effects))


def gains_from_profile(data: Mapping[str, Any]) -> dict[str, float]:
    profile = profile_from_dict(data)
    if profile.profile_type == "unmodeled":
        raise ValueError(f"{profile.name} has no representable perturbation in this simulator")
    gains = {transmitter: 1.0 for transmitter in TRANSMITTERS}
    for effect in profile.effects:
        gains[effect.transmitter] = effect.value
    return gains


def typed_profile_from_gain_values(
    name: str,
    values: Mapping[str, float],
    *,
    source: str,
    rationale: str,
    status: str = "hypothesis",
) -> dict[str, Any]:
    """Explicit adapter for legacy/DB gain columns; provenance is mandatory."""
    unknown = set(values) - set(TRANSMITTERS)
    if unknown:
        raise ValueError(f"unknown transmitter(s): {', '.join(sorted(unknown))}")
    effects = []
    for transmitter, raw_value in values.items():
        effect = TransmitterGainEffect(
            transmitter=transmitter,
            value=float(raw_value),
            evidence_status=status,
            evidence_source=source,
            rationale=rationale,
        )
        effects.append(effect.to_dict())
    profile_type = "null_model_arm" if all(float(values.get(t, 1.0)) == 1.0 for t in TRANSMITTERS) else "transmitter_gain_hypothesis"
    if profile_type == "null_model_arm":
        effects = []
    profile = {"schema_version": PROFILE_SCHEMA_VERSION, "name": name, "profile_type": profile_type, "effects": effects}
    profile_from_dict(profile)
    return profile
