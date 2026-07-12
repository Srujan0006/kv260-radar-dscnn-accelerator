"""Label utilities for mapping detailed target names to four superclasses."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

CLASS_NAMES = ["Drone", "Bird", "Human", "CornerReflector"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}

DRONE_KEYWORDS = [
    "drone",
    "uav",
    "mavic",
    "matrice",
    "phantom",
    "fpv",
    "tarot",
    "syma",
    "quadcopter",
]
BIRD_KEYWORDS = [
    "bird",
    "seagull",
    "gull",
    "pigeon",
    "raven",
    "heron",
]
HUMAN_KEYWORDS = ["human", "person", "walking", "running", "walk", "run"]
CR_KEYWORDS = ["corner", "reflector", "cr", "calibrator", "reference"]


def label_to_superclass(label: Any) -> int:
    """Map a label value to one of the four superclass IDs.

    Integers 0..3 are passed through. Strings are matched using known keywords.
    """
    if isinstance(label, (int,)):
        if 0 <= int(label) < len(CLASS_NAMES):
            return int(label)
        raise ValueError(f"Integer label {label} is outside expected range 0..3")

    # numpy scalar support without importing numpy here
    if hasattr(label, "item"):
        try:
            return label_to_superclass(label.item())
        except Exception:
            pass

    text = str(label).strip().lower().replace("_", " ").replace("-", " ")

    if any(key in text for key in DRONE_KEYWORDS):
        return CLASS_TO_ID["Drone"]
    if any(key in text for key in BIRD_KEYWORDS):
        return CLASS_TO_ID["Bird"]
    if any(key in text for key in HUMAN_KEYWORDS):
        return CLASS_TO_ID["Human"]
    if any(key in text for key in CR_KEYWORDS):
        return CLASS_TO_ID["CornerReflector"]

    raise ValueError(
        f"Cannot map label {label!r} to superclass. Provide labels as 0..3 or update labels.py."
    )


def map_labels(labels: Iterable[Any]) -> List[int]:
    return [label_to_superclass(label) for label in labels]


def class_distribution(labels: Iterable[int]) -> Dict[str, int]:
    counts: Dict[str, int] = {name: 0 for name in CLASS_NAMES}
    for label in labels:
        counts[CLASS_NAMES[int(label)]] += 1
    return counts
