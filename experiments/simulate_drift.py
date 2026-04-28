"""Create 8 data windows simulating ISOT → LIAR distribution shift."""

import numpy as np
import pandas as pd

MIXING_RATIOS = [
    (1.00, 0.00),
    (1.00, 0.00),
    (1.00, 0.00),
    (0.75, 0.25),
    (0.75, 0.25),
    (0.50, 0.50),
    (0.50, 0.50),
    (0.25, 0.75),
]


def create_drift_windows(
    isot_df: pd.DataFrame,
    liar_df: pd.DataFrame,
    window_size: int = 2000,
    seed: int = 42,
) -> list:
    rng = np.random.RandomState(seed)
    windows = []
    for isot_ratio, liar_ratio in MIXING_RATIOS:
        n_isot = int(window_size * isot_ratio)
        n_liar = int(window_size * liar_ratio)
        parts = []
        if n_isot > 0:
            idx = rng.choice(len(isot_df), min(n_isot, len(isot_df)), replace=False)
            parts.append(isot_df.iloc[idx])
        if n_liar > 0:
            idx = rng.choice(len(liar_df), min(n_liar, len(liar_df)), replace=False)
            parts.append(liar_df.iloc[idx])
        window = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed)
        windows.append(window.reset_index(drop=True))
    return windows


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from ml.dataset import load_isot, load_liar

    isot = load_isot("data/isot")
    liar = load_liar("data/liar")
    windows = create_drift_windows(isot, liar)
    for i, w in enumerate(windows):
        ratio = MIXING_RATIOS[i]
        print(
            f"Window {i}: {len(w)} samples, ISOT={ratio[0]:.0%}, LIAR={ratio[1]:.0%}, "
            f"fake={w['label'].eq(0).sum()}, real={w['label'].eq(1).sum()}"
        )
