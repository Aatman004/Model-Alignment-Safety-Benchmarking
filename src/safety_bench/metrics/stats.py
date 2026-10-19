"""Statistical analysis and confidence interval calculators."""

import math
from typing import Tuple


def wilson_score_interval(
    successes: int, total: int, confidence: float = 0.95
) -> Tuple[float, float]:
    """Calculate the Wilson score confidence interval for a binomial proportion.

    Widely recommended for safety evaluations because it does not fail near 0% or 100%.
    """
    if total == 0:
        return 0.0, 0.0

    # z-scores for common confidence levels
    z_map = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}
    z = z_map.get(confidence, 1.960)

    p_hat = successes / total
    denom = 1 + (z**2) / total
    center = (p_hat + (z**2) / (2 * total)) / denom
    margin = (z * math.sqrt((p_hat * (1 - p_hat) + (z**2) / (4 * total)) / total)) / denom

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return round(lower, 4), round(upper, 4)
