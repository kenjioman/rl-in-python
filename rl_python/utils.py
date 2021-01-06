import numpy as np


def get_rng_if_needed(
    random_number_generator: np.random.Generator = None, seed: int = None
) -> np.random.Generator:
    if (random_number_generator is None) or (seed is not None):
        return np.random.default_rng(seed)

    return random_number_generator
