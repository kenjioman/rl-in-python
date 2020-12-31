import typing as t

import numpy as np

import utils


class Bandit:
    def __init__(self, p: float, rng: t.Optional[np.random.Generator] = None, seed: t.Optional[int] = None):
        """
        Create a bandit with success probability `p`

        Args:
            p: Probability of a pull returning a reward
            rng: Random number generator
            seed: Seed to use to create a new rng, if none given. If both rng
                and seed are None, will use system random state to seed the new
                rng.
        """
        self._p = p
        self._rng = utils.get_rng_if_needed(rng, seed)

    def pull(self) -> bool:
        """
        "query" the bandit to see if we succeeded or not, based on the
        `p` given at init

        Returns:
            bool: True if we succeeded, False if not
        """
        return self._pull_uniform()

    def _pull_uniform(self) -> bool:
        return self._rng.random() < self._p


class GangOfBandits:
    def __init__(self,
                 p_actuals: t.List[float],
                 p_estimates: t.Optional[t.List[float]] = None,
                 rng: np.random.Generator = None,
                 seed: int = None) -> None:
        self._rng = utils.get_rng_if_needed(rng, seed)
        self._p_actuals = p_actuals
        self._n_bandits = len(p_actuals)

        assert (p_estimates is None) or (len(p_estimates) == self._n_bandits)
        self._p_initial_estimate = p_estimates

        self._bandits: np.ndarray[Bandit] = self._create_bandits()
        self._best_bandit: t.Set[int] = self._get_best_bandits()

        self._p_estimates: np.ndarray[float] = self._initialize_p_estimates()
        self._bandit_draws: np.ndarray[int] = np.zeros(self._n_bandits)

    def _create_bandits(self) -> np.ndarray[Bandit]:
        return np.array([Bandit(p, self._rng) for p in self._p_actuals])

    def _get_best_bandits(self) -> t.Set[int]:
        """
        Find the indicies of the bandits with the highest success probability
        """
        best_p = max(self._p_actuals)

        return set(i for i, p in enumerate(self._p_actuals) if p == best_p)

    def _initialize_p_estimates(self) -> np.ndarray[float]:
        return np.zeros(self._n_bandits) if self._p_initial_estimate is None else np.array(self._p_initial_estimate)

    def play_random_bandit(self) -> t.Tuple[int, bool, bool]:
        """
            Randomly choose a bandit to play from the list of available bandits,
            based on a simple uniform distribution

            Returns:
                bandit_chosen: The index of the bandit chosen
                bandit_is_best: True/False for if the chosen bandit has the highest
                    actual success probability
                reward: True/False for if a reward was given
        """

        bandit_chosen = self._rng.integers(self._n_bandits)
        bandit_is_best, reward = self._play_bandit_and_update_tallies(bandit_chosen)
        return bandit_chosen, bandit_is_best, reward

    def _play_bandit_and_update_tallies(self, bandit_chosen: int) -> t.Tuple[bool, bool]:
        bandit_is_best = bandit_chosen in self._best_bandit
        reward = self._bandits[bandit_chosen].pull()
        self._update_draws_and_estimates_for_bandit(bandit_chosen, reward)
        return bandit_is_best, reward

    def _update_draws_and_estimates_for_bandit(self, bandit_i: int, reward: bool) -> None:
        self._bandit_draws[bandit_i] += 1
        self._p_estimates[bandit_i] += (reward - self._p_estimates[bandit_i]) / self._bandit_draws[bandit_i]

    def play_current_best_bandit(self) -> t.Tuple[int, bool, bool]:
        """
        Based on the current estimates of bandit success probabilities, choose
        the best bandit to play

        Returns:
            bandit_chosen: The index of the bandit chosen
            bandit_is_best: True/False for if the chosen bandit has the highest
                actual success probability
            reward: True/False for if a reward was given
        """
        current_best_score = np.amax(self._p_estimates)
        bandits_with_best_score = np.argwhere(self._p_estimates == current_best_score)
        bandit_chosen = self._rng.choice(bandits_with_best_score)[0]
        bandit_is_best, reward = self._play_bandit_and_update_tallies(bandit_chosen)
        return bandit_chosen, bandit_is_best, reward
