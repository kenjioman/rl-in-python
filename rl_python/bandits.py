import abc
import inspect
from concurrent.futures import ProcessPoolExecutor
import dataclasses as dc
import typing as t

import numpy as np

from . import utils


class Bandit:
    def __init__(
        self,
        p: float,
        rng: t.Optional[np.random.Generator] = None,
        seed: t.Optional[int] = None,
    ):
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


@dc.dataclass(frozen=True)
class GangOfBanditsInputs:
    p_actuals: t.List[float]
    n_bandits: int = dc.field(init=False)
    p_estimates: t.Optional[t.List[float]] = None
    rng: np.random.Generator = None
    seed: int = None

    def __post_init__(self):
        object.__setattr__(self, "n_bandits", len(self.p_actuals))
        assert (self.p_estimates is None) or (len(self.p_estimates) == self.n_bandits)


class GangOfBandits:
    def __init__(self, args: GangOfBanditsInputs) -> None:
        self._store_params(args)

        self._bandits: np.ndarray[Bandit] = self._create_bandits()
        self._best_bandit: t.Set[int] = self._get_best_bandits()

        self.p_estimates: np.ndarray[float] = self._initialize_p_estimates()
        self._bandit_draw_count: np.ndarray[int] = (
            np.zeros(self.n_bandits)
            if args.p_estimates is None
            else np.ones(self.n_bandits)
        )

    def _store_params(self, args: GangOfBanditsInputs) -> None:
        self._rng = utils.get_rng_if_needed(args.rng, args.seed)
        self._p_actuals = args.p_actuals
        self.n_bandits = args.n_bandits
        self._p_initial_estimate = args.p_estimates

    def _create_bandits(self) -> "np.ndarray[Bandit]":
        return np.array([Bandit(p, self._rng) for p in self._p_actuals])

    def _get_best_bandits(self) -> t.Set[int]:
        """
        Find the indicies of the bandits with the highest success probability
        """
        best_p = max(self._p_actuals)

        return set(i for i, p in enumerate(self._p_actuals) if p == best_p)

    def _initialize_p_estimates(self) -> "np.ndarray[float]":
        return (
            np.zeros(self.n_bandits)
            if self._p_initial_estimate is None
            else np.array(self._p_initial_estimate, dtype=float)
        )

    def play_bandit(self, bandit_chosen: int) -> t.Tuple[bool, bool]:
        bandit_is_best = bandit_chosen in self._best_bandit
        reward = self._bandits[bandit_chosen].pull()
        self._update_draws_and_estimates_for_bandit(bandit_chosen, reward)
        return bandit_is_best, reward

    def _update_draws_and_estimates_for_bandit(
        self, bandit_i: int, reward: bool
    ) -> None:
        self._bandit_draw_count[bandit_i] += 1
        self.p_estimates[bandit_i] += (
            reward - self.p_estimates[bandit_i]
        ) / self._bandit_draw_count[bandit_i]


class AbstractStrategy(abc.ABC):
    def __init__(self, rng: np.random.Generator = None):
        self._rng = utils.get_rng_if_needed(rng, seed=None)

    @abc.abstractmethod
    def play_strategy(self, gob: GangOfBandits) -> t.Tuple[int, bool, bool]:
        raise NotImplementedError

    def play_random_bandit(self, gob: GangOfBandits) -> t.Tuple[int, bool, bool]:
        """
        Randomly choose a bandit to play from the available bandits,
        based on a simple uniform distribution

        Returns:
            bandit_chosen: The index of the bandit chosen
            bandit_is_best: True/False for if the chosen bandit has the highest
                actual success probability
            reward: True/False for if a reward was given
        """

        bandit_chosen = self._rng.integers(gob.n_bandits)
        bandit_is_best, reward = gob.play_bandit(bandit_chosen)
        return bandit_chosen, bandit_is_best, reward


@dc.dataclass
class StrategyWrapper:
    strategy_class: t.Type[AbstractStrategy]
    strategy_args: t.Dict[str, t.Any]


@dc.dataclass
class ReplicateResults:
    _best_chosen: t.List = dc.field(default_factory=list, init=False)
    _earnings: t.List = dc.field(default_factory=list, init=False)

    _best_chosen_record: np.ndarray = dc.field(init=False, default=None)
    _earnings_record: np.ndarray = dc.field(init=False, default=None)

    def add_best_chosen(self, values: np.ndarray) -> None:
        self._best_chosen.append(values)

    def add_earnings(self, values: np.ndarray) -> None:
        self._earnings.append(values)

    @property
    def best_chosen_record(self) -> np.ndarray:
        if self._best_chosen_record is None:
            self._best_chosen_record = np.stack(self._best_chosen)

        return self._best_chosen_record

    @property
    def earnings_record(self) -> np.ndarray:
        if self._earnings_record is None:
            self._earnings_record = np.stack(self._earnings)

        return self._earnings_record


class BanditExperiments:
    def __init__(
        self, gob_args: GangOfBanditsInputs, iterations: int, replicates: int
    ) -> None:
        self._rng = utils.get_rng_if_needed(gob_args.rng, gob_args.seed)
        _gob_args = self._update_gang_of_bandit_inputs_with_rng(gob_args, self._rng)
        self._gob_args = _gob_args
        self._iterations = iterations
        self._replicates = replicates

    @staticmethod
    def _update_gang_of_bandit_inputs_with_rng(
        gob_args: GangOfBanditsInputs, rng: np.random.Generator
    ) -> GangOfBanditsInputs:
        new_args = {**dc.asdict(gob_args), "rng": rng, "seed": None}
        new_args.pop("n_bandits", None)
        return GangOfBanditsInputs(**new_args)

    def run_iterations(
        self, strategy: AbstractStrategy, rng: np.random.Generator = None
    ) -> t.Dict[str, np.ndarray]:
        if rng is not None:
            _gob_args = self._update_gang_of_bandit_inputs_with_rng(self._gob_args, rng)
        else:
            _gob_args = self._gob_args
        gang_of_bandits = GangOfBandits(_gob_args)
        record = {
            "bandit_chosen": np.zeros(self._iterations),
            "bandit_is_best": np.zeros(self._iterations),
            "reward_given": np.zeros(self._iterations),
        }

        for i in range(self._iterations):
            self._record_iteration_results(
                record, i, strategy.play_strategy(gang_of_bandits)
            )

        return record

    @staticmethod
    def _record_iteration_results(
        record: t.Dict[str, "np.ndarray[int]"],
        _round: int,
        results: t.Tuple[int, bool, bool],
    ) -> None:
        chosen, is_best, reward = results
        record["bandit_chosen"][_round] = chosen
        record["bandit_is_best"][_round] = is_best
        record["reward_given"][_round] = reward

    def run_replicates(self, strategy_wrapper: StrategyWrapper) -> ReplicateResults:
        if "rng" in inspect.signature(strategy_wrapper.strategy_class).parameters:
            strategy_args = {**strategy_wrapper.strategy_args, "rng": self._rng}
        else:
            strategy_args = strategy_wrapper.strategy_args

        record = ReplicateResults()

        for replicate in range(self._replicates):
            strategy = strategy_wrapper.strategy_class(**strategy_args)
            self._record_replicate_results(record, self.run_iterations(strategy))

        return record

    @staticmethod
    def _record_replicate_results(
        record: ReplicateResults, results: t.Dict[str, np.ndarray]
    ) -> None:
        record.add_best_chosen(results["bandit_is_best"])
        record.add_earnings(results["reward_given"])

    def run_replicates_parallel(
        self, strategy_wrapper: StrategyWrapper
    ) -> ReplicateResults:
        record = ReplicateResults()

        # To get good parallel RNG results, need to do some special magic. See:
        # https://numpy.org/doc/stable/reference/random/parallel.html
        # https://albertcthomas.github.io/good-practices-random-number-generators/
        seed_sequence: np.random.SeedSequence = self._rng.bit_generator._seed_seq
        child_states = seed_sequence.spawn(self._replicates)
        rngs = [utils.get_rng_if_needed(seed=s) for s in child_states]
        if "rng" in inspect.signature(strategy_wrapper.strategy_class).parameters:
            strategies = [
                strategy_wrapper.strategy_class(
                    **{**strategy_wrapper.strategy_args, "rng": r}
                )
                for r in rngs
            ]
        else:
            strategies = [
                strategy_wrapper.strategy_class(**strategy_wrapper.strategy_args)
                for _ in range(self._replicates)
            ]

        with ProcessPoolExecutor() as executor:
            results = executor.map(self.run_iterations, strategies, rngs)
            for result in results:
                self._record_replicate_results(record, result)

        return record


class Greedy(AbstractStrategy):
    def __init__(self, rng: t.Optional[np.random.Generator] = None):
        super().__init__(rng)

    def play_strategy(self, gob: GangOfBandits) -> t.Tuple[int, bool, bool]:
        return self.play_random_bandit(gob)


class EpsilonGreedy(AbstractStrategy):
    def __init__(self, epsilon: float, rng: t.Optional[np.random.Generator] = None):
        super().__init__(rng)
        self._p = epsilon

    def play_strategy(self, gob: GangOfBandits) -> t.Tuple[int, bool, bool]:
        should_play_random_bandit = self._rng.random() < self._p
        if should_play_random_bandit:
            return self.play_random_bandit(gob)
        else:
            return self.play_current_best_bandit(gob)

    def play_current_best_bandit(self, gob: GangOfBandits) -> t.Tuple[int, bool, bool]:
        """
        Based on the current estimates of bandit success probabilities, choose
        the best bandit to play

        Returns:
            bandit_chosen: The index of the bandit chosen
            bandit_is_best: True/False for if the chosen bandit has the highest
                actual success probability
            reward: True/False for if a reward was given
        """
        current_best_score = np.amax(gob.p_estimates)
        bandits_with_best_score = np.argwhere(gob.p_estimates == current_best_score)
        bandit_chosen = self._rng.choice(bandits_with_best_score)[0]
        bandit_is_best, reward = gob.play_bandit(bandit_chosen)
        return bandit_chosen, bandit_is_best, reward


class ExponentialDecayEpsilonGreedy(EpsilonGreedy):
    def __init__(
        self,
        epsilon_0: float,
        alpha: float,
        rng: t.Optional[np.random.Generator] = None,
    ):
        super().__init__(epsilon_0, rng)

        self._epsilon_0 = epsilon_0
        assert 0 <= alpha <= 1
        self._alpha = alpha

        self._t = -1

    def play_strategy(self, gob: GangOfBandits) -> t.Tuple[int, bool, bool]:
        self._t += 1
        self._p = self._epsilon_0 * (self._alpha ** self._t)
        return super().play_strategy(gob)


def plot_p_best_bandit_chosen_vs_rounds_in_game(
    results: t.List[ReplicateResults],
    labels: t.List[str],
    **plot_args: t.Dict[str, t.Any]
) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=plot_args.get("figsize", (15, 10)))
    for result, label in zip(results, labels):
        plt.plot(np.mean(result.best_chosen_record, axis=0), label=label)

    plt.title("Probability best bandit was chosen over iterations in a replicate")
    plt.xlabel("Iterations")
    plt.ylabel("Probability of Optimal Action")
    plt.ylim((0, 1))
    plt.grid(which="major")
    plt.show()
