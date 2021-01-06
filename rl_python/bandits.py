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

        self._p_estimates: np.ndarray[float] = self._initialize_p_estimates()
        self._bandit_draws: np.ndarray[int] = (
            np.zeros(self._n_bandits)
            if args.p_estimates is None
            else np.ones(self._n_bandits)
        )

    def _store_params(self, args: GangOfBanditsInputs) -> None:
        self._rng = utils.get_rng_if_needed(args.rng, args.seed)
        self._p_actuals = args.p_actuals
        self._n_bandits = args.n_bandits
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
            np.zeros(self._n_bandits)
            if self._p_initial_estimate is None
            else np.array(self._p_initial_estimate, dtype=float)
        )

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

    def _play_bandit_and_update_tallies(
        self, bandit_chosen: int
    ) -> t.Tuple[bool, bool]:
        bandit_is_best = bandit_chosen in self._best_bandit
        reward = self._bandits[bandit_chosen].pull()
        self._update_draws_and_estimates_for_bandit(bandit_chosen, reward)
        return bandit_is_best, reward

    def _update_draws_and_estimates_for_bandit(
        self, bandit_i: int, reward: bool
    ) -> None:
        self._bandit_draws[bandit_i] += 1
        self._p_estimates[bandit_i] += (
            reward - self._p_estimates[bandit_i]
        ) / self._bandit_draws[bandit_i]

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


class AbstractIfRandomPicker(abc.ABC):
    def __init__(self, rng=None):
        self.rng = utils.get_rng_if_needed(rng, seed=None)

    @abc.abstractmethod
    def should_play_random(self) -> bool:
        pass


@dc.dataclass
class StrategyWrapper:
    strategy_class: t.Type[AbstractIfRandomPicker]
    strategy_args: t.Dict[str, t.Any]


@dc.dataclass
class SeasonResults:
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


class BanditGames:
    def __init__(self, gob_args: GangOfBanditsInputs, rounds: int, games: int) -> None:
        self._rng = utils.get_rng_if_needed(gob_args.rng, gob_args.seed)
        _gob_args = self._update_gang_of_bandit_inputs_with_rng(gob_args, self._rng)
        self._gob_args = _gob_args
        self._rounds = rounds
        self._games = games

    @staticmethod
    def _update_gang_of_bandit_inputs_with_rng(
        gob_args: GangOfBanditsInputs, rng: np.random.Generator
    ) -> GangOfBanditsInputs:
        new_args = {**dc.asdict(gob_args), "rng": rng, "seed": None}
        new_args.pop("n_bandits", None)
        return GangOfBanditsInputs(**new_args)

    def play_game(
        self, strategy: AbstractIfRandomPicker, rng: np.random.Generator = None
    ) -> t.Dict[str, np.ndarray]:
        if rng is not None:
            _gob_args = self._update_gang_of_bandit_inputs_with_rng(self._gob_args, rng)
        else:
            _gob_args = self._gob_args
        gang_of_bandits = GangOfBandits(_gob_args)
        record = {
            "bandit_chosen": np.zeros(self._rounds),
            "bandit_is_best": np.zeros(self._rounds),
            "reward_given": np.zeros(self._rounds),
        }

        for _round in range(self._rounds):
            if strategy.should_play_random():
                self._record_round_results(
                    record, _round, gang_of_bandits.play_random_bandit()
                )
            else:
                self._record_round_results(
                    record, _round, gang_of_bandits.play_current_best_bandit()
                )

        return record

    @staticmethod
    def _record_round_results(
        record: t.Dict[str, "np.ndarray[int]"],
        _round: int,
        results: t.Tuple[int, bool, bool],
    ) -> None:
        chosen, is_best, reward = results
        record["bandit_chosen"][_round] = chosen
        record["bandit_is_best"][_round] = is_best
        record["reward_given"][_round] = reward

    def play_season(self, strategy_wrapper: StrategyWrapper) -> SeasonResults:
        if "rng" in inspect.signature(strategy_wrapper.strategy_class).parameters:
            strategy_args = {**strategy_wrapper.strategy_args, "rng": self._rng}
        else:
            strategy_args = strategy_wrapper.strategy_args

        record = SeasonResults()

        for game in range(self._games):
            strategy = strategy_wrapper.strategy_class(**strategy_args)
            self._record_game_results(record, self.play_game(strategy))

        return record

    def _record_game_results(
        self, record: SeasonResults, results: t.Dict[str, np.ndarray]
    ) -> None:
        record.add_best_chosen(results["bandit_is_best"])
        record.add_earnings(results["reward_given"])

    def play_season_parallel(self, strategy_wrapper: StrategyWrapper) -> SeasonResults:
        record = SeasonResults()

        # To get good parallel RNG results, need to do some special magic. See:
        # https://numpy.org/doc/stable/reference/random/parallel.html
        # https://albertcthomas.github.io/good-practices-random-number-generators/
        seed_sequence: np.random.SeedSequence = self._rng.bit_generator._seed_seq
        child_states = seed_sequence.spawn(self._games)
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
                for _ in range(self._games)
            ]

        with ProcessPoolExecutor() as executor:
            results = executor.map(self.play_game, strategies, rngs)
            for result in results:
                self._record_game_results(record, result)

        return record


class Greedy(AbstractIfRandomPicker):
    def __init__(self):
        pass

    def should_play_random(self) -> bool:
        return False


class EpsilonGreedy(AbstractIfRandomPicker):
    def __init__(self, epsilon: float, rng: t.Optional[np.random.Generator] = None):
        super().__init__(rng)
        self._p = epsilon

    def should_play_random(self) -> bool:
        return self._rng.random() < self._p


class ExponentialDecayEpsilonGreedy(AbstractIfRandomPicker):
    def __init__(
        self,
        epsilon_0: float,
        alpha: float,
        rng: t.Optional[np.random.Generator] = None,
    ):
        super().__init__(rng)

        self._epsilon_0 = epsilon_0
        assert 0 <= alpha <= 1
        self._alpha = alpha

        self._t = -1

    def should_play_random(self) -> bool:
        self._t += 1
        epsilon = self._epsilon_0 * (self._alpha ** self._t)
        return self._rng.random() < epsilon


def plot_p_best_bandit_chosen_vs_rounds_in_game(
    results: t.List[SeasonResults], labels: t.List[str], **plot_args: t.Dict[str, t.Any]
) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=plot_args.get("figsize", (15, 10)))
    for result, label in zip(results, labels):
        plt.plot(np.mean(result.best_chosen_record, axis=0), label=label)

    plt.title("Probability best bandit was chosen over rounds in a game")
    plt.xlabel("Rounds")
    plt.ylabel("Probability of Optimal Action")
    plt.ylim((0, 1))
    plt.grid(which="major")
    plt.show()
