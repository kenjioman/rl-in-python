import abc
from concurrent.futures import ProcessPoolExecutor
import dataclasses as dc
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


@dc.dataclass(frozen=True)
class GangOfBanditsInputs:
    p_actuals: t.List[float]
    n_bandits: int = dc.field(init=False)
    p_estimates: t.Optional[t.List[float]] = None
    rng: np.random.Generator = None
    seed: int = None

    def __post_init__(self):
        object.__setattr__(self, 'n_bandits', len(self.p_actuals))
        assert (self.p_estimates is None) or (len(self.p_estimates) == self.n_bandits)


class GangOfBandits:
    def __init__(self, args: GangOfBanditsInputs) -> None:
        self._store_params(args)

        self._bandits: np.ndarray[Bandit] = self._create_bandits()
        self._best_bandit: t.Set[int] = self._get_best_bandits()

        self._p_estimates: np.ndarray[float] = self._initialize_p_estimates()
        self._bandit_draws: np.ndarray[int] = np.zeros(self._n_bandits)

    def _store_params(self, args: GangOfBanditsInputs) -> None:
        self._rng = utils.get_rng_if_needed(args.rng, args.seed)
        self._p_actuals = args.p_actuals
        self._n_bandits = args.n_bandits
        self._p_initial_estimate = args.p_estimates

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


class AbstractIfRandomPicker(abc.ABC):
    @abc.abstractmethod
    def should_play_random(self) -> bool:
        pass


class BanditGames:
    def __init__(self, gob_args: GangOfBanditsInputs, rounds: int, games: int):
        self._gob_args = gob_args
        self._rounds = rounds
        self._games = games

    def play_game(self, strategy: AbstractIfRandomPicker) -> t.Dict[str, np.ndarray]:
        gang_of_bandits = GangOfBandits(self._gob_args)
        record = {
            'bandit_chosen': np.zeros(self._rounds),
            'bandit_is_best': np.zeros(self._rounds),
            'reward_given': np.zeros(self._rounds)
        }

        for _round in range(self._rounds):
            if strategy.should_play_random():
                self._record_round_results(record, _round, gang_of_bandits.play_random_bandit())
            else:
                self._record_round_results(record, _round, gang_of_bandits.play_current_best_bandit())

        return record

    @staticmethod
    def _record_round_results(record: t.Dict[str, np.ndarray[int]], _round: int, results: t.Tuple[int, bool, bool]) -> None:
        chosen, is_best, reward = results
        record['bandit_chosen'][_round] = chosen
        record['bandit_is_best'][_round] = is_best
        record['reward_given'][_round] = reward

    def play_season(self, strategy_class: t.Type[AbstractIfRandomPicker], **strategy_args: t.Dict[str, t.Any]) -> t.Dict[str, np.ndarray]:
        record = {
            'best_chosen_record': [],
            'earnings_record': []
        }

        for game in range(self._games):
            strategy = strategy_class(**strategy_args)
            self._record_game_results(record, self.play_game(strategy))

        record = {k: np.stack(v) for k, v in record.items()}
        return record

    def _record_game_results(self, record: t.Dict[str, t.List[np.ndarray]], results: t.Dict[str, np.ndarray]) -> None:
        record['best_chosen_record'].append(results['bandit_is_best'])
        record['earnings_record'].append(results['reward_given'])

    def play_season_parallel(self,
                             strategy_class: t.Type[AbstractIfRandomPicker],
                             **strategy_args: t.Dict[str, t.Any]
                             ) -> t.Dict[str, np.ndarray]:
        record = {
            'best_chosen_record': [],
            'earnings_record': []
        }

        with ProcessPoolExecutor() as executor:
            results = executor.map(self.play_game, [strategy_class(**strategy_args) for _ in range(self._games)])
            for result in results:
                self._record_game_results(record, result)

        record = {k: np.stack(v) for k, v in record.items()}
        return record


class Greedy(AbstractIfRandomPicker):
    def should_play_random(self) -> bool:
        return False


class EpsilonGreedy(AbstractIfRandomPicker):
    def __init__(self, epsilon: float, rng: t.Optional[np.random.Generator] = None, seed: t.Optional[int] = None):
        self._p = epsilon
        self._rng = utils.get_rng_if_needed(rng, seed)

    def should_play_random(self) -> bool:
        return self._rng.random() < self._p

class ExponentialDecayEpsilonGreedy(AbstractIfRandomPicker):
    def __init__(self, epsilon_0: float, alpha: float, rng: t.Optional[np.random.Generator] = None, seed: t.Optional[int] = None):
        self._rng = utils.get_rng_if_needed(rng, seed)

        self._epsilon_0 = epsilon_0
        assert 0 <= alpha <= 1
        self._alpha = alpha

        self._t = -1

    def should_play_random(self) -> bool:
        self._t += 1
        epsilon = self._epsilon_0 * (self._alpha ** self._t)
        return self._rng.random() < epsilon
