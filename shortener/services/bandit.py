"""
Multi-Armed Bandit Optimization.
Dynamic traffic allocation using Thompson Sampling and UCB algorithms.
"""
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

from django.core.cache import cache
from django.utils import timezone

# Optional dependency
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class Arm:
    """Represents a single arm (variant) in the bandit."""
    id: int
    name: str
    successes: int = 0
    failures: int = 0
    total_reward: float = 0.0

    @property
    def trials(self) -> int:
        return self.successes + self.failures

    @property
    def conversion_rate(self) -> float:
        return self.successes / self.trials if self.trials > 0 else 0.0

    @property
    def average_reward(self) -> float:
        return self.total_reward / self.trials if self.trials > 0 else 0.0


class BanditAlgorithm(ABC):
    """Base class for bandit algorithms."""

    @abstractmethod
    def select_arm(self, arms: List[Arm]) -> Arm:
        """Select which arm to pull."""
        pass

    @abstractmethod
    def get_allocation(self, arms: List[Arm]) -> Dict[int, float]:
        """Get traffic allocation percentages."""
        pass


class ThompsonSampling(BanditAlgorithm):
    """
    Thompson Sampling algorithm.
    Samples from posterior Beta distributions to select arms.
    """

    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

    def select_arm(self, arms: List[Arm]) -> Arm:
        """
        Select arm by sampling from posterior distributions.
        """
        if not arms:
            raise ValueError("No arms provided")

        best_arm = None
        best_sample = -1

        for arm in arms:
            alpha = self.prior_alpha + arm.successes
            beta = self.prior_beta + arm.failures

            sample = np.random.beta(alpha, beta)

            if sample > best_sample:
                best_sample = sample
                best_arm = arm

        return best_arm

    def get_allocation(self, arms: List[Arm], num_simulations: int = 10000) -> Dict[int, float]:
        """
        Estimate traffic allocation based on probability of being best.
        """
        if not arms:
            return {}

        win_counts = {arm.id: 0 for arm in arms}

        for _ in range(num_simulations):
            samples = {}
            for arm in arms:
                alpha = self.prior_alpha + arm.successes
                beta = self.prior_beta + arm.failures
                samples[arm.id] = np.random.beta(alpha, beta)

            winner = max(samples, key=samples.get)
            win_counts[winner] += 1

        total = sum(win_counts.values())
        return {arm_id: count / total for arm_id, count in win_counts.items()}


class UCB1(BanditAlgorithm):
    """
    Upper Confidence Bound (UCB1) algorithm.
    Balances exploration and exploitation using confidence bounds.
    """

    def __init__(self, exploration_factor: float = 2.0):
        self.c = exploration_factor

    def select_arm(self, arms: List[Arm]) -> Arm:
        """
        Select arm with highest UCB score.
        """
        if not arms:
            raise ValueError("No arms provided")

        total_trials = sum(arm.trials for arm in arms)

        # First, try each arm at least once
        for arm in arms:
            if arm.trials == 0:
                return arm

        best_arm = None
        best_ucb = -1

        for arm in arms:
            exploitation = arm.conversion_rate
            exploration = np.sqrt(self.c * np.log(total_trials) / arm.trials)
            ucb = exploitation + exploration

            if ucb > best_ucb:
                best_ucb = ucb
                best_arm = arm

        return best_arm

    def get_allocation(self, arms: List[Arm]) -> Dict[int, float]:
        """
        Allocate traffic proportionally to UCB scores.
        """
        if not arms:
            return {}

        total_trials = sum(arm.trials for arm in arms) + len(arms)

        ucb_scores = {}
        for arm in arms:
            if arm.trials == 0:
                ucb_scores[arm.id] = float('inf')
            else:
                exploitation = arm.conversion_rate
                exploration = np.sqrt(self.c * np.log(total_trials) / arm.trials)
                ucb_scores[arm.id] = exploitation + exploration

        # Handle infinite scores
        if any(s == float('inf') for s in ucb_scores.values()):
            inf_count = sum(1 for s in ucb_scores.values() if s == float('inf'))
            return {
                arm_id: (1.0 / inf_count if score == float('inf') else 0)
                for arm_id, score in ucb_scores.items()
            }

        total_score = sum(ucb_scores.values())
        return {arm_id: score / total_score for arm_id, score in ucb_scores.items()}


class EpsilonGreedy(BanditAlgorithm):
    """
    Epsilon-Greedy algorithm.
    Exploits best arm most of the time, explores randomly with probability epsilon.
    """

    def __init__(self, epsilon: float = 0.1, decay: float = 0.999):
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.decay = decay
        self.rounds = 0

    def select_arm(self, arms: List[Arm]) -> Arm:
        """
        Select best arm with probability (1-epsilon), random otherwise.
        """
        if not arms:
            raise ValueError("No arms provided")

        self.rounds += 1
        current_epsilon = self.epsilon * (self.decay ** self.rounds)

        if np.random.random() < current_epsilon:
            return np.random.choice(arms)
        else:
            return max(arms, key=lambda a: a.conversion_rate)

    def get_allocation(self, arms: List[Arm]) -> Dict[int, float]:
        """
        Allocate epsilon to exploration, rest to best arm.
        """
        if not arms:
            return {}

        best_arm = max(arms, key=lambda a: a.conversion_rate)
        current_epsilon = self.epsilon * (self.decay ** self.rounds)

        allocation = {}
        explore_share = current_epsilon / len(arms)

        for arm in arms:
            if arm.id == best_arm.id:
                allocation[arm.id] = (1 - current_epsilon) + explore_share
            else:
                allocation[arm.id] = explore_share

        return allocation


class BanditOptimizer:
    """
    High-level optimizer that manages bandit-based A/B tests.
    """

    ALGORITHMS = {
        'thompson': ThompsonSampling,
        'ucb': UCB1,
        'epsilon_greedy': EpsilonGreedy,
    }

    def __init__(self, algorithm: str = 'thompson', **kwargs):
        if algorithm not in self.ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        self.algorithm = self.ALGORITHMS[algorithm](**kwargs)
        self.algorithm_name = algorithm

    def get_variant_for_visitor(self, test_id: int) -> Dict:
        """
        Select variant for a visitor using bandit algorithm.
        """
        from shortener.models import ABTest, ABTestVariant

        try:
            test = ABTest.objects.get(pk=test_id)
        except ABTest.DoesNotExist:
            return {'error': 'Test not found'}

        if test.status != 'running':
            return {'error': 'Test is not running'}

        # Build arms from variants
        arms = []
        for variant in test.variants.all():
            visitors = variant.get_visitors()
            conversions = variant.get_conversions()

            arm = Arm(
                id=variant.id,
                name=variant.name,
                successes=conversions,
                failures=visitors - conversions,
                total_reward=float(variant.clicks.aggregate(
                    total=models.Sum('conversion_value')
                )['total'] or 0)
            )
            arms.append(arm)

        if not arms:
            return {'error': 'No variants found'}

        # Select arm
        selected_arm = self.algorithm.select_arm(arms)

        # Get the variant
        selected_variant = ABTestVariant.objects.get(pk=selected_arm.id)

        return {
            'variant_id': selected_variant.id,
            'variant_name': selected_variant.name,
            'destination_url': selected_variant.destination_url,
            'algorithm': self.algorithm_name
        }

    def get_current_allocation(self, test_id: int) -> Dict:
        """
        Get current traffic allocation for all variants.
        """
        from shortener.models import ABTest

        try:
            test = ABTest.objects.get(pk=test_id)
        except ABTest.DoesNotExist:
            return {'error': 'Test not found'}

        arms = []
        variant_names = {}

        for variant in test.variants.all():
            visitors = variant.get_visitors()
            conversions = variant.get_conversions()

            arm = Arm(
                id=variant.id,
                name=variant.name,
                successes=conversions,
                failures=visitors - conversions
            )
            arms.append(arm)
            variant_names[variant.id] = variant.name

        allocation = self.algorithm.get_allocation(arms)

        return {
            'test_id': test_id,
            'algorithm': self.algorithm_name,
            'allocation': {
                variant_names[arm_id]: round(pct * 100, 1)
                for arm_id, pct in allocation.items()
            }
        }

    def get_regret_analysis(self, test_id: int) -> Dict:
        """
        Calculate cumulative regret for the test.
        Regret = difference between optimal and actual performance.
        """
        from shortener.models import ABTest, ClickData

        try:
            test = ABTest.objects.get(pk=test_id)
        except ABTest.DoesNotExist:
            return {'error': 'Test not found'}

        # Get all clicks assigned to variants
        clicks = ClickData.objects.filter(
            ab_test=test
        ).order_by('timestamp')

        if not clicks.exists():
            return {'cumulative_regret': 0, 'average_regret': 0}

        # Find best variant (hindsight)
        best_rate = 0
        for variant in test.variants.all():
            rate = variant.get_conversion_rate() / 100
            if rate > best_rate:
                best_rate = rate

        # Calculate regret over time
        cumulative_regret = 0
        regret_over_time = []

        for i, click in enumerate(clicks):
            if click.ab_variant:
                variant_rate = click.ab_variant.get_conversion_rate() / 100
                instant_regret = best_rate - variant_rate
                cumulative_regret += instant_regret

                if (i + 1) % 100 == 0:  # Sample every 100 clicks
                    regret_over_time.append({
                        'click_num': i + 1,
                        'cumulative_regret': round(cumulative_regret, 4)
                    })

        return {
            'cumulative_regret': round(cumulative_regret, 4),
            'average_regret': round(cumulative_regret / clicks.count(), 6),
            'regret_over_time': regret_over_time
        }


class AutoOptimizer:
    """
    Automatic optimization that switches between exploration and exploitation.
    """

    def __init__(
        self,
        exploration_threshold: int = 1000,
        exploitation_probability: float = 0.95
    ):
        self.exploration_threshold = exploration_threshold
        self.exploitation_prob = exploitation_probability
        self.thompson = ThompsonSampling()

    def optimize(self, test_id: int) -> Dict:
        """
        Run one optimization step for the test.
        """
        from shortener.models import ABTest, ABTestVariant

        try:
            test = ABTest.objects.get(pk=test_id)
        except ABTest.DoesNotExist:
            return {'error': 'Test not found'}

        # Build arms
        arms = []
        for variant in test.variants.all():
            visitors = variant.get_visitors()
            conversions = variant.get_conversions()

            arms.append(Arm(
                id=variant.id,
                name=variant.name,
                successes=conversions,
                failures=visitors - conversions
            ))

        total_trials = sum(a.trials for a in arms)

        # Determine phase
        if total_trials < self.exploration_threshold:
            phase = 'exploration'
            allocation = {a.id: 1.0 / len(arms) for a in arms}  # Equal split
        else:
            phase = 'exploitation'
            allocation = self.thompson.get_allocation(arms)

            # Check if we have a clear winner
            max_prob = max(allocation.values())
            if max_prob >= self.exploitation_prob:
                phase = 'winner_found'
                winner_id = max(allocation, key=allocation.get)
                winner = ABTestVariant.objects.get(pk=winner_id)

                return {
                    'phase': phase,
                    'winner': winner.name,
                    'winner_probability': round(max_prob * 100, 1),
                    'recommendation': 'Consider stopping the test and implementing winner'
                }

        return {
            'phase': phase,
            'total_samples': total_trials,
            'allocation': {
                next(a.name for a in arms if a.id == arm_id): round(pct * 100, 1)
                for arm_id, pct in allocation.items()
            }
        }

    def should_stop_test(self, test_id: int) -> Tuple[bool, str]:
        """
        Determine if the test should be stopped.
        """
        result = self.optimize(test_id)

        if result.get('phase') == 'winner_found':
            return True, f"Winner found: {result['winner']} ({result['winner_probability']}%)"

        return False, f"Continue testing - current phase: {result.get('phase')}"


# Import models at runtime to avoid circular imports
from django.db import models
