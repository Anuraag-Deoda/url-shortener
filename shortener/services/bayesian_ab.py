"""
Bayesian A/B Testing Service.
Provides probability distributions, credible intervals, and expected loss calculations.
"""
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from django.utils import timezone

# Optional dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    stats = None
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class VariantStats:
    """Statistics for a single variant."""
    name: str
    visitors: int
    conversions: int
    revenue: float = 0.0
    is_control: bool = False


class BayesianABTest:
    """
    Bayesian A/B testing with Beta-Binomial model.
    """

    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        """
        Initialize with prior parameters.
        Beta(1, 1) is uniform prior (no prior knowledge).
        """
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.num_samples = 100000

    def get_posterior(self, successes: int, trials: int) -> Tuple[float, float]:
        """
        Calculate posterior Beta distribution parameters.
        Posterior = Beta(alpha + successes, beta + failures)
        """
        alpha = self.prior_alpha + successes
        beta = self.prior_beta + (trials - successes)
        return alpha, beta

    def sample_posterior(self, successes: int, trials: int) -> np.ndarray:
        """
        Draw samples from the posterior distribution.
        """
        alpha, beta = self.get_posterior(successes, trials)
        return np.random.beta(alpha, beta, size=self.num_samples)

    def probability_to_beat(self, variant_a: VariantStats, variant_b: VariantStats) -> float:
        """
        Calculate probability that variant A beats variant B.
        P(A > B) using Monte Carlo simulation.
        """
        samples_a = self.sample_posterior(variant_a.conversions, variant_a.visitors)
        samples_b = self.sample_posterior(variant_b.conversions, variant_b.visitors)

        probability = np.mean(samples_a > samples_b)
        return float(probability)

    def probability_to_beat_all(self, variants: List[VariantStats]) -> Dict[str, float]:
        """
        Calculate probability of each variant being the best.
        """
        if len(variants) < 2:
            return {}

        # Sample all variants
        samples = {}
        for v in variants:
            samples[v.name] = self.sample_posterior(v.conversions, v.visitors)

        # Count wins for each variant
        results = {}
        sample_matrix = np.array([samples[v.name] for v in variants])
        best_indices = np.argmax(sample_matrix, axis=0)

        for i, v in enumerate(variants):
            results[v.name] = float(np.mean(best_indices == i))

        return results

    def expected_loss(self, variant: VariantStats, best_variant: VariantStats) -> float:
        """
        Calculate expected loss if choosing this variant over the best.
        E[max(0, best - variant)]
        """
        samples_variant = self.sample_posterior(variant.conversions, variant.visitors)
        samples_best = self.sample_posterior(best_variant.conversions, best_variant.visitors)

        loss = np.maximum(0, samples_best - samples_variant)
        return float(np.mean(loss))

    def credible_interval(
        self,
        successes: int,
        trials: int,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate credible interval for conversion rate.
        """
        alpha, beta = self.get_posterior(successes, trials)

        lower = (1 - confidence) / 2
        upper = 1 - lower

        ci_lower = stats.beta.ppf(lower, alpha, beta)
        ci_upper = stats.beta.ppf(upper, alpha, beta)

        return float(ci_lower), float(ci_upper)

    def relative_uplift_distribution(
        self,
        treatment: VariantStats,
        control: VariantStats
    ) -> Dict:
        """
        Calculate the distribution of relative uplift.
        (treatment - control) / control
        """
        samples_treatment = self.sample_posterior(treatment.conversions, treatment.visitors)
        samples_control = self.sample_posterior(control.conversions, control.visitors)

        # Avoid division by zero
        samples_control = np.maximum(samples_control, 1e-10)

        uplift = (samples_treatment - samples_control) / samples_control

        return {
            'mean_uplift': float(np.mean(uplift)),
            'median_uplift': float(np.median(uplift)),
            'std_uplift': float(np.std(uplift)),
            'ci_lower': float(np.percentile(uplift, 2.5)),
            'ci_upper': float(np.percentile(uplift, 97.5)),
            'prob_positive': float(np.mean(uplift > 0)),
            'prob_10pct_lift': float(np.mean(uplift > 0.1)),
        }

    def risk_of_choosing(self, variant: VariantStats, all_variants: List[VariantStats]) -> Dict:
        """
        Calculate risk metrics for choosing a specific variant.
        """
        # Find best variant by current conversion rate
        best = max(all_variants, key=lambda v: v.conversions / max(v.visitors, 1))

        # Expected loss
        exp_loss = self.expected_loss(variant, best)

        # Probability of being best
        prob_best = self.probability_to_beat_all(all_variants).get(variant.name, 0)

        # Value at risk (95th percentile of loss)
        samples_variant = self.sample_posterior(variant.conversions, variant.visitors)
        samples_best = self.sample_posterior(best.conversions, best.visitors)
        losses = np.maximum(0, samples_best - samples_variant)
        var_95 = float(np.percentile(losses, 95))

        return {
            'expected_loss': exp_loss,
            'probability_best': prob_best,
            'value_at_risk_95': var_95,
            'risk_level': 'low' if exp_loss < 0.005 else 'medium' if exp_loss < 0.02 else 'high'
        }


class BayesianABTestAnalyzer:
    """
    High-level analyzer for A/B tests using Bayesian methods.
    """

    def __init__(self):
        self.bayesian = BayesianABTest()

    def analyze_test(self, test_id: int) -> Dict:
        """
        Perform comprehensive Bayesian analysis of an A/B test.
        """
        from shortener.models import ABTest, ABTestVariant

        try:
            test = ABTest.objects.get(pk=test_id)
        except ABTest.DoesNotExist:
            return {'error': 'Test not found'}

        variants_qs = test.variants.all()
        if variants_qs.count() < 2:
            return {'error': 'Need at least 2 variants'}

        # Build variant stats
        variants = []
        control = None

        for v in variants_qs:
            stats = VariantStats(
                name=v.name,
                visitors=v.get_visitors(),
                conversions=v.get_conversions(),
                revenue=float(v.clicks.aggregate(
                    total=models.Sum('conversion_value')
                )['total'] or 0),
                is_control=v.is_control
            )
            variants.append(stats)
            if v.is_control:
                control = stats

        if not control:
            control = variants[0]  # Use first variant as control

        # Calculate probabilities
        prob_best = self.bayesian.probability_to_beat_all(variants)

        # Analyze each variant
        results = []
        for v in variants:
            cr = v.conversions / max(v.visitors, 1)
            ci_lower, ci_upper = self.bayesian.credible_interval(
                v.conversions, v.visitors
            )

            variant_result = {
                'name': v.name,
                'is_control': v.is_control,
                'visitors': v.visitors,
                'conversions': v.conversions,
                'conversion_rate': round(cr * 100, 2),
                'credible_interval': {
                    'lower': round(ci_lower * 100, 2),
                    'upper': round(ci_upper * 100, 2)
                },
                'probability_best': round(prob_best.get(v.name, 0) * 100, 1),
                'risk': self.bayesian.risk_of_choosing(v, variants)
            }

            # Calculate uplift vs control
            if not v.is_control and control:
                uplift = self.bayesian.relative_uplift_distribution(v, control)
                variant_result['uplift'] = {
                    'mean': round(uplift['mean_uplift'] * 100, 1),
                    'ci_lower': round(uplift['ci_lower'] * 100, 1),
                    'ci_upper': round(uplift['ci_upper'] * 100, 1),
                    'prob_positive': round(uplift['prob_positive'] * 100, 1)
                }

            results.append(variant_result)

        # Determine winner and recommendation
        best_variant = max(results, key=lambda x: x['probability_best'])
        recommendation = self._get_recommendation(results, best_variant)

        return {
            'test_name': test.name,
            'test_status': test.status,
            'total_visitors': sum(v.visitors for v in variants),
            'variants': results,
            'winner': best_variant['name'] if best_variant['probability_best'] > 95 else None,
            'recommendation': recommendation,
            'can_stop_early': self._can_stop_early(results)
        }

    def _get_recommendation(self, results: List[Dict], best: Dict) -> Dict:
        """Generate recommendation based on analysis."""
        prob_best = best['probability_best']
        exp_loss = best['risk']['expected_loss']

        if prob_best >= 95 and exp_loss < 0.005:
            return {
                'action': 'implement',
                'confidence': 'high',
                'message': f"Implement {best['name']} - {prob_best}% probability of being best with minimal risk"
            }
        elif prob_best >= 80:
            return {
                'action': 'monitor',
                'confidence': 'medium',
                'message': f"{best['name']} is leading but continue testing for higher confidence"
            }
        else:
            return {
                'action': 'continue',
                'confidence': 'low',
                'message': "No clear winner yet - continue collecting data"
            }

    def _can_stop_early(self, results: List[Dict]) -> Dict:
        """
        Determine if test can be stopped early.
        Based on probability of best exceeding threshold.
        """
        max_prob = max(r['probability_best'] for r in results)
        min_visitors = min(r['visitors'] for r in results)

        # Early stopping criteria
        can_stop = max_prob >= 99 or (max_prob >= 95 and min_visitors >= 1000)

        return {
            'can_stop': can_stop,
            'reason': 'High confidence achieved' if can_stop else 'More data needed',
            'confidence_level': max_prob
        }

    def compare_to_frequentist(self, test_id: int) -> Dict:
        """
        Compare Bayesian results to frequentist analysis.
        """
        from shortener.models import ABTest
        from shortener.services.analytics_service import ABTestAnalyticsService

        # Bayesian analysis
        bayesian_results = self.analyze_test(test_id)

        # Frequentist analysis
        try:
            test = ABTest.objects.get(pk=test_id)
            freq_results = ABTestAnalyticsService.get_test_results(test)
        except ABTest.DoesNotExist:
            return {'error': 'Test not found'}

        return {
            'bayesian': {
                'winner': bayesian_results.get('winner'),
                'recommendation': bayesian_results.get('recommendation'),
                'probability_best': max(
                    v['probability_best'] for v in bayesian_results.get('variants', [])
                ) if bayesian_results.get('variants') else 0
            },
            'frequentist': {
                'winner': freq_results.get('winner'),
                'is_significant': freq_results.get('is_significant'),
                'confidence_level': test.confidence_level * 100
            },
            'agreement': bayesian_results.get('winner') == freq_results.get('winner')
        }


class SequentialBayesianTest:
    """
    Sequential testing with optional stopping.
    Monitors test progress and determines when to stop.
    """

    def __init__(
        self,
        min_samples_per_variant: int = 100,
        probability_threshold: float = 0.95,
        expected_loss_threshold: float = 0.01
    ):
        self.min_samples = min_samples_per_variant
        self.prob_threshold = probability_threshold
        self.loss_threshold = expected_loss_threshold
        self.bayesian = BayesianABTest()

    def should_stop(self, variants: List[VariantStats]) -> Tuple[bool, str]:
        """
        Determine if the test should be stopped.
        Returns (should_stop, reason).
        """
        # Check minimum samples
        min_visitors = min(v.visitors for v in variants)
        if min_visitors < self.min_samples:
            return False, f"Minimum samples not reached ({min_visitors}/{self.min_samples})"

        # Calculate probabilities
        prob_best = self.bayesian.probability_to_beat_all(variants)
        max_prob = max(prob_best.values())

        if max_prob >= self.prob_threshold:
            winner = max(prob_best, key=prob_best.get)

            # Check expected loss
            winner_variant = next(v for v in variants if v.name == winner)
            best_other = max(
                (v for v in variants if v.name != winner),
                key=lambda x: x.conversions / max(x.visitors, 1)
            )
            exp_loss = self.bayesian.expected_loss(winner_variant, best_other)

            if exp_loss <= self.loss_threshold:
                return True, f"Winner found: {winner} ({max_prob*100:.1f}% confidence)"

        return False, "No conclusive winner yet"

    def get_required_samples(
        self,
        baseline_rate: float,
        minimum_effect: float,
        desired_probability: float = 0.95
    ) -> int:
        """
        Estimate required samples using simulation.
        """
        # Simulate test outcomes
        samples_needed = []

        for _ in range(100):  # 100 simulations
            n = 100
            while n < 100000:
                # Simulate control
                control_conv = np.random.binomial(n, baseline_rate)

                # Simulate treatment with effect
                treatment_rate = baseline_rate * (1 + minimum_effect)
                treatment_conv = np.random.binomial(n, treatment_rate)

                # Check probability
                control = VariantStats('control', n, control_conv, is_control=True)
                treatment = VariantStats('treatment', n, treatment_conv)

                prob = self.bayesian.probability_to_beat(treatment, control)

                if prob >= desired_probability:
                    samples_needed.append(n)
                    break

                n += 100

        return int(np.median(samples_needed)) if samples_needed else 10000


# Import models at runtime to avoid circular imports
from django.db import models
