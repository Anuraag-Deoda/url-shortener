"""
URL Shortener Services Package.
Advanced analytics, ML, and optimization services.
"""

from .geolocation import GeoLocationService
from .qrcode_service import QRCodeService
from .domain_verification import DomainVerificationService
from .ml_service import TrafficForecaster, AnomalyDetector, BotDetector, ConversionPredictor
from .bayesian_ab import BayesianABTest, BayesianABTestAnalyzer, SequentialBayesianTest
from .bandit import ThompsonSampling, UCB1, EpsilonGreedy, BanditOptimizer, AutoOptimizer
from .forecasting import ForecastingService, ProphetForecaster, ARIMAForecaster
from .bot_detection import BotDetectionService, BotSignatureDetector, MLBotClassifier
from .pandas_analytics import PandasAnalyticsService, SessionReconstructor, ConversionAttributor
from .conversion_prediction import ConversionPredictionService, LeadScorer
from .nlp_classification import NLPClassificationService, AutoTagger, SpamReferrerDetector

__all__ = [
    # Core services
    'GeoLocationService',
    'QRCodeService',
    'DomainVerificationService',

    # ML services
    'TrafficForecaster',
    'AnomalyDetector',
    'BotDetector',
    'ConversionPredictor',

    # Bayesian A/B Testing
    'BayesianABTest',
    'BayesianABTestAnalyzer',
    'SequentialBayesianTest',

    # Multi-Armed Bandits
    'ThompsonSampling',
    'UCB1',
    'EpsilonGreedy',
    'BanditOptimizer',
    'AutoOptimizer',

    # Forecasting
    'ForecastingService',
    'ProphetForecaster',
    'ARIMAForecaster',

    # Bot Detection
    'BotDetectionService',
    'BotSignatureDetector',
    'MLBotClassifier',

    # Pandas Analytics
    'PandasAnalyticsService',
    'SessionReconstructor',
    'ConversionAttributor',

    # Conversion Prediction
    'ConversionPredictionService',
    'LeadScorer',

    # NLP Classification
    'NLPClassificationService',
    'AutoTagger',
    'SpamReferrerDetector',
]
