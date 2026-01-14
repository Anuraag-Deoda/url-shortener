"""
URL Shortener Services Package.
Advanced analytics, ML, and optimization services.

All ML/advanced imports are optional and gracefully degrade
if dependencies are not installed.
"""

# Core services (always available)
from .geolocation import GeoLocationService
from .qrcode_service import QRCodeService
from .domain_verification import DomainVerificationService

__all__ = [
    'GeoLocationService',
    'QRCodeService',
    'DomainVerificationService',
]

# Optional ML services - import only if dependencies available
try:
    from .ml_service import TrafficForecaster, AnomalyDetector, BotDetector, ConversionPredictor
    __all__.extend([
        'TrafficForecaster',
        'AnomalyDetector',
        'BotDetector',
        'ConversionPredictor',
    ])
except ImportError:
    pass

try:
    from .bayesian_ab import BayesianABTest, BayesianABTestAnalyzer, SequentialBayesianTest
    __all__.extend([
        'BayesianABTest',
        'BayesianABTestAnalyzer',
        'SequentialBayesianTest',
    ])
except ImportError:
    pass

try:
    from .bandit import ThompsonSampling, UCB1, EpsilonGreedy, BanditOptimizer, AutoOptimizer
    __all__.extend([
        'ThompsonSampling',
        'UCB1',
        'EpsilonGreedy',
        'BanditOptimizer',
        'AutoOptimizer',
    ])
except ImportError:
    pass

try:
    from .forecasting import ForecastingService, ProphetForecaster, ARIMAForecaster
    __all__.extend([
        'ForecastingService',
        'ProphetForecaster',
        'ARIMAForecaster',
    ])
except ImportError:
    pass

try:
    from .bot_detection import BotDetectionService, BotSignatureDetector, MLBotClassifier
    __all__.extend([
        'BotDetectionService',
        'BotSignatureDetector',
        'MLBotClassifier',
    ])
except ImportError:
    pass

try:
    from .pandas_analytics import PandasAnalyticsService, SessionReconstructor, ConversionAttributor
    __all__.extend([
        'PandasAnalyticsService',
        'SessionReconstructor',
        'ConversionAttributor',
    ])
except ImportError:
    pass

try:
    from .conversion_prediction import ConversionPredictionService, LeadScorer
    __all__.extend([
        'ConversionPredictionService',
        'LeadScorer',
    ])
except ImportError:
    pass

try:
    from .nlp_classification import NLPClassificationService, AutoTagger, SpamReferrerDetector
    __all__.extend([
        'NLPClassificationService',
        'AutoTagger',
        'SpamReferrerDetector',
    ])
except ImportError:
    pass
