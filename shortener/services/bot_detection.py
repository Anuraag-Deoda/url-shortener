"""
ML-Based Bot Detection Service.
Advanced bot detection using behavioral analysis and machine learning.
"""
import logging
import hashlib
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
from django.utils import timezone
from django.core.cache import cache
from django.db.models import Count, Avg, F

logger = logging.getLogger(__name__)


@dataclass
class ClickFeatures:
    """Features extracted from click data for ML classification."""
    # Request features
    has_user_agent: bool = True
    user_agent_length: int = 0
    has_accept_language: bool = False
    has_referer: bool = False

    # Behavioral features
    time_since_last_click: float = 0.0
    clicks_per_minute: float = 0.0
    clicks_per_hour: float = 0.0
    clicks_today: int = 0
    unique_urls_today: int = 0
    session_duration: float = 0.0

    # Pattern features
    is_regular_interval: bool = False
    interval_variance: float = 0.0
    hour_of_day: int = 0
    day_of_week: int = 0

    # Known bot indicators
    contains_bot_keyword: bool = False
    is_known_datacenter: bool = False
    is_headless_browser: bool = False

    # Geographic features
    country_frequency: float = 0.0

    def to_vector(self) -> np.ndarray:
        """Convert features to numpy array for ML model."""
        return np.array([
            float(self.has_user_agent),
            self.user_agent_length / 200.0,  # Normalize
            float(self.has_accept_language),
            float(self.has_referer),
            min(self.time_since_last_click, 3600) / 3600.0,
            min(self.clicks_per_minute, 60) / 60.0,
            min(self.clicks_per_hour, 1000) / 1000.0,
            min(self.clicks_today, 100) / 100.0,
            min(self.unique_urls_today, 50) / 50.0,
            min(self.session_duration, 3600) / 3600.0,
            float(self.is_regular_interval),
            min(self.interval_variance, 100) / 100.0,
            self.hour_of_day / 24.0,
            self.day_of_week / 7.0,
            float(self.contains_bot_keyword),
            float(self.is_known_datacenter),
            float(self.is_headless_browser),
            self.country_frequency
        ])


@dataclass
class BotDetectionResult:
    """Result of bot detection analysis."""
    is_bot: bool
    confidence: float
    bot_type: Optional[str] = None
    reasons: List[str] = field(default_factory=list)
    features: Optional[ClickFeatures] = None

    def to_dict(self) -> Dict:
        return {
            'is_bot': self.is_bot,
            'confidence': round(self.confidence, 3),
            'bot_type': self.bot_type,
            'reasons': self.reasons
        }


class BotSignatureDetector:
    """
    Rule-based bot detection using known signatures.
    """

    # Known bot user agent patterns
    BOT_PATTERNS = [
        r'bot', r'crawl', r'spider', r'scrape', r'fetch',
        r'wget', r'curl', r'python-requests', r'axios',
        r'headless', r'phantom', r'selenium', r'puppeteer',
        r'lighthouse', r'pagespeed', r'gtmetrix',
        r'facebook.*bot', r'twitter.*bot', r'linkedin.*bot',
        r'googlebot', r'bingbot', r'yandex', r'baidu',
        r'slurp', r'duckduck', r'archive\.org'
    ]

    # Good bots that we should allow
    GOOD_BOTS = [
        r'googlebot', r'bingbot', r'yandexbot',
        r'slackbot', r'twitterbot', r'facebookexternalhit',
        r'linkedinbot', r'whatsapp', r'telegram'
    ]

    # Headless browser indicators
    HEADLESS_INDICATORS = [
        r'headless', r'phantomjs', r'slimerjs',
        r'puppeteer', r'playwright', r'selenium'
    ]

    # Known datacenter IP ranges (simplified)
    DATACENTER_ASNS = {
        'amazon': ['AS16509', 'AS14618'],
        'google': ['AS15169', 'AS396982'],
        'microsoft': ['AS8075'],
        'digitalocean': ['AS14061'],
        'ovh': ['AS16276'],
        'hetzner': ['AS24940'],
        'linode': ['AS63949'],
        'vultr': ['AS20473']
    }

    def __init__(self):
        self.bot_regex = re.compile(
            '|'.join(self.BOT_PATTERNS),
            re.IGNORECASE
        )
        self.good_bot_regex = re.compile(
            '|'.join(self.GOOD_BOTS),
            re.IGNORECASE
        )
        self.headless_regex = re.compile(
            '|'.join(self.HEADLESS_INDICATORS),
            re.IGNORECASE
        )

    def check_user_agent(self, user_agent: str) -> Tuple[bool, str, List[str]]:
        """
        Check user agent for bot signatures.
        Returns (is_bot, bot_type, reasons)
        """
        if not user_agent:
            return True, 'no_user_agent', ['Missing user agent']

        reasons = []

        # Check for good bots
        if self.good_bot_regex.search(user_agent):
            return True, 'search_engine', ['Identified as search engine bot']

        # Check for headless browsers
        if self.headless_regex.search(user_agent):
            reasons.append('Headless browser detected')
            return True, 'headless_browser', reasons

        # Check for bot patterns
        if self.bot_regex.search(user_agent):
            reasons.append('Bot keyword in user agent')
            return True, 'known_bot', reasons

        # Check for suspicious characteristics
        if len(user_agent) < 20:
            reasons.append('Unusually short user agent')

        if not re.search(r'mozilla|chrome|safari|firefox|edge|opera', user_agent, re.I):
            reasons.append('Missing browser identifier')

        if reasons:
            return True, 'suspicious', reasons

        return False, None, []

    def is_datacenter_ip(self, ip: str, asn: Optional[str] = None) -> bool:
        """
        Check if IP belongs to a known datacenter.
        """
        if asn:
            for provider, asns in self.DATACENTER_ASNS.items():
                if asn in asns:
                    return True

        # Check for private/local IPs
        if ip.startswith(('10.', '172.16.', '192.168.', '127.')):
            return False

        return False


class BehavioralAnalyzer:
    """
    Analyze click behavior patterns for bot detection.
    """

    def __init__(
        self,
        rate_limit_per_minute: int = 10,
        rate_limit_per_hour: int = 100,
        rate_limit_per_day: int = 500
    ):
        self.rate_limit_minute = rate_limit_per_minute
        self.rate_limit_hour = rate_limit_per_hour
        self.rate_limit_day = rate_limit_per_day

    def analyze_click_timing(
        self,
        ip: str,
        session_id: Optional[str] = None
    ) -> Dict:
        """
        Analyze click timing patterns.
        """
        from shortener.models import ClickData

        now = timezone.now()
        minute_ago = now - timedelta(minutes=1)
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)

        # Get recent clicks
        base_query = ClickData.objects.filter(ip_address=ip)

        clicks_minute = base_query.filter(timestamp__gte=minute_ago).count()
        clicks_hour = base_query.filter(timestamp__gte=hour_ago).count()
        clicks_day = base_query.filter(timestamp__gte=day_ago).count()

        # Get click timestamps for interval analysis
        recent_clicks = list(
            base_query.filter(timestamp__gte=hour_ago)
            .order_by('timestamp')
            .values_list('timestamp', flat=True)
        )

        # Calculate interval statistics
        intervals = []
        for i in range(1, len(recent_clicks)):
            interval = (recent_clicks[i] - recent_clicks[i-1]).total_seconds()
            intervals.append(interval)

        interval_variance = np.var(intervals) if len(intervals) > 1 else 0
        is_regular = interval_variance < 1.0 and len(intervals) > 5

        # Unique URLs
        unique_urls = base_query.filter(
            timestamp__gte=day_ago
        ).values('url').distinct().count()

        return {
            'clicks_per_minute': clicks_minute,
            'clicks_per_hour': clicks_hour,
            'clicks_today': clicks_day,
            'unique_urls_today': unique_urls,
            'interval_variance': interval_variance,
            'is_regular_interval': is_regular,
            'exceeds_minute_limit': clicks_minute > self.rate_limit_minute,
            'exceeds_hour_limit': clicks_hour > self.rate_limit_hour,
            'exceeds_day_limit': clicks_day > self.rate_limit_day
        }

    def get_session_behavior(self, session_id: str) -> Dict:
        """
        Analyze session-level behavior.
        """
        from shortener.models import ClickData

        if not session_id:
            return {'session_duration': 0, 'session_clicks': 0}

        clicks = ClickData.objects.filter(
            session_id=session_id
        ).order_by('timestamp')

        if not clicks.exists():
            return {'session_duration': 0, 'session_clicks': 0}

        first_click = clicks.first().timestamp
        last_click = clicks.last().timestamp

        return {
            'session_duration': (last_click - first_click).total_seconds(),
            'session_clicks': clicks.count()
        }


class MLBotClassifier:
    """
    Machine learning based bot classifier.
    Uses ensemble of models for robust detection.
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self._sklearn_available = self._check_sklearn()

    def _check_sklearn(self) -> bool:
        """Check if scikit-learn is available."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            return True
        except ImportError:
            logger.warning("scikit-learn not available, using rule-based detection only")
            return False

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Train the bot detection model.
        """
        if not self._sklearn_available:
            return {'error': 'scikit-learn not available'}

        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            class_weight='balanced'
        )

        self.model.fit(X_scaled, y)

        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5)

        return {
            'accuracy': round(np.mean(cv_scores), 4),
            'std': round(np.std(cv_scores), 4),
            'n_samples': len(y),
            'n_features': X.shape[1],
            'feature_importance': self._get_feature_importance()
        }

    def _get_feature_importance(self) -> Dict:
        """Get feature importance from trained model."""
        if self.model is None:
            return {}

        feature_names = [
            'has_user_agent', 'user_agent_length', 'has_accept_language',
            'has_referer', 'time_since_last_click', 'clicks_per_minute',
            'clicks_per_hour', 'clicks_today', 'unique_urls_today',
            'session_duration', 'is_regular_interval', 'interval_variance',
            'hour_of_day', 'day_of_week', 'contains_bot_keyword',
            'is_known_datacenter', 'is_headless_browser', 'country_frequency'
        ]

        importance = dict(zip(
            feature_names,
            self.model.feature_importances_.tolist()
        ))

        return dict(sorted(importance.items(), key=lambda x: -x[1]))

    def predict(self, features: ClickFeatures) -> Tuple[bool, float]:
        """
        Predict if click is from a bot.
        """
        if self.model is None or not self._sklearn_available:
            # Fallback to rule-based score
            return self._rule_based_predict(features)

        X = features.to_vector().reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        prob = self.model.predict_proba(X_scaled)[0, 1]
        is_bot = prob > 0.5

        return is_bot, prob

    def _rule_based_predict(self, features: ClickFeatures) -> Tuple[bool, float]:
        """
        Rule-based prediction fallback.
        """
        score = 0.0

        # User agent checks
        if not features.has_user_agent:
            score += 0.3
        if features.user_agent_length < 20:
            score += 0.1
        if features.contains_bot_keyword:
            score += 0.4
        if features.is_headless_browser:
            score += 0.5

        # Behavioral checks
        if features.clicks_per_minute > 10:
            score += 0.3
        if features.clicks_per_hour > 100:
            score += 0.2
        if features.is_regular_interval and features.interval_variance < 1:
            score += 0.3

        # Infrastructure checks
        if features.is_known_datacenter:
            score += 0.2

        # Request checks
        if not features.has_accept_language and not features.has_referer:
            score += 0.1

        score = min(score, 1.0)

        return score > 0.5, score

    def save_model(self, path: str):
        """Save trained model to disk."""
        if not self._sklearn_available or self.model is None:
            return

        import joblib
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler
        }, path)

    def load_model(self, path: str) -> bool:
        """Load trained model from disk."""
        if not self._sklearn_available:
            return False

        try:
            import joblib
            data = joblib.load(path)
            self.model = data['model']
            self.scaler = data['scaler']
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False


class FeatureExtractor:
    """
    Extract features from click data for ML model.
    """

    def __init__(self):
        self.signature_detector = BotSignatureDetector()
        self.behavioral_analyzer = BehavioralAnalyzer()

    def extract(
        self,
        ip: str,
        user_agent: str,
        referer: Optional[str] = None,
        accept_language: Optional[str] = None,
        session_id: Optional[str] = None,
        country: Optional[str] = None
    ) -> ClickFeatures:
        """
        Extract all features for a click.
        """
        # Get behavioral data
        timing = self.behavioral_analyzer.analyze_click_timing(ip, session_id)
        session = self.behavioral_analyzer.get_session_behavior(session_id) if session_id else {}

        # Check user agent
        is_bot_ua, bot_type, _ = self.signature_detector.check_user_agent(user_agent)

        # Check for bot keywords
        contains_bot = bool(re.search(
            r'bot|crawl|spider|scrape',
            user_agent or '',
            re.I
        ))

        # Check for headless browser
        is_headless = bool(re.search(
            r'headless|phantom|selenium|puppeteer',
            user_agent or '',
            re.I
        ))

        now = timezone.now()

        return ClickFeatures(
            has_user_agent=bool(user_agent),
            user_agent_length=len(user_agent) if user_agent else 0,
            has_accept_language=bool(accept_language),
            has_referer=bool(referer),
            time_since_last_click=timing.get('time_since_last', 0),
            clicks_per_minute=timing.get('clicks_per_minute', 0),
            clicks_per_hour=timing.get('clicks_per_hour', 0),
            clicks_today=timing.get('clicks_today', 0),
            unique_urls_today=timing.get('unique_urls_today', 0),
            session_duration=session.get('session_duration', 0),
            is_regular_interval=timing.get('is_regular_interval', False),
            interval_variance=timing.get('interval_variance', 0),
            hour_of_day=now.hour,
            day_of_week=now.weekday(),
            contains_bot_keyword=contains_bot,
            is_known_datacenter=self.signature_detector.is_datacenter_ip(ip),
            is_headless_browser=is_headless,
            country_frequency=0.0  # Would need global stats
        )


class FingerprintGenerator:
    """
    Generate browser fingerprints for bot detection.
    """

    def generate(
        self,
        user_agent: str,
        accept_language: str,
        accept_encoding: str,
        screen_resolution: Optional[str] = None,
        timezone_offset: Optional[int] = None,
        plugins: Optional[List[str]] = None
    ) -> str:
        """
        Generate a fingerprint hash from browser characteristics.
        """
        components = [
            user_agent or '',
            accept_language or '',
            accept_encoding or '',
            screen_resolution or '',
            str(timezone_offset) if timezone_offset else '',
            ','.join(plugins) if plugins else ''
        ]

        fingerprint_string = '|'.join(components)

        return hashlib.sha256(fingerprint_string.encode()).hexdigest()[:32]

    def analyze_fingerprint_consistency(
        self,
        ip: str,
        fingerprint: str
    ) -> Dict:
        """
        Check if IP has consistent fingerprints over time.
        """
        cache_key = f"fingerprints:{ip}"
        fingerprints = cache.get(cache_key, [])

        fingerprints.append({
            'hash': fingerprint,
            'time': timezone.now().isoformat()
        })

        # Keep last 100 fingerprints
        fingerprints = fingerprints[-100:]
        cache.set(cache_key, fingerprints, 86400)  # 24 hours

        # Analyze
        unique_fps = len(set(fp['hash'] for fp in fingerprints))
        total_fps = len(fingerprints)

        return {
            'total_fingerprints': total_fps,
            'unique_fingerprints': unique_fps,
            'consistency_ratio': (total_fps - unique_fps + 1) / total_fps if total_fps > 0 else 1,
            'is_suspicious': unique_fps > total_fps * 0.5 and total_fps > 10
        }


class BotDetectionService:
    """
    High-level bot detection service.
    Combines all detection methods.
    """

    def __init__(self):
        self.signature_detector = BotSignatureDetector()
        self.behavioral_analyzer = BehavioralAnalyzer()
        self.ml_classifier = MLBotClassifier()
        self.feature_extractor = FeatureExtractor()
        self.fingerprint_generator = FingerprintGenerator()

    def analyze_click(
        self,
        ip: str,
        user_agent: str,
        referer: Optional[str] = None,
        accept_language: Optional[str] = None,
        session_id: Optional[str] = None,
        country: Optional[str] = None
    ) -> BotDetectionResult:
        """
        Comprehensive bot detection for a single click.
        """
        reasons = []
        bot_type = None
        confidence = 0.0

        # 1. Signature-based detection
        is_bot_sig, sig_type, sig_reasons = self.signature_detector.check_user_agent(user_agent)
        if is_bot_sig:
            reasons.extend(sig_reasons)
            bot_type = sig_type
            confidence = 0.9 if sig_type != 'suspicious' else 0.6

        # 2. Behavioral analysis
        timing = self.behavioral_analyzer.analyze_click_timing(ip, session_id)

        if timing.get('exceeds_minute_limit'):
            reasons.append(f"Exceeded rate limit: {timing['clicks_per_minute']} clicks/minute")
            confidence = max(confidence, 0.8)

        if timing.get('is_regular_interval'):
            reasons.append("Suspicious regular click intervals")
            confidence = max(confidence, 0.7)

        # 3. ML-based detection
        features = self.feature_extractor.extract(
            ip=ip,
            user_agent=user_agent,
            referer=referer,
            accept_language=accept_language,
            session_id=session_id,
            country=country
        )

        ml_is_bot, ml_confidence = self.ml_classifier.predict(features)

        if ml_is_bot and ml_confidence > confidence:
            confidence = ml_confidence
            reasons.append(f"ML model confidence: {ml_confidence:.1%}")

        # Determine final result
        is_bot = confidence > 0.5

        if not bot_type and is_bot:
            if features.is_headless_browser:
                bot_type = 'headless_browser'
            elif features.is_regular_interval:
                bot_type = 'automated_tool'
            elif features.clicks_per_minute > 10:
                bot_type = 'rate_abuser'
            else:
                bot_type = 'suspicious'

        return BotDetectionResult(
            is_bot=is_bot,
            confidence=confidence,
            bot_type=bot_type,
            reasons=reasons,
            features=features
        )

    def get_training_data(self, days_back: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from historical clicks.
        Uses flagged fraud data for labeling.
        """
        from shortener.models import ClickData, FraudAlert

        start_date = timezone.now() - timedelta(days=days_back)

        # Get all clicks
        clicks = ClickData.objects.filter(
            timestamp__gte=start_date
        ).select_related('url')

        # Get fraud alerts for labeling
        fraud_ips = set(
            FraudAlert.objects.filter(
                created_at__gte=start_date,
                status='confirmed'
            ).values_list('ip_address', flat=True)
        )

        X = []
        y = []

        for click in clicks:
            features = self.feature_extractor.extract(
                ip=click.ip_address,
                user_agent=click.user_agent or '',
                referer=click.referrer,
                session_id=click.session_id,
                country=click.country_code
            )

            X.append(features.to_vector())

            # Label based on fraud alerts
            is_fraud = click.ip_address in fraud_ips
            y.append(1 if is_fraud else 0)

        return np.array(X), np.array(y)

    def retrain_model(self) -> Dict:
        """
        Retrain the ML model with recent data.
        """
        X, y = self.get_training_data()

        if len(y) < 100:
            return {'error': 'Insufficient training data'}

        # Need some positive labels
        if np.sum(y) < 10:
            logger.warning("Few positive samples, model may not perform well")

        result = self.ml_classifier.train(X, y)

        # Save model
        self.ml_classifier.save_model('bot_detection_model.joblib')

        return result

    def get_stats(self, days_back: int = 7) -> Dict:
        """
        Get bot detection statistics.
        """
        from shortener.models import ClickData, FraudAlert

        start_date = timezone.now() - timedelta(days=days_back)

        total_clicks = ClickData.objects.filter(
            timestamp__gte=start_date
        ).count()

        fraud_alerts = FraudAlert.objects.filter(
            created_at__gte=start_date
        )

        return {
            'total_clicks': total_clicks,
            'flagged_clicks': fraud_alerts.filter(status='pending').count(),
            'confirmed_bots': fraud_alerts.filter(status='confirmed').count(),
            'false_positives': fraud_alerts.filter(status='false_positive').count(),
            'detection_rate': round(
                fraud_alerts.count() / total_clicks * 100, 2
            ) if total_clicks > 0 else 0
        }
