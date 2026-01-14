"""
Conversion Prediction Service.
ML-based visitor scoring and conversion probability prediction.
"""
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

from django.utils import timezone
from django.core.cache import cache
from django.db.models import Count, Sum, Avg, Q

# Optional dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class VisitorFeatures:
    """Features for visitor conversion prediction."""
    # Session features
    session_duration: float = 0.0
    pages_viewed: int = 0
    time_on_page: float = 0.0
    scroll_depth: float = 0.0

    # Engagement features
    clicks_in_session: int = 0
    unique_urls_viewed: int = 0
    return_visitor: bool = False
    visits_count: int = 1

    # Traffic source features
    is_direct: bool = False
    is_organic: bool = False
    is_paid: bool = False
    is_social: bool = False
    is_referral: bool = False

    # Device features
    is_mobile: bool = False
    is_tablet: bool = False
    is_desktop: bool = True

    # Time features
    hour_of_day: int = 0
    day_of_week: int = 0
    is_weekend: bool = False
    is_business_hours: bool = False

    # Historical features
    previous_conversions: int = 0
    avg_time_to_convert: float = 0.0
    historical_conversion_rate: float = 0.0

    # Page features
    viewed_high_intent_page: bool = False
    viewed_pricing: bool = False

    def to_vector(self) -> np.ndarray:
        """Convert to feature vector."""
        return np.array([
            min(self.session_duration, 3600) / 3600.0,
            min(self.pages_viewed, 20) / 20.0,
            min(self.time_on_page, 600) / 600.0,
            self.scroll_depth,
            min(self.clicks_in_session, 50) / 50.0,
            min(self.unique_urls_viewed, 10) / 10.0,
            float(self.return_visitor),
            min(self.visits_count, 10) / 10.0,
            float(self.is_direct),
            float(self.is_organic),
            float(self.is_paid),
            float(self.is_social),
            float(self.is_referral),
            float(self.is_mobile),
            float(self.is_tablet),
            float(self.is_desktop),
            self.hour_of_day / 24.0,
            self.day_of_week / 7.0,
            float(self.is_weekend),
            float(self.is_business_hours),
            min(self.previous_conversions, 5) / 5.0,
            min(self.avg_time_to_convert, 86400) / 86400.0,
            self.historical_conversion_rate,
            float(self.viewed_high_intent_page),
            float(self.viewed_pricing)
        ])


@dataclass
class PredictionResult:
    """Result of conversion prediction."""
    probability: float
    score: int  # 1-100 lead score
    segment: str  # hot, warm, cold
    recommended_actions: List[str]
    features: Optional[VisitorFeatures] = None

    def to_dict(self) -> Dict:
        return {
            'probability': round(self.probability, 4),
            'score': self.score,
            'segment': self.segment,
            'recommended_actions': self.recommended_actions
        }


class FeatureExtractor:
    """
    Extract features for conversion prediction.
    """

    HIGH_INTENT_PATTERNS = ['pricing', 'demo', 'trial', 'signup', 'register', 'checkout', 'buy']

    def extract_from_session(
        self,
        ip_address: str,
        session_id: Optional[str] = None
    ) -> VisitorFeatures:
        """
        Extract features from session data.
        """
        from shortener.models import ClickData

        now = timezone.now()

        # Get session clicks
        if session_id:
            session_clicks = ClickData.objects.filter(session_id=session_id)
        else:
            # Reconstruct session from IP
            recent_window = now - timedelta(minutes=30)
            session_clicks = ClickData.objects.filter(
                ip_address=ip_address,
                timestamp__gte=recent_window
            )

        # Historical data
        all_clicks = ClickData.objects.filter(ip_address=ip_address)

        # Calculate session metrics
        session_data = list(session_clicks.order_by('timestamp').values(
            'timestamp', 'url__original_url', 'referrer', 'device_type', 'is_conversion'
        ))

        if not session_data:
            return self._default_features(now)

        # Session duration
        if len(session_data) > 1:
            duration = (session_data[-1]['timestamp'] - session_data[0]['timestamp']).total_seconds()
        else:
            duration = 0

        # Unique URLs
        unique_urls = len(set(d['url__original_url'] for d in session_data if d['url__original_url']))

        # Traffic source
        first_referrer = session_data[0].get('referrer', '') or ''
        traffic_source = self._classify_traffic_source(first_referrer)

        # Device
        device = session_data[0].get('device_type', 'desktop') or 'desktop'

        # Historical metrics
        total_visits = all_clicks.values('session_id').distinct().count()
        previous_conversions = all_clicks.filter(is_conversion=True).count()

        # High intent page
        viewed_high_intent = any(
            any(pattern in (d['url__original_url'] or '').lower() for pattern in self.HIGH_INTENT_PATTERNS)
            for d in session_data
        )

        return VisitorFeatures(
            session_duration=duration,
            pages_viewed=len(session_data),
            clicks_in_session=len(session_data),
            unique_urls_viewed=unique_urls,
            return_visitor=total_visits > 1,
            visits_count=total_visits,
            is_direct=traffic_source == 'direct',
            is_organic=traffic_source == 'organic',
            is_paid=traffic_source == 'paid',
            is_social=traffic_source == 'social',
            is_referral=traffic_source == 'referral',
            is_mobile=device == 'mobile',
            is_tablet=device == 'tablet',
            is_desktop=device == 'desktop',
            hour_of_day=now.hour,
            day_of_week=now.weekday(),
            is_weekend=now.weekday() >= 5,
            is_business_hours=9 <= now.hour <= 17 and now.weekday() < 5,
            previous_conversions=previous_conversions,
            historical_conversion_rate=previous_conversions / total_visits if total_visits > 0 else 0,
            viewed_high_intent_page=viewed_high_intent
        )

    def _classify_traffic_source(self, referrer: str) -> str:
        """Classify traffic source from referrer."""
        if not referrer:
            return 'direct'

        referrer_lower = referrer.lower()

        # Search engines
        search_engines = ['google', 'bing', 'yahoo', 'duckduckgo', 'baidu']
        if any(se in referrer_lower for se in search_engines):
            if 'ads' in referrer_lower or 'gclid' in referrer_lower:
                return 'paid'
            return 'organic'

        # Social networks
        social = ['facebook', 'twitter', 'linkedin', 'instagram', 'tiktok', 'pinterest']
        if any(s in referrer_lower for s in social):
            return 'social'

        return 'referral'

    def _default_features(self, now: datetime) -> VisitorFeatures:
        """Return default features for new visitors."""
        return VisitorFeatures(
            hour_of_day=now.hour,
            day_of_week=now.weekday(),
            is_weekend=now.weekday() >= 5,
            is_business_hours=9 <= now.hour <= 17 and now.weekday() < 5
        )


class ConversionPredictor:
    """
    ML model for conversion prediction.
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self._sklearn_available = self._check_sklearn()

    def _check_sklearn(self) -> bool:
        """Check if sklearn is available."""
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            return True
        except ImportError:
            return False

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Train the conversion prediction model.
        """
        if not self._sklearn_available:
            return {'error': 'scikit-learn not available'}

        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score, train_test_split
        from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train model
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            min_samples_split=10,
            random_state=42
        )

        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        return {
            'auc_roc': round(roc_auc_score(y_test, y_pred_proba), 4),
            'average_precision': round(average_precision_score(y_test, y_pred_proba), 4),
            'n_samples': len(y),
            'positive_rate': round(np.mean(y), 4),
            'feature_importance': self._get_feature_importance()
        }

    def _get_feature_importance(self) -> Dict:
        """Get feature importance."""
        if self.model is None:
            return {}

        feature_names = [
            'session_duration', 'pages_viewed', 'time_on_page', 'scroll_depth',
            'clicks_in_session', 'unique_urls_viewed', 'return_visitor', 'visits_count',
            'is_direct', 'is_organic', 'is_paid', 'is_social', 'is_referral',
            'is_mobile', 'is_tablet', 'is_desktop', 'hour_of_day', 'day_of_week',
            'is_weekend', 'is_business_hours', 'previous_conversions',
            'avg_time_to_convert', 'historical_conversion_rate',
            'viewed_high_intent_page', 'viewed_pricing'
        ]

        importance = dict(zip(feature_names, self.model.feature_importances_.tolist()))
        return dict(sorted(importance.items(), key=lambda x: -x[1]))

    def predict(self, features: VisitorFeatures) -> Tuple[float, int]:
        """
        Predict conversion probability.
        Returns (probability, lead_score).
        """
        if self.model is None or not self._sklearn_available:
            return self._rule_based_predict(features)

        X = features.to_vector().reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        probability = self.model.predict_proba(X_scaled)[0, 1]
        lead_score = int(probability * 100)

        return probability, lead_score

    def _rule_based_predict(self, features: VisitorFeatures) -> Tuple[float, int]:
        """
        Rule-based prediction fallback.
        """
        score = 0.0

        # Session engagement
        if features.session_duration > 60:
            score += 0.1
        if features.session_duration > 180:
            score += 0.1
        if features.pages_viewed > 2:
            score += 0.1
        if features.pages_viewed > 5:
            score += 0.1

        # Return visitor bonus
        if features.return_visitor:
            score += 0.15
        if features.previous_conversions > 0:
            score += 0.2

        # Traffic source
        if features.is_paid:
            score += 0.15
        elif features.is_organic:
            score += 0.1
        elif features.is_direct:
            score += 0.12

        # Intent signals
        if features.viewed_high_intent_page:
            score += 0.2
        if features.viewed_pricing:
            score += 0.15

        # Time factors
        if features.is_business_hours:
            score += 0.05

        probability = min(score, 1.0)
        lead_score = int(probability * 100)

        return probability, lead_score

    def save_model(self, path: str):
        """Save model to disk."""
        if not self._sklearn_available or self.model is None:
            return

        import joblib
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler
        }, path)

    def load_model(self, path: str) -> bool:
        """Load model from disk."""
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


class LeadScorer:
    """
    Lead scoring and segmentation.
    """

    SEGMENTS = {
        'hot': (70, 100),
        'warm': (40, 69),
        'cold': (0, 39)
    }

    def __init__(self):
        self.predictor = ConversionPredictor()
        self.extractor = FeatureExtractor()

    def score_visitor(
        self,
        ip_address: str,
        session_id: Optional[str] = None
    ) -> PredictionResult:
        """
        Score a visitor and return prediction.
        """
        features = self.extractor.extract_from_session(ip_address, session_id)
        probability, score = self.predictor.predict(features)

        # Determine segment
        segment = 'cold'
        for seg, (low, high) in self.SEGMENTS.items():
            if low <= score <= high:
                segment = seg
                break

        # Generate recommendations
        recommendations = self._get_recommendations(segment, features, score)

        return PredictionResult(
            probability=probability,
            score=score,
            segment=segment,
            recommended_actions=recommendations,
            features=features
        )

    def _get_recommendations(
        self,
        segment: str,
        features: VisitorFeatures,
        score: int
    ) -> List[str]:
        """Generate action recommendations."""
        recommendations = []

        if segment == 'hot':
            recommendations.append("Show targeted conversion offer")
            if not features.viewed_pricing:
                recommendations.append("Direct to pricing page")
            recommendations.append("Enable live chat proactively")

        elif segment == 'warm':
            if features.pages_viewed < 3:
                recommendations.append("Show content recommendations")
            if not features.return_visitor:
                recommendations.append("Offer email capture for nurturing")
            recommendations.append("Show social proof/testimonials")

        else:  # cold
            recommendations.append("Focus on awareness content")
            if features.is_direct:
                recommendations.append("Improve landing page clarity")
            recommendations.append("Consider retargeting campaign")

        return recommendations

    def get_segment_distribution(self, days_back: int = 7) -> Dict:
        """
        Get distribution of visitors across segments.
        """
        from shortener.models import ClickData

        start_date = timezone.now() - timedelta(days=days_back)

        # Get unique visitors
        visitors = ClickData.objects.filter(
            timestamp__gte=start_date
        ).values('ip_address').distinct()

        segments = {'hot': 0, 'warm': 0, 'cold': 0}

        for visitor in visitors[:1000]:  # Limit for performance
            result = self.score_visitor(visitor['ip_address'])
            segments[result.segment] += 1

        total = sum(segments.values())
        return {
            'distribution': segments,
            'percentages': {
                k: round(v / total * 100, 1) if total > 0 else 0
                for k, v in segments.items()
            }
        }


class ABTestWinnerPredictor:
    """
    Predict likely winner of an A/B test.
    """

    def predict_winner(self, test_id: int) -> Dict:
        """
        Predict which variant is likely to win.
        """
        from shortener.models import ABTest, ABTestVariant

        try:
            test = ABTest.objects.get(pk=test_id)
        except ABTest.DoesNotExist:
            return {'error': 'Test not found'}

        variants = test.variants.all()

        if not variants:
            return {'error': 'No variants found'}

        predictions = []

        for variant in variants:
            visitors = variant.get_visitors()
            conversions = variant.get_conversions()

            if visitors < 10:
                confidence = 'low'
                predicted_rate = 0.0
            else:
                # Bayesian estimation
                alpha = 1 + conversions
                beta = 1 + (visitors - conversions)

                # Expected value
                predicted_rate = alpha / (alpha + beta)

                # Confidence based on sample size
                if visitors > 1000:
                    confidence = 'high'
                elif visitors > 100:
                    confidence = 'medium'
                else:
                    confidence = 'low'

            predictions.append({
                'variant_id': variant.id,
                'variant_name': variant.name,
                'current_rate': variant.get_conversion_rate(),
                'predicted_rate': round(predicted_rate * 100, 2),
                'visitors': visitors,
                'confidence': confidence
            })

        # Sort by predicted rate
        predictions.sort(key=lambda x: -x['predicted_rate'])

        if len(predictions) > 1:
            leader = predictions[0]
            margin = predictions[0]['predicted_rate'] - predictions[1]['predicted_rate']

            recommendation = None
            if margin > 5 and leader['visitors'] > 100:
                recommendation = f"Consider implementing {leader['variant_name']}"
            elif leader['visitors'] < 100:
                recommendation = "Continue testing to gather more data"
        else:
            recommendation = None

        return {
            'predictions': predictions,
            'likely_winner': predictions[0]['variant_name'] if predictions else None,
            'recommendation': recommendation
        }


class ConversionPredictionService:
    """
    High-level conversion prediction service.
    """

    def __init__(self):
        self.predictor = ConversionPredictor()
        self.extractor = FeatureExtractor()
        self.lead_scorer = LeadScorer()
        self.ab_predictor = ABTestWinnerPredictor()

    def get_visitor_prediction(
        self,
        ip_address: str,
        session_id: Optional[str] = None
    ) -> Dict:
        """
        Get conversion prediction for a visitor.
        """
        result = self.lead_scorer.score_visitor(ip_address, session_id)
        return result.to_dict()

    def get_real_time_score_api(
        self,
        ip_address: str,
        session_id: Optional[str] = None,
        page_url: Optional[str] = None
    ) -> Dict:
        """
        Real-time scoring API for JavaScript integration.
        """
        result = self.lead_scorer.score_visitor(ip_address, session_id)

        # Simplified response for frontend
        return {
            'score': result.score,
            'segment': result.segment,
            'actions': result.recommended_actions[:3]  # Top 3 recommendations
        }

    def train_model(self, days_back: int = 90) -> Dict:
        """
        Train the prediction model on historical data.
        """
        from shortener.models import ClickData

        start_date = timezone.now() - timedelta(days=days_back)

        # Get sessions with conversion outcome
        clicks = ClickData.objects.filter(timestamp__gte=start_date)

        # Group by session
        sessions = clicks.values('session_id').annotate(
            has_conversion=Sum('is_conversion')
        ).filter(session_id__isnull=False)

        X = []
        y = []

        for session in sessions:
            features = self.extractor.extract_from_session(
                ip_address='',
                session_id=session['session_id']
            )

            X.append(features.to_vector())
            y.append(1 if session['has_conversion'] > 0 else 0)

        if len(y) < 100:
            return {'error': 'Insufficient training data'}

        X = np.array(X)
        y = np.array(y)

        result = self.predictor.train(X, y)

        # Save model
        self.predictor.save_model('conversion_model.joblib')

        return result

    def get_prediction_stats(self) -> Dict:
        """
        Get prediction model statistics.
        """
        distribution = self.lead_scorer.get_segment_distribution()

        return {
            'model_loaded': self.predictor.model is not None,
            'segment_distribution': distribution
        }

    def predict_ab_test_winner(self, test_id: int) -> Dict:
        """
        Predict A/B test winner.
        """
        return self.ab_predictor.predict_winner(test_id)
