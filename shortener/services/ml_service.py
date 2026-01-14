"""
Machine Learning services for analytics.
Provides traffic forecasting, anomaly detection, and bot classification.
"""
import logging
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
from django.core.cache import cache
from django.db.models import Count, Avg
from django.db.models.functions import TruncHour, TruncDay
from django.utils import timezone

logger = logging.getLogger(__name__)


class TrafficForecaster:
    """
    Time series forecasting for traffic prediction.
    Uses exponential smoothing and seasonal decomposition.
    """

    def __init__(self):
        self.cache_prefix = 'traffic_forecast'

    def forecast_traffic(self, url_id: int, days: int = 7) -> Dict:
        """
        Forecast traffic for a specific URL.
        """
        from shortener.models import ClickData

        # Get historical data
        thirty_days_ago = timezone.now() - timedelta(days=30)
        hourly_data = ClickData.objects.filter(
            url_id=url_id,
            timestamp__gte=thirty_days_ago
        ).annotate(
            hour=TruncHour('timestamp')
        ).values('hour').annotate(
            count=Count('id')
        ).order_by('hour')

        if not hourly_data:
            return {'forecast': [], 'confidence': 0}

        # Convert to time series
        data_dict = {item['hour']: item['count'] for item in hourly_data}

        # Fill missing hours with 0
        all_hours = []
        counts = []
        current = thirty_days_ago.replace(minute=0, second=0, microsecond=0)
        while current <= timezone.now():
            all_hours.append(current)
            counts.append(data_dict.get(current, 0))
            current += timedelta(hours=1)

        if len(counts) < 24:
            return {'forecast': [], 'confidence': 0}

        # Calculate seasonal pattern (24-hour cycle)
        seasonal = self._calculate_seasonal_pattern(counts)

        # Calculate trend using simple linear regression
        trend = self._calculate_trend(counts)

        # Generate forecast
        forecast = []
        forecast_hours = days * 24

        for i in range(forecast_hours):
            hour_of_day = (len(counts) + i) % 24
            trend_value = trend['slope'] * (len(counts) + i) + trend['intercept']
            seasonal_value = seasonal[hour_of_day]

            predicted = max(0, trend_value + seasonal_value)

            forecast.append({
                'hour': (timezone.now() + timedelta(hours=i+1)).isoformat(),
                'predicted_clicks': round(predicted),
                'lower_bound': round(max(0, predicted * 0.7)),
                'upper_bound': round(predicted * 1.3)
            })

        # Calculate confidence based on data consistency
        variance = np.var(counts) if counts else 0
        mean = np.mean(counts) if counts else 1
        cv = (np.sqrt(variance) / mean) if mean > 0 else 1
        confidence = max(0, min(1, 1 - cv))

        return {
            'forecast': forecast,
            'confidence': round(confidence, 2),
            'historical_avg': round(np.mean(counts), 1),
            'trend': 'increasing' if trend['slope'] > 0 else 'decreasing'
        }

    def forecast_aggregate_traffic(self, days: int = 7) -> Dict:
        """
        Forecast aggregate traffic across all URLs.
        """
        from shortener.models import ClickData

        thirty_days_ago = timezone.now() - timedelta(days=30)
        hourly_data = ClickData.objects.filter(
            timestamp__gte=thirty_days_ago
        ).annotate(
            hour=TruncHour('timestamp')
        ).values('hour').annotate(
            count=Count('id')
        ).order_by('hour')

        data_dict = {item['hour']: item['count'] for item in hourly_data}

        counts = []
        current = thirty_days_ago.replace(minute=0, second=0, microsecond=0)
        while current <= timezone.now():
            counts.append(data_dict.get(current, 0))
            current += timedelta(hours=1)

        if len(counts) < 24:
            return {'forecast': [], 'confidence': 0}

        seasonal = self._calculate_seasonal_pattern(counts)
        trend = self._calculate_trend(counts)

        forecast = []
        for i in range(days * 24):
            hour_of_day = (len(counts) + i) % 24
            trend_value = trend['slope'] * (len(counts) + i) + trend['intercept']
            seasonal_value = seasonal[hour_of_day]

            predicted = max(0, trend_value + seasonal_value)
            forecast.append({
                'hour': (timezone.now() + timedelta(hours=i+1)).isoformat(),
                'predicted_clicks': round(predicted)
            })

        # Daily aggregation
        daily_forecast = []
        for d in range(days):
            day_clicks = sum(f['predicted_clicks'] for f in forecast[d*24:(d+1)*24])
            daily_forecast.append({
                'date': (timezone.now() + timedelta(days=d+1)).date().isoformat(),
                'predicted_clicks': day_clicks
            })

        return {
            'hourly_forecast': forecast,
            'daily_forecast': daily_forecast,
            'total_predicted': sum(f['predicted_clicks'] for f in forecast)
        }

    def _calculate_seasonal_pattern(self, data: List[float]) -> List[float]:
        """Calculate hourly seasonal pattern."""
        if len(data) < 24:
            return [0] * 24

        hourly_avgs = defaultdict(list)
        for i, value in enumerate(data):
            hour = i % 24
            hourly_avgs[hour].append(value)

        overall_mean = np.mean(data)
        seasonal = []
        for hour in range(24):
            if hourly_avgs[hour]:
                hour_mean = np.mean(hourly_avgs[hour])
                seasonal.append(hour_mean - overall_mean)
            else:
                seasonal.append(0)

        return seasonal

    def _calculate_trend(self, data: List[float]) -> Dict:
        """Calculate linear trend using least squares."""
        if len(data) < 2:
            return {'slope': 0, 'intercept': 0}

        x = np.arange(len(data))
        y = np.array(data)

        # Simple linear regression
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        if denominator == 0:
            return {'slope': 0, 'intercept': y_mean}

        slope = numerator / denominator
        intercept = y_mean - slope * x_mean

        return {'slope': slope, 'intercept': intercept}


class AnomalyDetector:
    """
    Detects traffic anomalies using statistical methods.
    """

    def __init__(self):
        self.threshold_multiplier = 3.0  # Standard deviations for anomaly
        self.cache_prefix = 'anomaly'

    def check_ip_anomaly(self, ip_address: str, click_count: int) -> Tuple[bool, float]:
        """
        Check if IP click count is anomalous.
        Returns (is_anomaly, anomaly_score).
        """
        from shortener.models import ClickData

        # Get historical data for this IP
        cache_key = f'{self.cache_prefix}_ip_{hashlib.md5(ip_address.encode()).hexdigest()[:8]}'
        cached = cache.get(cache_key)

        if cached:
            mean, std = cached['mean'], cached['std']
        else:
            # Calculate from recent data
            seven_days_ago = timezone.now() - timedelta(days=7)
            historical = ClickData.objects.filter(
                ip_address=ip_address,
                timestamp__gte=seven_days_ago
            ).annotate(
                hour=TruncHour('timestamp')
            ).values('hour').annotate(
                count=Count('id')
            )

            counts = [h['count'] for h in historical]

            if not counts:
                # First time seeing this IP, use global baseline
                mean, std = 5, 10  # Conservative defaults
            else:
                mean = np.mean(counts)
                std = np.std(counts) if len(counts) > 1 else mean * 0.5

            cache.set(cache_key, {'mean': mean, 'std': std}, timeout=3600)

        # Calculate z-score
        if std == 0:
            std = 1

        z_score = (click_count - mean) / std
        is_anomaly = z_score > self.threshold_multiplier

        return is_anomaly, float(z_score)

    def get_expected_traffic(self, minutes: int = 5) -> int:
        """
        Get expected traffic volume for a time window.
        """
        from shortener.models import ClickData

        cache_key = f'{self.cache_prefix}_expected_{minutes}'
        cached = cache.get(cache_key)

        if cached:
            return cached

        # Calculate from last 7 days
        seven_days_ago = timezone.now() - timedelta(days=7)

        hourly_counts = ClickData.objects.filter(
            timestamp__gte=seven_days_ago
        ).annotate(
            hour=TruncHour('timestamp')
        ).values('hour').annotate(
            count=Count('id')
        )

        if not hourly_counts:
            return 10  # Default fallback

        # Average per hour, scaled to minutes
        avg_hourly = np.mean([h['count'] for h in hourly_counts])
        expected = int((avg_hourly / 60) * minutes)

        cache.set(cache_key, expected, timeout=300)
        return expected

    def detect_traffic_spike(self, current_count: int, window_minutes: int = 5) -> Dict:
        """
        Detect if current traffic represents a spike.
        """
        expected = self.get_expected_traffic(window_minutes)

        ratio = current_count / expected if expected > 0 else current_count

        return {
            'is_spike': ratio > 2.0,
            'severity': 'high' if ratio > 5 else 'medium' if ratio > 3 else 'low',
            'current': current_count,
            'expected': expected,
            'ratio': round(ratio, 2)
        }


class BotDetector:
    """
    ML-based bot detection using behavioral patterns.
    """

    def __init__(self):
        self.model = None
        self.feature_names = [
            'ua_length', 'ua_has_bot', 'ua_has_crawler',
            'ua_is_browser', 'referrer_present', 'referrer_suspicious',
            'hour_of_day', 'is_weekend', 'clicks_per_minute'
        ]
        self._load_model()

    def _load_model(self):
        """Load trained model from database."""
        from shortener.models import PredictiveModel

        try:
            model_record = PredictiveModel.objects.filter(
                name='bot_detector',
                is_active=True
            ).first()

            if model_record and model_record.model_data:
                self.model = pickle.loads(model_record.model_data)
        except Exception as e:
            logger.warning(f"Could not load bot detector model: {e}")
            self.model = None

    def extract_features(self, click_data: Dict) -> List[float]:
        """
        Extract features from click data for classification.
        """
        user_agent = click_data.get('user_agent', '') or ''
        referrer = click_data.get('referrer', '') or ''
        timestamp = click_data.get('timestamp', timezone.now())
        ip_address = click_data.get('ip_address', '')

        ua_lower = user_agent.lower()

        # Feature extraction
        features = [
            len(user_agent),  # ua_length
            1 if 'bot' in ua_lower else 0,  # ua_has_bot
            1 if 'crawler' in ua_lower or 'spider' in ua_lower else 0,  # ua_has_crawler
            1 if any(b in ua_lower for b in ['chrome', 'firefox', 'safari', 'edge']) else 0,  # ua_is_browser
            1 if referrer else 0,  # referrer_present
            1 if any(s in referrer.lower() for s in ['click', 'traffic', 'exchange']) else 0,  # referrer_suspicious
            timestamp.hour if hasattr(timestamp, 'hour') else 12,  # hour_of_day
            1 if hasattr(timestamp, 'weekday') and timestamp.weekday() >= 5 else 0,  # is_weekend
            self._get_clicks_per_minute(ip_address),  # clicks_per_minute
        ]

        return features

    def _get_clicks_per_minute(self, ip_address: str) -> float:
        """Get recent click rate for IP."""
        from shortener.models import ClickData

        if not ip_address:
            return 0

        one_minute_ago = timezone.now() - timedelta(minutes=1)
        count = ClickData.objects.filter(
            ip_address=ip_address,
            timestamp__gte=one_minute_ago
        ).count()

        return float(count)

    def predict(self, click_data: Dict) -> Tuple[bool, float]:
        """
        Predict if a click is from a bot.
        Returns (is_bot, confidence).
        """
        features = self.extract_features(click_data)

        # If no trained model, use rule-based detection
        if self.model is None:
            return self._rule_based_detection(features, click_data)

        try:
            features_array = np.array([features])
            prediction = self.model.predict(features_array)[0]
            probability = self.model.predict_proba(features_array)[0]

            return bool(prediction), float(max(probability))
        except Exception as e:
            logger.error(f"Model prediction error: {e}")
            return self._rule_based_detection(features, click_data)

    def _rule_based_detection(self, features: List[float], click_data: Dict) -> Tuple[bool, float]:
        """
        Fallback rule-based bot detection.
        """
        score = 0.0
        max_score = 5.0

        # Check user agent
        if features[1] > 0:  # ua_has_bot
            score += 2.0
        if features[2] > 0:  # ua_has_crawler
            score += 1.5
        if features[3] == 0:  # not a browser
            score += 1.0
        if features[0] < 20:  # very short UA
            score += 0.5

        # Check referrer
        if features[5] > 0:  # suspicious referrer
            score += 1.0

        # Check click rate
        if features[8] > 10:  # high click rate
            score += 1.5

        is_bot = score >= 2.5
        confidence = min(1.0, score / max_score)

        return is_bot, confidence

    def train(self, training_data: List[Dict]) -> float:
        """
        Train the bot detection model.
        Returns accuracy score.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from shortener.models import PredictiveModel

        if len(training_data) < 100:
            logger.warning("Not enough training data for bot detector")
            return 0.0

        # Prepare features and labels
        X = []
        y = []

        for click in training_data:
            features = self.extract_features(click)
            X.append(features)
            y.append(1 if click.get('is_bot', False) else 0)

        X = np.array(X)
        y = np.array(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Evaluate
        accuracy = model.score(X_test, y_test)

        # Save model
        self.model = model
        model_bytes = pickle.dumps(model)

        PredictiveModel.objects.update_or_create(
            name='bot_detector',
            defaults={
                'model_type': 'anomaly_detection',
                'model_data': model_bytes,
                'accuracy': accuracy,
                'last_trained': timezone.now(),
                'training_samples': len(training_data),
                'is_active': True
            }
        )

        return accuracy


class ConversionPredictor:
    """
    Predicts conversion probability for visitors.
    """

    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load trained model from database."""
        from shortener.models import PredictiveModel

        try:
            model_record = PredictiveModel.objects.filter(
                name='conversion_predictor',
                is_active=True
            ).first()

            if model_record and model_record.model_data:
                self.model = pickle.loads(model_record.model_data)
        except Exception:
            self.model = None

    def extract_features(self, visitor_data: Dict) -> List[float]:
        """Extract features for conversion prediction."""
        features = [
            visitor_data.get('total_clicks', 0),
            visitor_data.get('unique_urls', 0),
            visitor_data.get('session_duration', 0),
            visitor_data.get('pages_per_session', 0),
            1 if visitor_data.get('utm_source') else 0,
            1 if visitor_data.get('device_type') == 'Desktop' else 0,
            visitor_data.get('return_visits', 0),
            visitor_data.get('hour_of_first_visit', 12),
        ]
        return features

    def predict_conversion(self, visitor_data: Dict) -> Dict:
        """
        Predict conversion probability for a visitor.
        """
        features = self.extract_features(visitor_data)

        if self.model is None:
            # Rule-based fallback
            score = min(1.0, (
                features[0] * 0.1 +  # clicks
                features[2] * 0.001 +  # duration
                features[6] * 0.2  # return visits
            ))
            return {
                'conversion_probability': round(score, 3),
                'model_type': 'rule_based'
            }

        try:
            features_array = np.array([features])
            probability = self.model.predict_proba(features_array)[0][1]

            return {
                'conversion_probability': round(float(probability), 3),
                'model_type': 'ml'
            }
        except Exception as e:
            logger.error(f"Conversion prediction error: {e}")
            return {'conversion_probability': 0.5, 'model_type': 'fallback'}

    def train(self, training_data: List[Dict]) -> float:
        """Train conversion prediction model."""
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import train_test_split
        from shortener.models import PredictiveModel

        if len(training_data) < 100:
            return 0.0

        X = []
        y = []

        for visitor in training_data:
            features = self.extract_features(visitor)
            X.append(features)
            y.append(1 if visitor.get('converted', False) else 0)

        X = np.array(X)
        y = np.array(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)

        self.model = model
        model_bytes = pickle.dumps(model)

        PredictiveModel.objects.update_or_create(
            name='conversion_predictor',
            defaults={
                'model_type': 'conversion_prediction',
                'model_data': model_bytes,
                'accuracy': accuracy,
                'last_trained': timezone.now(),
                'training_samples': len(training_data),
                'is_active': True
            }
        )

        return accuracy
