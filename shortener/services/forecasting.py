"""
Traffic Forecasting Service.
Time series analysis using Prophet and ARIMA models.
"""
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from django.utils import timezone
from django.db.models import Count
from django.db.models.functions import TruncHour, TruncDay, TruncWeek

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


class ForecastGranularity(Enum):
    HOURLY = 'hourly'
    DAILY = 'daily'
    WEEKLY = 'weekly'


@dataclass
class ForecastResult:
    """Container for forecast results."""
    timestamp: datetime
    predicted: float
    lower_bound: float
    upper_bound: float

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'predicted': round(self.predicted, 2),
            'lower_bound': round(self.lower_bound, 2),
            'upper_bound': round(self.upper_bound, 2)
        }


@dataclass
class SeasonalComponent:
    """Seasonal decomposition component."""
    daily: Dict[int, float]  # Hour -> average
    weekly: Dict[int, float]  # Day of week -> average
    monthly: Dict[int, float]  # Day of month -> average


class TimeSeriesAnalyzer:
    """
    Time series analysis and decomposition.
    Identifies trends, seasonality, and residuals.
    """

    def __init__(self, min_data_points: int = 14):
        self.min_data_points = min_data_points

    def prepare_dataframe(
        self,
        url_id: Optional[int] = None,
        granularity: ForecastGranularity = ForecastGranularity.DAILY,
        days_back: int = 90
    ) -> pd.DataFrame:
        """
        Prepare time series DataFrame from click data.
        """
        from shortener.models import ClickData

        start_date = timezone.now() - timedelta(days=days_back)

        queryset = ClickData.objects.filter(timestamp__gte=start_date)
        if url_id:
            queryset = queryset.filter(url_id=url_id)

        # Aggregate based on granularity
        if granularity == ForecastGranularity.HOURLY:
            data = queryset.annotate(
                period=TruncHour('timestamp')
            ).values('period').annotate(
                clicks=Count('id')
            ).order_by('period')
        elif granularity == ForecastGranularity.DAILY:
            data = queryset.annotate(
                period=TruncDay('timestamp')
            ).values('period').annotate(
                clicks=Count('id')
            ).order_by('period')
        else:  # Weekly
            data = queryset.annotate(
                period=TruncWeek('timestamp')
            ).values('period').annotate(
                clicks=Count('id')
            ).order_by('period')

        df = pd.DataFrame(list(data))

        if df.empty:
            return df

        # Rename columns for Prophet compatibility
        df = df.rename(columns={'period': 'ds', 'clicks': 'y'})
        df['ds'] = pd.to_datetime(df['ds'])

        # Fill missing periods with zeros
        df = self._fill_missing_periods(df, granularity)

        return df

    def _fill_missing_periods(
        self,
        df: pd.DataFrame,
        granularity: ForecastGranularity
    ) -> pd.DataFrame:
        """Fill gaps in time series with zeros."""
        if df.empty:
            return df

        # Determine frequency
        freq_map = {
            ForecastGranularity.HOURLY: 'h',
            ForecastGranularity.DAILY: 'D',
            ForecastGranularity.WEEKLY: 'W'
        }
        freq = freq_map[granularity]

        # Create complete date range
        date_range = pd.date_range(
            start=df['ds'].min(),
            end=df['ds'].max(),
            freq=freq
        )

        # Reindex with complete range
        df = df.set_index('ds').reindex(date_range, fill_value=0).reset_index()
        df.columns = ['ds', 'y']

        return df

    def decompose(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Decompose time series into trend, seasonal, and residual components.
        Uses STL-like decomposition.
        """
        if len(df) < self.min_data_points:
            return {'error': 'Insufficient data for decomposition'}

        y = df['y'].values
        n = len(y)

        # Calculate trend using moving average
        window = min(7, n // 3)
        if window < 3:
            window = 3

        trend = pd.Series(y).rolling(window=window, center=True).mean().fillna(method='bfill').fillna(method='ffill').values

        # Detrended series
        detrended = y - trend

        # Calculate seasonal component (weekly pattern)
        seasonal = np.zeros(n)
        df_temp = df.copy()
        df_temp['detrended'] = detrended
        df_temp['dow'] = df_temp['ds'].dt.dayofweek

        dow_means = df_temp.groupby('dow')['detrended'].mean()
        for i, row in df_temp.iterrows():
            seasonal[i] = dow_means.get(row['dow'], 0)

        # Residual
        residual = y - trend - seasonal

        return {
            'trend': trend.tolist(),
            'seasonal': seasonal.tolist(),
            'residual': residual.tolist(),
            'dates': df['ds'].dt.strftime('%Y-%m-%d').tolist()
        }

    def detect_seasonality(self, df: pd.DataFrame) -> SeasonalComponent:
        """
        Detect seasonal patterns in the data.
        """
        if df.empty:
            return SeasonalComponent({}, {}, {})

        df = df.copy()
        df['hour'] = df['ds'].dt.hour
        df['dow'] = df['ds'].dt.dayofweek
        df['dom'] = df['ds'].dt.day

        # Calculate averages
        hourly = df.groupby('hour')['y'].mean().to_dict() if 'hour' in df.columns else {}
        weekly = df.groupby('dow')['y'].mean().to_dict()
        monthly = df.groupby('dom')['y'].mean().to_dict()

        return SeasonalComponent(
            daily=hourly,
            weekly=weekly,
            monthly=monthly
        )


class ProphetForecaster:
    """
    Prophet-based forecasting.
    Falls back to custom implementation if Prophet unavailable.
    """

    def __init__(
        self,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        yearly_seasonality: bool = False,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = True
    ):
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self._prophet_available = self._check_prophet()

    def _check_prophet(self) -> bool:
        """Check if Prophet is available."""
        try:
            from prophet import Prophet
            return True
        except ImportError:
            logger.warning("Prophet not available, using fallback forecasting")
            return False

    def fit_predict(
        self,
        df: pd.DataFrame,
        periods: int = 7,
        freq: str = 'D'
    ) -> List[ForecastResult]:
        """
        Fit model and generate forecasts.
        """
        if len(df) < 2:
            return []

        if self._prophet_available:
            return self._prophet_forecast(df, periods, freq)
        else:
            return self._fallback_forecast(df, periods, freq)

    def _prophet_forecast(
        self,
        df: pd.DataFrame,
        periods: int,
        freq: str
    ) -> List[ForecastResult]:
        """Use Prophet for forecasting."""
        from prophet import Prophet

        model = Prophet(
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality
        )

        model.fit(df)

        future = model.make_future_dataframe(periods=periods, freq=freq)
        forecast = model.predict(future)

        # Extract forecast for future periods only
        future_forecast = forecast.tail(periods)

        results = []
        for _, row in future_forecast.iterrows():
            results.append(ForecastResult(
                timestamp=row['ds'].to_pydatetime(),
                predicted=max(0, row['yhat']),
                lower_bound=max(0, row['yhat_lower']),
                upper_bound=max(0, row['yhat_upper'])
            ))

        return results

    def _fallback_forecast(
        self,
        df: pd.DataFrame,
        periods: int,
        freq: str
    ) -> List[ForecastResult]:
        """
        Fallback forecasting using exponential smoothing.
        """
        y = df['y'].values

        # Triple exponential smoothing (Holt-Winters)
        alpha = 0.3  # Level
        beta = 0.1   # Trend
        gamma = 0.2  # Seasonality

        season_length = 7 if freq == 'D' else 24 if freq == 'h' else 4

        # Initialize
        n = len(y)
        level = y[0]
        trend = np.mean(np.diff(y[:min(season_length, n)])) if n > 1 else 0

        # Initialize seasonal factors
        if n >= season_length:
            seasonal = list(y[:season_length] / np.mean(y[:season_length]))
        else:
            seasonal = [1.0] * season_length

        # Fit the model
        fitted = []
        for t in range(n):
            if t >= season_length:
                # Update components
                old_level = level
                level = alpha * (y[t] / seasonal[t % season_length]) + (1 - alpha) * (level + trend)
                trend = beta * (level - old_level) + (1 - beta) * trend
                seasonal[t % season_length] = gamma * (y[t] / level) + (1 - gamma) * seasonal[t % season_length]

            fitted.append(level * seasonal[t % season_length])

        # Calculate prediction intervals
        residuals = y - np.array(fitted[-len(y):])
        std_error = np.std(residuals) if len(residuals) > 1 else 1.0

        # Generate forecasts
        results = []
        last_date = df['ds'].max()

        freq_delta = {
            'D': timedelta(days=1),
            'h': timedelta(hours=1),
            'W': timedelta(weeks=1)
        }
        delta = freq_delta.get(freq, timedelta(days=1))

        for i in range(1, periods + 1):
            forecast_date = last_date + delta * i

            # Forecast
            pred = (level + i * trend) * seasonal[(n + i) % season_length]
            pred = max(0, pred)

            # Prediction interval widens with horizon
            interval_width = 1.96 * std_error * np.sqrt(1 + i * 0.1)

            results.append(ForecastResult(
                timestamp=forecast_date.to_pydatetime() if hasattr(forecast_date, 'to_pydatetime') else forecast_date,
                predicted=pred,
                lower_bound=max(0, pred - interval_width),
                upper_bound=pred + interval_width
            ))

        return results


class ARIMAForecaster:
    """
    ARIMA-based forecasting.
    Uses statsmodels if available, otherwise custom implementation.
    """

    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1)):
        self.order = order
        self._statsmodels_available = self._check_statsmodels()

    def _check_statsmodels(self) -> bool:
        """Check if statsmodels is available."""
        try:
            from statsmodels.tsa.arima.model import ARIMA
            return True
        except ImportError:
            logger.warning("statsmodels not available, using fallback ARIMA")
            return False

    def fit_predict(
        self,
        df: pd.DataFrame,
        periods: int = 7
    ) -> List[ForecastResult]:
        """
        Fit ARIMA model and generate forecasts.
        """
        if len(df) < max(self.order) + 2:
            return []

        y = df['y'].values

        if self._statsmodels_available:
            return self._statsmodels_forecast(df, y, periods)
        else:
            return self._fallback_forecast(df, y, periods)

    def _statsmodels_forecast(
        self,
        df: pd.DataFrame,
        y: np.ndarray,
        periods: int
    ) -> List[ForecastResult]:
        """Use statsmodels ARIMA."""
        from statsmodels.tsa.arima.model import ARIMA

        try:
            model = ARIMA(y, order=self.order)
            fitted = model.fit()

            forecast = fitted.get_forecast(steps=periods)
            mean = forecast.predicted_mean
            conf_int = forecast.conf_int()

            results = []
            last_date = df['ds'].max()

            for i in range(periods):
                forecast_date = last_date + timedelta(days=i + 1)

                results.append(ForecastResult(
                    timestamp=forecast_date.to_pydatetime() if hasattr(forecast_date, 'to_pydatetime') else forecast_date,
                    predicted=max(0, mean.iloc[i]),
                    lower_bound=max(0, conf_int.iloc[i, 0]),
                    upper_bound=max(0, conf_int.iloc[i, 1])
                ))

            return results

        except Exception as e:
            logger.error(f"ARIMA fitting error: {e}")
            return self._fallback_forecast(df, y, periods)

    def _fallback_forecast(
        self,
        df: pd.DataFrame,
        y: np.ndarray,
        periods: int
    ) -> List[ForecastResult]:
        """
        Simple AR(1) fallback.
        """
        n = len(y)

        # Fit AR(1): y_t = c + phi * y_{t-1} + e
        y_lag = y[:-1]
        y_curr = y[1:]

        # OLS estimation
        phi = np.corrcoef(y_lag, y_curr)[0, 1] if len(y_lag) > 1 else 0.5
        c = np.mean(y_curr) * (1 - phi)

        # Calculate residual variance
        fitted = c + phi * y_lag
        residuals = y_curr - fitted
        sigma = np.std(residuals) if len(residuals) > 1 else 1.0

        # Generate forecasts
        results = []
        last_date = df['ds'].max()
        last_y = y[-1]

        for i in range(1, periods + 1):
            forecast_date = last_date + timedelta(days=i)

            # Multi-step forecast
            pred = c * (1 - phi ** i) / (1 - phi) + (phi ** i) * last_y
            pred = max(0, pred)

            # Prediction interval
            var = sigma ** 2 * (1 - phi ** (2 * i)) / (1 - phi ** 2)
            interval = 1.96 * np.sqrt(var)

            results.append(ForecastResult(
                timestamp=forecast_date.to_pydatetime() if hasattr(forecast_date, 'to_pydatetime') else forecast_date,
                predicted=pred,
                lower_bound=max(0, pred - interval),
                upper_bound=pred + interval
            ))

        return results


class AnomalyDetector:
    """
    Detect anomalies in traffic data.
    """

    def __init__(
        self,
        z_threshold: float = 3.0,
        window_size: int = 7
    ):
        self.z_threshold = z_threshold
        self.window_size = window_size

    def detect(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect anomalies using rolling Z-score.
        """
        if len(df) < self.window_size:
            return []

        df = df.copy()

        # Calculate rolling statistics
        df['rolling_mean'] = df['y'].rolling(window=self.window_size).mean()
        df['rolling_std'] = df['y'].rolling(window=self.window_size).std()

        # Calculate Z-scores
        df['z_score'] = (df['y'] - df['rolling_mean']) / df['rolling_std']
        df['z_score'] = df['z_score'].fillna(0)

        # Identify anomalies
        anomalies = df[abs(df['z_score']) > self.z_threshold]

        results = []
        for _, row in anomalies.iterrows():
            anomaly_type = 'spike' if row['z_score'] > 0 else 'drop'

            results.append({
                'timestamp': row['ds'].isoformat(),
                'value': row['y'],
                'expected': round(row['rolling_mean'], 2),
                'z_score': round(row['z_score'], 2),
                'type': anomaly_type,
                'severity': 'high' if abs(row['z_score']) > 4 else 'medium'
            })

        return results

    def detect_pattern_anomalies(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect anomalies in seasonal patterns.
        """
        if len(df) < 14:
            return []

        df = df.copy()
        df['dow'] = df['ds'].dt.dayofweek

        # Calculate expected values per day of week
        dow_stats = df.groupby('dow')['y'].agg(['mean', 'std']).to_dict()

        anomalies = []
        for _, row in df.iterrows():
            dow = row['dow']
            expected = dow_stats['mean'].get(dow, row['y'])
            std = dow_stats['std'].get(dow, 1)

            if std > 0:
                z = (row['y'] - expected) / std
                if abs(z) > self.z_threshold:
                    anomalies.append({
                        'timestamp': row['ds'].isoformat(),
                        'value': row['y'],
                        'expected_for_day': round(expected, 2),
                        'z_score': round(z, 2),
                        'day_of_week': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][dow]
                    })

        return anomalies


class TrendAnalyzer:
    """
    Analyze trends in traffic data.
    """

    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Analyze overall trend and calculate statistics.
        """
        if len(df) < 2:
            return {'trend': 'insufficient_data'}

        y = df['y'].values
        x = np.arange(len(y))

        # Linear regression for trend
        slope, intercept = np.polyfit(x, y, 1)

        # Calculate R-squared
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Determine trend direction
        if abs(slope) < 0.01 * np.mean(y):
            trend = 'stable'
        elif slope > 0:
            trend = 'increasing'
        else:
            trend = 'decreasing'

        # Calculate growth rate
        if y[0] > 0:
            growth_rate = ((y[-1] - y[0]) / y[0]) * 100
        else:
            growth_rate = 0

        # Recent trend (last 7 days)
        recent = y[-min(7, len(y)):]
        recent_slope, _ = np.polyfit(np.arange(len(recent)), recent, 1)

        if abs(recent_slope) < 0.01 * np.mean(recent):
            recent_trend = 'stable'
        elif recent_slope > 0:
            recent_trend = 'increasing'
        else:
            recent_trend = 'decreasing'

        return {
            'trend': trend,
            'slope': round(slope, 4),
            'r_squared': round(r_squared, 4),
            'growth_rate_percent': round(growth_rate, 2),
            'recent_trend': recent_trend,
            'recent_slope': round(recent_slope, 4),
            'average_daily': round(np.mean(y), 2),
            'peak_value': round(np.max(y), 2),
            'min_value': round(np.min(y), 2),
            'volatility': round(np.std(y) / np.mean(y) * 100, 2) if np.mean(y) > 0 else 0
        }


class ForecastingService:
    """
    High-level forecasting service combining all components.
    """

    def __init__(self):
        self.analyzer = TimeSeriesAnalyzer()
        self.prophet = ProphetForecaster()
        self.arima = ARIMAForecaster()
        self.anomaly_detector = AnomalyDetector()
        self.trend_analyzer = TrendAnalyzer()

    def get_forecast(
        self,
        url_id: Optional[int] = None,
        periods: int = 7,
        granularity: ForecastGranularity = ForecastGranularity.DAILY,
        method: str = 'auto'
    ) -> Dict:
        """
        Generate traffic forecast.
        """
        freq_map = {
            ForecastGranularity.HOURLY: 'h',
            ForecastGranularity.DAILY: 'D',
            ForecastGranularity.WEEKLY: 'W'
        }
        freq = freq_map[granularity]

        # Prepare data
        df = self.analyzer.prepare_dataframe(
            url_id=url_id,
            granularity=granularity
        )

        if df.empty or len(df) < 7:
            return {
                'error': 'Insufficient data for forecasting',
                'data_points': len(df) if not df.empty else 0
            }

        # Choose method
        if method == 'auto':
            method = 'prophet' if len(df) > 30 else 'arima'

        # Generate forecast
        if method == 'prophet':
            forecasts = self.prophet.fit_predict(df, periods, freq)
        else:
            forecasts = self.arima.fit_predict(df, periods)

        # Get trend analysis
        trend = self.trend_analyzer.analyze(df)

        return {
            'historical': df.tail(30).to_dict('records'),
            'forecasts': [f.to_dict() for f in forecasts],
            'trend': trend,
            'method': method,
            'granularity': granularity.value
        }

    def get_anomalies(
        self,
        url_id: Optional[int] = None,
        days_back: int = 30
    ) -> Dict:
        """
        Detect anomalies in traffic.
        """
        df = self.analyzer.prepare_dataframe(
            url_id=url_id,
            granularity=ForecastGranularity.DAILY,
            days_back=days_back
        )

        if df.empty:
            return {'anomalies': [], 'pattern_anomalies': []}

        return {
            'anomalies': self.anomaly_detector.detect(df),
            'pattern_anomalies': self.anomaly_detector.detect_pattern_anomalies(df)
        }

    def get_seasonal_patterns(
        self,
        url_id: Optional[int] = None
    ) -> Dict:
        """
        Get seasonal pattern analysis.
        """
        # Hourly patterns
        df_hourly = self.analyzer.prepare_dataframe(
            url_id=url_id,
            granularity=ForecastGranularity.HOURLY,
            days_back=30
        )

        # Daily patterns
        df_daily = self.analyzer.prepare_dataframe(
            url_id=url_id,
            granularity=ForecastGranularity.DAILY,
            days_back=90
        )

        hourly_pattern = {}
        weekly_pattern = {}

        if not df_hourly.empty:
            df_hourly['hour'] = df_hourly['ds'].dt.hour
            hourly_pattern = df_hourly.groupby('hour')['y'].mean().round(2).to_dict()

        if not df_daily.empty:
            df_daily['dow'] = df_daily['ds'].dt.dayofweek
            weekly_pattern = df_daily.groupby('dow')['y'].mean().round(2).to_dict()
            weekly_pattern = {
                ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][k]: v
                for k, v in weekly_pattern.items()
            }

        return {
            'hourly_pattern': hourly_pattern,
            'weekly_pattern': weekly_pattern,
            'best_hour': max(hourly_pattern, key=hourly_pattern.get) if hourly_pattern else None,
            'best_day': max(weekly_pattern, key=weekly_pattern.get) if weekly_pattern else None
        }

    def get_decomposition(
        self,
        url_id: Optional[int] = None
    ) -> Dict:
        """
        Get time series decomposition.
        """
        df = self.analyzer.prepare_dataframe(
            url_id=url_id,
            granularity=ForecastGranularity.DAILY,
            days_back=90
        )

        return self.analyzer.decompose(df)
