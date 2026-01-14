"""
Advanced Pandas Analytics Service.
Complex data analysis, session reconstruction, and conversion attribution.
"""
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from io import BytesIO, StringIO

import numpy as np
import pandas as pd
from django.utils import timezone
from django.db.models import Count, Sum, Avg, F, Q
from django.db.models.functions import TruncDate, TruncHour

logger = logging.getLogger(__name__)


class DataFrameBuilder:
    """
    Build DataFrames from Django QuerySets efficiently.
    """

    @staticmethod
    def clicks_to_dataframe(
        url_id: Optional[int] = None,
        days_back: int = 30,
        include_fraud: bool = False
    ) -> pd.DataFrame:
        """
        Convert click data to pandas DataFrame.
        """
        from shortener.models import ClickData

        start_date = timezone.now() - timedelta(days=days_back)

        queryset = ClickData.objects.filter(timestamp__gte=start_date)

        if url_id:
            queryset = queryset.filter(url_id=url_id)

        if not include_fraud:
            queryset = queryset.filter(is_fraud=False)

        # Select specific fields for efficiency
        values = queryset.values(
            'id', 'url_id', 'url__short_code', 'timestamp',
            'ip_address', 'user_agent', 'referrer',
            'device_type', 'browser', 'os', 'country_code',
            'latitude', 'longitude', 'is_conversion', 'conversion_value',
            'session_id', 'is_fraud', 'served_url'
        )

        df = pd.DataFrame(list(values))

        if df.empty:
            return df

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Add derived columns
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6])

        return df

    @staticmethod
    def urls_to_dataframe() -> pd.DataFrame:
        """
        Convert URLs to DataFrame with aggregated stats.
        """
        from shortener.models import URL

        urls = URL.objects.annotate(
            total_clicks=Count('clicks'),
            total_conversions=Count('clicks', filter=Q(clicks__is_conversion=True)),
            total_revenue=Sum('clicks__conversion_value')
        ).values(
            'id', 'short_code', 'original_url', 'created_at',
            'enable_device_targeting', 'enable_rotation',
            'total_clicks', 'total_conversions', 'total_revenue'
        )

        return pd.DataFrame(list(urls))


class SessionReconstructor:
    """
    Reconstruct user sessions from click data.
    """

    def __init__(self, session_timeout_minutes: int = 30):
        self.session_timeout = timedelta(minutes=session_timeout_minutes)

    def reconstruct_sessions(
        self,
        df: pd.DataFrame,
        identifier: str = 'ip_address'
    ) -> pd.DataFrame:
        """
        Reconstruct sessions based on click timing.
        """
        if df.empty:
            return df

        df = df.sort_values(['ip_address', 'timestamp']).copy()

        # Calculate time since last click per user
        df['time_diff'] = df.groupby(identifier)['timestamp'].diff()

        # Start new session if gap > timeout
        df['new_session'] = (
            df['time_diff'].isna() |
            (df['time_diff'] > self.session_timeout)
        )

        # Generate session IDs
        df['reconstructed_session'] = df.groupby(identifier)['new_session'].cumsum()
        df['full_session_id'] = (
            df[identifier].astype(str) + '_' +
            df['reconstructed_session'].astype(str)
        )

        return df

    def get_session_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate metrics per session.
        """
        if df.empty or 'full_session_id' not in df.columns:
            df = self.reconstruct_sessions(df)

        if df.empty:
            return pd.DataFrame()

        sessions = df.groupby('full_session_id').agg({
            'id': 'count',
            'timestamp': ['min', 'max'],
            'url_id': 'nunique',
            'is_conversion': 'sum',
            'conversion_value': 'sum',
            'ip_address': 'first',
            'device_type': 'first',
            'country_code': 'first'
        })

        sessions.columns = [
            'clicks', 'start_time', 'end_time',
            'unique_urls', 'conversions', 'revenue',
            'ip_address', 'device_type', 'country'
        ]

        sessions['duration_seconds'] = (
            sessions['end_time'] - sessions['start_time']
        ).dt.total_seconds()

        sessions['is_bounced'] = sessions['clicks'] == 1
        sessions['pages_per_session'] = sessions['unique_urls']

        return sessions.reset_index()


class PathAnalyzer:
    """
    Analyze user navigation paths and journeys.
    """

    def get_click_paths(
        self,
        df: pd.DataFrame,
        min_path_length: int = 2,
        max_paths: int = 100
    ) -> List[Dict]:
        """
        Extract common click paths.
        """
        if df.empty:
            return []

        # Ensure sessions are reconstructed
        if 'full_session_id' not in df.columns:
            reconstructor = SessionReconstructor()
            df = reconstructor.reconstruct_sessions(df)

        # Build paths per session
        paths = []

        for session_id, group in df.groupby('full_session_id'):
            sorted_clicks = group.sort_values('timestamp')
            path = sorted_clicks['url__short_code'].tolist()

            if len(path) >= min_path_length:
                paths.append({
                    'session_id': session_id,
                    'path': path,
                    'length': len(path),
                    'converted': sorted_clicks['is_conversion'].any()
                })

        # Sort by length and limit
        paths.sort(key=lambda x: -x['length'])

        return paths[:max_paths]

    def get_path_frequencies(
        self,
        df: pd.DataFrame,
        path_length: int = 3
    ) -> pd.DataFrame:
        """
        Get frequency of specific path patterns.
        """
        paths = self.get_click_paths(df, min_path_length=path_length)

        if not paths:
            return pd.DataFrame()

        # Truncate paths to specified length and count
        truncated_paths = []
        for p in paths:
            truncated = tuple(p['path'][:path_length])
            truncated_paths.append({
                'path': ' -> '.join(truncated),
                'converted': p['converted']
            })

        path_df = pd.DataFrame(truncated_paths)

        result = path_df.groupby('path').agg({
            'converted': ['count', 'sum']
        })
        result.columns = ['occurrences', 'conversions']
        result['conversion_rate'] = (result['conversions'] / result['occurrences'] * 100).round(2)

        return result.sort_values('occurrences', ascending=False).reset_index()

    def get_entry_exit_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Analyze entry and exit points.
        """
        if df.empty:
            return {'entry_pages': {}, 'exit_pages': {}}

        if 'full_session_id' not in df.columns:
            reconstructor = SessionReconstructor()
            df = reconstructor.reconstruct_sessions(df)

        # Entry pages (first click in session)
        entry_clicks = df.loc[
            df.groupby('full_session_id')['timestamp'].idxmin()
        ]
        entry_pages = entry_clicks['url__short_code'].value_counts().to_dict()

        # Exit pages (last click in session)
        exit_clicks = df.loc[
            df.groupby('full_session_id')['timestamp'].idxmax()
        ]
        exit_pages = exit_clicks['url__short_code'].value_counts().to_dict()

        return {
            'entry_pages': entry_pages,
            'exit_pages': exit_pages
        }


class ConversionAttributor:
    """
    Multi-touch attribution modeling.
    """

    def __init__(self):
        self.models = ['first_touch', 'last_touch', 'linear', 'time_decay', 'position_based']

    def calculate_attribution(
        self,
        df: pd.DataFrame,
        model: str = 'linear'
    ) -> pd.DataFrame:
        """
        Calculate conversion attribution using specified model.
        """
        if model not in self.models:
            raise ValueError(f"Unknown model: {model}. Available: {self.models}")

        if df.empty:
            return pd.DataFrame()

        # Ensure sessions
        if 'full_session_id' not in df.columns:
            reconstructor = SessionReconstructor()
            df = reconstructor.reconstruct_sessions(df)

        # Get conversion sessions
        conversion_sessions = df[df['is_conversion']]['full_session_id'].unique()

        if len(conversion_sessions) == 0:
            return pd.DataFrame()

        attribution_data = []

        for session_id in conversion_sessions:
            session_clicks = df[df['full_session_id'] == session_id].sort_values('timestamp')
            conversion_value = session_clicks['conversion_value'].sum()

            touchpoints = session_clicks['url__short_code'].tolist()
            n = len(touchpoints)

            if n == 0:
                continue

            # Calculate weights based on model
            if model == 'first_touch':
                weights = [1.0] + [0.0] * (n - 1)
            elif model == 'last_touch':
                weights = [0.0] * (n - 1) + [1.0]
            elif model == 'linear':
                weights = [1.0 / n] * n
            elif model == 'time_decay':
                # More recent touchpoints get more credit
                decay_factor = 0.5
                weights = [decay_factor ** (n - i - 1) for i in range(n)]
                total = sum(weights)
                weights = [w / total for w in weights]
            elif model == 'position_based':
                # 40% first, 40% last, 20% middle
                if n == 1:
                    weights = [1.0]
                elif n == 2:
                    weights = [0.5, 0.5]
                else:
                    middle_weight = 0.2 / (n - 2) if n > 2 else 0
                    weights = [0.4] + [middle_weight] * (n - 2) + [0.4]

            # Attribute value
            for i, (url, weight) in enumerate(zip(touchpoints, weights)):
                attribution_data.append({
                    'url': url,
                    'session_id': session_id,
                    'position': i + 1,
                    'weight': weight,
                    'attributed_value': conversion_value * weight
                })

        attr_df = pd.DataFrame(attribution_data)

        # Aggregate by URL
        result = attr_df.groupby('url').agg({
            'attributed_value': 'sum',
            'weight': 'sum',
            'session_id': 'nunique'
        }).rename(columns={
            'attributed_value': 'total_attributed_value',
            'weight': 'total_credit',
            'session_id': 'assisted_conversions'
        })

        return result.sort_values('total_attributed_value', ascending=False).reset_index()

    def compare_models(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compare attribution across all models.
        """
        results = {}

        for model in self.models:
            attr = self.calculate_attribution(df, model)
            if not attr.empty:
                results[model] = attr.set_index('url')['total_attributed_value']

        if not results:
            return pd.DataFrame()

        comparison = pd.DataFrame(results)
        comparison = comparison.fillna(0)

        return comparison.reset_index()


class AggregationEngine:
    """
    Complex aggregations and pivots.
    """

    @staticmethod
    def pivot_by_dimensions(
        df: pd.DataFrame,
        rows: str,
        cols: str,
        values: str = 'id',
        aggfunc: str = 'count'
    ) -> pd.DataFrame:
        """
        Create pivot table by dimensions.
        """
        if df.empty:
            return pd.DataFrame()

        return pd.pivot_table(
            df,
            values=values,
            index=rows,
            columns=cols,
            aggfunc=aggfunc,
            fill_value=0
        )

    @staticmethod
    def rolling_metrics(
        df: pd.DataFrame,
        window: int = 7,
        metrics: List[str] = None
    ) -> pd.DataFrame:
        """
        Calculate rolling metrics.
        """
        if df.empty:
            return pd.DataFrame()

        if metrics is None:
            metrics = ['clicks', 'conversions', 'revenue']

        # Aggregate by date first
        daily = df.groupby('date').agg({
            'id': 'count',
            'is_conversion': 'sum',
            'conversion_value': 'sum'
        }).rename(columns={
            'id': 'clicks',
            'is_conversion': 'conversions',
            'conversion_value': 'revenue'
        })

        # Calculate rolling metrics
        for metric in metrics:
            if metric in daily.columns:
                daily[f'{metric}_rolling_{window}d'] = daily[metric].rolling(window).mean()
                daily[f'{metric}_rolling_sum_{window}d'] = daily[metric].rolling(window).sum()

        return daily.reset_index()

    @staticmethod
    def cohort_retention(
        df: pd.DataFrame,
        cohort_column: str = 'date',
        periods: int = 12
    ) -> pd.DataFrame:
        """
        Calculate cohort retention matrix.
        """
        if df.empty:
            return pd.DataFrame()

        # Get first visit date per user (IP)
        df['first_visit'] = df.groupby('ip_address')['timestamp'].transform('min')
        df['first_visit_date'] = pd.to_datetime(df['first_visit']).dt.to_period('W')
        df['visit_period'] = pd.to_datetime(df['timestamp']).dt.to_period('W')

        # Calculate period offset
        df['period_offset'] = (df['visit_period'] - df['first_visit_date']).apply(lambda x: x.n)

        # Create cohort matrix
        cohort_data = df.groupby(['first_visit_date', 'period_offset'])['ip_address'].nunique().reset_index()
        cohort_pivot = cohort_data.pivot(
            index='first_visit_date',
            columns='period_offset',
            values='ip_address'
        ).fillna(0)

        # Calculate retention percentages
        cohort_sizes = cohort_pivot[0]
        retention = cohort_pivot.divide(cohort_sizes, axis=0) * 100

        return retention.round(2)


class DataExporter:
    """
    Export data in various formats.
    """

    @staticmethod
    def to_csv(df: pd.DataFrame) -> str:
        """Export DataFrame to CSV string."""
        return df.to_csv(index=False)

    @staticmethod
    def to_excel(df: pd.DataFrame, sheet_name: str = 'Data') -> bytes:
        """Export DataFrame to Excel bytes."""
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        return output.getvalue()

    @staticmethod
    def to_json(df: pd.DataFrame, orient: str = 'records') -> str:
        """Export DataFrame to JSON string."""
        return df.to_json(orient=orient, date_format='iso')

    @staticmethod
    def to_parquet(df: pd.DataFrame) -> bytes:
        """Export DataFrame to Parquet bytes."""
        output = BytesIO()
        df.to_parquet(output, index=False)
        return output.getvalue()


class PandasAnalyticsService:
    """
    High-level service combining all Pandas analytics.
    """

    def __init__(self):
        self.df_builder = DataFrameBuilder()
        self.session_reconstructor = SessionReconstructor()
        self.path_analyzer = PathAnalyzer()
        self.attributor = ConversionAttributor()
        self.aggregator = AggregationEngine()
        self.exporter = DataExporter()

    def get_comprehensive_analytics(
        self,
        url_id: Optional[int] = None,
        days_back: int = 30
    ) -> Dict:
        """
        Get comprehensive analytics for a URL or all URLs.
        """
        df = self.df_builder.clicks_to_dataframe(url_id, days_back)

        if df.empty:
            return {'error': 'No data available'}

        # Reconstruct sessions
        df = self.session_reconstructor.reconstruct_sessions(df)
        sessions = self.session_reconstructor.get_session_metrics(df)

        # Basic metrics
        metrics = {
            'total_clicks': len(df),
            'unique_visitors': df['ip_address'].nunique(),
            'total_sessions': len(sessions),
            'total_conversions': df['is_conversion'].sum(),
            'total_revenue': df['conversion_value'].sum(),
            'conversion_rate': round(df['is_conversion'].mean() * 100, 2)
        }

        # Session metrics
        if not sessions.empty:
            metrics['avg_session_duration'] = round(sessions['duration_seconds'].mean(), 2)
            metrics['bounce_rate'] = round(sessions['is_bounced'].mean() * 100, 2)
            metrics['avg_pages_per_session'] = round(sessions['pages_per_session'].mean(), 2)

        # Device breakdown
        device_breakdown = df.groupby('device_type').agg({
            'id': 'count',
            'is_conversion': 'sum'
        }).to_dict()

        # Country breakdown
        country_breakdown = df.groupby('country_code').size().nlargest(10).to_dict()

        # Hourly pattern
        hourly_pattern = df.groupby('hour')['id'].count().to_dict()

        # Path analysis
        entry_exit = self.path_analyzer.get_entry_exit_analysis(df)

        return {
            'metrics': metrics,
            'device_breakdown': device_breakdown,
            'top_countries': country_breakdown,
            'hourly_pattern': hourly_pattern,
            'entry_pages': entry_exit['entry_pages'],
            'exit_pages': entry_exit['exit_pages']
        }

    def get_conversion_funnel(
        self,
        url_ids: List[int],
        days_back: int = 30
    ) -> Dict:
        """
        Analyze conversion funnel for specified URLs.
        """
        df = self.df_builder.clicks_to_dataframe(days_back=days_back)

        if df.empty:
            return {'error': 'No data available'}

        # Filter to funnel URLs
        df = df[df['url_id'].isin(url_ids)]
        df = self.session_reconstructor.reconstruct_sessions(df)

        funnel_data = []
        previous_visitors = None

        for i, url_id in enumerate(url_ids):
            url_df = df[df['url_id'] == url_id]

            visitors = url_df['ip_address'].nunique()
            sessions = url_df['full_session_id'].nunique()

            drop_off = 0
            if previous_visitors is not None and previous_visitors > 0:
                drop_off = round((1 - visitors / previous_visitors) * 100, 2)

            funnel_data.append({
                'step': i + 1,
                'url_id': url_id,
                'visitors': visitors,
                'sessions': sessions,
                'drop_off_percent': drop_off
            })

            previous_visitors = visitors

        return {
            'funnel': funnel_data,
            'overall_conversion': round(
                funnel_data[-1]['visitors'] / funnel_data[0]['visitors'] * 100, 2
            ) if funnel_data and funnel_data[0]['visitors'] > 0 else 0
        }

    def export_data(
        self,
        url_id: Optional[int] = None,
        days_back: int = 30,
        format: str = 'csv'
    ) -> Any:
        """
        Export click data in specified format.
        """
        df = self.df_builder.clicks_to_dataframe(url_id, days_back)

        if format == 'csv':
            return self.exporter.to_csv(df)
        elif format == 'json':
            return self.exporter.to_json(df)
        elif format == 'excel':
            return self.exporter.to_excel(df)
        elif format == 'parquet':
            return self.exporter.to_parquet(df)
        else:
            raise ValueError(f"Unknown format: {format}")

    def get_attribution_report(
        self,
        days_back: int = 30,
        model: str = 'linear'
    ) -> Dict:
        """
        Get attribution report.
        """
        df = self.df_builder.clicks_to_dataframe(days_back=days_back)

        attribution = self.attributor.calculate_attribution(df, model)
        comparison = self.attributor.compare_models(df)

        return {
            'attribution': attribution.to_dict('records') if not attribution.empty else [],
            'model_comparison': comparison.to_dict('records') if not comparison.empty else [],
            'model_used': model
        }
