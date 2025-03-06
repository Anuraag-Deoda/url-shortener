from datetime import datetime, timedelta
from django.db.models import Count, Sum
from django.db.models.functions import TruncDate, TruncHour
from django.core.cache import cache
from shortener.models import URL, ClickData
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

class AnalyticsService:
    CACHE_TTL = 3600  # 1 hour

    @staticmethod
    def get_url_stats(url_id: int) -> Dict[str, Any]:
        cache_key = f'url_stats_{url_id}'
        cached_stats = cache.get(cache_key)
        if cached_stats:
            return cached_stats

        url = URL.objects.get(id=url_id)
        clicks = ClickData.objects.filter(url=url)
        
        # Basic stats
        total_clicks = clicks.count()
        unique_visitors = clicks.values('visitor_id').distinct().count()
        
        # Time-based analysis
        hourly_clicks = clicks.annotate(
            hour=TruncHour('created_at')
        ).values('hour').annotate(
            count=Count('id')
        ).order_by('hour')
        
        # Convert to pandas for advanced analytics
        df = pd.DataFrame(hourly_clicks)
        if not df.empty:
            df['hour'] = pd.to_datetime(df['hour'])
            df.set_index('hour', inplace=True)
            
            # Calculate moving averages
            df['ma_4h'] = df['count'].rolling(window=4).mean()
            df['ma_24h'] = df['count'].rolling(window=24).mean()
            
            # Detect peak hours
            peak_hours = df[df['count'] > df['count'].mean() + df['count'].std()]
            
            # Predict next hour's clicks (simple)
            last_4h_avg = df['count'].tail(4).mean()
            prediction = max(0, last_4h_avg * (1 + np.random.normal(0, 0.1)))
        else:
            peak_hours = pd.DataFrame()
            prediction = 0

        stats = {
            'total_clicks': total_clicks,
            'unique_visitors': unique_visitors,
            'conversion_rate': (unique_visitors / total_clicks * 100) if total_clicks else 0,
            'peak_hours': peak_hours.to_dict('records') if not peak_hours.empty else [],
            'predicted_next_hour': round(prediction, 2),
            'hourly_data': [
                {
                    'hour': row['hour'].isoformat(),
                    'clicks': row['count'],
                    'ma_4h': row['ma_4h'] if 'ma_4h' in df else None,
                    'ma_24h': row['ma_24h'] if 'ma_24h' in df else None
                }
                for _, row in df.iterrows()
            ] if not df.empty else []
        }
        
        cache.set(cache_key, stats, AnalyticsService.CACHE_TTL)
        return stats

    @staticmethod
    def get_dashboard_stats() -> Dict[str, Any]:
        cache_key = 'dashboard_stats'
        cached_stats = cache.get(cache_key)
        if cached_stats:
            return cached_stats

        # Time ranges
        now = datetime.now()
        today = now.date()
        yesterday = today - timedelta(days=1)
        last_week = today - timedelta(days=7)
        last_month = today - timedelta(days=30)

        # Aggregate queries
        urls = URL.objects.all()
        clicks = ClickData.objects.all()
        
        stats = {
            'total_urls': urls.count(),
            'total_clicks': clicks.count(),
            'today_clicks': clicks.filter(created_at__date=today).count(),
            'yesterday_clicks': clicks.filter(created_at__date=yesterday).count(),
            'growth_rate': {
                'daily': AnalyticsService._calculate_growth_rate(
                    clicks.filter(created_at__date=today).count(),
                    clicks.filter(created_at__date=yesterday).count()
                ),
                'weekly': AnalyticsService._calculate_growth_rate(
                    clicks.filter(created_at__date__gte=today - timedelta(days=7)).count(),
                    clicks.filter(created_at__date__gte=today - timedelta(days=14), 
                                created_at__date__lt=today - timedelta(days=7)).count()
                ),
                'monthly': AnalyticsService._calculate_growth_rate(
                    clicks.filter(created_at__date__gte=today - timedelta(days=30)).count(),
                    clicks.filter(created_at__date__gte=today - timedelta(days=60),
                                created_at__date__lt=today - timedelta(days=30)).count()
                )
            },
            'top_urls': AnalyticsService._get_top_urls(),
            'engagement_metrics': AnalyticsService._calculate_engagement_metrics(),
            'retention_rate': AnalyticsService._calculate_retention_rate()
        }
        
        cache.set(cache_key, stats, AnalyticsService.CACHE_TTL)
        return stats

    @staticmethod
    def _calculate_growth_rate(current: int, previous: int) -> float:
        if previous == 0:
            return 100 if current > 0 else 0
        return ((current - previous) / previous) * 100

    @staticmethod
    def _get_top_urls(limit: int = 10) -> List[Dict[str, Any]]:
        return list(URL.objects.annotate(
            click_count=Count('clicks'),
            unique_visitors=Count('clicks__visitor_id', distinct=True)
        ).values('original_url', 'short_code', 'click_count', 'unique_visitors')
        .order_by('-click_count')[:limit])

    @staticmethod
    def _calculate_engagement_metrics() -> Dict[str, Any]:
        clicks = ClickData.objects.all()
        total_clicks = clicks.count()
        
        if total_clicks == 0:
            return {
                'avg_session_duration': 0,
                'bounce_rate': 0,
                'peak_hours': []
            }

        # Calculate average session duration
        sessions = clicks.values('visitor_id', 'created_at').order_by('visitor_id', 'created_at')
        session_durations = []
        
        for visitor_id, group in pd.DataFrame(sessions).groupby('visitor_id'):
            if len(group) > 1:
                duration = (group['created_at'].max() - group['created_at'].min()).total_seconds()
                session_durations.append(duration)
        
        avg_session_duration = sum(session_durations) / len(session_durations) if session_durations else 0
        
        # Calculate bounce rate (visitors with only one click)
        single_click_visitors = clicks.values('visitor_id').annotate(
            click_count=Count('id')
        ).filter(click_count=1).count()
        
        total_visitors = clicks.values('visitor_id').distinct().count()
        bounce_rate = (single_click_visitors / total_visitors * 100) if total_visitors > 0 else 0
        
        # Find peak hours
        peak_hours = clicks.annotate(
            hour=TruncHour('created_at')
        ).values('hour').annotate(
            count=Count('id')
        ).order_by('-count')[:5]
        
        return {
            'avg_session_duration': round(avg_session_duration, 2),
            'bounce_rate': round(bounce_rate, 2),
            'peak_hours': list(peak_hours)
        }

    @staticmethod
    def _calculate_retention_rate() -> Dict[str, float]:
        today = datetime.now().date()
        periods = [7, 14, 30]  # days
        retention_rates = {}
        
        for period in periods:
            period_start = today - timedelta(days=period)
            period_end = today - timedelta(days=period-7)
            
            # Get visitors from initial period
            initial_visitors = set(ClickData.objects.filter(
                created_at__date__gte=period_start,
                created_at__date__lt=period_end
            ).values_list('visitor_id', flat=True))
            
            # Get returning visitors
            returning_visitors = set(ClickData.objects.filter(
                created_at__date__gte=period_end,
                visitor_id__in=initial_visitors
            ).values_list('visitor_id', flat=True))
            
            retention_rate = (len(returning_visitors) / len(initial_visitors) * 100) if initial_visitors else 0
            retention_rates[f'{period}d'] = round(retention_rate, 2)
        
        return retention_rates 