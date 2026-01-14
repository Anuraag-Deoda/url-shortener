"""
Real-time analytics service for broadcasting click events via WebSockets
"""
import json
from datetime import datetime, timedelta
from django.utils import timezone
from django.core.cache import cache
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync


class RealTimeAnalyticsService:
    """Service for broadcasting real-time analytics events"""

    CHANNEL_GROUP = 'analytics_live'
    CACHE_KEY_CLICKS = 'realtime_clicks_{}'
    CACHE_KEY_VISITORS = 'realtime_visitors_{}'
    CACHE_TTL = 300  # 5 minutes

    @classmethod
    def broadcast_click(cls, url, click_data: dict):
        """Broadcast a click event to all connected clients"""
        channel_layer = get_channel_layer()

        event_data = {
            'type': 'click',
            'timestamp': datetime.now().isoformat(),
            'short_code': url.short_code,
            'original_url': url.original_url[:50],
            'total_clicks': url.clicks,
            'country': click_data.get('country'),
            'city': click_data.get('city'),
            'device_type': click_data.get('device_type'),
            'browser': click_data.get('browser'),
            'os': click_data.get('os'),
            'latitude': click_data.get('latitude'),
            'longitude': click_data.get('longitude'),
        }

        try:
            async_to_sync(channel_layer.group_send)(
                cls.CHANNEL_GROUP,
                {
                    'type': 'analytics_event',
                    'data': event_data
                }
            )
        except Exception:
            # Silently fail if channel layer is not available
            pass

        # Update cache for recent activity
        cls._update_recent_clicks(url.short_code, event_data)

    @classmethod
    def _update_recent_clicks(cls, short_code: str, event_data: dict):
        """Update cached recent clicks"""
        cache_key = cls.CACHE_KEY_CLICKS.format('all')
        recent = cache.get(cache_key, [])

        recent.insert(0, event_data)
        recent = recent[:50]  # Keep last 50

        cache.set(cache_key, recent, cls.CACHE_TTL)

    @classmethod
    def get_recent_clicks(cls, limit: int = 20) -> list:
        """Get recent click events from cache"""
        cache_key = cls.CACHE_KEY_CLICKS.format('all')
        recent = cache.get(cache_key, [])
        return recent[:limit]

    @classmethod
    def get_live_stats(cls) -> dict:
        """Get live statistics snapshot"""
        from shortener.models import ClickData, URL

        now = timezone.now()
        one_hour_ago = now - timedelta(hours=1)
        five_mins_ago = now - timedelta(minutes=5)

        # Last hour clicks
        hourly_clicks = ClickData.objects.filter(timestamp__gte=one_hour_ago).count()

        # Last 5 minutes clicks (for "right now" metric)
        recent_clicks = ClickData.objects.filter(timestamp__gte=five_mins_ago).count()

        # Active URLs (clicked in last hour)
        active_urls = ClickData.objects.filter(
            timestamp__gte=one_hour_ago
        ).values('url').distinct().count()

        # Unique visitors in last hour
        unique_visitors = ClickData.objects.filter(
            timestamp__gte=one_hour_ago
        ).values('ip_address').distinct().count()

        # Top URLs right now
        from django.db.models import Count
        top_urls = list(
            ClickData.objects.filter(timestamp__gte=one_hour_ago)
            .values('url__short_code', 'url__original_url')
            .annotate(clicks=Count('id'))
            .order_by('-clicks')[:5]
        )

        return {
            'hourly_clicks': hourly_clicks,
            'recent_clicks': recent_clicks,
            'active_urls': active_urls,
            'unique_visitors': unique_visitors,
            'top_urls': top_urls,
            'timestamp': now.isoformat(),
        }

    @classmethod
    def get_clicks_per_minute(cls, minutes: int = 60) -> list:
        """Get clicks per minute for the last N minutes"""
        from shortener.models import ClickData
        from django.db.models import Count
        from django.db.models.functions import TruncMinute

        now = timezone.now()
        start_time = now - timedelta(minutes=minutes)

        clicks_by_minute = list(
            ClickData.objects.filter(timestamp__gte=start_time)
            .annotate(minute=TruncMinute('timestamp'))
            .values('minute')
            .annotate(count=Count('id'))
            .order_by('minute')
        )

        return [
            {
                'minute': item['minute'].isoformat(),
                'count': item['count']
            }
            for item in clicks_by_minute
        ]
