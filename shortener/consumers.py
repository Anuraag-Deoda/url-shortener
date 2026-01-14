"""
WebSocket consumers for real-time analytics
"""
import json
from channels.generic.websocket import AsyncWebsocketConsumer
from asgiref.sync import sync_to_async


class AnalyticsConsumer(AsyncWebsocketConsumer):
    """WebSocket consumer for real-time analytics updates"""

    async def connect(self):
        self.room_group_name = 'analytics_live'

        # Join room group
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )

        await self.accept()

        # Send initial stats on connect
        await self.send_initial_stats()

    async def disconnect(self, close_code):
        # Leave room group
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )

    async def receive(self, text_data):
        """Handle incoming messages from client"""
        try:
            data = json.loads(text_data)
            message_type = data.get('type')

            if message_type == 'get_stats':
                await self.send_live_stats()
            elif message_type == 'get_recent':
                await self.send_recent_clicks()
            elif message_type == 'get_per_minute':
                await self.send_clicks_per_minute()
        except json.JSONDecodeError:
            pass

    async def analytics_event(self, event):
        """Handle analytics event from channel layer"""
        await self.send(text_data=json.dumps(event['data']))

    async def send_initial_stats(self):
        """Send initial statistics on connection"""
        stats = await self.get_live_stats()
        await self.send(text_data=json.dumps({
            'type': 'initial',
            'stats': stats
        }))

    async def send_live_stats(self):
        """Send current live statistics"""
        stats = await self.get_live_stats()
        await self.send(text_data=json.dumps({
            'type': 'stats',
            'data': stats
        }))

    async def send_recent_clicks(self):
        """Send recent click events"""
        from .services.realtime import RealTimeAnalyticsService
        recent = await sync_to_async(RealTimeAnalyticsService.get_recent_clicks)(20)
        await self.send(text_data=json.dumps({
            'type': 'recent',
            'data': recent
        }))

    async def send_clicks_per_minute(self):
        """Send clicks per minute data"""
        from .services.realtime import RealTimeAnalyticsService
        data = await sync_to_async(RealTimeAnalyticsService.get_clicks_per_minute)()
        await self.send(text_data=json.dumps({
            'type': 'per_minute',
            'data': data
        }))

    @sync_to_async
    def get_live_stats(self):
        from .services.realtime import RealTimeAnalyticsService
        return RealTimeAnalyticsService.get_live_stats()
