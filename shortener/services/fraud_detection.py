"""
Click Fraud Detection Service
Detects and prevents click fraud using configurable rules
"""
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from django.db.models import Count
from django.utils import timezone

from shortener.models import ClickData, ClickFraudRule, FraudAlert


class FraudDetectionService:
    """Service for detecting click fraud"""

    # Known bot user agent patterns
    BOT_PATTERNS = [
        r'bot', r'spider', r'crawler', r'scraper', r'curl', r'wget',
        r'python-requests', r'httpx', r'aiohttp', r'scrapy',
        r'headless', r'phantom', r'selenium', r'puppeteer',
        r'slurp', r'baiduspider', r'yandex', r'sogou',
        r'exabot', r'facebot', r'facebookexternalhit',
        r'archive\.org', r'ia_archiver', r'mj12bot',
        r'ahrefsbot', r'semrush', r'dotbot', r'rogerbot',
        r'screaming frog', r'gtmetrix', r'pingdom',
        r'pagespeed', r'google-structured-data-testing-tool'
    ]

    # Datacenter IP ranges (simplified - in production use full lists)
    DATACENTER_ASN_PATTERNS = [
        'amazonaws', 'googlecloud', 'digitalocean', 'linode',
        'vultr', 'ovh', 'hetzner', 'azure', 'cloudflare'
    ]

    def __init__(self):
        self.active_rules = ClickFraudRule.objects.filter(is_active=True).order_by('-priority')

    def check_click(self, click_data: dict) -> Tuple[bool, List[Dict]]:
        """
        Check if a click is fraudulent.
        Returns (is_fraud, list of triggered rules)
        """
        triggered_rules = []
        is_fraud = False

        # Run all active rules
        for rule in self.active_rules:
            matched, reason = rule.check_click(click_data)
            if matched:
                triggered_rules.append({
                    'rule_id': rule.id,
                    'rule_name': rule.name,
                    'rule_type': rule.rule_type,
                    'action': rule.action,
                    'reason': reason,
                    'redirect_url': rule.redirect_url
                })
                if rule.action in ['block', 'challenge']:
                    is_fraud = True

        # Additional built-in checks
        additional_checks = self._run_built_in_checks(click_data)
        triggered_rules.extend(additional_checks)
        if any(check.get('action') == 'block' for check in additional_checks):
            is_fraud = True

        return is_fraud, triggered_rules

    def _run_built_in_checks(self, click_data: dict) -> List[Dict]:
        """Run built-in fraud detection checks"""
        triggered = []

        # Check for bot user agent
        if self._is_bot_user_agent(click_data.get('user_agent', '')):
            triggered.append({
                'rule_name': 'Built-in Bot Detection',
                'rule_type': 'bot_detection',
                'action': 'flag',
                'reason': 'User agent matches known bot pattern'
            })

        # Check for empty/suspicious user agent
        user_agent = click_data.get('user_agent', '')
        if not user_agent or len(user_agent) < 20:
            triggered.append({
                'rule_name': 'Suspicious User Agent',
                'rule_type': 'user_agent_pattern',
                'action': 'flag',
                'reason': 'Empty or suspiciously short user agent'
            })

        # Check for rapid succession clicks
        rapid_clicks = self._check_rapid_clicks(
            click_data.get('ip_address'),
            click_data.get('url_id')
        )
        if rapid_clicks:
            triggered.append({
                'rule_name': 'Rapid Click Detection',
                'rule_type': 'ip_frequency',
                'action': 'flag',
                'reason': f'Multiple rapid clicks detected ({rapid_clicks} in last minute)'
            })

        # Check click timing patterns
        if self._check_suspicious_timing(click_data):
            triggered.append({
                'rule_name': 'Suspicious Timing Pattern',
                'rule_type': 'timing_analysis',
                'action': 'flag',
                'reason': 'Click timing suggests automated behavior'
            })

        return triggered

    def _is_bot_user_agent(self, user_agent: str) -> bool:
        """Check if user agent matches bot patterns"""
        if not user_agent:
            return False

        user_agent_lower = user_agent.lower()
        for pattern in self.BOT_PATTERNS:
            if re.search(pattern, user_agent_lower):
                return True
        return False

    def _check_rapid_clicks(self, ip_address: str, url_id: int = None) -> int:
        """Check for rapid succession clicks from same IP"""
        if not ip_address:
            return 0

        one_minute_ago = timezone.now() - timedelta(minutes=1)
        queryset = ClickData.objects.filter(
            ip_address=ip_address,
            timestamp__gte=one_minute_ago
        )

        if url_id:
            queryset = queryset.filter(url_id=url_id)

        count = queryset.count()

        # More than 10 clicks per minute is suspicious
        return count if count > 10 else 0

    def _check_suspicious_timing(self, click_data: dict) -> bool:
        """Check for suspicious timing patterns (e.g., perfectly regular intervals)"""
        ip_address = click_data.get('ip_address')
        if not ip_address:
            return False

        # Get last 10 clicks from this IP
        recent_clicks = ClickData.objects.filter(
            ip_address=ip_address
        ).order_by('-timestamp')[:10]

        if recent_clicks.count() < 5:
            return False

        # Calculate intervals between clicks
        timestamps = [c.timestamp for c in recent_clicks]
        intervals = []

        for i in range(len(timestamps) - 1):
            interval = (timestamps[i] - timestamps[i + 1]).total_seconds()
            intervals.append(interval)

        if not intervals:
            return False

        # Check for suspiciously regular intervals (within 1 second variance)
        avg_interval = sum(intervals) / len(intervals)
        variance = sum((i - avg_interval) ** 2 for i in intervals) / len(intervals)

        # Very low variance suggests automated clicking
        return variance < 1 and avg_interval < 10

    def create_fraud_alert(
        self,
        click: ClickData,
        rule: Optional[ClickFraudRule],
        reason: str,
        severity: str = 'medium'
    ) -> FraudAlert:
        """Create a fraud alert for a suspicious click"""
        return FraudAlert.objects.create(
            rule=rule,
            click=click,
            severity=severity,
            reason=reason,
            ip_address=click.ip_address,
            user_agent=click.user_agent or ''
        )

    def get_fraud_summary(self, days: int = 7) -> Dict:
        """Get fraud detection summary for dashboard"""
        start_date = timezone.now() - timedelta(days=days)

        alerts = FraudAlert.objects.filter(created_at__gte=start_date)

        # Count by severity
        severity_counts = alerts.values('severity').annotate(
            count=Count('id')
        )

        # Count by rule type
        rule_counts = alerts.values('rule__rule_type').annotate(
            count=Count('id')
        )

        # Count by status
        status_counts = alerts.values('status').annotate(
            count=Count('id')
        )

        # Top offending IPs
        top_ips = alerts.values('ip_address').annotate(
            count=Count('id')
        ).order_by('-count')[:10]

        # Daily trend
        daily_trend = alerts.extra(
            select={'date': 'date(created_at)'}
        ).values('date').annotate(
            count=Count('id')
        ).order_by('date')

        return {
            'total_alerts': alerts.count(),
            'severity_breakdown': {
                item['severity']: item['count'] for item in severity_counts
            },
            'rule_breakdown': {
                item['rule__rule_type'] or 'built_in': item['count']
                for item in rule_counts
            },
            'status_breakdown': {
                item['status']: item['count'] for item in status_counts
            },
            'top_offending_ips': list(top_ips),
            'daily_trend': list(daily_trend),
            'blocked_clicks': alerts.filter(
                rule__action='block'
            ).count() if alerts.filter(rule__isnull=False).exists() else 0,
        }

    def analyze_ip(self, ip_address: str, days: int = 30) -> Dict:
        """Analyze click patterns from a specific IP"""
        start_date = timezone.now() - timedelta(days=days)

        clicks = ClickData.objects.filter(
            ip_address=ip_address,
            timestamp__gte=start_date
        )

        alerts = FraudAlert.objects.filter(
            ip_address=ip_address,
            created_at__gte=start_date
        )

        # Click distribution by hour
        hourly_distribution = clicks.extra(
            select={'hour': 'extract(hour from timestamp)'}
        ).values('hour').annotate(count=Count('id'))

        # URLs clicked
        urls_clicked = clicks.values('url__short_code').annotate(
            count=Count('id')
        ).order_by('-count')[:10]

        # User agents used
        user_agents = clicks.values('user_agent').annotate(
            count=Count('id')
        ).order_by('-count')[:5]

        # Calculate risk score (0-100)
        risk_factors = []

        # Factor: Alert count
        alert_count = alerts.count()
        if alert_count > 10:
            risk_factors.append(30)
        elif alert_count > 5:
            risk_factors.append(20)
        elif alert_count > 0:
            risk_factors.append(10)

        # Factor: High frequency clicking
        click_count = clicks.count()
        if click_count > 1000:
            risk_factors.append(25)
        elif click_count > 500:
            risk_factors.append(15)
        elif click_count > 100:
            risk_factors.append(5)

        # Factor: Bot-like user agents
        bot_clicks = 0
        for click in clicks:
            if click.user_agent and self._is_bot_user_agent(click.user_agent):
                bot_clicks += 1

        if bot_clicks > 0:
            bot_ratio = bot_clicks / click_count if click_count > 0 else 0
            risk_factors.append(int(bot_ratio * 30))

        # Factor: Multiple user agents
        ua_count = clicks.values('user_agent').distinct().count()
        if ua_count > 10:
            risk_factors.append(15)
        elif ua_count > 5:
            risk_factors.append(10)

        risk_score = min(100, sum(risk_factors))

        return {
            'ip_address': ip_address,
            'total_clicks': click_count,
            'total_alerts': alert_count,
            'risk_score': risk_score,
            'risk_level': 'high' if risk_score > 70 else 'medium' if risk_score > 40 else 'low',
            'first_seen': clicks.order_by('timestamp').first().timestamp if clicks.exists() else None,
            'last_seen': clicks.order_by('-timestamp').first().timestamp if clicks.exists() else None,
            'hourly_distribution': list(hourly_distribution),
            'top_urls': list(urls_clicked),
            'user_agents': list(user_agents),
            'countries': list(clicks.values('country').distinct()[:5]),
        }

    def get_recommended_rules(self) -> List[Dict]:
        """Get recommended fraud detection rules based on patterns"""
        recommendations = []

        # Analyze recent click patterns
        recent_clicks = ClickData.objects.filter(
            timestamp__gte=timezone.now() - timedelta(days=7)
        )

        # Check for high-frequency IPs
        high_freq_ips = recent_clicks.values('ip_address').annotate(
            count=Count('id')
        ).filter(count__gt=100).order_by('-count')

        if high_freq_ips.exists():
            recommendations.append({
                'rule_type': 'ip_frequency',
                'reason': f'Found {high_freq_ips.count()} IPs with >100 clicks/week',
                'suggested_config': {
                    'max_per_minute': 10,
                    'max_daily': 100
                }
            })

        # Check for bot traffic
        bot_clicks = 0
        total_clicks = recent_clicks.count()

        for click in recent_clicks[:1000]:  # Sample
            if click.user_agent and self._is_bot_user_agent(click.user_agent):
                bot_clicks += 1

        if bot_clicks > total_clicks * 0.1:  # >10% bot traffic
            recommendations.append({
                'rule_type': 'bot_detection',
                'reason': f'Detected ~{int(bot_clicks/10)}% bot traffic',
                'suggested_config': {
                    'action': 'block'
                }
            })

        # Check for suspicious referrers
        suspicious_referrers = recent_clicks.filter(
            referrer__icontains='click'
        ).values('referrer').annotate(
            count=Count('id')
        ).filter(count__gt=10)

        if suspicious_referrers.exists():
            recommendations.append({
                'rule_type': 'referrer_pattern',
                'reason': f'Found {suspicious_referrers.count()} suspicious referrer patterns',
                'suggested_config': {
                    'patterns': ['click.*exchange', 'traffic.*bot', 'auto.*click']
                }
            })

        return recommendations


class RealTimeFraudMonitor:
    """Real-time fraud monitoring with scoring"""

    def __init__(self):
        self.service = FraudDetectionService()
        self.ip_scores = {}  # In-memory cache for real-time scoring

    def process_click(self, click_data: dict) -> Tuple[bool, Dict]:
        """
        Process a click in real-time.
        Returns (should_block, response_data)
        """
        is_fraud, triggered_rules = self.service.check_click(click_data)

        response = {
            'fraud_detected': is_fraud,
            'triggered_rules': triggered_rules,
            'action': 'allow'
        }

        if triggered_rules:
            # Determine action based on highest priority rule
            actions_priority = {'block': 3, 'challenge': 2, 'redirect': 1, 'flag': 0}
            max_action = max(
                triggered_rules,
                key=lambda x: actions_priority.get(x.get('action', 'flag'), 0)
            )

            response['action'] = max_action.get('action', 'flag')

            if response['action'] == 'redirect':
                response['redirect_url'] = max_action.get('redirect_url')
            elif response['action'] == 'challenge':
                response['challenge_type'] = 'captcha'

        return is_fraud, response

    def update_ip_score(self, ip_address: str, delta: int):
        """Update real-time IP risk score"""
        current = self.ip_scores.get(ip_address, 50)
        self.ip_scores[ip_address] = max(0, min(100, current + delta))

    def get_ip_score(self, ip_address: str) -> int:
        """Get current IP risk score"""
        return self.ip_scores.get(ip_address, 50)
