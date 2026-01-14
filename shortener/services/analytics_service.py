"""
Advanced Analytics Service
Provides cohort analysis, attribution modeling, and campaign analytics
"""
import hashlib
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from django.db import models
from django.db.models import Count, Sum, Avg, F, Q
from django.db.models.functions import TruncDate, TruncWeek, TruncMonth
from django.utils import timezone

from shortener.models import (
    ClickData, Campaign, CampaignURL, ABTest, ABTestVariant,
    Cohort, Attribution, TouchPoint
)


class CampaignAnalyticsService:
    """Analytics service for marketing campaigns"""

    @staticmethod
    def get_campaign_performance(campaign: Campaign, days: int = 30) -> Dict:
        """Get comprehensive campaign performance metrics"""
        start_date = timezone.now() - timedelta(days=days)

        clicks = ClickData.objects.filter(
            utm_campaign=campaign.utm_campaign,
            timestamp__gte=start_date
        )

        total_clicks = clicks.count()
        unique_visitors = clicks.values('visitor_id').distinct().count()
        conversions = clicks.filter(conversion_value__isnull=False).count()
        revenue = clicks.aggregate(total=Sum('conversion_value'))['total'] or Decimal('0')

        # Daily breakdown
        daily_stats = clicks.annotate(
            date=TruncDate('timestamp')
        ).values('date').annotate(
            clicks=Count('id'),
            visitors=Count('visitor_id', distinct=True),
            conversions=Count('id', filter=Q(conversion_value__isnull=False)),
            revenue=Sum('conversion_value')
        ).order_by('date')

        # Source breakdown
        source_stats = clicks.values('utm_source').annotate(
            clicks=Count('id'),
            visitors=Count('visitor_id', distinct=True),
            conversions=Count('id', filter=Q(conversion_value__isnull=False))
        ).order_by('-clicks')

        # Medium breakdown
        medium_stats = clicks.values('utm_medium').annotate(
            clicks=Count('id'),
            visitors=Count('visitor_id', distinct=True),
            conversions=Count('id', filter=Q(conversion_value__isnull=False))
        ).order_by('-clicks')

        conversion_rate = (conversions / unique_visitors * 100) if unique_visitors > 0 else 0
        cpc = (float(campaign.spent) / total_clicks) if total_clicks > 0 and campaign.spent else 0
        cpa = (float(campaign.spent) / conversions) if conversions > 0 and campaign.spent else 0

        return {
            'total_clicks': total_clicks,
            'unique_visitors': unique_visitors,
            'conversions': conversions,
            'revenue': float(revenue),
            'conversion_rate': round(conversion_rate, 2),
            'cpc': round(cpc, 2),
            'cpa': round(cpa, 2),
            'roi': campaign.get_roi(),
            'daily_stats': list(daily_stats),
            'source_stats': list(source_stats),
            'medium_stats': list(medium_stats),
        }

    @staticmethod
    def compare_campaigns(campaign_ids: List[int], days: int = 30) -> List[Dict]:
        """Compare multiple campaigns"""
        results = []
        for campaign_id in campaign_ids:
            try:
                campaign = Campaign.objects.get(pk=campaign_id)
                perf = CampaignAnalyticsService.get_campaign_performance(campaign, days)
                perf['campaign_name'] = campaign.name
                perf['campaign_id'] = campaign.id
                results.append(perf)
            except Campaign.DoesNotExist:
                continue
        return results


class CohortAnalyticsService:
    """Service for cohort analysis"""

    @staticmethod
    def generate_retention_cohorts(
        start_date: datetime,
        end_date: datetime,
        period: str = 'week'  # 'day', 'week', 'month'
    ) -> Dict:
        """Generate retention cohort matrix"""
        trunc_func = {
            'day': TruncDate,
            'week': TruncWeek,
            'month': TruncMonth,
        }.get(period, TruncWeek)

        # Get first click date for each visitor
        first_clicks = ClickData.objects.filter(
            timestamp__gte=start_date,
            timestamp__lte=end_date,
            visitor_id__isnull=False
        ).values('visitor_id').annotate(
            first_click=models.Min('timestamp')
        )

        # Build cohort lookup
        visitor_cohorts = {}
        for item in first_clicks:
            visitor_cohorts[item['visitor_id']] = item['first_click']

        # Get all clicks with period info
        clicks = ClickData.objects.filter(
            timestamp__gte=start_date,
            timestamp__lte=end_date,
            visitor_id__isnull=False
        ).annotate(
            period=trunc_func('timestamp')
        ).values('visitor_id', 'period')

        # Build retention matrix
        cohorts = defaultdict(lambda: defaultdict(set))

        for click in clicks:
            visitor_id = click['visitor_id']
            click_period = click['period']

            if visitor_id not in visitor_cohorts:
                continue

            cohort_date = visitor_cohorts[visitor_id]

            # Calculate period difference based on period type
            if period == 'day':
                period_diff = (click_period.date() - cohort_date.date()).days
            elif period == 'week':
                period_diff = (click_period.date() - cohort_date.date()).days // 7
            else:  # month
                period_diff = (
                    (click_period.year - cohort_date.year) * 12 +
                    click_period.month - cohort_date.month
                )

            cohorts[cohort_date.date()][period_diff].add(visitor_id)

        # Convert to retention percentages
        retention_matrix = []
        sorted_cohorts = sorted(cohorts.keys())

        for cohort_date in sorted_cohorts:
            cohort_data = cohorts[cohort_date]
            initial_size = len(cohort_data.get(0, set()))

            if initial_size == 0:
                continue

            row = {
                'cohort_date': cohort_date,
                'size': initial_size,
                'retention': []
            }

            max_periods = max(cohort_data.keys()) + 1 if cohort_data else 1
            for period_num in range(max_periods):
                retained = len(cohort_data.get(period_num, set()))
                retention_pct = round((retained / initial_size) * 100, 1)
                row['retention'].append({
                    'period': period_num,
                    'count': retained,
                    'percentage': retention_pct
                })

            retention_matrix.append(row)

        return {
            'period_type': period,
            'start_date': start_date,
            'end_date': end_date,
            'cohorts': retention_matrix
        }

    @staticmethod
    def analyze_cohort(cohort: Cohort, metrics: List[str] = None) -> Dict:
        """Analyze a specific cohort's behavior"""
        if metrics is None:
            metrics = ['clicks', 'conversions', 'revenue', 'bounce_rate']

        members = cohort.get_members()

        result = {
            'cohort_name': cohort.name,
            'cohort_type': cohort.cohort_type,
            'total_visitors': members.values('visitor_id').distinct().count(),
            'total_clicks': members.count(),
        }

        if 'conversions' in metrics:
            result['conversions'] = members.filter(conversion_value__isnull=False).count()
            result['conversion_rate'] = round(
                (result['conversions'] / result['total_visitors'] * 100)
                if result['total_visitors'] > 0 else 0, 2
            )

        if 'revenue' in metrics:
            revenue = members.aggregate(total=Sum('conversion_value'))['total']
            result['revenue'] = float(revenue) if revenue else 0
            result['avg_revenue_per_visitor'] = round(
                result['revenue'] / result['total_visitors']
                if result['total_visitors'] > 0 else 0, 2
            )

        if 'bounce_rate' in metrics:
            single_visit = members.values('visitor_id').annotate(
                visits=Count('id')
            ).filter(visits=1).count()
            result['bounce_rate'] = round(
                (single_visit / result['total_visitors'] * 100)
                if result['total_visitors'] > 0 else 0, 2
            )

        # Time-based breakdown
        result['daily_activity'] = list(
            members.annotate(date=TruncDate('timestamp')).values('date').annotate(
                clicks=Count('id'),
                visitors=Count('visitor_id', distinct=True)
            ).order_by('date')[:30]
        )

        return result


class ABTestAnalyticsService:
    """Service for A/B test analytics"""

    @staticmethod
    def get_test_results(ab_test: ABTest) -> Dict:
        """Get comprehensive A/B test results"""
        variants = ab_test.variants.all()

        variant_results = []
        for variant in variants:
            clicks = variant.clicks.all()

            variant_data = {
                'id': variant.id,
                'name': variant.name,
                'is_control': variant.is_control,
                'destination_url': variant.destination_url,
                'visitors': variant.get_visitors(),
                'clicks': variant.get_total_clicks(),
                'conversions': variant.get_conversions(),
                'conversion_rate': variant.get_conversion_rate(),
                'ctr': variant.get_ctr(),
                'bounce_rate': variant.get_bounce_rate(),
                'revenue': float(clicks.aggregate(
                    total=Sum('conversion_value')
                )['total'] or 0),
            }

            # Calculate confidence interval (using Wilson score)
            if variant_data['visitors'] > 0:
                p = variant_data['conversion_rate'] / 100
                n = variant_data['visitors']
                z = 1.96  # 95% confidence

                denominator = 1 + z**2 / n
                center = (p + z**2 / (2*n)) / denominator
                margin = z * ((p*(1-p) + z**2/(4*n)) / n)**0.5 / denominator

                variant_data['ci_lower'] = round(max(0, (center - margin)) * 100, 2)
                variant_data['ci_upper'] = round(min(1, (center + margin)) * 100, 2)
            else:
                variant_data['ci_lower'] = 0
                variant_data['ci_upper'] = 0

            variant_results.append(variant_data)

        # Calculate statistical significance
        is_significant = ab_test.is_statistically_significant()
        winner = ab_test.determine_winner()

        # Calculate lift
        control = next((v for v in variant_results if v['is_control']), None)
        for variant in variant_results:
            if control and not variant['is_control'] and control['conversion_rate'] > 0:
                variant['lift'] = round(
                    ((variant['conversion_rate'] - control['conversion_rate']) /
                     control['conversion_rate']) * 100, 2
                )
            else:
                variant['lift'] = 0

        return {
            'test_id': ab_test.id,
            'test_name': ab_test.name,
            'status': ab_test.status,
            'goal_type': ab_test.goal_type,
            'confidence_level': ab_test.confidence_level,
            'minimum_sample_size': ab_test.minimum_sample_size,
            'total_visitors': ab_test.get_total_visitors(),
            'is_significant': is_significant,
            'winner': winner.name if winner else None,
            'winner_id': winner.id if winner else None,
            'variants': variant_results,
        }

    @staticmethod
    def calculate_sample_size(
        baseline_rate: float,
        minimum_detectable_effect: float,
        confidence_level: float = 0.95,
        power: float = 0.8
    ) -> int:
        """Calculate required sample size for A/B test"""
        from scipy import stats

        alpha = 1 - confidence_level
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)

        p1 = baseline_rate
        p2 = baseline_rate * (1 + minimum_detectable_effect)

        pooled_p = (p1 + p2) / 2
        effect_size = abs(p2 - p1)

        if effect_size == 0:
            return float('inf')

        n = (2 * pooled_p * (1 - pooled_p) * (z_alpha + z_beta)**2) / effect_size**2

        return int(n) + 1


class AttributionService:
    """Service for multi-touch attribution"""

    @staticmethod
    def calculate_attribution(
        visitor_id: str,
        conversion_value: Decimal,
        model: str = 'linear'
    ) -> Dict:
        """Calculate attribution for a conversion"""
        # Get all touchpoints for this visitor
        touchpoints = ClickData.objects.filter(
            visitor_id=visitor_id
        ).order_by('timestamp')

        if not touchpoints.exists():
            return {}

        touchpoint_list = list(touchpoints)
        n = len(touchpoint_list)

        if n == 0:
            return {}

        credits = {}

        if model == 'first_touch':
            credits[touchpoint_list[0].id] = float(conversion_value)

        elif model == 'last_touch':
            credits[touchpoint_list[-1].id] = float(conversion_value)

        elif model == 'linear':
            credit_per_touch = float(conversion_value) / n
            for tp in touchpoint_list:
                credits[tp.id] = credit_per_touch

        elif model == 'time_decay':
            # Decay factor: 50% half-life of 7 days
            half_life = 7 * 24 * 3600  # seconds
            conversion_time = touchpoint_list[-1].timestamp

            weights = []
            for tp in touchpoint_list:
                time_diff = (conversion_time - tp.timestamp).total_seconds()
                weight = 2 ** (-time_diff / half_life)
                weights.append(weight)

            total_weight = sum(weights)
            for i, tp in enumerate(touchpoint_list):
                credits[tp.id] = float(conversion_value) * weights[i] / total_weight

        elif model == 'position_based':
            # 40% first, 40% last, 20% distributed among middle
            if n == 1:
                credits[touchpoint_list[0].id] = float(conversion_value)
            elif n == 2:
                credits[touchpoint_list[0].id] = float(conversion_value) * 0.5
                credits[touchpoint_list[-1].id] = float(conversion_value) * 0.5
            else:
                first_credit = float(conversion_value) * 0.4
                last_credit = float(conversion_value) * 0.4
                middle_credit = float(conversion_value) * 0.2 / (n - 2)

                credits[touchpoint_list[0].id] = first_credit
                credits[touchpoint_list[-1].id] = last_credit

                for tp in touchpoint_list[1:-1]:
                    credits[tp.id] = middle_credit

        return credits

    @staticmethod
    def get_channel_attribution(
        start_date: datetime,
        end_date: datetime,
        model: str = 'linear'
    ) -> Dict:
        """Get attribution by channel/source"""
        attributions = Attribution.objects.filter(
            converted_at__gte=start_date,
            converted_at__lte=end_date
        )

        channel_credits = defaultdict(float)
        channel_conversions = defaultdict(int)

        for attr in attributions:
            model_credits = attr.attributions.get(model, {})

            for touchpoint in attr.touchpoints.all():
                credit = model_credits.get(str(touchpoint.click_id), 0)
                channel = touchpoint.channel or 'direct'

                channel_credits[channel] += credit
                channel_conversions[channel] += 1

        results = []
        for channel in channel_credits:
            results.append({
                'channel': channel,
                'attributed_revenue': round(channel_credits[channel], 2),
                'conversions': channel_conversions[channel],
            })

        return sorted(results, key=lambda x: x['attributed_revenue'], reverse=True)

    @staticmethod
    def create_attribution_record(
        visitor_id: str,
        conversion_value: Decimal,
        currency: str = 'USD'
    ) -> Attribution:
        """Create attribution record with all models calculated"""
        conversion_id = hashlib.sha256(
            f"{visitor_id}:{timezone.now().isoformat()}".encode()
        ).hexdigest()[:32]

        # Calculate attribution for all models
        models_list = ['first_touch', 'last_touch', 'linear', 'time_decay', 'position_based']
        attributions = {}

        for model in models_list:
            credits = AttributionService.calculate_attribution(
                visitor_id, conversion_value, model
            )
            attributions[model] = {str(k): v for k, v in credits.items()}

        # Create attribution record
        attr = Attribution.objects.create(
            visitor_id=visitor_id,
            conversion_id=conversion_id,
            conversion_value=conversion_value,
            conversion_currency=currency,
            converted_at=timezone.now(),
            attributions=attributions
        )

        # Create touchpoint records
        clicks = ClickData.objects.filter(visitor_id=visitor_id).order_by('timestamp')

        for i, click in enumerate(clicks):
            TouchPoint.objects.create(
                attribution=attr,
                click=click,
                position=i,
                channel=click.utm_source or 'direct',
                campaign=click.utm_campaign or '',
                timestamp=click.timestamp
            )

        return attr
