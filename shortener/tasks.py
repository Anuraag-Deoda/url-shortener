"""
Celery tasks for URL shortener analytics.
"""
import logging
from datetime import timedelta
from decimal import Decimal

from celery import shared_task
from django.core.cache import cache
from django.db.models import Count, Sum, Avg
from django.db.models.functions import TruncHour, TruncDay
from django.utils import timezone

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=3)
def process_click_async(self, click_data_id: int):
    """
    Process click data asynchronously.
    Performs heavy analytics computations in background.
    """
    from shortener.models import ClickData
    from shortener.services.fraud_detection import FraudDetectionService

    try:
        click = ClickData.objects.get(pk=click_data_id)

        # Run fraud detection
        service = FraudDetectionService()
        is_fraud, triggered_rules = service.check_click({
            'ip_address': click.ip_address,
            'user_agent': click.user_agent,
            'referrer': click.referrer,
            'country_code': click.country_code,
            'url_id': click.url_id,
        })

        if is_fraud:
            click.is_bot = True
            click.save(update_fields=['is_bot'])

            # Create fraud alerts
            for rule_info in triggered_rules:
                from shortener.models import ClickFraudRule, FraudAlert
                rule = None
                if rule_info.get('rule_id'):
                    rule = ClickFraudRule.objects.filter(pk=rule_info['rule_id']).first()

                FraudAlert.objects.create(
                    rule=rule,
                    click=click,
                    severity='high' if rule_info.get('action') == 'block' else 'medium',
                    reason=rule_info.get('reason', 'Suspicious activity detected'),
                    ip_address=click.ip_address,
                    user_agent=click.user_agent or ''
                )

        # Invalidate relevant caches
        cache.delete(f'click_stats_{click.url_id}')
        cache.delete(f'campaign_stats_{click.utm_campaign}')

        logger.info(f"Processed click {click_data_id}, fraud={is_fraud}")
        return {'click_id': click_data_id, 'is_fraud': is_fraud}

    except ClickData.DoesNotExist:
        logger.error(f"Click {click_data_id} not found")
        return None
    except Exception as exc:
        logger.error(f"Error processing click {click_data_id}: {exc}")
        raise self.retry(exc=exc, countdown=60)


@shared_task
def update_traffic_forecasts():
    """
    Update traffic forecasts using historical data.
    Runs hourly via Celery Beat.
    """
    from shortener.models import URL, ClickData, PredictiveModel
    from shortener.services.ml_service import TrafficForecaster

    logger.info("Starting traffic forecast update")

    try:
        forecaster = TrafficForecaster()

        # Get URLs with enough data
        urls_with_data = ClickData.objects.values('url_id').annotate(
            count=Count('id')
        ).filter(count__gte=100)

        for url_info in urls_with_data[:50]:  # Limit to 50 URLs
            url_id = url_info['url_id']

            try:
                # Generate forecast
                forecast = forecaster.forecast_traffic(url_id, days=7)

                # Cache the forecast
                cache_key = f'traffic_forecast_{url_id}'
                cache.set(cache_key, forecast, timeout=3600)

                logger.info(f"Updated forecast for URL {url_id}")
            except Exception as e:
                logger.error(f"Error forecasting URL {url_id}: {e}")
                continue

        # Update aggregate forecast
        try:
            aggregate_forecast = forecaster.forecast_aggregate_traffic(days=7)
            cache.set('aggregate_traffic_forecast', aggregate_forecast, timeout=3600)
        except Exception as e:
            logger.error(f"Error with aggregate forecast: {e}")

        logger.info("Traffic forecast update completed")
        return {'status': 'success', 'urls_processed': len(urls_with_data)}

    except Exception as e:
        logger.error(f"Traffic forecast update failed: {e}")
        return {'status': 'error', 'message': str(e)}


@shared_task
def detect_anomalies():
    """
    Detect traffic anomalies in real-time.
    Runs every 5 minutes via Celery Beat.
    """
    from shortener.models import ClickData, FraudAlert
    from shortener.services.ml_service import AnomalyDetector

    logger.info("Running anomaly detection")

    try:
        detector = AnomalyDetector()

        # Get last 5 minutes of clicks
        five_min_ago = timezone.now() - timedelta(minutes=5)
        recent_clicks = ClickData.objects.filter(timestamp__gte=five_min_ago)

        # Check for IP-based anomalies
        ip_counts = recent_clicks.values('ip_address').annotate(
            count=Count('id')
        ).filter(count__gt=20)  # More than 20 clicks in 5 min

        anomalies_found = 0

        for ip_info in ip_counts:
            ip = ip_info['ip_address']
            count = ip_info['count']

            # Check if this is a known anomaly pattern
            is_anomaly, score = detector.check_ip_anomaly(ip, count)

            if is_anomaly:
                anomalies_found += 1

                # Create alert if not already exists
                existing = FraudAlert.objects.filter(
                    ip_address=ip,
                    created_at__gte=five_min_ago,
                    reason__contains='anomaly'
                ).exists()

                if not existing:
                    click = recent_clicks.filter(ip_address=ip).first()
                    if click:
                        FraudAlert.objects.create(
                            click=click,
                            severity='high',
                            reason=f'Traffic anomaly detected: {count} clicks in 5 min (score: {score:.2f})',
                            ip_address=ip,
                            user_agent=click.user_agent or ''
                        )

        # Check for overall traffic anomalies
        total_recent = recent_clicks.count()
        expected = detector.get_expected_traffic(minutes=5)

        if total_recent > expected * 3:  # 3x expected traffic
            logger.warning(f"Traffic spike detected: {total_recent} vs expected {expected}")
            cache.set('traffic_anomaly_alert', {
                'actual': total_recent,
                'expected': expected,
                'timestamp': timezone.now().isoformat()
            }, timeout=300)

        logger.info(f"Anomaly detection completed: {anomalies_found} anomalies found")
        return {'anomalies_found': anomalies_found, 'total_clicks': total_recent}

    except Exception as e:
        logger.error(f"Anomaly detection failed: {e}")
        return {'status': 'error', 'message': str(e)}


@shared_task
def cleanup_old_sessions():
    """
    Clean up old session data and expired cache entries.
    Runs daily via Celery Beat.
    """
    from shortener.models import ClickData, FraudAlert, LinkAccessLog

    logger.info("Starting session cleanup")

    try:
        # Archive old click data (older than 90 days)
        ninety_days_ago = timezone.now() - timedelta(days=90)

        # Delete old resolved fraud alerts
        deleted_alerts = FraudAlert.objects.filter(
            status__in=['dismissed', 'confirmed'],
            resolved_at__lt=ninety_days_ago
        ).delete()

        # Delete old access logs
        deleted_logs = LinkAccessLog.objects.filter(
            timestamp__lt=ninety_days_ago
        ).delete()

        # Clear old cache entries
        cache.delete_pattern('click_stats_*')
        cache.delete_pattern('session_*')

        logger.info(f"Cleanup completed: {deleted_alerts[0]} alerts, {deleted_logs[0]} logs deleted")
        return {
            'deleted_alerts': deleted_alerts[0],
            'deleted_logs': deleted_logs[0]
        }

    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return {'status': 'error', 'message': str(e)}


@shared_task
def retrain_bot_detector():
    """
    Retrain the ML bot detection model.
    Runs daily via Celery Beat.
    """
    from shortener.models import ClickData, PredictiveModel
    from shortener.services.ml_service import BotDetector

    logger.info("Starting bot detector retraining")

    try:
        detector = BotDetector()

        # Get training data from last 30 days
        thirty_days_ago = timezone.now() - timedelta(days=30)
        training_data = ClickData.objects.filter(
            timestamp__gte=thirty_days_ago
        ).values(
            'user_agent', 'ip_address', 'referrer',
            'device_type', 'is_bot'
        )

        # Retrain model
        accuracy = detector.train(list(training_data))

        # Save model metadata
        model_record, created = PredictiveModel.objects.update_or_create(
            name='bot_detector',
            model_type='anomaly_detection',
            defaults={
                'accuracy': accuracy,
                'last_trained': timezone.now(),
                'training_samples': training_data.count(),
                'is_active': True
            }
        )

        logger.info(f"Bot detector retrained with accuracy: {accuracy:.2%}")
        return {'accuracy': accuracy, 'samples': training_data.count()}

    except Exception as e:
        logger.error(f"Bot detector retraining failed: {e}")
        return {'status': 'error', 'message': str(e)}


@shared_task
def calculate_campaign_roi(campaign_id: int):
    """
    Calculate ROI for a campaign asynchronously.
    """
    from shortener.models import Campaign, ClickData

    try:
        campaign = Campaign.objects.get(pk=campaign_id)

        clicks = ClickData.objects.filter(utm_campaign=campaign.utm_campaign)

        stats = {
            'total_clicks': clicks.count(),
            'unique_visitors': clicks.values('visitor_id').distinct().count(),
            'conversions': clicks.filter(conversion_value__isnull=False).count(),
            'revenue': float(clicks.aggregate(
                total=Sum('conversion_value')
            )['total'] or 0),
        }

        if campaign.spent and campaign.spent > 0:
            stats['roi'] = ((stats['revenue'] - float(campaign.spent)) /
                           float(campaign.spent)) * 100
        else:
            stats['roi'] = None

        # Cache the results
        cache.set(f'campaign_stats_{campaign_id}', stats, timeout=300)

        return stats

    except Campaign.DoesNotExist:
        return None


@shared_task
def update_ab_test_results(test_id: int):
    """
    Update A/B test results and check for significance.
    """
    from shortener.models import ABTest
    from shortener.services.analytics_service import ABTestAnalyticsService

    try:
        test = ABTest.objects.get(pk=test_id)

        if test.status != 'running':
            return {'status': 'not_running'}

        results = ABTestAnalyticsService.get_test_results(test)

        # Check if we should auto-stop
        if results['is_significant']:
            winner = test.determine_winner()
            if winner:
                test.winner_variant = winner
                test.save(update_fields=['winner_variant'])

        # Cache results
        cache.set(f'abtest_results_{test_id}', results, timeout=60)

        return results

    except ABTest.DoesNotExist:
        return None


@shared_task
def generate_analytics_report(start_date: str, end_date: str, report_type: str = 'summary'):
    """
    Generate comprehensive analytics report.
    """
    from datetime import datetime
    from shortener.models import ClickData, Campaign, ABTest

    try:
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)

        clicks = ClickData.objects.filter(
            timestamp__gte=start,
            timestamp__lte=end
        )

        report = {
            'period': {'start': start_date, 'end': end_date},
            'summary': {
                'total_clicks': clicks.count(),
                'unique_visitors': clicks.values('visitor_id').distinct().count(),
                'conversions': clicks.filter(conversion_value__isnull=False).count(),
                'total_revenue': float(clicks.aggregate(
                    total=Sum('conversion_value')
                )['total'] or 0),
                'bot_clicks': clicks.filter(is_bot=True).count(),
            },
            'top_campaigns': list(clicks.exclude(
                utm_campaign__isnull=True
            ).values('utm_campaign').annotate(
                clicks=Count('id')
            ).order_by('-clicks')[:10]),
            'top_sources': list(clicks.exclude(
                utm_source__isnull=True
            ).values('utm_source').annotate(
                clicks=Count('id')
            ).order_by('-clicks')[:10]),
            'device_breakdown': list(clicks.values('device_type').annotate(
                clicks=Count('id')
            )),
            'country_breakdown': list(clicks.values('country').annotate(
                clicks=Count('id')
            ).order_by('-clicks')[:20]),
        }

        if report_type == 'detailed':
            report['hourly_trend'] = list(clicks.annotate(
                hour=TruncHour('timestamp')
            ).values('hour').annotate(
                clicks=Count('id')
            ).order_by('hour'))

            report['daily_trend'] = list(clicks.annotate(
                day=TruncDay('timestamp')
            ).values('day').annotate(
                clicks=Count('id'),
                visitors=Count('visitor_id', distinct=True)
            ).order_by('day'))

        # Cache the report
        cache_key = f'report_{start_date}_{end_date}_{report_type}'
        cache.set(cache_key, report, timeout=3600)

        return report

    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        return {'status': 'error', 'message': str(e)}


@shared_task
def export_data_to_csv(query_params: dict, user_email: str = None):
    """
    Export analytics data to CSV format.
    """
    import csv
    import io
    from shortener.models import ClickData

    try:
        queryset = ClickData.objects.all()

        # Apply filters
        if query_params.get('url_id'):
            queryset = queryset.filter(url_id=query_params['url_id'])
        if query_params.get('start_date'):
            queryset = queryset.filter(timestamp__gte=query_params['start_date'])
        if query_params.get('end_date'):
            queryset = queryset.filter(timestamp__lte=query_params['end_date'])
        if query_params.get('campaign'):
            queryset = queryset.filter(utm_campaign=query_params['campaign'])

        # Limit to prevent memory issues
        queryset = queryset[:100000]

        # Generate CSV
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            'timestamp', 'url_code', 'ip_address', 'country', 'city',
            'device_type', 'browser', 'os', 'referrer',
            'utm_source', 'utm_medium', 'utm_campaign',
            'conversion_value', 'is_bot'
        ])

        # Data rows
        for click in queryset:
            writer.writerow([
                click.timestamp.isoformat(),
                click.url.short_code,
                click.ip_address,
                click.country,
                click.city,
                click.device_type,
                click.browser,
                click.os,
                click.referrer,
                click.utm_source,
                click.utm_medium,
                click.utm_campaign,
                click.conversion_value,
                click.is_bot
            ])

        csv_content = output.getvalue()

        # Cache for download
        cache_key = f'export_{hash(str(query_params))}'
        cache.set(cache_key, csv_content, timeout=3600)

        # TODO: Send email with download link if user_email provided

        return {
            'status': 'success',
            'cache_key': cache_key,
            'row_count': queryset.count()
        }

    except Exception as e:
        logger.error(f"Data export failed: {e}")
        return {'status': 'error', 'message': str(e)}
