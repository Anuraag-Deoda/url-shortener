from django.urls import path
from . import views

app_name = 'shortener'

urlpatterns = [
    path('', views.HomeView.as_view(), name='home'),
    path('success/<int:pk>/', views.URLSuccessView.as_view(), name='success'),
    path('list/', views.URLListView.as_view(), name='url_list'),
    path('api/create/', views.api_create_short_url, name='api_create'),

    # Device Targeting
    path('url/<int:pk>/device-targets/', views.URLDeviceTargetView.as_view(), name='device_targets'),

    # Link Rotation
    path('url/<int:pk>/rotation/', views.RotationManagementView.as_view(), name='rotation'),

    # Time-Based Redirects
    path('url/<int:pk>/time-based/', views.TimeBasedRedirectView.as_view(), name='time_based'),

    # Link Settings (expiration, password, captcha)
    path('url/<int:pk>/settings/', views.URLSettingsView.as_view(), name='link_settings'),

    # Custom Domains
    path('domains/', views.CustomDomainListView.as_view(), name='custom_domains'),
    path('domains/add/', views.CustomDomainCreateView.as_view(), name='custom_domain_add'),
    path('domains/<int:pk>/verify/', views.verify_custom_domain, name='verify_domain'),

    # QR Code
    path('qr/<str:short_code>/', views.generate_qr_code, name='qr_code'),

    # Funnel Analytics
    path('funnels/', views.FunnelListView.as_view(), name='funnel_list'),
    path('funnels/create/', views.FunnelCreateView.as_view(), name='funnel_create'),
    path('funnels/<int:pk>/', views.FunnelDetailView.as_view(), name='funnel_detail'),
    path('funnels/<int:pk>/edit/', views.FunnelEditView.as_view(), name='funnel_edit'),

    # Real-Time Analytics
    path('realtime/', views.realtime_dashboard, name='realtime_dashboard'),
    path('api/realtime/stats/', views.api_realtime_stats, name='api_realtime_stats'),
    path('api/realtime/clicks/', views.api_realtime_clicks, name='api_realtime_clicks'),
    path('api/realtime/per-minute/', views.api_clicks_per_minute, name='api_clicks_per_minute'),

    # Gate (password/captcha)
    path('gate/<str:short_code>/', views.gate_view, name='gate'),

    # ============= Advanced Analytics =============

    # Campaign Management
    path('campaigns/', views.CampaignListView.as_view(), name='campaign_list'),
    path('campaigns/create/', views.CampaignCreateView.as_view(), name='campaign_create'),
    path('campaigns/<int:pk>/', views.CampaignDetailView.as_view(), name='campaign_detail'),
    path('campaigns/<int:pk>/edit/', views.CampaignEditView.as_view(), name='campaign_edit'),
    path('campaigns/<int:pk>/urls/', views.CampaignURLsView.as_view(), name='campaign_urls'),
    path('api/campaigns/compare/', views.api_campaign_compare, name='api_campaign_compare'),

    # A/B Testing
    path('ab-tests/', views.ABTestListView.as_view(), name='abtest_list'),
    path('ab-tests/create/', views.ABTestCreateView.as_view(), name='abtest_create'),
    path('ab-tests/<int:pk>/', views.ABTestDetailView.as_view(), name='abtest_detail'),
    path('ab-tests/<int:pk>/edit/', views.ABTestEditView.as_view(), name='abtest_edit'),
    path('ab-tests/<int:pk>/start/', views.abtest_start, name='abtest_start'),
    path('ab-tests/<int:pk>/stop/', views.abtest_stop, name='abtest_stop'),
    path('api/ab-tests/sample-size/', views.api_abtest_sample_size, name='api_abtest_sample_size'),

    # Cohort Analysis
    path('cohorts/', views.CohortListView.as_view(), name='cohort_list'),
    path('cohorts/create/', views.CohortCreateView.as_view(), name='cohort_create'),
    path('cohorts/<int:pk>/', views.CohortDetailView.as_view(), name='cohort_detail'),
    path('retention/', views.RetentionDashboardView.as_view(), name='retention_dashboard'),
    path('api/retention/', views.api_retention_cohorts, name='api_retention_cohorts'),

    # Fraud Detection
    path('fraud/', views.FraudDashboardView.as_view(), name='fraud_dashboard'),
    path('fraud/rules/', views.FraudRuleListView.as_view(), name='fraud_rules'),
    path('fraud/rules/create/', views.FraudRuleCreateView.as_view(), name='fraud_rule_create'),
    path('fraud/rules/<int:pk>/edit/', views.FraudRuleEditView.as_view(), name='fraud_rule_edit'),
    path('fraud/alerts/', views.FraudAlertListView.as_view(), name='fraud_alerts'),
    path('fraud/alerts/<int:pk>/status/', views.fraud_alert_update_status, name='fraud_alert_status'),
    path('api/fraud/ip-analysis/', views.api_fraud_ip_analysis, name='api_fraud_ip_analysis'),

    # Attribution
    path('attribution/', views.AttributionDashboardView.as_view(), name='attribution_dashboard'),
    path('api/attribution/by-model/', views.api_attribution_by_model, name='api_attribution_by_model'),
    path('api/attribution/conversion/', views.record_conversion, name='api_record_conversion'),

    # Advanced Analytics Dashboard
    path('analytics/advanced/', views.AdvancedAnalyticsDashboardView.as_view(), name='advanced_analytics'),
    path('utm-builder/', views.utm_builder, name='utm_builder'),
    path('api/utm/generate/', views.api_generate_utm_url, name='api_generate_utm'),

    # ============= ML Analytics API =============

    # Traffic Forecasting
    path('api/ml/forecast/', views.api_traffic_forecast, name='api_forecast'),
    path('api/ml/anomalies/', views.api_anomaly_detection, name='api_anomalies'),
    path('api/ml/seasonal/', views.api_seasonal_patterns, name='api_seasonal'),

    # Bot Detection
    path('api/ml/bot-analysis/', views.api_bot_analysis, name='api_bot_analysis'),
    path('api/ml/bot-stats/', views.api_bot_stats, name='api_bot_stats'),

    # Bayesian A/B Testing
    path('api/ml/bayesian/<int:test_id>/', views.api_bayesian_ab_results, name='api_bayesian_results'),

    # Multi-Armed Bandit
    path('api/ml/bandit/<int:test_id>/allocation/', views.api_bandit_allocation, name='api_bandit_allocation'),
    path('api/ml/bandit/<int:test_id>/select/', views.api_bandit_select_variant, name='api_bandit_select'),

    # Conversion Prediction
    path('api/ml/conversion-prediction/', views.api_conversion_prediction, name='api_conversion_prediction'),
    path('api/ml/lead-score/', views.api_lead_score, name='api_lead_score'),
    path('api/ml/ab-winner/<int:test_id>/', views.api_ab_winner_prediction, name='api_ab_winner'),

    # NLP Classification
    path('api/ml/classify-referrer/', views.api_nlp_classify_referrer, name='api_classify_referrer'),
    path('api/ml/spam-referrers/', views.api_spam_referrers, name='api_spam_referrers'),
    path('api/ml/traffic-breakdown/', views.api_traffic_breakdown, name='api_traffic_breakdown'),
    path('api/ml/auto-tag/', views.api_auto_tag_url, name='api_auto_tag'),

    # Pandas Analytics
    path('api/analytics/comprehensive/', views.api_pandas_analytics, name='api_pandas_analytics'),
    path('api/analytics/attribution/', views.api_attribution_report, name='api_attribution_report'),
    path('api/analytics/export/', views.api_export_data, name='api_export_data'),

    # Redirect (must be last - catches all short codes)
    path('<str:short_code>/', views.redirect_to_original, name='redirect'),
]
