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

    # Redirect (must be last - catches all short codes)
    path('<str:short_code>/', views.redirect_to_original, name='redirect'),
]
