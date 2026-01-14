import random
import hashlib
from datetime import timedelta
from decimal import Decimal
from django.shortcuts import render, redirect, get_object_or_404
from django.views.generic import ListView, CreateView, DetailView, UpdateView, DeleteView, TemplateView
from django.urls import reverse_lazy, reverse
from django.contrib import messages
from django.http import JsonResponse, HttpResponse
from django.forms import inlineformset_factory
from django.views.decorators.http import require_http_methods
from django.utils import timezone
from django.db.models import Count, Sum, Avg, Q
from django.db.models.functions import TruncDate, TruncHour
from .models import (
    URL, ClickData, DeviceTarget, RotationGroup,
    RotationURL, CustomDomain, DomainURL, TimeSchedule,
    Funnel, FunnelStep, FunnelEvent, LinkAccessLog,
    Campaign, CampaignURL, ABTest, ABTestVariant,
    Cohort, ClickFraudRule, FraudAlert, Attribution
)
from .forms import URLShortenerForm
from .services.geolocation import GeoLocationService
from .services.qrcode_service import QRCodeService
from .services.domain_verification import DomainVerificationService
from .services.realtime import RealTimeAnalyticsService
from .services.analytics_service import (
    CampaignAnalyticsService, CohortAnalyticsService,
    ABTestAnalyticsService, AttributionService
)
from .services.fraud_detection import FraudDetectionService, RealTimeFraudMonitor
import user_agents


def get_client_ip(request):
    """Extract real IP from request (handles proxies)"""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        return x_forwarded_for.split(',')[0].strip()
    return request.META.get('REMOTE_ADDR', '')


def extract_client_info(request):
    """Extract client information from request for analytics"""
    user_agent_string = request.META.get('HTTP_USER_AGENT', '')
    user_agent = user_agents.parse(user_agent_string)

    ip_address = get_client_ip(request)

    # Get geolocation data
    geo_data = GeoLocationService.lookup_ip(ip_address) or {}

    client_info = {
        'ip_address': ip_address,
        'user_agent': user_agent_string,
        'referrer': request.META.get('HTTP_REFERER'),
        'browser': f"{user_agent.browser.family} {user_agent.browser.version_string}",
        'os': f"{user_agent.os.family} {user_agent.os.version_string}",
        'device_type': 'Mobile' if user_agent.is_mobile else 'Tablet' if user_agent.is_tablet else 'Desktop',
        # Geolocation data
        'country': geo_data.get('country'),
        'country_code': geo_data.get('country_code'),
        'city': geo_data.get('city'),
        'region': geo_data.get('region'),
        'latitude': geo_data.get('latitude'),
        'longitude': geo_data.get('longitude'),
        'timezone_name': geo_data.get('timezone'),
    }

    return client_info


class HomeView(CreateView):
    model = URL
    form_class = URLShortenerForm
    template_name = 'shortener/home.html'
    success_url = reverse_lazy('shortener:success')

    def form_valid(self, form):
        # Check if URL already exists
        original_url = form.cleaned_data['original_url']
        existing_url = URL.objects.filter(original_url=original_url).first()

        if existing_url:
            self.object = existing_url
            return redirect('shortener:success', pk=existing_url.pk)

        # Generate a unique short code
        short_code = URL.generate_short_code()
        while URL.objects.filter(short_code=short_code).exists():
            short_code = URL.generate_short_code()

        # Create new URL record
        self.object = form.save(commit=False)
        self.object.short_code = short_code
        self.object.save()

        return redirect('shortener:success', pk=self.object.pk)


class URLSuccessView(DetailView):
    model = URL
    template_name = 'shortener/success.html'
    context_object_name = 'url'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['short_url'] = self.request.build_absolute_uri(
            reverse_lazy('shortener:redirect', kwargs={'short_code': self.object.short_code})
        )
        return context


def redirect_to_original(request, short_code):
    """Redirect to the original URL and record click data"""
    url = get_object_or_404(URL, short_code=short_code)
    client_info = extract_client_info(request)

    # Check if link is expired
    if url.is_expired():
        LinkAccessLog.objects.create(
            url=url,
            access_type='expired',
            ip_address=client_info.get('ip_address'),
            user_agent=client_info.get('user_agent', '')
        )
        if url.expired_redirect_url:
            return redirect(url.expired_redirect_url)
        return render(request, 'shortener/expired.html', {'url': url})

    # Check if gate is required (password or captcha)
    if url.requires_gate():
        # Check if user has already passed the gate (via session)
        gate_key = f'gate_passed_{url.short_code}'
        if not request.session.get(gate_key):
            return redirect('shortener:gate', short_code=short_code)

    # Determine destination URL (priority: time-based > rotation > device targeting > original)
    destination_url = url.original_url
    rotation_url = None
    device_target = None

    # Time-based redirect takes highest priority
    if url.enable_time_based:
        destination_url = url.get_time_based_destination()

    # Then check rotation
    elif url.enable_rotation:
        destination_url, rotation_url = url.get_rotation_destination()
        if rotation_url:
            rotation_url.increment_clicks()

    # Device targeting
    elif url.enable_device_targeting:
        destination_url = url.get_destination_for_device({
            'device_type': client_info.get('device_type'),
            'os': client_info.get('os'),
        })
        device_target = url.device_targets.filter(
            destination_url=destination_url,
            is_active=True
        ).first()

    # Record click data for analytics
    click_data = ClickData.objects.create(
        url=url,
        served_url=destination_url,
        device_target=device_target,
        rotation_url=rotation_url,
        **client_info
    )

    # Increment click counter
    url.increment_clicks()

    # Broadcast to real-time analytics
    RealTimeAnalyticsService.broadcast_click(url, client_info)

    # Track funnel events if URL is part of a funnel
    _track_funnel_event(url, click_data, client_info)

    return redirect(destination_url)


def _track_funnel_event(url, click_data, client_info):
    """Track funnel events for this URL"""
    funnel_steps = url.funnel_steps.filter(funnel__is_active=True)

    for step in funnel_steps:
        visitor_id = FunnelEvent.generate_visitor_id(
            client_info.get('ip_address', ''),
            client_info.get('user_agent', '')
        )

        FunnelEvent.objects.create(
            funnel=step.funnel,
            step=step,
            visitor_id=visitor_id,
            ip_address=client_info.get('ip_address'),
            click_data=click_data
        )


class URLListView(ListView):
    model = URL
    template_name = 'shortener/url_list.html'
    context_object_name = 'urls'
    paginate_by = 10


def api_create_short_url(request):
    """API endpoint for creating short URLs"""
    if request.method == 'POST':
        form = URLShortenerForm(request.POST)
        if form.is_valid():
            original_url = form.cleaned_data['original_url']

            # Check if URL already exists
            existing_url = URL.objects.filter(original_url=original_url).first()
            if existing_url:
                short_url = request.build_absolute_uri(
                    reverse_lazy('shortener:redirect', kwargs={'short_code': existing_url.short_code})
                )
                return JsonResponse({
                    'success': True,
                    'short_url': short_url,
                    'original_url': existing_url.original_url,
                    'short_code': existing_url.short_code,
                })

            # Generate a unique short code
            short_code = URL.generate_short_code()
            while URL.objects.filter(short_code=short_code).exists():
                short_code = URL.generate_short_code()

            # Create new URL record
            url = form.save(commit=False)
            url.short_code = short_code
            url.save()

            short_url = request.build_absolute_uri(
                reverse_lazy('shortener:redirect', kwargs={'short_code': short_code})
            )

            return JsonResponse({
                'success': True,
                'short_url': short_url,
                'original_url': url.original_url,
                'short_code': url.short_code,
            })
        else:
            return JsonResponse({
                'success': False,
                'errors': form.errors,
            }, status=400)

    return JsonResponse({
        'success': False,
        'message': 'Only POST requests are allowed',
    }, status=405)


# ============= Device Targeting Views =============

DeviceTargetFormSet = inlineformset_factory(
    URL, DeviceTarget,
    fields=['device_type', 'destination_url', 'priority', 'is_active'],
    extra=1,
    can_delete=True,
)


class URLDeviceTargetView(UpdateView):
    model = URL
    template_name = 'shortener/device_targets.html'
    fields = ['enable_device_targeting']

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        if self.request.POST:
            context['formset'] = DeviceTargetFormSet(self.request.POST, instance=self.object)
        else:
            context['formset'] = DeviceTargetFormSet(instance=self.object)
        context['short_url'] = self.request.build_absolute_uri(
            reverse('shortener:redirect', kwargs={'short_code': self.object.short_code})
        )
        return context

    def form_valid(self, form):
        context = self.get_context_data()
        formset = context['formset']
        if formset.is_valid():
            self.object = form.save()
            formset.instance = self.object
            formset.save()
            messages.success(self.request, 'Device targeting settings saved successfully.')
            return redirect('shortener:success', pk=self.object.pk)
        return self.render_to_response(context)


# ============= Link Rotation Views =============

RotationURLFormSet = inlineformset_factory(
    RotationGroup, RotationURL,
    fields=['destination_url', 'label', 'weight', 'order', 'is_active'],
    extra=2,
    can_delete=True,
)


class RotationManagementView(UpdateView):
    model = URL
    template_name = 'shortener/rotation_management.html'
    fields = ['enable_rotation']

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        # Get or create rotation group
        rotation_group, created = RotationGroup.objects.get_or_create(url=self.object)
        context['rotation_group'] = rotation_group

        if self.request.POST:
            context['url_formset'] = RotationURLFormSet(self.request.POST, instance=rotation_group)
            context['strategy'] = self.request.POST.get('strategy', rotation_group.strategy)
        else:
            context['url_formset'] = RotationURLFormSet(instance=rotation_group)
            context['strategy'] = rotation_group.strategy

        # Statistics
        context['rotation_stats'] = self._get_rotation_stats(rotation_group)
        context['short_url'] = self.request.build_absolute_uri(
            reverse('shortener:redirect', kwargs={'short_code': self.object.short_code})
        )

        return context

    def _get_rotation_stats(self, rotation_group):
        urls = rotation_group.rotation_urls.filter(is_active=True)
        total_clicks = sum(u.clicks for u in urls)

        stats = []
        for url in urls:
            percentage = (url.clicks / total_clicks * 100) if total_clicks > 0 else 0
            stats.append({
                'url': url,
                'clicks': url.clicks,
                'percentage': round(percentage, 1)
            })

        return stats

    def form_valid(self, form):
        context = self.get_context_data()
        url_formset = context['url_formset']

        if url_formset.is_valid():
            self.object = form.save()

            # Update rotation group strategy
            rotation_group = context['rotation_group']
            rotation_group.strategy = self.request.POST.get('strategy', 'round_robin')
            rotation_group.save()

            url_formset.save()
            messages.success(self.request, 'Rotation settings saved successfully.')
            return redirect('shortener:success', pk=self.object.pk)

        return self.render_to_response(context)


# ============= Custom Domains Views =============

class CustomDomainListView(ListView):
    model = CustomDomain
    template_name = 'shortener/custom_domains.html'
    context_object_name = 'domains'


class CustomDomainCreateView(CreateView):
    model = CustomDomain
    template_name = 'shortener/custom_domain_add.html'
    fields = ['domain']
    success_url = reverse_lazy('shortener:custom_domains')

    def form_valid(self, form):
        response = super().form_valid(form)
        messages.info(
            self.request,
            f'Domain added. Please add the following TXT record to verify ownership: '
            f'{self.object.get_dns_txt_record()}'
        )
        return response


def verify_custom_domain(request, pk):
    """Verify a custom domain"""
    domain = get_object_or_404(CustomDomain, pk=pk)

    success, message = DomainVerificationService.verify_domain(domain)

    if success:
        messages.success(request, message)
    else:
        messages.error(request, message)

    return redirect('shortener:custom_domains')


# ============= QR Code Views =============

def generate_qr_code(request, short_code):
    """Generate and return QR code for a URL"""
    url = get_object_or_404(URL, short_code=short_code)

    # Get options from query params
    style = request.GET.get('style', 'default')
    size = int(request.GET.get('size', 10))
    fg_color = request.GET.get('fg', '#000000')
    bg_color = request.GET.get('bg', '#FFFFFF')
    format_type = request.GET.get('format', 'png')

    # Build full URL
    full_url = request.build_absolute_uri(
        reverse('shortener:redirect', kwargs={'short_code': short_code})
    )

    if format_type == 'base64':
        base64_data = QRCodeService.generate_qr_base64(
            full_url, size=size, style=style,
            foreground_color=fg_color, background_color=bg_color
        )
        return JsonResponse({'qr_code': base64_data})
    else:
        buffer = QRCodeService.generate_qr_code(
            full_url, size=size, style=style,
            foreground_color=fg_color, background_color=bg_color
        )
        response = HttpResponse(buffer.getvalue(), content_type='image/png')
        response['Content-Disposition'] = f'attachment; filename="{short_code}_qr.png"'
        return response


# ============= Gate Views (Password/CAPTCHA) =============

def gate_view(request, short_code):
    """Handle password and CAPTCHA gates"""
    url = get_object_or_404(URL, short_code=short_code)
    client_info = extract_client_info(request)

    # Check if expired
    if url.is_expired():
        return render(request, 'shortener/expired.html', {'url': url})

    # Generate simple math CAPTCHA
    captcha_data = None
    if url.enable_captcha and url.captcha_type == 'simple':
        num1 = random.randint(1, 10)
        num2 = random.randint(1, 10)
        request.session['captcha_answer'] = num1 + num2
        captcha_data = {'num1': num1, 'num2': num2}

    context = {
        'url': url,
        'requires_password': url.is_password_protected(),
        'requires_captcha': url.enable_captcha,
        'captcha_type': url.captcha_type,
        'captcha_data': captcha_data,
        'password_hint': url.password_hint,
        'error': None,
    }

    if request.method == 'POST':
        password_ok = True
        captcha_ok = True

        # Verify password
        if url.is_password_protected():
            password = request.POST.get('password', '')
            if not url.check_password(password):
                password_ok = False
                context['error'] = 'Incorrect password'
                LinkAccessLog.objects.create(
                    url=url,
                    access_type='password_fail',
                    ip_address=client_info.get('ip_address'),
                    user_agent=client_info.get('user_agent', '')
                )

        # Verify CAPTCHA
        if url.enable_captcha and password_ok:
            if url.captcha_type == 'simple':
                user_answer = request.POST.get('captcha_answer', '')
                correct_answer = request.session.get('captcha_answer')
                try:
                    if int(user_answer) != correct_answer:
                        captcha_ok = False
                        context['error'] = 'Incorrect CAPTCHA answer'
                except (ValueError, TypeError):
                    captcha_ok = False
                    context['error'] = 'Invalid CAPTCHA answer'

                if not captcha_ok:
                    LinkAccessLog.objects.create(
                        url=url,
                        access_type='captcha_fail',
                        ip_address=client_info.get('ip_address'),
                        user_agent=client_info.get('user_agent', '')
                    )
            # Add reCAPTCHA/hCaptcha verification here if needed

        if password_ok and captcha_ok:
            # Mark gate as passed in session
            request.session[f'gate_passed_{url.short_code}'] = True

            # Log success
            if url.is_password_protected():
                LinkAccessLog.objects.create(
                    url=url,
                    access_type='password_success',
                    ip_address=client_info.get('ip_address'),
                    user_agent=client_info.get('user_agent', '')
                )
            if url.enable_captcha:
                LinkAccessLog.objects.create(
                    url=url,
                    access_type='captcha_success',
                    ip_address=client_info.get('ip_address'),
                    user_agent=client_info.get('user_agent', '')
                )

            return redirect('shortener:redirect', short_code=short_code)

        # Regenerate CAPTCHA on failure
        if url.enable_captcha and url.captcha_type == 'simple':
            num1 = random.randint(1, 10)
            num2 = random.randint(1, 10)
            request.session['captcha_answer'] = num1 + num2
            context['captcha_data'] = {'num1': num1, 'num2': num2}

    return render(request, 'shortener/gate.html', context)


# ============= Link Settings Views =============

class URLSettingsView(UpdateView):
    """View for managing link expiration, password, and CAPTCHA settings"""
    model = URL
    template_name = 'shortener/link_settings.html'
    fields = [
        'expires_at', 'max_clicks', 'expired_redirect_url',
        'enable_captcha', 'captcha_type', 'password_hint'
    ]

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['short_url'] = self.request.build_absolute_uri(
            reverse('shortener:redirect', kwargs={'short_code': self.object.short_code})
        )
        context['has_password'] = self.object.is_password_protected()
        return context

    def form_valid(self, form):
        # Handle password separately
        new_password = self.request.POST.get('new_password')
        remove_password = self.request.POST.get('remove_password')

        self.object = form.save(commit=False)

        if remove_password:
            self.object.password_hash = ''
        elif new_password:
            self.object.set_password(new_password)

        self.object.save()
        messages.success(self.request, 'Link settings saved successfully.')
        return redirect('shortener:success', pk=self.object.pk)


# ============= Time-Based Redirect Views =============

TimeScheduleFormSet = inlineformset_factory(
    URL, TimeSchedule,
    fields=['destination_url', 'label', 'start_time', 'end_time',
            'start_date', 'end_date', 'weekdays', 'timezone_name', 'priority', 'is_active'],
    extra=1,
    can_delete=True,
)


class TimeBasedRedirectView(UpdateView):
    """View for managing time-based redirects"""
    model = URL
    template_name = 'shortener/time_based.html'
    fields = ['enable_time_based']

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        if self.request.POST:
            context['formset'] = TimeScheduleFormSet(self.request.POST, instance=self.object)
        else:
            context['formset'] = TimeScheduleFormSet(instance=self.object)
        context['short_url'] = self.request.build_absolute_uri(
            reverse('shortener:redirect', kwargs={'short_code': self.object.short_code})
        )
        context['current_time'] = timezone.now()
        return context

    def form_valid(self, form):
        context = self.get_context_data()
        formset = context['formset']
        if formset.is_valid():
            self.object = form.save()
            formset.instance = self.object
            formset.save()
            messages.success(self.request, 'Time-based redirect settings saved.')
            return redirect('shortener:success', pk=self.object.pk)
        return self.render_to_response(context)


# ============= Funnel Views =============

class FunnelListView(ListView):
    """List all funnels"""
    model = Funnel
    template_name = 'shortener/funnel_list.html'
    context_object_name = 'funnels'


class FunnelDetailView(DetailView):
    """View funnel analytics"""
    model = Funnel
    template_name = 'shortener/funnel_detail.html'
    context_object_name = 'funnel'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        funnel = self.object

        # Get step-by-step analytics
        steps = funnel.steps.order_by('order')
        step_data = []

        for step in steps:
            step_data.append({
                'step': step,
                'unique_visitors': step.get_unique_visitors(),
                'total_clicks': step.get_total_clicks(),
                'conversion_to_next': step.get_conversion_to_next(),
            })

        context['step_data'] = step_data
        context['overall_conversion'] = funnel.get_conversion_rate()

        return context


class FunnelCreateView(CreateView):
    """Create a new funnel"""
    model = Funnel
    template_name = 'shortener/funnel_create.html'
    fields = ['name', 'description']
    success_url = reverse_lazy('shortener:funnel_list')


FunnelStepFormSet = inlineformset_factory(
    Funnel, FunnelStep,
    fields=['url', 'name', 'order'],
    extra=3,
    can_delete=True,
)


class FunnelEditView(UpdateView):
    """Edit funnel steps"""
    model = Funnel
    template_name = 'shortener/funnel_edit.html'
    fields = ['name', 'description', 'is_active']

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        if self.request.POST:
            context['step_formset'] = FunnelStepFormSet(self.request.POST, instance=self.object)
        else:
            context['step_formset'] = FunnelStepFormSet(instance=self.object)
        return context

    def form_valid(self, form):
        context = self.get_context_data()
        step_formset = context['step_formset']
        if step_formset.is_valid():
            self.object = form.save()
            step_formset.instance = self.object
            step_formset.save()
            messages.success(self.request, 'Funnel updated successfully.')
            return redirect('shortener:funnel_detail', pk=self.object.pk)
        return self.render_to_response(context)


# ============= Real-Time Analytics Views =============

def realtime_dashboard(request):
    """Real-time analytics dashboard"""
    stats = RealTimeAnalyticsService.get_live_stats()
    recent_clicks = RealTimeAnalyticsService.get_recent_clicks(20)

    return render(request, 'shortener/realtime_dashboard.html', {
        'stats': stats,
        'recent_clicks': recent_clicks,
    })


def api_realtime_stats(request):
    """API endpoint for real-time statistics"""
    stats = RealTimeAnalyticsService.get_live_stats()
    return JsonResponse({'success': True, 'data': stats})


def api_realtime_clicks(request):
    """API endpoint for recent clicks"""
    limit = int(request.GET.get('limit', 20))
    clicks = RealTimeAnalyticsService.get_recent_clicks(limit)
    return JsonResponse({'success': True, 'data': clicks})


def api_clicks_per_minute(request):
    """API endpoint for clicks per minute"""
    minutes = int(request.GET.get('minutes', 60))
    data = RealTimeAnalyticsService.get_clicks_per_minute(minutes)
    return JsonResponse({'success': True, 'data': data})


# ============= Campaign Management Views =============

class CampaignListView(ListView):
    """List all marketing campaigns"""
    model = Campaign
    template_name = 'shortener/campaign_list.html'
    context_object_name = 'campaigns'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # Add summary stats for each campaign
        for campaign in context['campaigns']:
            campaign.stats = {
                'total_clicks': campaign.get_total_clicks(),
                'unique_visitors': campaign.get_unique_visitors(),
                'conversions': campaign.get_conversion_count(),
                'revenue': campaign.get_total_revenue(),
                'roi': campaign.get_roi(),
            }
        return context


class CampaignCreateView(CreateView):
    """Create a new campaign"""
    model = Campaign
    template_name = 'shortener/campaign_form.html'
    fields = ['name', 'description', 'utm_source', 'utm_medium', 'utm_campaign',
              'budget', 'currency', 'start_date', 'end_date']
    success_url = reverse_lazy('shortener:campaign_list')

    def form_valid(self, form):
        messages.success(self.request, 'Campaign created successfully.')
        return super().form_valid(form)


class CampaignDetailView(DetailView):
    """Campaign analytics dashboard"""
    model = Campaign
    template_name = 'shortener/campaign_detail.html'
    context_object_name = 'campaign'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        days = int(self.request.GET.get('days', 30))
        context['performance'] = CampaignAnalyticsService.get_campaign_performance(
            self.object, days
        )
        context['days'] = days
        context['urls'] = self.object.urls.all()
        return context


class CampaignEditView(UpdateView):
    """Edit campaign"""
    model = Campaign
    template_name = 'shortener/campaign_form.html'
    fields = ['name', 'description', 'utm_source', 'utm_medium', 'utm_campaign',
              'budget', 'spent', 'currency', 'start_date', 'end_date', 'is_active']

    def get_success_url(self):
        return reverse('shortener:campaign_detail', kwargs={'pk': self.object.pk})


CampaignURLFormSet = inlineformset_factory(
    Campaign, CampaignURL,
    fields=['url', 'utm_source', 'utm_medium', 'utm_content', 'utm_term'],
    extra=2,
    can_delete=True,
)


class CampaignURLsView(UpdateView):
    """Manage URLs in a campaign"""
    model = Campaign
    template_name = 'shortener/campaign_urls.html'
    fields = []

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        if self.request.POST:
            context['formset'] = CampaignURLFormSet(self.request.POST, instance=self.object)
        else:
            context['formset'] = CampaignURLFormSet(instance=self.object)
        context['all_urls'] = URL.objects.all()
        return context

    def form_valid(self, form):
        context = self.get_context_data()
        formset = context['formset']
        if formset.is_valid():
            formset.save()
            messages.success(self.request, 'Campaign URLs updated.')
            return redirect('shortener:campaign_detail', pk=self.object.pk)
        return self.render_to_response(context)


def api_campaign_compare(request):
    """API: Compare multiple campaigns"""
    campaign_ids = request.GET.getlist('ids')
    days = int(request.GET.get('days', 30))

    if not campaign_ids:
        return JsonResponse({'success': False, 'error': 'No campaign IDs provided'}, status=400)

    try:
        ids = [int(id) for id in campaign_ids]
    except ValueError:
        return JsonResponse({'success': False, 'error': 'Invalid campaign IDs'}, status=400)

    results = CampaignAnalyticsService.compare_campaigns(ids, days)
    return JsonResponse({'success': True, 'data': results})


# ============= A/B Test Views =============

class ABTestListView(ListView):
    """List all A/B tests"""
    model = ABTest
    template_name = 'shortener/abtest_list.html'
    context_object_name = 'tests'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        for test in context['tests']:
            test.visitors = test.get_total_visitors()
            test.is_significant = test.is_statistically_significant()
        return context


class ABTestCreateView(CreateView):
    """Create a new A/B test"""
    model = ABTest
    template_name = 'shortener/abtest_form.html'
    fields = ['name', 'description', 'url', 'goal_type', 'confidence_level', 'minimum_sample_size']
    success_url = reverse_lazy('shortener:abtest_list')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['urls'] = URL.objects.all()
        return context


ABTestVariantFormSet = inlineformset_factory(
    ABTest, ABTestVariant,
    fields=['name', 'destination_url', 'weight', 'is_control'],
    extra=2,
    can_delete=True,
)


class ABTestDetailView(DetailView):
    """A/B test results dashboard"""
    model = ABTest
    template_name = 'shortener/abtest_detail.html'
    context_object_name = 'test'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['results'] = ABTestAnalyticsService.get_test_results(self.object)
        return context


class ABTestEditView(UpdateView):
    """Edit A/B test variants"""
    model = ABTest
    template_name = 'shortener/abtest_edit.html'
    fields = ['name', 'description', 'status', 'goal_type', 'confidence_level', 'minimum_sample_size']

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        if self.request.POST:
            context['variant_formset'] = ABTestVariantFormSet(self.request.POST, instance=self.object)
        else:
            context['variant_formset'] = ABTestVariantFormSet(instance=self.object)
        return context

    def form_valid(self, form):
        context = self.get_context_data()
        formset = context['variant_formset']
        if formset.is_valid():
            self.object = form.save()
            formset.save()
            messages.success(self.request, 'A/B test updated.')
            return redirect('shortener:abtest_detail', pk=self.object.pk)
        return self.render_to_response(context)


def abtest_start(request, pk):
    """Start an A/B test"""
    test = get_object_or_404(ABTest, pk=pk)
    if test.variants.count() < 2:
        messages.error(request, 'A/B test needs at least 2 variants to start.')
    else:
        test.status = 'running'
        test.start_date = timezone.now()
        test.save()
        messages.success(request, f'A/B test "{test.name}" started.')
    return redirect('shortener:abtest_detail', pk=pk)


def abtest_stop(request, pk):
    """Stop an A/B test"""
    test = get_object_or_404(ABTest, pk=pk)
    test.status = 'completed'
    test.end_date = timezone.now()

    # Determine winner if significant
    winner = test.determine_winner()
    if winner:
        test.winner_variant = winner

    test.save()
    messages.success(request, f'A/B test "{test.name}" completed.')
    return redirect('shortener:abtest_detail', pk=pk)


def api_abtest_sample_size(request):
    """API: Calculate required sample size"""
    baseline_rate = float(request.GET.get('baseline', 0.05))
    mde = float(request.GET.get('mde', 0.1))  # Minimum detectable effect
    confidence = float(request.GET.get('confidence', 0.95))

    sample_size = ABTestAnalyticsService.calculate_sample_size(
        baseline_rate, mde, confidence
    )

    return JsonResponse({
        'success': True,
        'sample_size': sample_size,
        'total_needed': sample_size * 2  # For 2 variants
    })


# ============= Cohort Analysis Views =============

class CohortListView(ListView):
    """List all cohorts"""
    model = Cohort
    template_name = 'shortener/cohort_list.html'
    context_object_name = 'cohorts'


class CohortCreateView(CreateView):
    """Create a new cohort"""
    model = Cohort
    template_name = 'shortener/cohort_form.html'
    fields = ['name', 'cohort_type', 'description', 'criteria']
    success_url = reverse_lazy('shortener:cohort_list')


class CohortDetailView(DetailView):
    """Cohort analysis dashboard"""
    model = Cohort
    template_name = 'shortener/cohort_detail.html'
    context_object_name = 'cohort'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['analysis'] = CohortAnalyticsService.analyze_cohort(self.object)
        return context


class RetentionDashboardView(TemplateView):
    """Retention cohort analysis dashboard"""
    template_name = 'shortener/retention_dashboard.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        days = int(self.request.GET.get('days', 60))
        period = self.request.GET.get('period', 'week')

        start_date = timezone.now() - timedelta(days=days)
        end_date = timezone.now()

        context['retention_data'] = CohortAnalyticsService.generate_retention_cohorts(
            start_date, end_date, period
        )
        context['days'] = days
        context['period'] = period
        return context


def api_retention_cohorts(request):
    """API: Get retention cohort data"""
    days = int(request.GET.get('days', 60))
    period = request.GET.get('period', 'week')

    start_date = timezone.now() - timedelta(days=days)
    end_date = timezone.now()

    data = CohortAnalyticsService.generate_retention_cohorts(start_date, end_date, period)

    # Convert dates to strings for JSON
    for cohort in data.get('cohorts', []):
        cohort['cohort_date'] = str(cohort['cohort_date'])

    return JsonResponse({'success': True, 'data': data})


# ============= Fraud Detection Views =============

class FraudDashboardView(TemplateView):
    """Fraud detection dashboard"""
    template_name = 'shortener/fraud_dashboard.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        days = int(self.request.GET.get('days', 7))

        service = FraudDetectionService()
        context['summary'] = service.get_fraud_summary(days)
        context['recent_alerts'] = FraudAlert.objects.all()[:20]
        context['rules'] = ClickFraudRule.objects.all()
        context['recommendations'] = service.get_recommended_rules()
        context['days'] = days
        return context


class FraudRuleListView(ListView):
    """List fraud detection rules"""
    model = ClickFraudRule
    template_name = 'shortener/fraud_rules.html'
    context_object_name = 'rules'


class FraudRuleCreateView(CreateView):
    """Create a fraud detection rule"""
    model = ClickFraudRule
    template_name = 'shortener/fraud_rule_form.html'
    fields = ['name', 'rule_type', 'action', 'parameters', 'redirect_url', 'priority', 'is_active']
    success_url = reverse_lazy('shortener:fraud_rules')


class FraudRuleEditView(UpdateView):
    """Edit fraud detection rule"""
    model = ClickFraudRule
    template_name = 'shortener/fraud_rule_form.html'
    fields = ['name', 'rule_type', 'action', 'parameters', 'redirect_url', 'priority', 'is_active']
    success_url = reverse_lazy('shortener:fraud_rules')


class FraudAlertListView(ListView):
    """List fraud alerts"""
    model = FraudAlert
    template_name = 'shortener/fraud_alerts.html'
    context_object_name = 'alerts'
    paginate_by = 50

    def get_queryset(self):
        queryset = super().get_queryset()
        status = self.request.GET.get('status')
        severity = self.request.GET.get('severity')

        if status:
            queryset = queryset.filter(status=status)
        if severity:
            queryset = queryset.filter(severity=severity)

        return queryset


def fraud_alert_update_status(request, pk):
    """Update fraud alert status"""
    alert = get_object_or_404(FraudAlert, pk=pk)
    new_status = request.POST.get('status')

    if new_status in ['new', 'investigating', 'confirmed', 'dismissed']:
        alert.status = new_status
        if new_status in ['confirmed', 'dismissed']:
            alert.resolved_at = timezone.now()
        alert.save()
        messages.success(request, f'Alert status updated to {new_status}.')

    return redirect('shortener:fraud_alerts')


def api_fraud_ip_analysis(request):
    """API: Analyze IP for fraud patterns"""
    ip_address = request.GET.get('ip')
    if not ip_address:
        return JsonResponse({'success': False, 'error': 'IP address required'}, status=400)

    service = FraudDetectionService()
    analysis = service.analyze_ip(ip_address)

    # Convert dates to strings
    if analysis.get('first_seen'):
        analysis['first_seen'] = analysis['first_seen'].isoformat()
    if analysis.get('last_seen'):
        analysis['last_seen'] = analysis['last_seen'].isoformat()

    return JsonResponse({'success': True, 'data': analysis})


# ============= Attribution Views =============

class AttributionDashboardView(TemplateView):
    """Multi-touch attribution dashboard"""
    template_name = 'shortener/attribution_dashboard.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        days = int(self.request.GET.get('days', 30))
        model = self.request.GET.get('model', 'linear')

        start_date = timezone.now() - timedelta(days=days)
        end_date = timezone.now()

        context['channel_attribution'] = AttributionService.get_channel_attribution(
            start_date, end_date, model
        )
        context['days'] = days
        context['model'] = model
        context['models'] = [
            ('first_touch', 'First Touch'),
            ('last_touch', 'Last Touch'),
            ('linear', 'Linear'),
            ('time_decay', 'Time Decay'),
            ('position_based', 'Position Based'),
        ]

        # Recent conversions
        context['recent_conversions'] = Attribution.objects.all()[:20]

        return context


def api_attribution_by_model(request):
    """API: Get attribution by different models"""
    days = int(request.GET.get('days', 30))

    start_date = timezone.now() - timedelta(days=days)
    end_date = timezone.now()

    models_data = {}
    for model in ['first_touch', 'last_touch', 'linear', 'time_decay', 'position_based']:
        models_data[model] = AttributionService.get_channel_attribution(
            start_date, end_date, model
        )

    return JsonResponse({'success': True, 'data': models_data})


def record_conversion(request):
    """API: Record a conversion for attribution"""
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'POST required'}, status=405)

    visitor_id = request.POST.get('visitor_id')
    value = request.POST.get('value')
    currency = request.POST.get('currency', 'USD')

    if not visitor_id or not value:
        return JsonResponse({
            'success': False,
            'error': 'visitor_id and value required'
        }, status=400)

    try:
        conversion_value = Decimal(value)
    except Exception:
        return JsonResponse({'success': False, 'error': 'Invalid value'}, status=400)

    attribution = AttributionService.create_attribution_record(
        visitor_id, conversion_value, currency
    )

    return JsonResponse({
        'success': True,
        'attribution_id': attribution.id,
        'conversion_id': attribution.conversion_id
    })


# ============= Advanced Analytics Dashboard =============

class AdvancedAnalyticsDashboardView(TemplateView):
    """Main advanced analytics dashboard"""
    template_name = 'shortener/advanced_analytics.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        days = int(self.request.GET.get('days', 30))
        start_date = timezone.now() - timedelta(days=days)

        # Summary statistics
        clicks = ClickData.objects.filter(timestamp__gte=start_date)

        context['summary'] = {
            'total_clicks': clicks.count(),
            'unique_visitors': clicks.values('visitor_id').distinct().count(),
            'conversions': clicks.filter(conversion_value__isnull=False).count(),
            'total_revenue': float(clicks.aggregate(Sum('conversion_value'))['conversion_value__sum'] or 0),
            'bot_clicks': clicks.filter(is_bot=True).count(),
        }

        # Top campaigns
        context['top_campaigns'] = clicks.exclude(
            utm_campaign__isnull=True
        ).exclude(utm_campaign='').values('utm_campaign').annotate(
            clicks=Count('id'),
            visitors=Count('visitor_id', distinct=True)
        ).order_by('-clicks')[:10]

        # Top sources
        context['top_sources'] = clicks.exclude(
            utm_source__isnull=True
        ).exclude(utm_source='').values('utm_source').annotate(
            clicks=Count('id'),
            visitors=Count('visitor_id', distinct=True)
        ).order_by('-clicks')[:10]

        # Fraud summary
        context['fraud_summary'] = FraudDetectionService().get_fraud_summary(days)

        # Active A/B tests
        context['active_tests'] = ABTest.objects.filter(status='running')[:5]

        # Recent alerts
        context['recent_alerts'] = FraudAlert.objects.filter(status='new')[:5]

        context['days'] = days
        return context


# ============= UTM Builder =============

def utm_builder(request):
    """UTM parameter builder tool"""
    return render(request, 'shortener/utm_builder.html', {
        'urls': URL.objects.all()[:50]
    })


def api_generate_utm_url(request):
    """API: Generate URL with UTM parameters"""
    base_url = request.GET.get('url', '')
    utm_source = request.GET.get('utm_source', '')
    utm_medium = request.GET.get('utm_medium', '')
    utm_campaign = request.GET.get('utm_campaign', '')
    utm_term = request.GET.get('utm_term', '')
    utm_content = request.GET.get('utm_content', '')

    if not base_url:
        return JsonResponse({'success': False, 'error': 'URL required'}, status=400)

    from urllib.parse import urlencode, urlparse, parse_qs, urlunparse

    parsed = urlparse(base_url)
    params = parse_qs(parsed.query)

    if utm_source:
        params['utm_source'] = [utm_source]
    if utm_medium:
        params['utm_medium'] = [utm_medium]
    if utm_campaign:
        params['utm_campaign'] = [utm_campaign]
    if utm_term:
        params['utm_term'] = [utm_term]
    if utm_content:
        params['utm_content'] = [utm_content]

    new_query = urlencode(params, doseq=True)
    final_url = urlunparse(parsed._replace(query=new_query))

    return JsonResponse({'success': True, 'url': final_url})
