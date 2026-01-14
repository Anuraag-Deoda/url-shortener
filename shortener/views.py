import random
from django.shortcuts import render, redirect, get_object_or_404
from django.views.generic import ListView, CreateView, DetailView, UpdateView
from django.urls import reverse_lazy, reverse
from django.contrib import messages
from django.http import JsonResponse, HttpResponse
from django.forms import inlineformset_factory
from django.views.decorators.http import require_http_methods
from django.utils import timezone
from .models import (
    URL, ClickData, DeviceTarget, RotationGroup,
    RotationURL, CustomDomain, DomainURL, TimeSchedule,
    Funnel, FunnelStep, FunnelEvent, LinkAccessLog
)
from .forms import URLShortenerForm
from .services.geolocation import GeoLocationService
from .services.qrcode_service import QRCodeService
from .services.domain_verification import DomainVerificationService
from .services.realtime import RealTimeAnalyticsService
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
