from django.shortcuts import render, redirect, get_object_or_404
from django.views.generic import ListView, CreateView, DetailView, UpdateView
from django.urls import reverse_lazy, reverse
from django.contrib import messages
from django.http import JsonResponse, HttpResponse
from django.forms import inlineformset_factory
from .models import (
    URL, ClickData, DeviceTarget, RotationGroup,
    RotationURL, CustomDomain, DomainURL
)
from .forms import URLShortenerForm
from .services.geolocation import GeoLocationService
from .services.qrcode_service import QRCodeService
from .services.domain_verification import DomainVerificationService
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

    # Extract client info including geolocation
    client_info = extract_client_info(request)

    # Determine destination URL (priority: rotation > device targeting > original)
    destination_url = url.original_url
    rotation_url = None
    device_target = None

    # Check rotation first
    if url.enable_rotation:
        destination_url, rotation_url = url.get_rotation_destination()
        if rotation_url:
            rotation_url.increment_clicks()

    # Device targeting can work if rotation is not enabled
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
    ClickData.objects.create(
        url=url,
        served_url=destination_url,
        device_target=device_target,
        rotation_url=rotation_url,
        **client_info
    )

    # Increment click counter
    url.increment_clicks()

    return redirect(destination_url)


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
