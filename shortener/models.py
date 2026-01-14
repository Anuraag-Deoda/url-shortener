import string
import random
import secrets
from django.db import models
from django.utils import timezone


class URL(models.Model):
    original_url = models.URLField(max_length=500)
    short_code = models.CharField(max_length=10, unique=True)
    created_at = models.DateTimeField(default=timezone.now)
    clicks = models.PositiveIntegerField(default=0)

    # Feature flags
    enable_device_targeting = models.BooleanField(default=False)
    enable_rotation = models.BooleanField(default=False)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.short_code} -> {self.original_url[:50]}"

    @staticmethod
    def generate_short_code(length=6):
        """Generate a random short code of specified length"""
        chars = string.ascii_letters + string.digits
        return ''.join(random.choice(chars) for _ in range(length))

    def increment_clicks(self):
        """Increment the click counter for this URL"""
        self.clicks += 1
        self.save(update_fields=['clicks'])

    def get_absolute_url(self):
        from django.urls import reverse
        return reverse('shortener:redirect', kwargs={'short_code': self.short_code})

    def get_stats_url(self):
        from django.urls import reverse
        return reverse('analytics:url_stats', kwargs={'short_code': self.short_code})

    def get_destination_for_device(self, device_info: dict) -> str:
        """Get the appropriate destination URL based on device info"""
        if not self.enable_device_targeting:
            return self.original_url

        targets = self.device_targets.filter(is_active=True).order_by('-priority')

        if not targets.exists():
            return self.original_url

        # Check for specific OS matches first (higher specificity)
        os_name = device_info.get('os', '').lower()
        for target in targets:
            if target.device_type == 'ios' and 'ios' in os_name:
                return target.destination_url
            elif target.device_type == 'android' and 'android' in os_name:
                return target.destination_url
            elif target.device_type == 'windows' and 'windows' in os_name:
                return target.destination_url
            elif target.device_type == 'macos' and ('mac' in os_name or 'darwin' in os_name):
                return target.destination_url
            elif target.device_type == 'linux' and 'linux' in os_name:
                return target.destination_url

        # Check for device type matches
        device_type = device_info.get('device_type', '').lower()
        for target in targets:
            if target.device_type == device_type:
                return target.destination_url

        # Fallback to default target or original URL
        default_target = targets.filter(device_type='default').first()
        return default_target.destination_url if default_target else self.original_url

    def get_rotation_destination(self) -> tuple:
        """Get destination URL from rotation group. Returns (url, rotation_url_object)"""
        if not self.enable_rotation or not hasattr(self, 'rotation_group'):
            return self.original_url, None

        rotation_group = self.rotation_group
        if not rotation_group.is_active:
            return self.original_url, None

        rotation_url = rotation_group.get_next_url()
        if rotation_url:
            return rotation_url.destination_url, rotation_url

        return self.original_url, None


class ClickData(models.Model):
    """Model to store detailed click data for analytics"""
    url = models.ForeignKey(URL, on_delete=models.CASCADE, related_name='click_data')
    timestamp = models.DateTimeField(default=timezone.now)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(blank=True, null=True)
    referrer = models.URLField(max_length=500, blank=True, null=True)
    country = models.CharField(max_length=100, blank=True, null=True)
    city = models.CharField(max_length=100, blank=True, null=True)
    device_type = models.CharField(max_length=20, blank=True, null=True)
    browser = models.CharField(max_length=100, blank=True, null=True)
    os = models.CharField(max_length=100, blank=True, null=True)

    # Geolocation fields
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)
    country_code = models.CharField(max_length=2, blank=True, null=True)
    region = models.CharField(max_length=100, blank=True, null=True)
    timezone_name = models.CharField(max_length=50, blank=True, null=True)

    # Tracking which URL was served (for device targeting/rotation)
    served_url = models.URLField(max_length=500, blank=True, null=True)
    device_target = models.ForeignKey(
        'DeviceTarget', on_delete=models.SET_NULL,
        null=True, blank=True, related_name='clicks'
    )
    rotation_url = models.ForeignKey(
        'RotationURL', on_delete=models.SET_NULL,
        null=True, blank=True, related_name='click_data'
    )

    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        return f"Click on {self.url.short_code} at {self.timestamp}"


class DeviceTarget(models.Model):
    """Maps device types to specific destination URLs for a shortened link"""

    DEVICE_CHOICES = [
        ('desktop', 'Desktop'),
        ('mobile', 'Mobile'),
        ('tablet', 'Tablet'),
        ('ios', 'iOS'),
        ('android', 'Android'),
        ('windows', 'Windows'),
        ('macos', 'macOS'),
        ('linux', 'Linux'),
        ('default', 'Default/Fallback'),
    ]

    url = models.ForeignKey(URL, on_delete=models.CASCADE, related_name='device_targets')
    device_type = models.CharField(max_length=20, choices=DEVICE_CHOICES)
    destination_url = models.URLField(max_length=500)
    priority = models.PositiveIntegerField(default=0, help_text="Higher priority targets are matched first")
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-priority', 'device_type']
        unique_together = ['url', 'device_type']

    def __str__(self):
        return f"{self.url.short_code} -> {self.device_type} -> {self.destination_url[:30]}"


class RotationGroup(models.Model):
    """A group of URLs that can be rotated for a single short link"""

    ROTATION_STRATEGIES = [
        ('round_robin', 'Round Robin'),
        ('random', 'Random'),
        ('weighted', 'Weighted'),
    ]

    url = models.OneToOneField(URL, on_delete=models.CASCADE, related_name='rotation_group')
    strategy = models.CharField(max_length=20, choices=ROTATION_STRATEGIES, default='round_robin')
    is_active = models.BooleanField(default=True)
    current_index = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Rotation for {self.url.short_code} ({self.strategy})"

    def get_next_url(self):
        """Get the next URL based on rotation strategy"""
        active_urls = self.rotation_urls.filter(is_active=True).order_by('order')

        if not active_urls.exists():
            return None

        if self.strategy == 'random':
            return self._get_random_url(active_urls)
        elif self.strategy == 'weighted':
            return self._get_weighted_url(active_urls)
        else:  # round_robin (default)
            return self._get_round_robin_url(active_urls)

    def _get_round_robin_url(self, urls):
        url_list = list(urls)
        selected = url_list[self.current_index % len(url_list)]

        # Update index atomically
        RotationGroup.objects.filter(pk=self.pk).update(
            current_index=(self.current_index + 1) % len(url_list)
        )

        return selected

    def _get_random_url(self, urls):
        return random.choice(list(urls))

    def _get_weighted_url(self, urls):
        total_weight = sum(u.weight for u in urls)
        if total_weight == 0:
            return self._get_random_url(urls)

        r = random.uniform(0, total_weight)
        cumulative = 0
        for url in urls:
            cumulative += url.weight
            if r <= cumulative:
                return url
        return urls.last()


class RotationURL(models.Model):
    """Individual URL within a rotation group"""

    rotation_group = models.ForeignKey(
        RotationGroup, on_delete=models.CASCADE,
        related_name='rotation_urls'
    )
    destination_url = models.URLField(max_length=500)
    weight = models.PositiveIntegerField(default=1, help_text="Weight for weighted rotation")
    order = models.PositiveIntegerField(default=0, help_text="Order for round-robin rotation")
    is_active = models.BooleanField(default=True)
    label = models.CharField(max_length=100, blank=True, help_text="Optional label (e.g., 'Version A')")
    created_at = models.DateTimeField(auto_now_add=True)
    clicks = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ['order', 'created_at']

    def __str__(self):
        label = f" ({self.label})" if self.label else ""
        return f"{self.destination_url[:40]}{label}"

    def increment_clicks(self):
        RotationURL.objects.filter(pk=self.pk).update(clicks=models.F('clicks') + 1)


class CustomDomain(models.Model):
    """Custom domain configuration for URL shortening"""

    VERIFICATION_STATUS = [
        ('pending', 'Pending Verification'),
        ('verified', 'Verified'),
        ('failed', 'Verification Failed'),
    ]

    domain = models.CharField(max_length=255, unique=True)
    verification_status = models.CharField(max_length=20, choices=VERIFICATION_STATUS, default='pending')
    verification_token = models.CharField(max_length=64, unique=True, editable=False)
    verified_at = models.DateTimeField(null=True, blank=True)
    is_active = models.BooleanField(default=False)
    ssl_enabled = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def save(self, *args, **kwargs):
        if not self.verification_token:
            self.verification_token = secrets.token_urlsafe(32)
        super().save(*args, **kwargs)

    def __str__(self):
        status = "verified" if self.verification_status == 'verified' else "pending"
        return f"{self.domain} ({status})"

    def get_dns_txt_record(self) -> str:
        """Get the TXT record value for DNS verification"""
        return f"url-shortener-verify={self.verification_token}"

    def get_cname_target(self) -> str:
        """Get the CNAME target for the custom domain"""
        from django.conf import settings
        return getattr(settings, 'SHORT_URL_DOMAIN', 'short.example.com')


class DomainURL(models.Model):
    """Association between URLs and custom domains"""

    url = models.ForeignKey(URL, on_delete=models.CASCADE, related_name='domain_urls')
    custom_domain = models.ForeignKey(CustomDomain, on_delete=models.CASCADE, related_name='urls')
    custom_path = models.CharField(
        max_length=100, blank=True,
        help_text="Optional custom path (uses short_code if empty)"
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ['custom_domain', 'custom_path']

    def __str__(self):
        path = self.custom_path or self.url.short_code
        return f"{self.custom_domain.domain}/{path}"

    def get_full_url(self):
        protocol = 'https' if self.custom_domain.ssl_enabled else 'http'
        path = self.custom_path or self.url.short_code
        return f"{protocol}://{self.custom_domain.domain}/{path}"
