import string
import random
import secrets
import hashlib
from django.db import models
from django.utils import timezone
from django.contrib.auth.hashers import make_password, check_password


class URL(models.Model):
    original_url = models.URLField(max_length=500)
    short_code = models.CharField(max_length=10, unique=True)
    created_at = models.DateTimeField(default=timezone.now)
    clicks = models.PositiveIntegerField(default=0)

    # Feature flags
    enable_device_targeting = models.BooleanField(default=False)
    enable_rotation = models.BooleanField(default=False)
    enable_time_based = models.BooleanField(default=False)

    # Link expiration
    expires_at = models.DateTimeField(null=True, blank=True, help_text="Link expires after this date")
    max_clicks = models.PositiveIntegerField(null=True, blank=True, help_text="Link expires after this many clicks")
    expired_redirect_url = models.URLField(max_length=500, blank=True, help_text="Redirect here when expired")

    # Password protection
    password_hash = models.CharField(max_length=128, blank=True, help_text="Hashed password for protection")
    password_hint = models.CharField(max_length=100, blank=True, help_text="Optional hint for password")

    # CAPTCHA gate
    enable_captcha = models.BooleanField(default=False)
    captcha_type = models.CharField(max_length=20, default='simple', choices=[
        ('simple', 'Simple Math'),
        ('recaptcha', 'Google reCAPTCHA'),
        ('hcaptcha', 'hCaptcha'),
    ])

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

    def get_time_based_destination(self) -> str:
        """Get destination URL based on current time schedule"""
        if not self.enable_time_based:
            return self.original_url

        now = timezone.now()
        schedules = self.time_schedules.filter(is_active=True)

        for schedule in schedules:
            if schedule.is_active_now(now):
                return schedule.destination_url

        return self.original_url

    def is_expired(self) -> bool:
        """Check if the link has expired"""
        now = timezone.now()

        # Check date expiration
        if self.expires_at and now > self.expires_at:
            return True

        # Check click limit
        if self.max_clicks and self.clicks >= self.max_clicks:
            return True

        return False

    def set_password(self, raw_password: str):
        """Set password for the link"""
        self.password_hash = make_password(raw_password)

    def check_password(self, raw_password: str) -> bool:
        """Check if provided password matches"""
        if not self.password_hash:
            return True
        return check_password(raw_password, self.password_hash)

    def is_password_protected(self) -> bool:
        """Check if link is password protected"""
        return bool(self.password_hash)

    def requires_captcha(self) -> bool:
        """Check if link requires CAPTCHA"""
        return self.enable_captcha

    def requires_gate(self) -> bool:
        """Check if any gate (password/captcha) is required"""
        return self.is_password_protected() or self.requires_captcha()


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

    # UTM Parameters for campaign tracking
    utm_source = models.CharField(max_length=100, blank=True, null=True, db_index=True)
    utm_medium = models.CharField(max_length=100, blank=True, null=True, db_index=True)
    utm_campaign = models.CharField(max_length=100, blank=True, null=True, db_index=True)
    utm_term = models.CharField(max_length=100, blank=True, null=True)
    utm_content = models.CharField(max_length=100, blank=True, null=True)

    # Session and visitor tracking
    session_id = models.CharField(max_length=64, blank=True, null=True, db_index=True)
    visitor_id = models.CharField(max_length=64, blank=True, null=True, db_index=True)
    is_unique = models.BooleanField(default=True)  # First visit from this visitor
    is_bot = models.BooleanField(default=False)

    # Conversion tracking
    conversion_value = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    conversion_currency = models.CharField(max_length=3, default='USD')

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

    # A/B test tracking
    ab_test = models.ForeignKey(
        'ABTest', on_delete=models.SET_NULL,
        null=True, blank=True, related_name='clicks'
    )
    ab_variant = models.ForeignKey(
        'ABTestVariant', on_delete=models.SET_NULL,
        null=True, blank=True, related_name='clicks'
    )

    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['url', 'timestamp']),
            models.Index(fields=['visitor_id', 'timestamp']),
            models.Index(fields=['utm_campaign', 'timestamp']),
        ]

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


class TimeSchedule(models.Model):
    """Time-based redirect schedules"""

    WEEKDAYS = [
        (0, 'Monday'),
        (1, 'Tuesday'),
        (2, 'Wednesday'),
        (3, 'Thursday'),
        (4, 'Friday'),
        (5, 'Saturday'),
        (6, 'Sunday'),
    ]

    url = models.ForeignKey(URL, on_delete=models.CASCADE, related_name='time_schedules')
    destination_url = models.URLField(max_length=500)
    label = models.CharField(max_length=100, blank=True, help_text="e.g., 'Business Hours', 'Weekend'")

    # Time constraints
    start_time = models.TimeField(null=True, blank=True, help_text="Start time (leave empty for all day)")
    end_time = models.TimeField(null=True, blank=True, help_text="End time (leave empty for all day)")

    # Date constraints
    start_date = models.DateField(null=True, blank=True, help_text="Start date (leave empty for no limit)")
    end_date = models.DateField(null=True, blank=True, help_text="End date (leave empty for no limit)")

    # Weekday constraints (stored as comma-separated integers)
    weekdays = models.CharField(max_length=20, blank=True, help_text="Comma-separated weekday numbers (0=Mon, 6=Sun)")

    # Timezone for the schedule
    timezone_name = models.CharField(max_length=50, default='UTC')

    priority = models.PositiveIntegerField(default=0, help_text="Higher priority schedules are checked first")
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-priority', 'created_at']

    def __str__(self):
        return f"{self.url.short_code}: {self.label or self.destination_url[:30]}"

    def get_weekdays_list(self):
        """Get list of active weekdays as integers"""
        if not self.weekdays:
            return list(range(7))  # All days if not specified
        return [int(d.strip()) for d in self.weekdays.split(',') if d.strip().isdigit()]

    def is_active_now(self, now=None) -> bool:
        """Check if this schedule is currently active"""
        import pytz

        if now is None:
            now = timezone.now()

        # Convert to schedule's timezone
        try:
            tz = pytz.timezone(self.timezone_name)
            local_now = now.astimezone(tz)
        except Exception:
            local_now = now

        # Check date range
        today = local_now.date()
        if self.start_date and today < self.start_date:
            return False
        if self.end_date and today > self.end_date:
            return False

        # Check weekday
        if local_now.weekday() not in self.get_weekdays_list():
            return False

        # Check time range
        current_time = local_now.time()
        if self.start_time and self.end_time:
            if self.start_time <= self.end_time:
                # Normal range (e.g., 9:00 - 17:00)
                if not (self.start_time <= current_time <= self.end_time):
                    return False
            else:
                # Overnight range (e.g., 22:00 - 06:00)
                if not (current_time >= self.start_time or current_time <= self.end_time):
                    return False
        elif self.start_time:
            if current_time < self.start_time:
                return False
        elif self.end_time:
            if current_time > self.end_time:
                return False

        return True


class Funnel(models.Model):
    """Funnel for tracking multi-step conversions"""

    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return self.name

    def get_conversion_rate(self):
        """Calculate overall funnel conversion rate"""
        steps = self.steps.order_by('order')
        if steps.count() < 2:
            return 0

        first_step = steps.first()
        last_step = steps.last()

        first_count = first_step.get_unique_visitors()
        last_count = last_step.get_unique_visitors()

        if first_count == 0:
            return 0
        return round((last_count / first_count) * 100, 2)


class FunnelStep(models.Model):
    """Individual step in a funnel"""

    funnel = models.ForeignKey(Funnel, on_delete=models.CASCADE, related_name='steps')
    url = models.ForeignKey(URL, on_delete=models.CASCADE, related_name='funnel_steps')
    name = models.CharField(max_length=100, help_text="e.g., 'Landing Page', 'Signup', 'Purchase'")
    order = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['funnel', 'order']
        unique_together = ['funnel', 'order']

    def __str__(self):
        return f"{self.funnel.name} - Step {self.order}: {self.name}"

    def get_unique_visitors(self):
        """Get count of unique visitors to this step"""
        return ClickData.objects.filter(url=self.url).values('ip_address').distinct().count()

    def get_total_clicks(self):
        """Get total clicks for this step"""
        return self.url.clicks

    def get_conversion_to_next(self):
        """Get conversion rate to next step"""
        next_step = self.funnel.steps.filter(order__gt=self.order).order_by('order').first()
        if not next_step:
            return None

        current_visitors = self.get_unique_visitors()
        next_visitors = next_step.get_unique_visitors()

        if current_visitors == 0:
            return 0
        return round((next_visitors / current_visitors) * 100, 2)


class FunnelEvent(models.Model):
    """Track visitor progress through funnel"""

    funnel = models.ForeignKey(Funnel, on_delete=models.CASCADE, related_name='events')
    step = models.ForeignKey(FunnelStep, on_delete=models.CASCADE, related_name='events')
    visitor_id = models.CharField(max_length=64, help_text="Hashed identifier for visitor")
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    timestamp = models.DateTimeField(default=timezone.now)

    # Reference to click data
    click_data = models.ForeignKey(
        ClickData, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='funnel_events'
    )

    class Meta:
        ordering = ['funnel', 'visitor_id', 'timestamp']

    def __str__(self):
        return f"{self.funnel.name} - {self.step.name} - {self.visitor_id[:8]}"

    @staticmethod
    def generate_visitor_id(ip_address: str, user_agent: str) -> str:
        """Generate a consistent visitor ID from IP and user agent"""
        raw = f"{ip_address}:{user_agent}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]


class LinkAccessLog(models.Model):
    """Log for password/captcha attempts and access"""

    ACCESS_TYPES = [
        ('password_success', 'Password Success'),
        ('password_fail', 'Password Failed'),
        ('captcha_success', 'CAPTCHA Success'),
        ('captcha_fail', 'CAPTCHA Failed'),
        ('expired', 'Expired Link'),
    ]

    url = models.ForeignKey(URL, on_delete=models.CASCADE, related_name='access_logs')
    access_type = models.CharField(max_length=20, choices=ACCESS_TYPES)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(blank=True)
    timestamp = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        return f"{self.url.short_code}: {self.access_type} at {self.timestamp}"


# ============= Advanced Analytics Models =============

class Campaign(models.Model):
    """Marketing campaign for tracking UTM parameters"""

    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)

    # UTM defaults for this campaign
    utm_source = models.CharField(max_length=100, blank=True)
    utm_medium = models.CharField(max_length=100, blank=True)
    utm_campaign = models.CharField(max_length=100)

    # Budget tracking
    budget = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    spent = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    currency = models.CharField(max_length=3, default='USD')

    # Date range
    start_date = models.DateField(null=True, blank=True)
    end_date = models.DateField(null=True, blank=True)

    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return self.name

    def get_total_clicks(self):
        return ClickData.objects.filter(utm_campaign=self.utm_campaign).count()

    def get_unique_visitors(self):
        return ClickData.objects.filter(
            utm_campaign=self.utm_campaign
        ).values('visitor_id').distinct().count()

    def get_conversion_count(self):
        return ClickData.objects.filter(
            utm_campaign=self.utm_campaign,
            conversion_value__isnull=False
        ).count()

    def get_total_revenue(self):
        result = ClickData.objects.filter(
            utm_campaign=self.utm_campaign
        ).aggregate(total=models.Sum('conversion_value'))
        return result['total'] or 0

    def get_roi(self):
        if not self.spent or self.spent == 0:
            return None
        revenue = self.get_total_revenue()
        return round(((revenue - float(self.spent)) / float(self.spent)) * 100, 2)


class CampaignURL(models.Model):
    """Links associated with a campaign"""

    campaign = models.ForeignKey(Campaign, on_delete=models.CASCADE, related_name='urls')
    url = models.ForeignKey(URL, on_delete=models.CASCADE, related_name='campaigns')
    utm_source = models.CharField(max_length=100, blank=True)
    utm_medium = models.CharField(max_length=100, blank=True)
    utm_content = models.CharField(max_length=100, blank=True)
    utm_term = models.CharField(max_length=100, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ['campaign', 'url']

    def __str__(self):
        return f"{self.campaign.name} - {self.url.short_code}"

    def get_utm_url(self):
        """Generate full URL with UTM parameters"""
        from urllib.parse import urlencode, urlparse, parse_qs, urlunparse

        base_url = self.url.original_url
        parsed = urlparse(base_url)

        # Build UTM params
        utm_params = {}
        if self.utm_source or self.campaign.utm_source:
            utm_params['utm_source'] = self.utm_source or self.campaign.utm_source
        if self.utm_medium or self.campaign.utm_medium:
            utm_params['utm_medium'] = self.utm_medium or self.campaign.utm_medium
        if self.campaign.utm_campaign:
            utm_params['utm_campaign'] = self.campaign.utm_campaign
        if self.utm_content:
            utm_params['utm_content'] = self.utm_content
        if self.utm_term:
            utm_params['utm_term'] = self.utm_term

        # Merge with existing query params
        existing_params = parse_qs(parsed.query)
        existing_params.update(utm_params)

        new_query = urlencode(existing_params, doseq=True)
        return urlunparse(parsed._replace(query=new_query))


class ABTest(models.Model):
    """A/B Test configuration"""

    STATUS_CHOICES = [
        ('draft', 'Draft'),
        ('running', 'Running'),
        ('paused', 'Paused'),
        ('completed', 'Completed'),
    ]

    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    url = models.ForeignKey(URL, on_delete=models.CASCADE, related_name='ab_tests')

    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='draft')
    start_date = models.DateTimeField(null=True, blank=True)
    end_date = models.DateTimeField(null=True, blank=True)

    # Statistical settings
    confidence_level = models.FloatField(default=0.95, help_text="e.g., 0.95 for 95%")
    minimum_sample_size = models.PositiveIntegerField(default=100)

    # Goal tracking
    goal_type = models.CharField(max_length=50, default='clicks', choices=[
        ('clicks', 'Click-through Rate'),
        ('conversions', 'Conversion Rate'),
        ('revenue', 'Revenue per Click'),
        ('bounce', 'Bounce Rate'),
    ])

    winner_variant = models.ForeignKey(
        'ABTestVariant', on_delete=models.SET_NULL,
        null=True, blank=True, related_name='won_tests'
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.name} ({self.status})"

    def get_total_visitors(self):
        return self.clicks.values('visitor_id').distinct().count()

    def is_statistically_significant(self):
        """Check if test has reached statistical significance"""
        variants = self.variants.all()
        if variants.count() < 2:
            return False

        # Check minimum sample size
        for variant in variants:
            if variant.get_visitors() < self.minimum_sample_size:
                return False

        # Perform chi-square test
        return self._perform_significance_test()

    def _perform_significance_test(self):
        """Perform chi-square test for significance"""
        from scipy import stats
        import numpy as np

        variants = list(self.variants.all())
        if len(variants) < 2:
            return False

        # Build contingency table
        observed = []
        for variant in variants:
            visitors = variant.get_visitors()
            conversions = variant.get_conversions()
            observed.append([conversions, visitors - conversions])

        observed = np.array(observed)

        if observed.min() < 5:  # Chi-square requires min 5 expected
            return False

        try:
            chi2, p_value, dof, expected = stats.chi2_contingency(observed)
            return p_value < (1 - self.confidence_level)
        except Exception:
            return False

    def determine_winner(self):
        """Determine the winning variant based on goal type"""
        if not self.is_statistically_significant():
            return None

        variants = self.variants.all()
        best_variant = None
        best_value = -float('inf')

        for variant in variants:
            if self.goal_type == 'clicks':
                value = variant.get_ctr()
            elif self.goal_type == 'conversions':
                value = variant.get_conversion_rate()
            elif self.goal_type == 'revenue':
                value = variant.get_revenue_per_click()
            else:  # bounce - lower is better
                value = -variant.get_bounce_rate()

            if value > best_value:
                best_value = value
                best_variant = variant

        return best_variant


class ABTestVariant(models.Model):
    """Individual variant in an A/B test"""

    ab_test = models.ForeignKey(ABTest, on_delete=models.CASCADE, related_name='variants')
    name = models.CharField(max_length=100, help_text="e.g., 'Control', 'Variant A'")
    destination_url = models.URLField(max_length=500)
    weight = models.PositiveIntegerField(default=50, help_text="Traffic percentage (0-100)")
    is_control = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['ab_test', '-is_control', 'created_at']

    def __str__(self):
        control = " (Control)" if self.is_control else ""
        return f"{self.ab_test.name} - {self.name}{control}"

    def get_visitors(self):
        return self.clicks.values('visitor_id').distinct().count()

    def get_total_clicks(self):
        return self.clicks.count()

    def get_conversions(self):
        return self.clicks.filter(conversion_value__isnull=False).count()

    def get_ctr(self):
        """Click-through rate"""
        visitors = self.get_visitors()
        if visitors == 0:
            return 0
        return round((self.get_total_clicks() / visitors) * 100, 2)

    def get_conversion_rate(self):
        visitors = self.get_visitors()
        if visitors == 0:
            return 0
        return round((self.get_conversions() / visitors) * 100, 2)

    def get_revenue_per_click(self):
        clicks = self.get_total_clicks()
        if clicks == 0:
            return 0
        revenue = self.clicks.aggregate(total=models.Sum('conversion_value'))['total'] or 0
        return round(float(revenue) / clicks, 2)

    def get_bounce_rate(self):
        """Percentage of single-page visits"""
        visitors = self.get_visitors()
        if visitors == 0:
            return 0
        # Consider bounced if only one click from visitor
        single_clicks = self.clicks.values('visitor_id').annotate(
            count=models.Count('id')
        ).filter(count=1).count()
        return round((single_clicks / visitors) * 100, 2)


class Cohort(models.Model):
    """Cohort definition for analysis"""

    COHORT_TYPES = [
        ('first_click', 'First Click Date'),
        ('campaign', 'Campaign'),
        ('source', 'Traffic Source'),
        ('country', 'Country'),
        ('device', 'Device Type'),
    ]

    name = models.CharField(max_length=200)
    cohort_type = models.CharField(max_length=50, choices=COHORT_TYPES)
    description = models.TextField(blank=True)

    # Filter criteria (JSON)
    criteria = models.JSONField(default=dict, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return self.name

    def get_members(self, start_date=None, end_date=None):
        """Get click data matching this cohort's criteria"""
        queryset = ClickData.objects.all()

        if start_date:
            queryset = queryset.filter(timestamp__gte=start_date)
        if end_date:
            queryset = queryset.filter(timestamp__lte=end_date)

        # Apply cohort-specific filters
        if self.cohort_type == 'campaign' and self.criteria.get('utm_campaign'):
            queryset = queryset.filter(utm_campaign=self.criteria['utm_campaign'])
        elif self.cohort_type == 'source' and self.criteria.get('utm_source'):
            queryset = queryset.filter(utm_source=self.criteria['utm_source'])
        elif self.cohort_type == 'country' and self.criteria.get('country_code'):
            queryset = queryset.filter(country_code=self.criteria['country_code'])
        elif self.cohort_type == 'device' and self.criteria.get('device_type'):
            queryset = queryset.filter(device_type=self.criteria['device_type'])

        return queryset


class ClickFraudRule(models.Model):
    """Rules for detecting click fraud"""

    RULE_TYPES = [
        ('ip_frequency', 'IP Frequency (clicks per minute)'),
        ('ip_daily', 'IP Daily Limit'),
        ('user_agent_pattern', 'User Agent Pattern'),
        ('referrer_pattern', 'Referrer Pattern'),
        ('country_blacklist', 'Country Blacklist'),
        ('ip_range', 'IP Range Blacklist'),
        ('bot_detection', 'Known Bot Detection'),
    ]

    ACTION_TYPES = [
        ('flag', 'Flag for Review'),
        ('block', 'Block Click'),
        ('redirect', 'Redirect to Different URL'),
        ('challenge', 'Show CAPTCHA Challenge'),
    ]

    name = models.CharField(max_length=200)
    rule_type = models.CharField(max_length=50, choices=RULE_TYPES)
    action = models.CharField(max_length=20, choices=ACTION_TYPES, default='flag')

    # Rule parameters (JSON)
    parameters = models.JSONField(default=dict)

    # Optional redirect URL for 'redirect' action
    redirect_url = models.URLField(max_length=500, blank=True)

    is_active = models.BooleanField(default=True)
    priority = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-priority', 'created_at']

    def __str__(self):
        return f"{self.name} ({self.rule_type})"

    def check_click(self, click_data: dict) -> tuple:
        """Check if click matches this fraud rule. Returns (is_fraud, reason)"""
        if self.rule_type == 'ip_frequency':
            return self._check_ip_frequency(click_data)
        elif self.rule_type == 'ip_daily':
            return self._check_ip_daily(click_data)
        elif self.rule_type == 'user_agent_pattern':
            return self._check_user_agent_pattern(click_data)
        elif self.rule_type == 'referrer_pattern':
            return self._check_referrer_pattern(click_data)
        elif self.rule_type == 'country_blacklist':
            return self._check_country_blacklist(click_data)
        elif self.rule_type == 'bot_detection':
            return self._check_bot_detection(click_data)
        return False, None

    def _check_ip_frequency(self, click_data: dict) -> tuple:
        """Check clicks per minute from same IP"""
        import datetime
        max_per_minute = self.parameters.get('max_per_minute', 10)
        ip = click_data.get('ip_address')

        if not ip:
            return False, None

        one_minute_ago = timezone.now() - datetime.timedelta(minutes=1)
        recent_clicks = ClickData.objects.filter(
            ip_address=ip,
            timestamp__gte=one_minute_ago
        ).count()

        if recent_clicks >= max_per_minute:
            return True, f"IP {ip} exceeded {max_per_minute} clicks/minute"
        return False, None

    def _check_ip_daily(self, click_data: dict) -> tuple:
        """Check daily clicks from same IP"""
        max_daily = self.parameters.get('max_daily', 100)
        ip = click_data.get('ip_address')

        if not ip:
            return False, None

        today = timezone.now().date()
        daily_clicks = ClickData.objects.filter(
            ip_address=ip,
            timestamp__date=today
        ).count()

        if daily_clicks >= max_daily:
            return True, f"IP {ip} exceeded {max_daily} clicks/day"
        return False, None

    def _check_user_agent_pattern(self, click_data: dict) -> tuple:
        """Check user agent against patterns"""
        import re
        patterns = self.parameters.get('patterns', [])
        user_agent = click_data.get('user_agent', '')

        for pattern in patterns:
            if re.search(pattern, user_agent, re.IGNORECASE):
                return True, f"User agent matched fraud pattern: {pattern}"
        return False, None

    def _check_referrer_pattern(self, click_data: dict) -> tuple:
        """Check referrer against suspicious patterns"""
        import re
        patterns = self.parameters.get('patterns', [])
        referrer = click_data.get('referrer', '')

        for pattern in patterns:
            if re.search(pattern, referrer, re.IGNORECASE):
                return True, f"Referrer matched fraud pattern: {pattern}"
        return False, None

    def _check_country_blacklist(self, click_data: dict) -> tuple:
        """Check if country is blacklisted"""
        blacklist = self.parameters.get('countries', [])
        country_code = click_data.get('country_code', '')

        if country_code in blacklist:
            return True, f"Country {country_code} is blacklisted"
        return False, None

    def _check_bot_detection(self, click_data: dict) -> tuple:
        """Check for known bot signatures"""
        user_agent = click_data.get('user_agent', '').lower()

        bot_signatures = [
            'bot', 'spider', 'crawler', 'scraper', 'curl', 'wget',
            'python-requests', 'httpx', 'aiohttp', 'scrapy',
            'headless', 'phantom', 'selenium', 'puppeteer'
        ]

        for sig in bot_signatures:
            if sig in user_agent:
                return True, f"Bot signature detected: {sig}"
        return False, None


class FraudAlert(models.Model):
    """Fraud detection alerts"""

    SEVERITY_LEVELS = [
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High'),
        ('critical', 'Critical'),
    ]

    STATUS_CHOICES = [
        ('new', 'New'),
        ('investigating', 'Investigating'),
        ('confirmed', 'Confirmed Fraud'),
        ('dismissed', 'Dismissed'),
    ]

    rule = models.ForeignKey(ClickFraudRule, on_delete=models.SET_NULL, null=True, related_name='alerts')
    click = models.ForeignKey(ClickData, on_delete=models.CASCADE, related_name='fraud_alerts')

    severity = models.CharField(max_length=20, choices=SEVERITY_LEVELS, default='medium')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='new')
    reason = models.TextField()

    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    resolved_at = models.DateTimeField(null=True, blank=True)
    notes = models.TextField(blank=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Fraud Alert: {self.reason[:50]} ({self.severity})"


class PredictiveModel(models.Model):
    """Store predictive model configurations and results"""

    MODEL_TYPES = [
        ('traffic_forecast', 'Traffic Forecast'),
        ('conversion_prediction', 'Conversion Prediction'),
        ('churn_prediction', 'Churn Prediction'),
        ('anomaly_detection', 'Anomaly Detection'),
    ]

    name = models.CharField(max_length=200)
    model_type = models.CharField(max_length=50, choices=MODEL_TYPES)
    description = models.TextField(blank=True)

    # Model configuration
    config = models.JSONField(default=dict)

    # Serialized model (pickle/joblib)
    model_data = models.BinaryField(null=True, blank=True)

    # Performance metrics
    accuracy = models.FloatField(null=True, blank=True)
    last_trained = models.DateTimeField(null=True, blank=True)
    training_samples = models.PositiveIntegerField(default=0)

    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.name} ({self.model_type})"


class Attribution(models.Model):
    """Multi-touch attribution tracking"""

    ATTRIBUTION_MODELS = [
        ('first_touch', 'First Touch'),
        ('last_touch', 'Last Touch'),
        ('linear', 'Linear'),
        ('time_decay', 'Time Decay'),
        ('position_based', 'Position Based'),
        ('data_driven', 'Data-Driven'),
    ]

    visitor_id = models.CharField(max_length=64, db_index=True)
    conversion_id = models.CharField(max_length=64, unique=True)

    # Conversion details
    conversion_value = models.DecimalField(max_digits=10, decimal_places=2)
    conversion_currency = models.CharField(max_length=3, default='USD')
    converted_at = models.DateTimeField()

    # Attribution results (JSON: {model: {touchpoint_id: credit}})
    attributions = models.JSONField(default=dict)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-converted_at']

    def __str__(self):
        return f"Attribution {self.conversion_id[:8]} - ${self.conversion_value}"


class TouchPoint(models.Model):
    """Individual touch point in attribution path"""

    attribution = models.ForeignKey(Attribution, on_delete=models.CASCADE, related_name='touchpoints')
    click = models.ForeignKey(ClickData, on_delete=models.CASCADE, related_name='touchpoints')

    position = models.PositiveIntegerField()  # Order in the path
    channel = models.CharField(max_length=100)  # e.g., utm_source
    campaign = models.CharField(max_length=100, blank=True)

    timestamp = models.DateTimeField()

    class Meta:
        ordering = ['attribution', 'position']

    def __str__(self):
        return f"Touchpoint {self.position}: {self.channel}"
