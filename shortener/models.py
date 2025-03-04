import string
import random
from django.db import models
from django.utils import timezone


class URL(models.Model):
    original_url = models.URLField(max_length=500)
    short_code = models.CharField(max_length=10, unique=True)
    created_at = models.DateTimeField(default=timezone.now)
    clicks = models.PositiveIntegerField(default=0)
    
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
    
    class Meta:
        ordering = ['-timestamp']
        
    def __str__(self):
        return f"Click on {self.url.short_code} at {self.timestamp}"
