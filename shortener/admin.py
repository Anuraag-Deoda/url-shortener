from django.contrib import admin
from .models import (
    URL, ClickData, DeviceTarget, RotationGroup,
    RotationURL, CustomDomain, DomainURL
)


@admin.register(URL)
class URLAdmin(admin.ModelAdmin):
    list_display = ('short_code', 'original_url', 'clicks', 'enable_device_targeting', 'enable_rotation', 'created_at')
    search_fields = ('short_code', 'original_url')
    list_filter = ('created_at', 'enable_device_targeting', 'enable_rotation')
    readonly_fields = ('clicks',)
    date_hierarchy = 'created_at'


@admin.register(ClickData)
class ClickDataAdmin(admin.ModelAdmin):
    list_display = ('url', 'timestamp', 'ip_address', 'country', 'city', 'device_type', 'browser')
    list_filter = ('timestamp', 'device_type', 'browser', 'os', 'country')
    search_fields = ('url__short_code', 'ip_address', 'country', 'city')
    date_hierarchy = 'timestamp'
    readonly_fields = ('latitude', 'longitude', 'served_url')


@admin.register(DeviceTarget)
class DeviceTargetAdmin(admin.ModelAdmin):
    list_display = ('url', 'device_type', 'destination_url', 'priority', 'is_active')
    list_filter = ('device_type', 'is_active')
    search_fields = ('url__short_code', 'destination_url')


@admin.register(RotationGroup)
class RotationGroupAdmin(admin.ModelAdmin):
    list_display = ('url', 'strategy', 'is_active', 'current_index', 'created_at')
    list_filter = ('strategy', 'is_active')
    search_fields = ('url__short_code',)


@admin.register(RotationURL)
class RotationURLAdmin(admin.ModelAdmin):
    list_display = ('rotation_group', 'destination_url', 'label', 'weight', 'order', 'clicks', 'is_active')
    list_filter = ('is_active',)
    search_fields = ('destination_url', 'label')


@admin.register(CustomDomain)
class CustomDomainAdmin(admin.ModelAdmin):
    list_display = ('domain', 'verification_status', 'is_active', 'ssl_enabled', 'created_at')
    list_filter = ('verification_status', 'is_active', 'ssl_enabled')
    search_fields = ('domain',)
    readonly_fields = ('verification_token', 'verified_at')


@admin.register(DomainURL)
class DomainURLAdmin(admin.ModelAdmin):
    list_display = ('custom_domain', 'url', 'custom_path', 'created_at')
    search_fields = ('custom_domain__domain', 'url__short_code', 'custom_path')
