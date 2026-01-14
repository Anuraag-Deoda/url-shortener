from django.contrib import admin
from .models import (
    URL, ClickData, DeviceTarget, RotationGroup,
    RotationURL, CustomDomain, DomainURL, TimeSchedule,
    Funnel, FunnelStep, FunnelEvent, LinkAccessLog
)


@admin.register(URL)
class URLAdmin(admin.ModelAdmin):
    list_display = ('short_code', 'original_url', 'clicks', 'enable_device_targeting',
                    'enable_rotation', 'enable_time_based', 'enable_captcha', 'expires_at', 'created_at')
    search_fields = ('short_code', 'original_url')
    list_filter = ('created_at', 'enable_device_targeting', 'enable_rotation',
                   'enable_time_based', 'enable_captcha')
    readonly_fields = ('clicks',)
    date_hierarchy = 'created_at'
    fieldsets = (
        (None, {
            'fields': ('original_url', 'short_code', 'clicks', 'created_at')
        }),
        ('Feature Flags', {
            'fields': ('enable_device_targeting', 'enable_rotation', 'enable_time_based')
        }),
        ('Expiration', {
            'fields': ('expires_at', 'max_clicks', 'expired_redirect_url'),
            'classes': ('collapse',)
        }),
        ('Password Protection', {
            'fields': ('password_hash', 'password_hint'),
            'classes': ('collapse',)
        }),
        ('CAPTCHA', {
            'fields': ('enable_captcha', 'captcha_type'),
            'classes': ('collapse',)
        }),
    )


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


@admin.register(TimeSchedule)
class TimeScheduleAdmin(admin.ModelAdmin):
    list_display = ('url', 'label', 'destination_url', 'start_time', 'end_time', 'priority', 'is_active')
    list_filter = ('is_active', 'timezone_name')
    search_fields = ('url__short_code', 'label', 'destination_url')


class FunnelStepInline(admin.TabularInline):
    model = FunnelStep
    extra = 1


@admin.register(Funnel)
class FunnelAdmin(admin.ModelAdmin):
    list_display = ('name', 'is_active', 'created_at')
    list_filter = ('is_active',)
    search_fields = ('name', 'description')
    inlines = [FunnelStepInline]


@admin.register(FunnelStep)
class FunnelStepAdmin(admin.ModelAdmin):
    list_display = ('funnel', 'name', 'url', 'order')
    list_filter = ('funnel',)
    search_fields = ('name', 'url__short_code')


@admin.register(FunnelEvent)
class FunnelEventAdmin(admin.ModelAdmin):
    list_display = ('funnel', 'step', 'visitor_id', 'ip_address', 'timestamp')
    list_filter = ('funnel', 'step', 'timestamp')
    search_fields = ('visitor_id', 'ip_address')
    date_hierarchy = 'timestamp'


@admin.register(LinkAccessLog)
class LinkAccessLogAdmin(admin.ModelAdmin):
    list_display = ('url', 'access_type', 'ip_address', 'timestamp')
    list_filter = ('access_type', 'timestamp')
    search_fields = ('url__short_code', 'ip_address')
    date_hierarchy = 'timestamp'
