from django.contrib import admin
from .models import URL, ClickData

@admin.register(URL)
class URLAdmin(admin.ModelAdmin):
    list_display = ('short_code', 'original_url', 'clicks', 'created_at')
    search_fields = ('short_code', 'original_url')
    list_filter = ('created_at',)
    readonly_fields = ('clicks',)
    date_hierarchy = 'created_at'

@admin.register(ClickData)
class ClickDataAdmin(admin.ModelAdmin):
    list_display = ('url', 'timestamp', 'ip_address', 'country', 'device_type', 'browser')
    list_filter = ('timestamp', 'device_type', 'browser', 'os')
    search_fields = ('url__short_code', 'ip_address', 'country', 'city')
    date_hierarchy = 'timestamp'
