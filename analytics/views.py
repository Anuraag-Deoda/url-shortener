from django.shortcuts import render, get_object_or_404
from django.views.generic import TemplateView, DetailView
from django.http import JsonResponse
from django.db.models import Count
from django.db.models.functions import TruncDay, TruncHour
from django.utils import timezone
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_http_methods
from django.core.paginator import Paginator

from shortener.models import URL, ClickData
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
from django_plotly_dash import DjangoDash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import json
import datetime
from .services import AnalyticsService

# Create a Dash app for the dashboard
app = DjangoDash('URLAnalyticsDashboard')
app.layout = html.Div([
    html.H1("URL Analytics Dashboard"),
    dcc.Dropdown(
        id='url-dropdown',
        placeholder="Select a URL to analyze"
    ),
    html.Div(id='dashboard-content')
])

@app.callback(
    Output('dashboard-content', 'children'),
    [Input('url-dropdown', 'value')]
)
def update_dashboard(short_code):
    if not short_code:
        return html.Div("Please select a URL to view analytics")
    
    try:
        url = URL.objects.get(short_code=short_code)
        clicks = ClickData.objects.filter(url=url)
        
        if not clicks.exists():
            return html.Div("No click data available for this URL yet")
        
        # Create visualizations
        return html.Div([
            html.H2(f"Analytics for {url.short_code}"),
            html.P(f"Original URL: {url.original_url}"),
            html.P(f"Total Clicks: {url.clicks}"),
            html.Hr(),
            
            html.Div([
                html.H3("Click Timeline"),
                dcc.Graph(id='timeline-graph')
            ]),
            
            html.Div([
                html.H3("Device Distribution"),
                dcc.Graph(id='device-graph')
            ]),
            
            html.Div([
                html.H3("Browser Distribution"),
                dcc.Graph(id='browser-graph')
            ]),
            
            html.Div([
                html.H3("OS Distribution"),
                dcc.Graph(id='os-graph')
            ])
        ])
    except URL.DoesNotExist:
        return html.Div("URL not found")


class DashboardView(TemplateView):
    template_name = 'analytics/dashboard.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Get top URLs by clicks
        top_urls = URL.objects.order_by('-clicks')[:10]
        
        # Get recent activity
        recent_clicks = ClickData.objects.select_related('url').order_by('-timestamp')[:20]
        
        # Get total stats
        total_urls = URL.objects.count()
        total_clicks = ClickData.objects.count()
        
        # Get clicks over time
        thirty_days_ago = timezone.now() - datetime.timedelta(days=30)
        clicks_by_day = ClickData.objects.filter(timestamp__gte=thirty_days_ago) \
            .annotate(day=TruncDay('timestamp')) \
            .values('day') \
            .annotate(count=Count('id')) \
            .order_by('day')
        
        # Convert to DataFrame for Plotly
        df = pd.DataFrame(list(clicks_by_day))
        if not df.empty:
            fig = px.line(df, x='day', y='count', title='Clicks Over Time (Last 30 Days)')
            clicks_chart = plot(fig, output_type='div', include_plotlyjs=False)
        else:
            clicks_chart = "<p>No click data available for the last 30 days</p>"
        
        context.update({
            'top_urls': top_urls,
            'recent_clicks': recent_clicks,
            'total_urls': total_urls,
            'total_clicks': total_clicks,
            'clicks_chart': clicks_chart,
        })
        
        return context


class URLStatsView(DetailView):
    model = URL
    template_name = 'analytics/url_stats.html'
    context_object_name = 'url'
    
    def get_object(self):
        return get_object_or_404(URL, short_code=self.kwargs['short_code'])
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        url = self.get_object()
        
        # Get click data
        clicks = ClickData.objects.filter(url=url)
        
        # Prepare data for charts
        if clicks.exists():
            # Clicks over time
            clicks_by_day = clicks.annotate(day=TruncDay('timestamp')) \
                .values('day') \
                .annotate(count=Count('id')) \
                .order_by('day')
            
            df_time = pd.DataFrame(list(clicks_by_day))
            if not df_time.empty:
                fig_time = px.line(df_time, x='day', y='count', title='Clicks Over Time')
                time_chart = plot(fig_time, output_type='div', include_plotlyjs=False)
            else:
                time_chart = "<p>No time-based click data available</p>"
            
            # Device distribution
            device_data = clicks.values('device_type') \
                .annotate(count=Count('id')) \
                .order_by('-count')
            
            df_device = pd.DataFrame(list(device_data))
            if not df_device.empty:
                fig_device = px.pie(df_device, values='count', names='device_type', title='Device Distribution')
                device_chart = plot(fig_device, output_type='div', include_plotlyjs=False)
            else:
                device_chart = "<p>No device data available</p>"
            
            # Browser distribution
            browser_data = clicks.values('browser') \
                .annotate(count=Count('id')) \
                .order_by('-count')
            
            df_browser = pd.DataFrame(list(browser_data))
            if not df_browser.empty:
                fig_browser = px.bar(df_browser, x='browser', y='count', title='Browser Distribution')
                browser_chart = plot(fig_browser, output_type='div', include_plotlyjs=False)
            else:
                browser_chart = "<p>No browser data available</p>"
            
            # OS distribution
            os_data = clicks.values('os') \
                .annotate(count=Count('id')) \
                .order_by('-count')
            
            df_os = pd.DataFrame(list(os_data))
            if not df_os.empty:
                fig_os = px.bar(df_os, x='os', y='count', title='Operating System Distribution')
                os_chart = plot(fig_os, output_type='div', include_plotlyjs=False)
            else:
                os_chart = "<p>No OS data available</p>"
            
            context.update({
                'time_chart': time_chart,
                'device_chart': device_chart,
                'browser_chart': browser_chart,
                'os_chart': os_chart,
            })
        
        return context


def url_data_api(request, short_code):
    """API endpoint for URL analytics data"""
    url = get_object_or_404(URL, short_code=short_code)
    clicks = ClickData.objects.filter(url=url)
    
    if not clicks.exists():
        return JsonResponse({
            'success': False,
            'message': 'No click data available for this URL'
        })
    
    # Get clicks over time
    clicks_by_day = clicks.annotate(day=TruncDay('timestamp')) \
        .values('day') \
        .annotate(count=Count('id')) \
        .order_by('day')
    
    # Get device distribution
    device_data = clicks.values('device_type') \
        .annotate(count=Count('id')) \
        .order_by('-count')
    
    # Get browser distribution
    browser_data = clicks.values('browser') \
        .annotate(count=Count('id')) \
        .order_by('-count')
    
    # Get OS distribution
    os_data = clicks.values('os') \
        .annotate(count=Count('id')) \
        .order_by('-count')
    
    # Format data for API response
    time_data = [{'date': item['day'].strftime('%Y-%m-%d'), 'count': item['count']} for item in clicks_by_day]
    device_data_list = [{'device': item['device_type'], 'count': item['count']} for item in device_data]
    browser_data_list = [{'browser': item['browser'], 'count': item['count']} for item in browser_data]
    os_data_list = [{'os': item['os'], 'count': item['count']} for item in os_data]
    
    return JsonResponse({
        'success': True,
        'url_info': {
            'short_code': url.short_code,
            'original_url': url.original_url,
            'total_clicks': url.clicks,
            'created_at': url.created_at.strftime('%Y-%m-%d %H:%M:%S'),
        },
        'time_data': time_data,
        'device_data': device_data_list,
        'browser_data': browser_data_list,
        'os_data': os_data_list,
    })


def ai_insights_api(request, short_code):
    """API endpoint for AI-generated insights about URL analytics"""
    url = get_object_or_404(URL, short_code=short_code)
    clicks = ClickData.objects.filter(url=url)
    
    if not clicks.exists() or not settings.OPENAI_API_KEY:
        return JsonResponse({
            'success': False,
            'message': 'No click data available or OpenAI API key not configured'
        })
    
    try:
        # Prepare data for AI analysis
        clicks_by_day = clicks.annotate(day=TruncDay('timestamp')) \
            .values('day') \
            .annotate(count=Count('id')) \
            .order_by('day')
        
        device_data = list(clicks.values('device_type')
            .annotate(count=Count('id'))
            .order_by('-count'))
        
        browser_data = list(clicks.values('browser')
            .annotate(count=Count('id'))
            .order_by('-count'))
        
        os_data = list(clicks.values('os')
            .annotate(count=Count('id'))
            .order_by('-count'))
        
        # Format data for AI prompt
        time_data = [{'date': item['day'].strftime('%Y-%m-%d'), 'count': item['count']} for item in clicks_by_day]
        
        # Create a summary of the data
        data_summary = {
            'url_info': {
                'short_code': url.short_code,
                'original_url': url.original_url,
                'total_clicks': url.clicks,
                'created_at': url.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            },
            'time_data': time_data,
            'device_data': device_data,
            'browser_data': browser_data,
            'os_data': os_data,
        }
        
        # Create prompt for LLM
        template = """
        You are an analytics expert. Analyze the following URL click data and provide insights:
        
        URL Information:
        - Short Code: {short_code}
        - Original URL: {original_url}
        - Total Clicks: {total_clicks}
        - Created: {created_at}
        
        Click Data Over Time:
        {time_data}
        
        Device Distribution:
        {device_data}
        
        Browser Distribution:
        {browser_data}
        
        Operating System Distribution:
        {os_data}
        
        Provide 3-5 key insights about this data, including:
        1. Patterns in click behavior over time
        2. User demographics based on device/browser/OS
        3. Recommendations for optimizing for the most common platforms
        4. Any unusual patterns or outliers
        
        Format your response as a JSON array of insight objects, each with a "title" and "description" field.
        """
        
        prompt = PromptTemplate(
            input_variables=["short_code", "original_url", "total_clicks", "created_at", 
                            "time_data", "device_data", "browser_data", "os_data"],
            template=template,
        )
        
        # Initialize LLM
        llm = OpenAI(temperature=0.7, api_key=settings.OPENAI_API_KEY)
        chain = LLMChain(llm=llm, prompt=prompt)
        
        # Run the chain
        result = chain.run(
            short_code=url.short_code,
            original_url=url.original_url,
            total_clicks=url.clicks,
            created_at=url.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            time_data=json.dumps(time_data),
            device_data=json.dumps(device_data),
            browser_data=json.dumps(browser_data),
            os_data=json.dumps(os_data),
        )
        
        # Parse the result
        try:
            insights = json.loads(result)
        except json.JSONDecodeError:
            # If the LLM doesn't return valid JSON, create a fallback response
            insights = [
                {
                    "title": "AI Analysis",
                    "description": result
                }
            ]
        
        return JsonResponse({
            'success': True,
            'insights': insights
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'message': f'Error generating insights: {str(e)}'
        }, status=500)

@login_required
def dashboard(request):
    """Main analytics dashboard view"""
    stats = AnalyticsService.get_dashboard_stats()
    
    # Paginate top URLs
    paginator = Paginator(stats['top_urls'], 10)
    page = request.GET.get('page', 1)
    top_urls = paginator.get_page(page)
    
    context = {
        'stats': stats,
        'top_urls': top_urls,
        'growth_rates': stats['growth_rate'],
        'engagement': stats['engagement_metrics'],
        'retention': stats['retention_rate']
    }
    return render(request, 'analytics/dashboard.html', context)

@login_required
def url_stats(request, url_id):
    """Detailed statistics for a specific URL"""
    url = get_object_or_404(URL, id=url_id)
    stats = AnalyticsService.get_url_stats(url_id)
    
    context = {
        'url': url,
        'stats': stats,
        'hourly_data': stats['hourly_data'],
        'peak_hours': stats['peak_hours']
    }
    return render(request, 'analytics/url_stats.html', context)

@login_required
@require_http_methods(['GET'])
def api_url_stats(request, url_id):
    """API endpoint for URL statistics"""
    try:
        stats = AnalyticsService.get_url_stats(url_id)
        return JsonResponse({
            'success': True,
            'data': stats
        })
    except URL.DoesNotExist:
        return JsonResponse({
            'success': False,
            'error': 'URL not found'
        }, status=404)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

@login_required
@require_http_methods(['GET'])
def api_dashboard_stats(request):
    """API endpoint for dashboard statistics"""
    try:
        stats = AnalyticsService.get_dashboard_stats()
        return JsonResponse({
            'success': True,
            'data': stats
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)
