from django.shortcuts import render, redirect, get_object_or_404
from django.views.generic import ListView, CreateView, DetailView
from django.urls import reverse_lazy
from django.contrib import messages
from django.http import JsonResponse
from .models import URL, ClickData
from .forms import URLShortenerForm
import user_agents


def extract_client_info(request):
    """Extract client information from request for analytics"""
    user_agent_string = request.META.get('HTTP_USER_AGENT', '')
    user_agent = user_agents.parse(user_agent_string)
    
    client_info = {
        'ip_address': request.META.get('REMOTE_ADDR'),
        'user_agent': user_agent_string,
        'referrer': request.META.get('HTTP_REFERER'),
        'browser': f"{user_agent.browser.family} {user_agent.browser.version_string}",
        'os': f"{user_agent.os.family} {user_agent.os.version_string}",
        'device_type': 'Mobile' if user_agent.is_mobile else 'Tablet' if user_agent.is_tablet else 'Desktop',
    }
    
    # In a real app, you might use a geolocation service here
    # client_info['country'] = geo_lookup(client_info['ip_address']).country
    # client_info['city'] = geo_lookup(client_info['ip_address']).city
    
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
    
    # Record click data for analytics
    client_info = extract_client_info(request)
    ClickData.objects.create(
        url=url,
        **client_info
    )
    
    # Increment click counter
    url.increment_clicks()
    
    return redirect(url.original_url)


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
