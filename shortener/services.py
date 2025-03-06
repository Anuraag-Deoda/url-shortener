from typing import Dict, Optional, Tuple
from django.core.cache import cache
from django.db import transaction
from .models import URL, Click
import hashlib
import base64
import re
from datetime import datetime
from user_agents import parse
import tldextract
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from django.conf import settings

class URLService:
    CACHE_TTL = 3600  # 1 hour
    
    @staticmethod
    def create_short_url(original_url: str, custom_code: Optional[str] = None) -> Tuple[URL, bool]:
        """
        Create a new shortened URL or return existing one
        Returns: (URL object, bool indicating if it was created)
        """
        # Normalize URL
        if not original_url.startswith(('http://', 'https://')):
            original_url = 'https://' + original_url
            
        # Check cache first
        cache_key = f'url_{original_url}'
        cached_url = cache.get(cache_key)
        if cached_url:
            return cached_url, False
            
        with transaction.atomic():
            # Check if URL already exists
            existing_url = URL.objects.filter(original_url=original_url).first()
            if existing_url:
                cache.set(cache_key, existing_url, URLService.CACHE_TTL)
                return existing_url, False
                
            # Generate short code
            if custom_code:
                if URL.objects.filter(short_code=custom_code).exists():
                    raise ValueError("Custom code already exists")
                short_code = custom_code
            else:
                short_code = URLService._generate_short_code(original_url)
                
            # Fetch metadata
            metadata = URLService._fetch_url_metadata(original_url)
            
            # Create new URL
            url = URL.objects.create(
                original_url=original_url,
                short_code=short_code,
                title=metadata.get('title', ''),
                description=metadata.get('description', ''),
                domain=metadata.get('domain', ''),
                is_active=True
            )
            
            cache.set(cache_key, url, URLService.CACHE_TTL)
            return url, True
    
    @staticmethod
    def record_click(url: URL, request) -> Click:
        """Record a click event with detailed analytics"""
        user_agent_string = request.META.get('HTTP_USER_AGENT', '')
        user_agent = parse(user_agent_string)
        
        click = Click.objects.create(
            url=url,
            visitor_id=URLService._get_visitor_id(request),
            ip_address=URLService._get_client_ip(request),
            referer=request.META.get('HTTP_REFERER', ''),
            user_agent=user_agent_string,
            browser=user_agent.browser.family,
            browser_version=user_agent.browser.version_string,
            os=user_agent.os.family,
            device=user_agent.device.family,
            is_mobile=user_agent.is_mobile,
            is_tablet=user_agent.is_tablet,
            is_bot=user_agent.is_bot
        )
        
        # Update URL click count in cache
        cache_key = f'url_clicks_{url.id}'
        cache.delete(cache_key)  # Invalidate cache
        
        return click
    
    @staticmethod
    def _generate_short_code(url: str, length: int = 6) -> str:
        """Generate a unique short code for a URL"""
        while True:
            # Create a hash of the URL + timestamp
            hash_input = f"{url}{datetime.now().timestamp()}"
            hash_object = hashlib.sha256(hash_input.encode())
            
            # Take first 'length' characters of base64 encoded hash
            short_code = base64.urlsafe_b64encode(hash_object.digest()).decode()[:length]
            
            # Ensure the code is URL-safe and unique
            short_code = re.sub(r'[^a-zA-Z0-9]', '', short_code)
            
            if not URL.objects.filter(short_code=short_code).exists():
                return short_code
    
    @staticmethod
    def _get_visitor_id(request) -> str:
        """Generate a consistent visitor ID based on IP and User Agent"""
        ip = URLService._get_client_ip(request)
        user_agent = request.META.get('HTTP_USER_AGENT', '')
        visitor_string = f"{ip}{user_agent}"
        return hashlib.md5(visitor_string.encode()).hexdigest()
    
    @staticmethod
    def _get_client_ip(request) -> str:
        """Get the client's real IP address"""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            return x_forwarded_for.split(',')[0]
        return request.META.get('REMOTE_ADDR', '')
    
    @staticmethod
    def _fetch_url_metadata(url: str) -> Dict[str, str]:
        """Fetch metadata about the URL (title, description, etc.)"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; URLShortenerBot/1.0)'
            }
            response = requests.get(url, headers=headers, timeout=5)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract domain
            extracted = tldextract.extract(url)
            domain = f"{extracted.domain}.{extracted.suffix}"
            
            # Get title
            title = soup.title.string if soup.title else ''
            
            # Get description
            description = ''
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc:
                description = meta_desc.get('content', '')
            
            return {
                'title': title[:200] if title else '',  # Limit length
                'description': description[:500] if description else '',
                'domain': domain
            }
            
        except Exception as e:
            # Log error but don't fail
            print(f"Error fetching metadata for {url}: {str(e)}")
            return {
                'title': '',
                'description': '',
                'domain': urlparse(url).netloc
            }

class URLValidator:
    @staticmethod
    def validate_url(url: str) -> Tuple[bool, str]:
        """
        Validate a URL and return (is_valid, error_message)
        """
        if not url:
            return False, "URL cannot be empty"
            
        # Add scheme if missing
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
            
        try:
            result = urlparse(url)
            if not all([result.scheme, result.netloc]):
                return False, "Invalid URL format"
                
            # Check against blacklist
            domain = result.netloc.lower()
            if URLValidator._is_domain_blacklisted(domain):
                return False, "This domain is not allowed"
                
            return True, ""
            
        except Exception:
            return False, "Invalid URL format"
    
    @staticmethod
    def validate_custom_code(code: str) -> Tuple[bool, str]:
        """
        Validate a custom short code
        """
        if not code:
            return False, "Custom code cannot be empty"
            
        if len(code) < 4:
            return False, "Custom code must be at least 4 characters"
            
        if len(code) > 10:
            return False, "Custom code cannot be longer than 10 characters"
            
        if not re.match(r'^[a-zA-Z0-9-_]+$', code):
            return False, "Custom code can only contain letters, numbers, hyphens and underscores"
            
        if URL.objects.filter(short_code=code).exists():
            return False, "This custom code is already in use"
            
        return True, ""
    
    @staticmethod
    def _is_domain_blacklisted(domain: str) -> bool:
        """
        Check if a domain is blacklisted
        """
        blacklist = getattr(settings, 'BLACKLISTED_DOMAINS', [])
        return domain in blacklist 