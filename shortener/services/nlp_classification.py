"""
NLP Traffic Classification Service.
Natural language processing for traffic categorization and analysis.
"""
import logging
import re
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import Counter
from urllib.parse import urlparse, parse_qs

import numpy as np
from django.utils import timezone
from django.core.cache import cache

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Result of URL/referrer classification."""
    category: str
    subcategory: Optional[str] = None
    confidence: float = 0.0
    keywords: List[str] = field(default_factory=list)
    features: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'category': self.category,
            'subcategory': self.subcategory,
            'confidence': round(self.confidence, 3),
            'keywords': self.keywords
        }


class TextPreprocessor:
    """
    Text preprocessing for NLP analysis.
    """

    # Common stop words
    STOP_WORDS = {
        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
        'this', 'that', 'these', 'those', 'it', 'its', 'www', 'http', 'https',
        'com', 'org', 'net', 'html', 'php', 'asp', 'jsp'
    }

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        if not text:
            return []

        # Convert to lowercase and split
        text = text.lower()

        # Remove special characters but keep spaces
        text = re.sub(r'[^a-z0-9\s]', ' ', text)

        # Split and filter
        tokens = text.split()
        tokens = [t for t in tokens if len(t) > 2 and t not in self.STOP_WORDS]

        return tokens

    def extract_url_components(self, url: str) -> Dict:
        """Extract meaningful components from URL."""
        if not url:
            return {}

        try:
            parsed = urlparse(url)

            # Extract domain parts
            domain = parsed.netloc.lower()
            domain_parts = domain.replace('www.', '').split('.')

            # Extract path tokens
            path = parsed.path
            path_tokens = self.tokenize(path.replace('/', ' '))

            # Extract query params
            query_tokens = []
            if parsed.query:
                params = parse_qs(parsed.query)
                for key, values in params.items():
                    query_tokens.append(key.lower())
                    for v in values:
                        query_tokens.extend(self.tokenize(v))

            return {
                'domain': domain,
                'domain_parts': domain_parts,
                'path_tokens': path_tokens,
                'query_tokens': query_tokens,
                'all_tokens': domain_parts + path_tokens + query_tokens
            }
        except Exception:
            return {}


class ReferrerClassifier:
    """
    Classify referrer URLs into categories.
    """

    CATEGORIES = {
        'search_engine': {
            'domains': ['google', 'bing', 'yahoo', 'duckduckgo', 'baidu', 'yandex'],
            'subcategories': {
                'organic': [],  # Default
                'paid': ['ads', 'adwords', 'gclid', 'msclkid', 'utm_medium=cpc']
            }
        },
        'social': {
            'domains': ['facebook', 'twitter', 'linkedin', 'instagram', 'pinterest',
                       'tiktok', 'reddit', 'youtube', 'snapchat', 'whatsapp', 'telegram'],
            'subcategories': {
                'paid': ['ads', 'sponsored', 'promoted'],
                'organic': []
            }
        },
        'email': {
            'patterns': ['mail', 'email', 'newsletter', 'mailchimp', 'sendgrid',
                        'constantcontact', 'mailgun'],
            'subcategories': {
                'newsletter': ['newsletter', 'digest', 'weekly'],
                'campaign': ['campaign', 'promo', 'sale']
            }
        },
        'news': {
            'domains': ['news', 'cnn', 'bbc', 'nytimes', 'wsj', 'reuters',
                       'huffpost', 'buzzfeed', 'medium', 'techcrunch', 'theverge'],
            'subcategories': {}
        },
        'forum': {
            'domains': ['reddit', 'quora', 'stackoverflow', 'hackernews', 'ycombinator',
                       'discourse', 'forum'],
            'subcategories': {}
        },
        'ecommerce': {
            'domains': ['amazon', 'ebay', 'shopify', 'etsy', 'alibaba', 'aliexpress'],
            'subcategories': {}
        },
        'affiliate': {
            'patterns': ['affiliate', 'partner', 'referral', 'ref=', 'aff='],
            'subcategories': {}
        }
    }

    def __init__(self):
        self.preprocessor = TextPreprocessor()

    def classify(self, referrer: str) -> ClassificationResult:
        """
        Classify a referrer URL.
        """
        if not referrer:
            return ClassificationResult(category='direct', confidence=1.0)

        referrer_lower = referrer.lower()
        components = self.preprocessor.extract_url_components(referrer)

        if not components:
            return ClassificationResult(category='unknown', confidence=0.0)

        domain = components.get('domain', '')
        all_tokens = components.get('all_tokens', [])

        # Check each category
        best_match = None
        best_confidence = 0.0

        for category, config in self.CATEGORIES.items():
            confidence = 0.0
            matched_keywords = []

            # Check domains
            if 'domains' in config:
                for d in config['domains']:
                    if d in domain:
                        confidence = max(confidence, 0.9)
                        matched_keywords.append(d)

            # Check patterns
            if 'patterns' in config:
                for p in config['patterns']:
                    if p in referrer_lower:
                        confidence = max(confidence, 0.8)
                        matched_keywords.append(p)

            if confidence > best_confidence:
                best_confidence = confidence
                subcategory = self._determine_subcategory(
                    referrer_lower, config.get('subcategories', {})
                )
                best_match = ClassificationResult(
                    category=category,
                    subcategory=subcategory,
                    confidence=confidence,
                    keywords=matched_keywords
                )

        if best_match:
            return best_match

        return ClassificationResult(
            category='referral',
            confidence=0.5,
            features={'domain': domain}
        )

    def _determine_subcategory(
        self,
        referrer: str,
        subcategories: Dict
    ) -> Optional[str]:
        """Determine subcategory from patterns."""
        for subcat, patterns in subcategories.items():
            if patterns:  # Skip empty pattern lists
                for pattern in patterns:
                    if pattern in referrer:
                        return subcat
        return None


class UTMAnalyzer:
    """
    Analyze and categorize UTM parameters.
    """

    UTM_PARAMS = ['utm_source', 'utm_medium', 'utm_campaign', 'utm_content', 'utm_term']

    # Common UTM values mapping
    MEDIUM_CATEGORIES = {
        'cpc': 'paid_search',
        'ppc': 'paid_search',
        'paidsearch': 'paid_search',
        'cpm': 'display',
        'banner': 'display',
        'display': 'display',
        'email': 'email',
        'newsletter': 'email',
        'social': 'social',
        'organic': 'organic',
        'referral': 'referral',
        'affiliate': 'affiliate',
        'partner': 'affiliate'
    }

    SOURCE_CATEGORIES = {
        'google': 'search',
        'bing': 'search',
        'facebook': 'social',
        'instagram': 'social',
        'twitter': 'social',
        'linkedin': 'social',
        'email': 'email',
        'newsletter': 'email'
    }

    def extract_utm_params(self, url: str) -> Dict:
        """Extract UTM parameters from URL."""
        try:
            parsed = urlparse(url)
            params = parse_qs(parsed.query)

            utm_data = {}
            for param in self.UTM_PARAMS:
                if param in params:
                    utm_data[param] = params[param][0]

            return utm_data
        except Exception:
            return {}

    def categorize_campaign(self, utm_data: Dict) -> ClassificationResult:
        """
        Categorize campaign based on UTM parameters.
        """
        if not utm_data:
            return ClassificationResult(category='unknown', confidence=0.0)

        source = utm_data.get('utm_source', '').lower()
        medium = utm_data.get('utm_medium', '').lower()
        campaign = utm_data.get('utm_campaign', '')

        # Determine category from medium
        category = self.MEDIUM_CATEGORIES.get(medium, 'other')

        # Fallback to source
        if category == 'other':
            for key, cat in self.SOURCE_CATEGORIES.items():
                if key in source:
                    category = cat
                    break

        # Extract keywords from campaign name
        preprocessor = TextPreprocessor()
        keywords = preprocessor.tokenize(campaign)

        return ClassificationResult(
            category=category,
            subcategory=source if source else None,
            confidence=0.9 if medium else 0.7,
            keywords=keywords,
            features=utm_data
        )


class SpamReferrerDetector:
    """
    Detect spam referrers using NLP patterns.
    """

    SPAM_PATTERNS = [
        # Generic spam patterns
        r'free.*traffic', r'buy.*traffic', r'cheap.*traffic',
        r'seo.*service', r'rank.*website', r'backlink',
        r'casino', r'poker', r'gambling', r'betting',
        r'viagra', r'cialis', r'pharmacy', r'pills',
        r'make.*money.*fast', r'work.*from.*home',

        # Known spam domains
        r'buttons-for-website', r'semalt', r'ranksonic',
        r'floating-share-buttons', r'event-tracking',
        r'trafficbot', r'ghostvisitor'
    ]

    SPAM_TLDS = ['.xyz', '.top', '.gq', '.ml', '.cf', '.tk', '.pw']

    def __init__(self):
        self.spam_regex = re.compile(
            '|'.join(self.SPAM_PATTERNS),
            re.IGNORECASE
        )

    def is_spam(self, referrer: str) -> Tuple[bool, float, List[str]]:
        """
        Check if referrer is spam.
        Returns (is_spam, confidence, reasons).
        """
        if not referrer:
            return False, 0.0, []

        referrer_lower = referrer.lower()
        reasons = []
        confidence = 0.0

        # Check spam patterns
        if self.spam_regex.search(referrer_lower):
            reasons.append("Matches spam pattern")
            confidence = max(confidence, 0.9)

        # Check spam TLDs
        for tld in self.SPAM_TLDS:
            if referrer_lower.endswith(tld):
                reasons.append(f"Suspicious TLD: {tld}")
                confidence = max(confidence, 0.6)

        # Check for abnormal URL structure
        if referrer.count('/') > 10:
            reasons.append("Abnormally deep URL path")
            confidence = max(confidence, 0.5)

        # Check for excessive query parameters
        if referrer.count('&') > 5 or referrer.count('?') > 1:
            reasons.append("Excessive URL parameters")
            confidence = max(confidence, 0.4)

        is_spam = confidence > 0.5

        return is_spam, confidence, reasons


class KeywordExtractor:
    """
    Extract keywords from URLs and content.
    """

    def __init__(self):
        self.preprocessor = TextPreprocessor()

    def extract_keywords(
        self,
        text: str,
        max_keywords: int = 10
    ) -> List[Tuple[str, int]]:
        """
        Extract most frequent keywords from text.
        """
        tokens = self.preprocessor.tokenize(text)
        counter = Counter(tokens)

        return counter.most_common(max_keywords)

    def extract_from_urls(
        self,
        urls: List[str],
        max_keywords: int = 20
    ) -> List[Tuple[str, int]]:
        """
        Extract keywords from multiple URLs.
        """
        all_tokens = []

        for url in urls:
            components = self.preprocessor.extract_url_components(url)
            all_tokens.extend(components.get('all_tokens', []))

        counter = Counter(all_tokens)

        return counter.most_common(max_keywords)

    def get_url_topics(self, url: str) -> List[str]:
        """
        Identify main topics from URL.
        """
        components = self.preprocessor.extract_url_components(url)
        tokens = components.get('all_tokens', [])

        # Simple topic identification based on common patterns
        topics = []

        topic_patterns = {
            'technology': ['tech', 'software', 'app', 'digital', 'cloud', 'data', 'ai', 'ml'],
            'business': ['business', 'enterprise', 'company', 'corporate', 'industry'],
            'marketing': ['marketing', 'ads', 'campaign', 'seo', 'content', 'brand'],
            'finance': ['finance', 'money', 'investment', 'bank', 'trading', 'crypto'],
            'health': ['health', 'medical', 'fitness', 'wellness', 'doctor'],
            'education': ['education', 'learn', 'course', 'training', 'tutorial'],
            'entertainment': ['entertainment', 'music', 'video', 'game', 'movie'],
            'ecommerce': ['shop', 'store', 'buy', 'product', 'cart', 'checkout']
        }

        for topic, patterns in topic_patterns.items():
            if any(p in tokens for p in patterns):
                topics.append(topic)

        return topics


class CampaignNameParser:
    """
    Parse and analyze campaign names using NLP.
    """

    # Common campaign name patterns
    PATTERNS = {
        'date': r'\d{4}[-_]?\d{2}[-_]?\d{2}',
        'quarter': r'q[1-4][-_]?\d{4}|q[1-4]',
        'version': r'v\d+|version[-_]?\d+',
        'variant': r'variant[-_]?[a-z]|test[-_]?[a-z0-9]+',
        'region': r'us|uk|eu|apac|latam|emea|global',
        'channel': r'email|social|paid|organic|display|search|affiliate'
    }

    def parse_campaign_name(self, name: str) -> Dict:
        """
        Parse campaign name to extract structured information.
        """
        if not name:
            return {}

        name_lower = name.lower()
        result = {}

        for pattern_name, pattern in self.PATTERNS.items():
            matches = re.findall(pattern, name_lower)
            if matches:
                result[pattern_name] = matches[0] if len(matches) == 1 else matches

        # Extract remaining keywords
        preprocessor = TextPreprocessor()
        tokens = preprocessor.tokenize(name)

        # Remove already identified tokens
        identified = set()
        for v in result.values():
            if isinstance(v, list):
                identified.update(v)
            else:
                identified.add(v)

        remaining_tokens = [t for t in tokens if t not in identified]
        result['keywords'] = remaining_tokens

        return result


class AutoTagger:
    """
    Automatically tag URLs and campaigns.
    """

    def __init__(self):
        self.referrer_classifier = ReferrerClassifier()
        self.utm_analyzer = UTMAnalyzer()
        self.keyword_extractor = KeywordExtractor()
        self.spam_detector = SpamReferrerDetector()
        self.campaign_parser = CampaignNameParser()

    def auto_tag(
        self,
        url: str,
        referrer: Optional[str] = None
    ) -> Dict:
        """
        Automatically generate tags for a URL/referrer combination.
        """
        tags = set()
        metadata = {}

        # Classify referrer
        if referrer:
            ref_result = self.referrer_classifier.classify(referrer)
            tags.add(f"source:{ref_result.category}")
            if ref_result.subcategory:
                tags.add(f"source:{ref_result.subcategory}")
            metadata['referrer_classification'] = ref_result.to_dict()

            # Check for spam
            is_spam, spam_conf, _ = self.spam_detector.is_spam(referrer)
            if is_spam:
                tags.add("warning:spam_referrer")
                metadata['spam_confidence'] = spam_conf

        # Analyze UTM parameters
        utm_data = self.utm_analyzer.extract_utm_params(url)
        if utm_data:
            utm_result = self.utm_analyzer.categorize_campaign(utm_data)
            tags.add(f"campaign:{utm_result.category}")
            metadata['utm_classification'] = utm_result.to_dict()

            # Parse campaign name
            if 'utm_campaign' in utm_data:
                parsed = self.campaign_parser.parse_campaign_name(utm_data['utm_campaign'])
                if 'region' in parsed:
                    tags.add(f"region:{parsed['region']}")
                if 'channel' in parsed:
                    tags.add(f"channel:{parsed['channel']}")
                metadata['campaign_parsed'] = parsed

        # Extract topics
        topics = self.keyword_extractor.get_url_topics(url)
        for topic in topics:
            tags.add(f"topic:{topic}")

        return {
            'tags': list(tags),
            'metadata': metadata
        }


class NLPClassificationService:
    """
    High-level NLP classification service.
    """

    def __init__(self):
        self.referrer_classifier = ReferrerClassifier()
        self.utm_analyzer = UTMAnalyzer()
        self.spam_detector = SpamReferrerDetector()
        self.keyword_extractor = KeywordExtractor()
        self.auto_tagger = AutoTagger()

    def classify_referrer(self, referrer: str) -> Dict:
        """
        Classify a referrer URL.
        """
        result = self.referrer_classifier.classify(referrer)
        is_spam, spam_conf, spam_reasons = self.spam_detector.is_spam(referrer)

        return {
            'classification': result.to_dict(),
            'is_spam': is_spam,
            'spam_confidence': spam_conf,
            'spam_reasons': spam_reasons
        }

    def analyze_campaign(self, url: str) -> Dict:
        """
        Analyze campaign from URL.
        """
        utm_data = self.utm_analyzer.extract_utm_params(url)
        classification = self.utm_analyzer.categorize_campaign(utm_data)

        return {
            'utm_params': utm_data,
            'classification': classification.to_dict()
        }

    def get_traffic_breakdown(self, days_back: int = 30) -> Dict:
        """
        Get traffic breakdown by classification.
        """
        from shortener.models import ClickData
        from datetime import timedelta

        start_date = timezone.now() - timedelta(days=days_back)

        clicks = ClickData.objects.filter(
            timestamp__gte=start_date
        ).values_list('referrer', flat=True)

        categories = Counter()
        subcategories = Counter()

        for referrer in clicks:
            result = self.referrer_classifier.classify(referrer)
            categories[result.category] += 1
            if result.subcategory:
                subcategories[f"{result.category}:{result.subcategory}"] += 1

        total = sum(categories.values())

        return {
            'categories': dict(categories),
            'subcategories': dict(subcategories),
            'percentages': {
                k: round(v / total * 100, 2) if total > 0 else 0
                for k, v in categories.items()
            }
        }

    def detect_spam_referrers(self, days_back: int = 7) -> Dict:
        """
        Detect spam referrers in recent traffic.
        """
        from shortener.models import ClickData
        from datetime import timedelta

        start_date = timezone.now() - timedelta(days=days_back)

        referrers = ClickData.objects.filter(
            timestamp__gte=start_date,
            referrer__isnull=False
        ).values('referrer').annotate(
            count=Count('id')
        ).order_by('-count')

        spam_referrers = []

        for item in referrers:
            is_spam, confidence, reasons = self.spam_detector.is_spam(item['referrer'])
            if is_spam:
                spam_referrers.append({
                    'referrer': item['referrer'],
                    'count': item['count'],
                    'confidence': confidence,
                    'reasons': reasons
                })

        return {
            'spam_referrers': spam_referrers[:50],  # Top 50
            'total_spam_clicks': sum(s['count'] for s in spam_referrers)
        }

    def auto_tag_url(self, url: str, referrer: Optional[str] = None) -> Dict:
        """
        Auto-generate tags for a URL.
        """
        return self.auto_tagger.auto_tag(url, referrer)

    def extract_keywords_from_traffic(self, days_back: int = 30) -> Dict:
        """
        Extract common keywords from traffic data.
        """
        from shortener.models import ClickData
        from datetime import timedelta

        start_date = timezone.now() - timedelta(days=days_back)

        urls = list(ClickData.objects.filter(
            timestamp__gte=start_date
        ).values_list('url__original_url', flat=True)[:10000])

        referrers = list(ClickData.objects.filter(
            timestamp__gte=start_date,
            referrer__isnull=False
        ).values_list('referrer', flat=True)[:10000])

        url_keywords = self.keyword_extractor.extract_from_urls(urls, 20)
        referrer_keywords = self.keyword_extractor.extract_from_urls(referrers, 20)

        return {
            'url_keywords': [{'keyword': k, 'count': c} for k, c in url_keywords],
            'referrer_keywords': [{'keyword': k, 'count': c} for k, c in referrer_keywords]
        }
