from typing import Optional, Dict
import os
import logging

logger = logging.getLogger(__name__)


class GeoLocationService:
    """Service for IP-based geolocation using MaxMind GeoLite2"""

    _reader = None

    @classmethod
    def get_reader(cls):
        """Get or create the GeoIP2 database reader"""
        if cls._reader is None:
            try:
                import geoip2.database
                from django.conf import settings

                db_path = getattr(
                    settings, 'GEOIP_DATABASE_PATH',
                    os.path.join(settings.BASE_DIR, 'data', 'GeoLite2-City.mmdb')
                )

                if os.path.exists(db_path):
                    cls._reader = geoip2.database.Reader(db_path)
                else:
                    logger.warning(f"GeoIP database not found at {db_path}")
                    return None
            except ImportError:
                logger.warning("geoip2 library not installed")
                return None
            except Exception as e:
                logger.error(f"Error initializing GeoIP reader: {e}")
                return None

        return cls._reader

    @classmethod
    def lookup_ip(cls, ip_address: str) -> Optional[Dict]:
        """
        Look up geographic information for an IP address.

        Args:
            ip_address: The IP address to look up

        Returns:
            Dictionary with geolocation data or None if lookup fails
        """
        # Skip private/local IP addresses
        if cls._is_private_ip(ip_address):
            return None

        reader = cls.get_reader()
        if reader is None:
            return None

        try:
            response = reader.city(ip_address)

            return {
                'country': response.country.name,
                'country_code': response.country.iso_code,
                'city': response.city.name,
                'region': response.subdivisions.most_specific.name if response.subdivisions else None,
                'latitude': response.location.latitude,
                'longitude': response.location.longitude,
                'timezone': response.location.time_zone,
            }
        except Exception as e:
            logger.debug(f"GeoIP lookup failed for {ip_address}: {e}")
            return None

    @classmethod
    def _is_private_ip(cls, ip_address: str) -> bool:
        """Check if an IP address is private/local"""
        if not ip_address:
            return True

        # Common private/local patterns
        private_patterns = [
            '127.',
            '10.',
            '172.16.', '172.17.', '172.18.', '172.19.',
            '172.20.', '172.21.', '172.22.', '172.23.',
            '172.24.', '172.25.', '172.26.', '172.27.',
            '172.28.', '172.29.', '172.30.', '172.31.',
            '192.168.',
            '::1',
            'fe80:',
            'fc00:',
            'fd00:',
        ]

        for pattern in private_patterns:
            if ip_address.startswith(pattern):
                return True

        return False

    @classmethod
    def close(cls):
        """Close the GeoIP database reader"""
        if cls._reader:
            cls._reader.close()
            cls._reader = None
