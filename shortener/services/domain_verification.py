from typing import Tuple
from django.utils import timezone
import logging

logger = logging.getLogger(__name__)


class DomainVerificationService:
    """Service for verifying custom domain ownership via DNS"""

    @staticmethod
    def verify_domain(custom_domain) -> Tuple[bool, str]:
        """
        Verify domain ownership via TXT record.

        Args:
            custom_domain: CustomDomain model instance

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            import dns.resolver
        except ImportError:
            return False, "DNS verification library not available"

        expected_record = custom_domain.get_dns_txt_record()

        try:
            # Check TXT records
            txt_records = dns.resolver.resolve(custom_domain.domain, 'TXT')

            for record in txt_records:
                record_value = record.to_text().strip('"')
                if record_value == expected_record:
                    custom_domain.verification_status = 'verified'
                    custom_domain.verified_at = timezone.now()
                    custom_domain.is_active = True
                    custom_domain.save()
                    return True, "Domain verified successfully"

            custom_domain.verification_status = 'failed'
            custom_domain.save()
            return False, "TXT record not found. Please add the verification record to your DNS."

        except dns.resolver.NXDOMAIN:
            custom_domain.verification_status = 'failed'
            custom_domain.save()
            return False, "Domain does not exist"

        except dns.resolver.NoAnswer:
            custom_domain.verification_status = 'failed'
            custom_domain.save()
            return False, "No TXT records found for this domain"

        except dns.resolver.Timeout:
            return False, "DNS lookup timed out. Please try again."

        except Exception as e:
            logger.error(f"Domain verification error for {custom_domain.domain}: {e}")
            return False, f"Verification failed: {str(e)}"

    @staticmethod
    def check_cname(custom_domain) -> Tuple[bool, str]:
        """
        Check if CNAME is properly configured.

        Args:
            custom_domain: CustomDomain model instance

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            import dns.resolver
        except ImportError:
            return False, "DNS verification library not available"

        expected_target = custom_domain.get_cname_target()

        try:
            cname_records = dns.resolver.resolve(custom_domain.domain, 'CNAME')

            for record in cname_records:
                if record.target.to_text().rstrip('.') == expected_target:
                    return True, "CNAME configured correctly"

            return False, f"CNAME should point to {expected_target}"

        except dns.resolver.NoAnswer:
            return False, f"No CNAME record found. Please add a CNAME pointing to {expected_target}"

        except dns.resolver.NXDOMAIN:
            return False, "Domain does not exist"

        except Exception as e:
            logger.error(f"CNAME check error for {custom_domain.domain}: {e}")
            return False, f"CNAME check failed: {str(e)}"
