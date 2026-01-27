"""
PII Log Redactor.

Ensures that user secrets (Email, Phone, generic identifiers) are redacted
from application logs while preserving debug utility.

Data in the Database remains RAW/ENCRYPTED (Source of Truth).
Logs are REDACTED (Safe for Ops).
"""

import re
import logging

class PIILogRedactor:
    """Regex-based PII Redactor for Logging."""
    
    # Patterns
    EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    PHONE_PATTERN = r'\b(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b'
    IP_PATTERN = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    
    def redact(self, message: str) -> str:
        """Redact PII from string."""
        if not message:
            return message
            
        redacted = re.sub(self.EMAIL_PATTERN, '[EMAIL_REDACTED]', message)
        redacted = re.sub(self.PHONE_PATTERN, '[PHONE_REDACTED]', redacted)
        # IP redaction can be partial if needed, but full is safer
        # redacted = re.sub(self.IP_PATTERN, '[IP_REDACTED]', redacted)
        
        return redacted

_redactor = PIILogRedactor()

def redact_log(msg: str) -> str:
    return _redactor.redact(msg)
