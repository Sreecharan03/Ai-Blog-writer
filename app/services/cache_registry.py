from enum import Enum

from app.api.cache_registry import CacheRegistryService, CacheRecord, fingerprint_bytes, fingerprint_text, normalize_text_for_fingerprint


class CacheScope(str, Enum):
    GLOBAL_PUBLIC = "PUBLIC_GLOBAL"
    TENANT_PRIVATE = "TENANT_PRIVATE"

__all__ = [
    "CacheRegistryService",
    "CacheScope",
    "CacheRecord",
    "fingerprint_bytes",
    "fingerprint_text",
    "normalize_text_for_fingerprint",
]
