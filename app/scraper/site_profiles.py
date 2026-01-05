"""
Site Profiles for Scraper Plug-and-Play Support.
Defines domain-specific settings and interaction rules.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable

@dataclass
class SiteProfile:
    """Configuration profile for a specific website."""
    domain: str
    selectors_to_wait_for: List[str] = field(default_factory=list)
    selectors_to_hover: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=list)
    sitemap_url: Optional[str] = None
    custom_extraction_logic: Optional[Callable] = None

# Registry of site profiles
SITE_PROFILES: Dict[str, SiteProfile] = {
    "www.ditstek.com": SiteProfile(
        domain="www.ditstek.com",
        selectors_to_wait_for=[
            ".nav-item-dropdown", 
            ".portfolio-item"
        ],
        selectors_to_hover=[
            "a.nav-link.dropdown-toggle",
            ".services-grid-item"
        ],
        exclude_patterns=[
            "/blog/tag/",
            "/portfolio/category/"
        ],
        sitemap_url="https://www.ditstek.com/sitemap.xml"
    )
}

def get_site_profile(url: str) -> Optional[SiteProfile]:
    """Get the site profile for a given URL."""
    from urllib.parse import urlparse
    domain = urlparse(url).netloc
    return SITE_PROFILES.get(domain)
