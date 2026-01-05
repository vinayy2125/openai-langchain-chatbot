"""
Content Extractor Module

Extracts and processes content from HTML pages using BeautifulSoup.
Creates structured chunks suitable for embedding generation.
"""
import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup, NavigableString, Comment

logger = logging.getLogger(__name__)


@dataclass
class PageData:
    """Structured data extracted from a web page."""
    url: str
    title: str = ""
    description: str = ""
    headings: List[str] = field(default_factory=list)
    content: str = ""
    chunks: List[str] = field(default_factory=list)
    links: List[str] = field(default_factory=list)
    images: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    sections: List['Section'] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "url": self.url,
            "title": self.title,
            "description": self.description,
            "headings": self.headings,
            "content": self.content,
            "chunks": self.chunks,
            "links": self.links,
            "images": self.images,
            "metadata": self.metadata,
            "sections": [s.to_dict() for s in self.sections] if self.sections else [],
        }


@dataclass
class Section:
    """A content section with heading and body text."""
    heading: str
    level: int  # h1=1, h2=2, etc.
    content: str
    url_anchor: str = ""  # URL fragment for direct linking
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "heading": self.heading,
            "level": self.level,
            "content": self.content,
            "url_anchor": self.url_anchor,
        }


class ContentExtractor:
    """Extract structured content from HTML pages with section-based chunking."""
    
    # Tags to exclude from content extraction - MINIMAL set to preserve max content
    EXCLUDED_TAGS = {
        'script', 'style', 'noscript', 'form', 'button', 'input', 'select', 'textarea',
        'iframe', 'embed', 'object', 'svg', 'canvas', 'video', 'audio'
    }
    
    # Tags that contain navigation/menu content we want to preserve
    NAV_TAGS = {'nav', 'header', 'footer', 'aside'}
    
    # Tags that typically contain main content
    CONTENT_TAGS = {
        'p', 'article', 'section', 'main', 'div', 'span', 'li', 'td', 'th', 'a',
        'footer', 'header', 'nav', 'aside'
    }
    
    # Page type classification patterns
    PAGE_TYPE_PATTERNS = {
        'service': ['services/', '/service', 'consulting', 'development', 'solutions'],
        'portfolio': ['portfolio', 'case-study', 'case-studies', 'project', 'work'],
        'blog': ['blog', 'article', 'news', 'post', 'insights'],
        'about': ['about', 'team', 'company', 'who-we-are', 'our-story'],
        'contact': ['contact', 'get-in-touch', 'reach-us', 'location'],
        'career': ['career', 'jobs', 'hiring', 'join-us'],
        'technology': ['technology', 'tech-stack', 'tools'],
    }
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the content extractor.
        
        Args:
            chunk_size: Target size for content chunks (in characters)
            chunk_overlap: Overlap between chunks to maintain context
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract(self, html: str, url: str) -> PageData:
        """
        Extract structured content from HTML with section-based organization.
        
        Args:
            html: Raw HTML content
            url: Source URL of the page
            
        Returns:
            PageData object with extracted content and sections
        """
        soup = BeautifulSoup(html, 'lxml')
        
        page_data = PageData(url=url)
        
        # Classify page type based on URL
        page_type = self._classify_page_type(url)
        page_data.metadata['page_type'] = page_type
        page_data.metadata['service_category'] = self._extract_service_category(url)
        page_data.metadata['source_url'] = url
        
        # FIRST: Extract navigation content BEFORE removing any elements
        nav_content = self._extract_nav_content(soup)
        
        # Extract metadata before cleanup
        page_data.title = self._extract_title(soup)
        page_data.description = self._extract_description(soup)
        page_data.headings = self._extract_headings(soup)
        page_data.metadata.update(self._extract_metadata(soup))
        
        # Extract links for crawling (before removing elements)
        page_data.links = self._extract_links(soup, url)
        
        # Extract images with alt text
        page_data.images = self._extract_images(soup, url)
        
        # Extract sections with heading hierarchy
        page_data.sections = self._extract_sections(soup, url)
        
        # Now remove truly unwanted elements (scripts, styles, forms)
        for tag in soup(self.EXCLUDED_TAGS):
            tag.decompose()
        
        # Remove comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()
        
        # Extract main content
        main_content = self._extract_text_content(soup)
        
        # Combine navigation content with main content
        if nav_content:
            page_data.content = f"Navigation & Services:\n{nav_content}\n\nMain Content:\n{main_content}"
            page_data.metadata['nav_items'] = nav_content
        else:
            page_data.content = main_content
        
        # Create section-based chunks for embeddings with source tracking
        page_data.chunks = self._chunk_content_by_sections(page_data)
        
        logger.debug(f"Extracted {len(page_data.chunks)} chunks from {url} (type: {page_type})")
        
        return page_data
    
    def _classify_page_type(self, url: str) -> str:
        """Classify page type based on URL patterns."""
        url_lower = url.lower()
        
        # Check if it's the homepage
        parsed = urlparse(url)
        if parsed.path in ('', '/', '/index.html', '/home'):
            return 'home'
        
        # Match against patterns
        for page_type, patterns in self.PAGE_TYPE_PATTERNS.items():
            if any(pattern in url_lower for pattern in patterns):
                return page_type
        
        return 'general'
    
    def _extract_service_category(self, url: str) -> str:
        """Extract service/capability category from URL patterns."""
        url_lower = url.lower()
        
        # Service category patterns mapped to capability areas
        SERVICE_CATEGORIES = {
            'ai_ml': ['ai-', 'artificial-intelligence', 'machine-learning', 'chatbot', 'ai-agent', 'generative-ai'],
            'web_development': ['web-development', 'full-stack', 'frontend', 'backend', 'asp-net', 'php', 'react', 'angular', 'node'],
            'mobile_development': ['mobile-app', 'cross-platform', 'ios', 'android', 'flutter', 'react-native'],
            'cloud_devops': ['cloud', 'devops', 'aws', 'azure', 'docker', 'kubernetes', 'ci-cd'],
            'healthcare': ['healthcare', 'hipaa', 'ehr', 'telemedicine', 'medical', 'patient', 'health-insurance'],
            'consulting': ['consulting', 'it-consulting', 'digital-transformation', 'strategy'],
            'qa_testing': ['qa', 'testing', 'quality-assurance', 'automation-testing'],
            'enterprise': ['enterprise', 'erp', 'crm', 'saas', 'legacy-modernization'],
            'iot': ['iot', 'internet-of-things', 'embedded', 'smart-'],
            'fintech': ['fintech', 'payment', 'banking', 'financial'],
        }
        
        for category, patterns in SERVICE_CATEGORIES.items():
            if any(pattern in url_lower for pattern in patterns):
                return category
        
        return ''
    
    def _extract_sections(self, soup: BeautifulSoup, url: str) -> List[Section]:
        """
        Extract content organized by heading sections.
        Groups content under each heading for structured retrieval.
        """
        sections = []
        
        # Find all headings in order
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        
        for i, heading in enumerate(headings):
            heading_text = heading.get_text(strip=True)
            if not heading_text or len(heading_text) < 2:
                continue
            
            level = int(heading.name[1])
            
            # Create URL-safe anchor
            anchor = re.sub(r'[^a-z0-9]+', '-', heading_text.lower()).strip('-')
            
            # Get content between this heading and the next
            content_parts = []
            sibling = heading.find_next_sibling()
            
            while sibling:
                # Stop at next heading of same or higher level
                if sibling.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    sibling_level = int(sibling.name[1])
                    if sibling_level <= level:
                        break
                
                # Extract text from this element
                if sibling.name not in self.EXCLUDED_TAGS:
                    text = sibling.get_text(strip=True)
                    if text and len(text) > 3:
                        content_parts.append(text)
                
                sibling = sibling.find_next_sibling()
            
            section_content = ' '.join(content_parts)
            if section_content or heading_text:  # Keep even if content is empty
                sections.append(Section(
                    heading=heading_text,
                    level=level,
                    content=section_content,
                    url_anchor=anchor
                ))
        
        return sections
    
    def _extract_nav_content(self, soup: BeautifulSoup) -> str:
        """
        Extract content from navigation elements (menus, headers, footers).
        Captures services, products, and important site-wide links.
        """
        nav_items = []
        seen = set()
        
        # Extract from nav, header, footer, aside elements
        for tag_name in self.NAV_TAGS:
            for element in soup.find_all(tag_name):
                # Get all links with meaningful text
                for link in element.find_all('a', href=True):
                    text = link.get_text(strip=True)
                    # Also check for title or aria-label if text is empty
                    if not text:
                        text = link.get('title') or link.get('aria-label') or ''
                    
                    # Filter criteria: meaningful length, not seen
                    if text and len(text) >= 2 and len(text) <= 150:
                        text_lower = text.lower()
                        # Skip very common generic navigation items
                        skip_terms = {'menu', 'toggle', 'close', 'open'}
                        if text_lower not in skip_terms and text_lower not in seen:
                            nav_items.append(text)
                            seen.add(text_lower)
                
                # Also get non-link text in lists and divs (often service descriptions)
                for item_tag in ['li', 'div', 'span', 'p']:
                    for item in element.find_all(item_tag):
                        # Only take items that are direct or near-direct children to avoid duplication
                        if len(item.get_text(strip=True)) < 5: continue
                        text = item.get_text(strip=True)
                        if text and len(text) >= 5 and len(text) <= 200:
                            text_lower = text.lower()
                            if text_lower not in seen:
                                nav_items.append(text)
                                seen.add(text_lower)
        
        # Return as formatted string
        if nav_items:
            return "\n".join(f"• {item}" for item in nav_items)
        return ""
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title."""
        # Try <title> tag first
        title_tag = soup.find('title')
        if title_tag and title_tag.string:
            return title_tag.string.strip()
        
        # Try <h1> as fallback
        h1 = soup.find('h1')
        if h1:
            return h1.get_text(strip=True)
        
        # Try og:title
        og_title = soup.find('meta', property='og:title')
        if og_title and og_title.get('content'):
            return og_title['content'].strip()
        
        return ""
    
    def _extract_description(self, soup: BeautifulSoup) -> str:
        """Extract page description from meta tags."""
        # Try standard meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            return meta_desc['content'].strip()
        
        # Try og:description
        og_desc = soup.find('meta', property='og:description')
        if og_desc and og_desc.get('content'):
            return og_desc['content'].strip()
        
        return ""
    
    def _extract_headings(self, soup: BeautifulSoup) -> List[str]:
        """Extract all headings (h1-h6) in order."""
        headings = []
        for level in range(1, 7):
            for heading in soup.find_all(f'h{level}'):
                text = heading.get_text(strip=True)
                if text:
                    headings.append(text)
        return headings
    
    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract useful metadata from the page."""
        metadata = {}
        
        # Open Graph metadata
        for og in soup.find_all('meta', property=re.compile(r'^og:')):
            prop = og.get('property', '').replace('og:', '')
            content = og.get('content', '')
            if prop and content:
                metadata[f'og_{prop}'] = content
        
        # Twitter Card metadata
        for twitter in soup.find_all('meta', attrs={'name': re.compile(r'^twitter:')}):
            name = twitter.get('name', '').replace('twitter:', '')
            content = twitter.get('content', '')
            if name and content:
                metadata[f'twitter_{name}'] = content
        
        # Canonical URL
        canonical = soup.find('link', rel='canonical')
        if canonical and canonical.get('href'):
            metadata['canonical_url'] = canonical['href']
        
        # Language
        html_tag = soup.find('html')
        if html_tag and html_tag.get('lang'):
            metadata['language'] = html_tag['lang']
        
        return metadata
    
    def _extract_text_content(self, soup: BeautifulSoup) -> str:
        """Extract clean text content from the page with improved coverage."""
        # Try to find main content area first
        main_content = (
            soup.find('main') or 
            soup.find('article') or 
            soup.find(id=re.compile(r'content|main|article', re.I)) or
            soup.find(class_=re.compile(r'content|main|article', re.I)) or
            soup.body
        )
        
        if not main_content:
            return ""
        
        # Extract text from meaningful elements in order (improved coverage)
        text_parts = []
        
        # Target elements that typically contain meaningful content
        # Added 'a' to capture link text and 'span' for inline content
        content_selectors = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'td', 'th', 
                            'blockquote', 'figcaption', 'article', 'section', 'a', 'span']
        
        for element in main_content.find_all(content_selectors):
            # Skip if parent is an excluded tag
            if element.parent and element.parent.name in self.EXCLUDED_TAGS:
                continue
            
            text = element.get_text(strip=True)
            # Reduced minimum from 15 to 5 chars to capture short service names
            if text and len(text) > 5:
                # Add heading markers for context
                if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    text = f"\n## {text}\n"
                text_parts.append(text)
        
        # If structured extraction yields little, fallback to raw text
        if len(text_parts) < 3:
            # Get text with proper spacing from all descendants
            lines = []
            for element in main_content.descendants:
                if isinstance(element, NavigableString):
                    text = str(element).strip()
                    if text and element.parent.name not in self.EXCLUDED_TAGS:
                        lines.append(text)
            content = ' '.join(lines)
            content = re.sub(r'\s+', ' ', content)
            return content.strip()
        
        # Join and clean up whitespace
        content = ' '.join(text_parts)
        content = re.sub(r'\s+', ' ', content)
        content = content.strip()
        
        return content
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract all valid links from the page."""
        links = []
        seen: Set[str] = set()
        
        for a in soup.find_all('a', href=True):
            href = a['href']
            
            # Skip anchors, javascript, mailto, etc.
            if href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
                continue
            
            # Convert relative URLs to absolute
            absolute_url = urljoin(base_url, href)
            
            # Remove fragments
            parsed = urlparse(absolute_url)
            clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            if parsed.query:
                clean_url += f"?{parsed.query}"
            
            # Deduplicate
            if clean_url not in seen and parsed.scheme in ('http', 'https'):
                seen.add(clean_url)
                links.append(clean_url)
        
        return links
    
    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        """Extract images with their alt text."""
        images = []
        
        for img in soup.find_all('img', src=True):
            src = urljoin(base_url, img['src'])
            alt = img.get('alt', '').strip()
            
            images.append({
                'src': src,
                'alt': alt,
            })
        
        return images
    
    def _chunk_content_by_sections(self, page_data: PageData) -> List[str]:
        """
        Create structured chunks organized by sections with source tracking.
        
        Each chunk includes:
        - Page title and type
        - Section heading for context
        - Source URL for transparency
        - Structured content
        """
        chunks = []
        page_type = page_data.metadata.get('page_type', 'general')
        source_url = page_data.metadata.get('source_url', page_data.url)
        
        # Create header template for chunks
        def create_chunk_header(section_heading: str = "", section_anchor: str = "") -> str:
            header_parts = [
                f"[Page: {page_data.title}]",
                f"[Type: {page_type}]",
            ]
            if section_heading:
                header_parts.append(f"[Section: {section_heading}]")
            
            # Include direct link to section if anchor exists
            if section_anchor:
                header_parts.append(f"[Source: {source_url}#{section_anchor}]")
            else:
                header_parts.append(f"[Source: {source_url}]")
            
            return "\n".join(header_parts) + "\n\n"
        
        # First, create chunks from structured sections
        if page_data.sections:
            for section in page_data.sections:
                if not section.content and not section.heading:
                    continue
                
                section_header = create_chunk_header(section.heading, section.url_anchor)
                section_text = f"## {section.heading}\n\n{section.content}" if section.content else f"## {section.heading}"
                
                # If section content fits in one chunk
                if len(section_header) + len(section_text) <= self.chunk_size:
                    chunks.append(section_header + section_text)
                else:
                    # Split large sections into multiple chunks, preserving section context
                    sentences = re.split(r'(?<=[.!?])\s+', section.content)
                    current_chunk = []
                    current_len = len(section_header) + len(f"## {section.heading}\n\n")
                    
                    for sentence in sentences:
                        if current_len + len(sentence) > self.chunk_size and current_chunk:
                            chunk_content = section_header + f"## {section.heading}\n\n" + ' '.join(current_chunk)
                            chunks.append(chunk_content)
                            # Keep overlap
                            overlap = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk
                            current_chunk = overlap.copy()
                            current_len = len(section_header) + len(f"## {section.heading}\n\n") + sum(len(s) for s in current_chunk)
                        
                        current_chunk.append(sentence)
                        current_len += len(sentence) + 1
                    
                    if current_chunk:
                        chunk_content = section_header + f"## {section.heading}\n\n" + ' '.join(current_chunk)
                        chunks.append(chunk_content)
        
        # Also create a navigation/services chunk if available
        nav_items = page_data.metadata.get('nav_items', '')
        if nav_items:
            nav_header = create_chunk_header("Navigation & Services")
            nav_chunk = nav_header + "Available Services and Navigation:\n" + nav_items
            if len(nav_chunk) <= self.chunk_size * 2:  # Allow larger nav chunk
                chunks.insert(0, nav_chunk)  # Put at beginning
        
        # Fallback to original chunking if no sections found
        if not chunks:
            chunks = self._chunk_content(page_data)
        
        return chunks
    
    def _chunk_content(self, page_data: PageData) -> List[str]:
        """
        Split content into chunks suitable for embeddings (fallback method).
        
        Optimized for RAG retrieval with:
        1. Page context prefix on EVERY chunk (title, URL) for source attribution
        2. Sentence-based overlap (not character-based) to preserve meaning
        3. Token-aware sizing (~400 tokens/chunk, ~80 tokens overlap)
        """
        chunks = []
        
        # Build page context prefix for attribution (added to every chunk)
        page_type = page_data.metadata.get('page_type', 'general')
        page_context = f"[Page: {page_data.title}]\n[Type: {page_type}]\n[Source: {page_data.url}]\n\n"
        
        # Build full text with metadata context for first chunk only
        full_text_parts = []
        
        if page_data.description:
            full_text_parts.append(f"Description: {page_data.description}")
        
        if page_data.headings:
            # Include top headings as topic indicators
            top_headings = [h for h in page_data.headings[:8] if h and len(h) < 100]
            if top_headings:
                full_text_parts.append(f"Topics: {', '.join(top_headings)}")
        
        if page_data.content:
            full_text_parts.append(page_data.content)
        
        full_text = '\n\n'.join(full_text_parts)
        
        if not full_text.strip():
            return []
        
        # Split into sentences (improved regex for better accuracy)
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', full_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            # Fallback: if no sentence boundaries found, split on double newlines
            sentences = [s.strip() for s in full_text.split('\n\n') if s.strip()]
        
        if not sentences:
            # Last resort: return as single chunk with context
            return [page_context + full_text[:self.chunk_size]]
        
        # Calculate overlap in terms of sentences (roughly 2-3 sentences for 80 tokens)
        overlap_sentence_count = max(1, self.chunk_overlap // 150)  # ~150 chars per sentence avg
        
        current_chunk_sentences = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence exceeds chunk size and we have content
            if current_length + sentence_length > self.chunk_size and current_chunk_sentences:
                # Create chunk with page context prefix
                chunk_text = page_context + ' '.join(current_chunk_sentences)
                chunks.append(chunk_text)
                
                # Keep last N sentences for overlap (sentence-based, not character-based)
                overlap_sentences = current_chunk_sentences[-overlap_sentence_count:]
                current_chunk_sentences = overlap_sentences.copy()
                current_length = sum(len(s) for s in current_chunk_sentences)
            
            current_chunk_sentences.append(sentence)
            current_length += sentence_length + 1  # +1 for space
        
        # Don't forget the last chunk
        if current_chunk_sentences:
            chunk_text = page_context + ' '.join(current_chunk_sentences)
            chunks.append(chunk_text)
        
        # If no chunks were created but we have content, create one chunk
        if not chunks and full_text.strip():
            chunks.append(page_context + full_text[:self.chunk_size])
        
        return chunks

