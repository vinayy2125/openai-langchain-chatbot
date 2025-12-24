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
        }


class ContentExtractor:
    """Extract structured content from HTML pages."""
    
    # Tags to exclude from content extraction
    EXCLUDED_TAGS = {
        'script', 'style', 'noscript', 'header', 'footer', 'nav',
        'aside', 'form', 'button', 'input', 'select', 'textarea',
        'iframe', 'embed', 'object', 'svg', 'canvas', 'video', 'audio'
    }
    
    # Tags that typically contain main content
    CONTENT_TAGS = {'p', 'article', 'section', 'main', 'div', 'span', 'li', 'td', 'th'}
    
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
        Extract structured content from HTML.
        
        Args:
            html: Raw HTML content
            url: Source URL of the page
            
        Returns:
            PageData object with extracted content
        """
        soup = BeautifulSoup(html, 'lxml')
        
        # Remove unwanted elements
        for tag in soup(self.EXCLUDED_TAGS):
            tag.decompose()
        
        # Remove comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()
        
        page_data = PageData(url=url)
        
        # Extract metadata
        page_data.title = self._extract_title(soup)
        page_data.description = self._extract_description(soup)
        page_data.headings = self._extract_headings(soup)
        page_data.metadata = self._extract_metadata(soup)
        
        # Extract main content
        page_data.content = self._extract_text_content(soup)
        
        # Extract links for crawling
        page_data.links = self._extract_links(soup, url)
        
        # Extract images with alt text
        page_data.images = self._extract_images(soup, url)
        
        # Create chunks for embeddings
        page_data.chunks = self._chunk_content(page_data)
        
        logger.debug(f"Extracted {len(page_data.chunks)} chunks from {url}")
        
        return page_data
    
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
        content_selectors = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'td', 'th', 
                            'blockquote', 'figcaption', 'article', 'section']
        
        for element in main_content.find_all(content_selectors):
            # Skip if parent is an excluded tag
            if element.parent and element.parent.name in self.EXCLUDED_TAGS:
                continue
            
            text = element.get_text(strip=True)
            # Filter out very short fragments (likely noise) and duplicates
            if text and len(text) > 15:
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
    
    def _chunk_content(self, page_data: PageData) -> List[str]:
        """
        Split content into chunks suitable for embeddings.
        
        Optimized for RAG retrieval with:
        1. Page context prefix on EVERY chunk (title, URL) for source attribution
        2. Sentence-based overlap (not character-based) to preserve meaning
        3. Token-aware sizing (~400 tokens/chunk, ~80 tokens overlap)
        """
        chunks = []
        
        # Build page context prefix for attribution (added to every chunk)
        page_context = f"[Source: {page_data.title}]\n[URL: {page_data.url}]\n\n"
        
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

