"""
STEP 2: PRUNE WEBSITE DATA
- Remove SEO filler, legal pages, repeated content, non-sales blogs, deep tech docs
- Retain only content supporting: service explanation, differentiation, common questions, fit/non-fit, pricing/objections
"""
import json
from collections import defaultdict

# Load the scraped data
with open('scraped_data/scrape_20251229_131253.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

pages = data['pages']

# Categories for pruning
RETAINED = defaultdict(list)
DISCARDED = defaultdict(list)

# Keywords to identify blog topics that ARE useful for sales conversations
SALES_RELEVANT_BLOG_KEYWORDS = [
    'how to choose', 'cost', 'pricing', 'benefits', 'vs', 'versus', 
    'why hire', 'why choose', 'comparison', 'guide to hiring',
    'outsourcing', 'offshore', 'dedicated team', 'when to use',
    'roi', 'investment', 'what is', 'difference between'
]

# Keywords to identify deep technical documentation (to exclude)
DEEP_TECH_KEYWORDS = [
    'tutorial', 'code example', 'implementation guide', 'step-by-step',
    'api documentation', 'sdk', 'library', 'framework tutorial'
]

# Analyze each page
for page in pages:
    url = page.get('url', '')
    title = page.get('title', 'No title').lower()
    content = page.get('content', '').lower()
    
    # Skip 404 pages
    if '404' in title:
        DISCARDED['404/Error Pages'].append(page.get('title', url))
        continue
    
    # ----- DISCARD CRITERIA -----
    
    # 1. Legal/Policy pages
    if '/privacy-policy' in url or '/terms' in url or 'privacy policy' in title or 'terms of service' in title:
        DISCARDED['Legal/Policy Pages'].append(page.get('title', url))
        continue
    
    # 2. Regional SEO pages (duplicate content for different locations)
    regional_patterns = ['/california', '/florida', '/atlanta', '/australia', '/japan', '/dubai', '/south-africa', '-uk', '-canada']
    is_regional_seo = any(pattern in url.lower() for pattern in regional_patterns) and '/services/' not in url and '/industries/' not in url
    if is_regional_seo:
        DISCARDED['Regional SEO Pages'].append(page.get('title', url))
        continue
    
    # 3. Blog posts - evaluate individually
    if '/blog' in url:
        # Check if blog is sales-relevant
        is_sales_relevant = any(kw in title for kw in SALES_RELEVANT_BLOG_KEYWORDS)
        is_deep_tech = any(kw in title for kw in DEEP_TECH_KEYWORDS)
        
        if is_deep_tech:
            DISCARDED['Deep Technical Blog Posts'].append(page.get('title', url))
            continue
        elif not is_sales_relevant:
            # Generic blog posts, tutorials, news etc
            DISCARDED['Non-Sales Blog Posts'].append(page.get('title', url))
            continue
        else:
            RETAINED['Sales-Relevant Blog Posts'].append(page.get('title', url))
            continue
    
    # 4. Events pages (not directly useful for conversational sales)
    if '/events/' in url:
        DISCARDED['Event Pages'].append(page.get('title', url))
        continue
    
    # ----- RETAIN CRITERIA -----
    
    # Core service pages
    if '/services/' in url:
        RETAINED['Core Service Pages'].append(page.get('title', url))
        continue
    
    # Industry vertical pages
    if '/industries/' in url:
        RETAINED['Industry Vertical Pages'].append(page.get('title', url))
        continue
    
    # Homepage (core messaging)
    if url.endswith('.com/') or url.endswith('.ca/'):
        RETAINED['Homepage'].append(page.get('title', url))
        continue
    
    # About page (company differentiation)
    if '/about' in url:
        RETAINED['About/Company Info'].append(page.get('title', url))
        continue
    
    # Portfolio/Case studies (proof points)
    if '/portfolios' in url or '/innovation-stories' in url or 'case study' in title:
        RETAINED['Portfolio/Case Studies'].append(page.get('title', url))
        continue
    
    # Technology page (capabilities)
    if '/technology' in url:
        RETAINED['Technology Stack'].append(page.get('title', url))
        continue
    
    # Enterprise software development
    if 'enterprise' in url.lower():
        RETAINED['Enterprise Solutions'].append(page.get('title', url))
        continue
    
    # Everything else goes to "Other - Review Needed"
    DISCARDED['Other/Uncategorized'].append(page.get('title', url))

# Print results
print("=" * 70)
print("STEP 2: PRUNE WEBSITE DATA - RESULTS")
print("=" * 70)

print("\n" + "=" * 70)
print("RETAINED TOPICS (for fine-tuning)")
print("=" * 70)

total_retained = 0
for category, items in sorted(RETAINED.items()):
    print(f"\n[{category}] - {len(items)} pages")
    print("-" * 50)
    for item in items[:10]:  # Show first 10
        print(f"   + {item}")
    if len(items) > 10:
        print(f"   ... and {len(items) - 10} more")
    total_retained += len(items)

print(f"\n>>> TOTAL RETAINED: {total_retained} pages")

print("\n" + "=" * 70)
print("DISCARDED CATEGORIES (NOT for fine-tuning)")
print("=" * 70)

total_discarded = 0 
for category, items in sorted(DISCARDED.items()):
    print(f"\n[{category}] - {len(items)} pages")
    print("-" * 50)
    for item in items[:5]:  # Show first 5
        print(f"   - {item}")
    if len(items) > 5:
        print(f"   ... and {len(items) - 5} more")
    total_discarded += len(items)

print(f"\n>>> TOTAL DISCARDED: {total_discarded} pages")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
RETAINED FOR FINE-TUNING:
  - Core Service Pages: {len(RETAINED['Core Service Pages'])}
  - Industry Vertical Pages: {len(RETAINED['Industry Vertical Pages'])}
  - Homepage: {len(RETAINED['Homepage'])}
  - About/Company Info: {len(RETAINED['About/Company Info'])}
  - Portfolio/Case Studies: {len(RETAINED['Portfolio/Case Studies'])}
  - Technology Stack: {len(RETAINED['Technology Stack'])}
  - Enterprise Solutions: {len(RETAINED['Enterprise Solutions'])}
  - Sales-Relevant Blog Posts: {len(RETAINED['Sales-Relevant Blog Posts'])}
  
DISCARDED:
  - Legal/Policy Pages: {len(DISCARDED['Legal/Policy Pages'])}
  - Regional SEO Pages: {len(DISCARDED['Regional SEO Pages'])}
  - Non-Sales Blog Posts: {len(DISCARDED['Non-Sales Blog Posts'])}
  - Deep Technical Blog Posts: {len(DISCARDED['Deep Technical Blog Posts'])}
  - Event Pages: {len(DISCARDED['Event Pages'])}
  - 404/Error Pages: {len(DISCARDED['404/Error Pages'])}
  - Other/Uncategorized: {len(DISCARDED['Other/Uncategorized'])}

RETENTION RATE: {total_retained}/{total_retained + total_discarded} ({100*total_retained/(total_retained + total_discarded):.1f}%)
""")

print("=" * 70)
print("STEP 2 COMPLETE - STOPPING AS INSTRUCTED")
print("=" * 70)
print("\nAwaiting instruction to proceed to STEP 3.")

# Save retained page URLs for next steps
retained_urls = []
for category, items in RETAINED.items():
    for page in pages:
        if page.get('title', '') in items:
            retained_urls.append({
                'url': page.get('url'),
                'title': page.get('title'),
                'category': category,
                'chunks': page.get('chunks', [])
            })

with open('step2_retained_pages.json', 'w', encoding='utf-8') as f:
    json.dump(retained_urls, f, indent=2, ensure_ascii=False)
    
print(f"\nRetained pages saved to: step2_retained_pages.json")
