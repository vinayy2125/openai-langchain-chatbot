"""Analyze scraped website data for fine-tuning preparation"""
import json
from collections import Counter

# Load the scraped data
with open('scraped_data/scrape_20251229_131253.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print("=" * 60)
print("STEP 1: WEBSITE DATA INSPECTION (READ-ONLY)")
print("=" * 60)

# Basic stats
print("\n[BASIC STATISTICS]")
print(f"   Total pages scraped: {data['total_pages']}")
print(f"   Total URLs found: {data['total_urls_found']}")
print(f"   Total errors: {data['total_errors']}")
print(f"   Max pages reached: {data['max_pages_reached']}")
print(f"   Base URL: {data['config']['base_urls']}")

# Extract unique page URLs and titles
pages = data['pages']
page_categories = Counter()
page_types = Counter()

print(f"\n[WEBSITE SECTIONS/PAGES]")
print("-" * 60)

# Categorize pages by URL pattern
for page in pages:
    url = page.get('url', '')
    title = page.get('title', 'No title')
    
    # Classify by URL path
    if '/services/' in url:
        page_categories['Services'] += 1
    elif '/industries/' in url:
        page_categories['Industries'] += 1
    elif '/blog' in url:
        page_categories['Blog'] += 1
    elif '/innovation-stories' in url:
        page_categories['Case Studies/Portfolio'] += 1
    elif '/portfolios' in url:
        page_categories['Portfolio'] += 1
    elif '/about' in url:
        page_categories['About'] += 1
    elif '/contact' in url:
        page_categories['Contact'] += 1
    elif '/events/' in url:
        page_categories['Events'] += 1
    elif '/technology' in url:
        page_categories['Technology'] += 1
    elif '/privacy-policy' in url or '/terms' in url:
        page_categories['Legal/Policy'] += 1
    elif url.endswith('.com/') or url.endswith('.ca/'):
        page_categories['Homepage'] += 1
    else:
        page_categories['Other'] += 1
        
print("\n[PAGE CATEGORIES] (by URL pattern):")
for cat, count in sorted(page_categories.items(), key=lambda x: -x[1]):
    print(f"   {cat}: {count} pages")

# Extract unique page types/titles
print("\n[SAMPLE PAGE TITLES BY CATEGORY]")
print("-" * 60)

# Group by category
services_pages = []
industries_pages = []
blog_pages = []
other_pages = []

for page in pages:
    url = page.get('url', '')
    title = page.get('title', 'No title')
    
    if '/services/' in url:
        services_pages.append(title)
    elif '/industries/' in url:
        industries_pages.append(title)
    elif '/blog' in url:
        blog_pages.append(title)
    else:
        other_pages.append(title)

print("\n[SERVICES PAGES]:")
for title in sorted(set(services_pages))[:25]:
    print(f"   - {title}")

print(f"\n[INDUSTRIES PAGES]:")
for title in sorted(set(industries_pages))[:15]:
    print(f"   - {title}")

print(f"\n[BLOG PAGES] (sample - first 10):")
for title in sorted(set(blog_pages))[:10]:
    print(f"   - {title[:70]}...")

print("\n" + "=" * 60)
print("HIGH-LEVEL THEMES COVERED BY THE WEBSITE:")
print("=" * 60)

themes = """
1. OFFSHORE SOFTWARE DEVELOPMENT
   - Custom software development
   - Dedicated development teams
   - Hiring models (Dedicated Team, Fixed Price, Hourly)
   
2. TECHNOLOGY SERVICES
   - Full-stack development (React, Angular, Vue, .NET, Node, PHP, Laravel)
   - API & Microservices
   - SaaS Development
   - MVP Development
   - Legacy App Modernization
   - Cloud Services
   - Mobile App Development
   - Web Application Development
   - AI Software Development
   - AI Chatbot Development
   - AI Agent Development
   - QA & Testing Services
   - Backend Development
   - Cross-Platform Development
   - Product Engineering
   - Digital Transformation
   - IT Consulting
   
3. INDUSTRY VERTICALS
   - Healthcare (HIPAA, HL7 compliance, RPM, Asset Management, Insurance)
   - Fintech
   - Real Estate
   - EdTech
   - Retail
   - IoT
   - Logistics & Transportation
   - Workflow Automation
   - Mining
   - Agriculture
   - Automotive
   - Insurance
   
4. COMPANY INFORMATION
   - About company
   - Case studies/Portfolio
   - Client testimonials
   - Company locations (India, Canada, USA, South Africa)
   - Events (GITEX, HIMSS)
   
5. HIRING/ENGAGEMENT
   - Hire remote developers
   - Hire ASP.NET developers
   - Hire PHP developers
   - Hire UI/UX developers
   - Hire Front-end developers
   - Hiring models and pricing
   
6. MARKETING/SEO CONTENT
   - Blog posts
   - Technology pages
   - Regional pages (California, Florida, Atlanta, Australia, Japan, Dubai)
"""
print(themes)

print("\n" + "=" * 60)
print("STEP 1 COMPLETE - STOPPING AS INSTRUCTED")
print("=" * 60)
print("\nAwaiting instruction to proceed to STEP 2.")
