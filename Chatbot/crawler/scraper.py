import asyncio
import os
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from urllib.parse import urlparse


# -------------------------------
# Scrape a single page
# -------------------------------
async def scrape_page(url: str) -> str:
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, timeout=30000)  # allow more time
            await page.wait_for_load_state("networkidle")  

            # wait extra for lazy/dynamic content (tweak as needed)
            await page.wait_for_timeout(5000)  

            html = await page.content()
            await browser.close()
    except Exception as e:
        print(f"❌ Playwright failed for {url}: {e}")
        return ""

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)


# -------------------------------
# Recursive crawler + manual links
# -------------------------------
async def scrape_website_recursive(start_url: str, max_pages: int = 1500, max_depth: int = 300) -> dict:
    visited = set()
    scraped_data = {}
    base_domain = urlparse(start_url).netloc

    # start queue: auto + manual URLs
    to_visit = [(start_url, 0)]

    while to_visit and len(visited) < max_pages:
        url, depth = to_visit.pop(0)
        if url in visited or depth > max_depth:
            continue

        print(f"🌐 Crawling: {url} (depth {depth}) | Queue size: {len(to_visit)}")
        text = await scrape_page(url)
        if not text.strip():
            continue

        scraped_data[url] = text
        visited.add(url)
        print(f"✅ Crawled: {url}")

        # discover new internal links automatically
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto(url, timeout=20000)
                links = await page.eval_on_selector_all("a[href]", "els => els.map(el => el.href)")
                await browser.close()
        except:
            links = []

        for link in links:
            if urlparse(link).netloc == base_domain and link not in visited:
                to_visit.append((link, depth + 1))

    return scraped_data


# -------------------------------
# Runner
# -------------------------------
if __name__ == "__main__":
    start_url = "https://www.ditstek.com/"
    site_data = asyncio.run(scrape_website_recursive(start_url, max_pages=1500, max_depth=300))
    print(f"Scraped {len(site_data)} pages.")
