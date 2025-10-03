# D:\Chatbot\scripts\ingest_ditstek.py

import sys
from pathlib import Path

# Add the project root (D:\Chatbot) to Python path
project_root = Path(__file__).parent.parent  # Go up from scripts/ to Chatbot/
sys.path.insert(0, str(project_root))

from app.ingestion.scrape_to_redis import ingest_website_data

if __name__ == "__main__":
    ingest_website_data(r"D:\sam\data\scraped_content.json")