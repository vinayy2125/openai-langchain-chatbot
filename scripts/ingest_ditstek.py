import sys
from pathlib import Path

project_root = Path(__file__).parent.parent  
sys.path.insert(0, str(project_root))

from app.ingestion.scrape_to_redis import ingest_website_data

if __name__ == "__main__":
    ingest_website_data(r"D:\sam\data\scraped_content.json")