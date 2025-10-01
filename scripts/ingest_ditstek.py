# scripts/ingest_ditstek.py
from app.ingestion.scrape_to_redis import ingest_website_data


if __name__ == "__main__":
    ingest_website_data(r"D:\sam\data\scraped_content.json")