import os
import re
import logging
import pandas as pd
import feedparser
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NLPCollector:
    """
    Scrapes maritime and supply chain news RSS feeds.
    Captures headlines and publication dates to be scored by FinBERT later.
    """
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.raw_dir = os.path.join(self.base_dir, "data", "raw")

        # Public, reliable RSS feeds for global shipping and supply chain logistics
        self.rss_feeds = {
            "Hellenic Shipping": "https://www.hellenicshippingnews.com/feed/",
            "Supply Chain Brain": "https://www.supplychainbrain.com/rss",
            "Google News Maritime": "https://news.google.com/rss/search?q=supply+chain+shipping+freight"
        }

    def clean_html(self, raw_text):
        """Removes messy HTML tags from feed summaries for clean ML processing."""
        cleanr = re.compile('<.*?>')
        return re.sub(cleanr, '', str(raw_text)).strip()

    def fetch_news(self) -> pd.DataFrame:
        """Main execution method to pull RSS feeds and save to Parquet."""
        logger.info("Starting RSS feed scraping for market sentiment...")
        articles = []

        for source, url in self.rss_feeds.items():
            logger.info(f"Fetching from {source}...")
            try:
                # Parse the live XML feed
                feed = feedparser.parse(url)
                
                # Grab the top 20 most recent articles per feed
                for entry in feed.entries[:20]:  
                    articles.append({
                        "source": source,
                        "published_date": entry.get("published", datetime.now().isoformat()),
                        "title": self.clean_html(entry.get("title", "")),
                        "summary": self.clean_html(entry.get("summary", ""))
                    })
            except Exception as e:
                logger.error(f"Failed to fetch {source}: {str(e)}")

        # Fallback for Portfolio/Demo mode if offline
        if not articles:
            logger.warning("No articles fetched. Generating portfolio mock data.")
            articles = [
                {"source": "Mock News", "published_date": datetime.now().isoformat(), "title": "Port congestion eases in Shanghai as lockdown lifts", "summary": ""},
                {"source": "Mock News", "published_date": datetime.now().isoformat(), "title": "Freight rates spike amid sudden canal blockage", "summary": ""}
            ]

        df = pd.DataFrame(articles)
        
        # Standardize the date formats so pandas can merge them later
        try:
            df['published_date'] = pd.to_datetime(df['published_date'], utc=True)
        except Exception:
            df['published_date'] = pd.Timestamp.now(tz='UTC')

        # Save to raw data folder
        output_file = os.path.join(self.raw_dir, "news_data_raw.parquet")
        df.to_parquet(output_file)
        
        logger.info(f"Successfully saved {len(df)} news articles to {output_file}")
        return df

if __name__ == "__main__":
    collector = NLPCollector()
    df = collector.fetch_news()
    print("\nLatest News Snippets:")
    # Print just the source and title of the first 5 articles so it fits in the terminal nicely
    pd.set_option('display.max_colwidth', 80)
    print(df[['source', 'title']].head())