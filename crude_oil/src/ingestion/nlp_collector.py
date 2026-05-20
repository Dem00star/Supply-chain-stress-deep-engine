import os
import yaml
import logging
import feedparser
import pandas as pd
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnergyNewsCollector:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_path = os.path.join(self.base_dir, 'config', 'config.yaml')
        
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
        self.keywords = self.config['data_sources'].get('nlp_keywords', ['OPEC', 'Oil', 'Crude'])
        self.raw_data_dir = os.path.join(self.base_dir, 'data', 'raw')
        
        # High-signal energy news feeds
        self.rss_feeds = [
            "https://feeds.a.dj.com/rss/RSSWorldNews.xml", # WSJ World (Geopolitics)
            "https://www.reutersagency.com/feed/?best-sectors=energy", # Reuters Energy
            "https://oilprice.com/rss/main" # Dedicated Oil Markets
        ]

    def fetch_news_headlines(self, days_back=7):
        """Scrapes RSS feeds and filters for energy-specific geopolitical events."""
        logger.info("Initializing Energy NLP Scraper...")
        cutoff_date = datetime.now() - timedelta(days=days_back)
        news_data = []

        for url in self.rss_feeds:
            logger.info(f"Parsing feed: {url}")
            feed = feedparser.parse(url)
            
            for entry in feed.entries:
                try:
                    # Convert RSS published date to datetime object
                    # Fallback to current time if feed lacks standard timestamp
                    pub_date = pd.to_datetime(entry.get('published', datetime.now())).tz_localize(None)
                    
                    if pub_date >= cutoff_date:
                        title = entry.title
                        summary = entry.get('summary', '')
                        
                        # Filter: Only keep articles mentioning our config keywords (e.g., OPEC, Aramco)
                        text_to_check = (title + " " + summary).lower()
                        if any(kw.lower() in text_to_check for kw in self.keywords):
                            news_data.append({
                                'Date': pub_date.date(),
                                'Headline': title,
                                'Source': url
                            })
                except Exception as e:
                    logger.warning(f"Failed to parse an entry: {e}")

        if news_data:
            df = pd.DataFrame(news_data)
            output_path = os.path.join(self.raw_data_dir, 'news_data_raw.parquet')
            
            # If historical data exists, append to it to build our corpus over time
            if os.path.exists(output_path):
                old_df = pd.read_parquet(output_path)
                df = pd.concat([old_df, df]).drop_duplicates(subset=['Headline']).reset_index(drop=True)
                
            df.to_parquet(output_path)
            logger.info(f"SUCCESS: Extracted {len(df)} energy-specific headlines. Saved to {output_path}")
            return df
        else:
            logger.warning("No relevant energy news found in the specified timeframe.")
            return None

if __name__ == "__main__":
    collector = EnergyNewsCollector()
    collector.fetch_news_headlines(days_back=30) # Pull a month of history for testing