import os
import logging
import pandas as pd
import torch
from transformers import pipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Reads raw maritime news headlines and scores them using FinBERT.
    Aggregates sentiment into a daily numeric stress score.
    """
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.raw_dir = os.path.join(self.base_dir, "data", "raw")
        self.processed_dir = os.path.join(self.base_dir, "data", "processed")
        
        # We use ProsusAI's FinBERT, which is specifically trained on financial texts
        logger.info("Loading FinBERT model (this may take a moment the first time)...")
        # Check if MPS (Apple Silicon GPU) or standard CPU is available
        device = 0 if torch.backends.mps.is_available() else -1 
        self.analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=device)
        
    def _map_score(self, result: dict) -> float:
        """Converts FinBERT text labels into a numerical gradient [-1.0 to 1.0]."""
        label = result['label']
        score = result['score'] # Confidence of the model
        
        if label == "positive":
            return score
        elif label == "negative":
            return -score
        else:
            return 0.0 # Neutral

    def process_news_sentiment(self):
        """Loads raw news, scores it, and outputs a daily aggregated feature."""
        input_file = os.path.join(self.raw_dir, "news_data_raw.parquet")
        
        if not os.path.exists(input_file):
            logger.error(f"Input file missing: {input_file}. Run nlp_collector.py first.")
            return None
            
        logger.info("Processing news headlines...")
        df = pd.read_parquet(input_file)
        
        if df.empty:
            logger.warning("News dataframe is empty.")
            return df
            
        # Run headlines through the neural network
        # We only pass the title to keep it fast and highly concentrated
        headlines = df['title'].tolist()
        sentiment_results = self.analyzer(headlines)
        
        # Map results back to the dataframe
        df['sentiment_raw'] = [res for res in sentiment_results]
        df['sentiment_score'] = [self._map_score(res) for res in sentiment_results]
        
        # Strip timezone info to make it compatible with our financial data dates
        df['Date'] = pd.to_datetime(df['published_date']).dt.tz_localize(None).dt.normalize()
        
        # Aggregate by day (average sentiment of all articles that day)
        daily_sentiment = df.groupby('Date')['sentiment_score'].mean().reset_index()
        
        # Save to processed folder
        output_file = os.path.join(self.processed_dir, "daily_sentiment.parquet")
        daily_sentiment.to_parquet(output_file)
        
        logger.info(f"Successfully saved daily sentiment scores to {output_file}")
        return daily_sentiment

if __name__ == "__main__":
    processor = SentimentAnalyzer()
    df = processor.process_news_sentiment()
    
    if df is not None:
        print("\nDaily Sentiment Feature Snapshot:")
        print(df.head())