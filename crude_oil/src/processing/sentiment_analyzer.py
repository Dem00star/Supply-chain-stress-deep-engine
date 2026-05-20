import os
import logging
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnergySentimentAnalyzer:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.raw_data_path = os.path.join(self.base_dir, 'data', 'raw', 'news_data_raw.parquet')
        self.processed_dir = os.path.join(self.base_dir, 'data', 'processed')
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Load FinBERT (The industry standard for financial/macro NLP)
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

    def analyze_sentiment(self):
        """Calculates a daily geopolitical sentiment score."""
        logger.info("Loading raw geopolitical headlines...")
        if not os.path.exists(self.raw_data_path):
            logger.error("No raw news data found. Run nlp_collector.py first.")
            return

        df = pd.read_parquet(self.raw_data_path)
        
        logger.info("Pushing headlines through FinBERT...")
        sentiment_scores = []
        
        for headline in df['Headline']:
            inputs = self.tokenizer(headline, return_tensors="pt", padding=True, truncation=True)
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # FinBERT outputs: [Positive, Negative, Neutral]
            pos_score = predictions[0][0].item()
            neg_score = predictions[0][1].item()
            
            # Calculate net sentiment (-1.0 to 1.0)
            net_score = pos_score - neg_score
            sentiment_scores.append(net_score)
            
        df['sentiment_score'] = sentiment_scores
        
        # Aggregate the scores by Day (we want a daily macro signal)
        daily_sentiment = df.groupby('Date')['sentiment_score'].mean().reset_index()
        daily_sentiment['Date'] = pd.to_datetime(daily_sentiment['Date'])
        
        output_path = os.path.join(self.processed_dir, 'daily_sentiment.parquet')
        daily_sentiment.to_parquet(output_path)
        logger.info(f"SUCCESS: Saved daily geopolitical sentiment scores to {output_path}")

if __name__ == "__main__":
    analyzer = EnergySentimentAnalyzer()
    analyzer.analyze_sentiment()