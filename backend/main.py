from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from io import StringIO
import logging

# Initialize FastAPI
app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React development server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Function for enhanced sentiment analysis with logging
def analyze_sentiment(text: str):
    score = analyzer.polarity_scores(text)
    compound_score = score['compound']
    
    # Log the text and compound score
    logging.debug(f"Text: {text}")
    logging.debug(f"Compound Score: {compound_score}")
    
    # Classify sentiment based on compound score
    if compound_score >= 0.80:
        return "very positive"
    elif 0.60 <= compound_score < 0.80:
        return "positive"
    elif 0.10 <= compound_score < 0.60:
        return "mildly positive"
    elif -0.10 <= compound_score < 0.10:
        return "neutral"
    elif -0.60 <= compound_score < -0.10:
        return "negative"
    else:
        return "very negative"

@app.post("/sentiment/csv")
async def analyze_csv(file: UploadFile = File(...)):
    # Read the file content
    content = await file.read()
    df = pd.read_csv(StringIO(content.decode('utf-8')))
    
    # Check if required columns exist
    if 'id' not in df.columns or 'text' not in df.columns or 'timestamp' not in df.columns:
        return {"error": "CSV file must contain 'id', 'text', and 'timestamp' columns"}

    # Apply sentiment analysis to each row in the 'text' column
    df['sentiment'] = df['text'].apply(analyze_sentiment)
    
    # Return the analyzed data as a dictionary
    return df.to_dict(orient="records")
