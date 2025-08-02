# YouTube Video Intelligence App

A comprehensive machine learning application that analyzes YouTube videos to provide key insights for creators and analysts.

## Features

- **Virality Prediction**: Binary classification model predicting if a video will go viral within 48 hours
- **Title Quality Scoring**: Regression model scoring video titles from 0-100
- **Engagement Forecasting**: Predicts expected views, likes, and comments after 48 hours
- **Tag Recommendation**: Suggests 5 relevant tags based on title and category
- **Description Summarization**: Summarizes long video descriptions into concise sentences

## Tech Stack

- **Frontend**: Streamlit
- **ML Models**: Scikit-learn, XGBoost, LightGBM
- **NLP**: Hugging Face Transformers (T5-small)
- **Embeddings**: Sentence-transformers
- **YouTube Data**: YouTube API v3 with yt-dlp fallback
- **Deployment**: Streamlit Cloud

## Installation

### Option 1: Automated Setup (Recommended)

1. Clone the repository:
```bash
git clone <repository-url>
cd YouTube_Video_Intelligence_App
```

2. Run the automated setup script:
```bash
python setup_venv.py
```

This will:
- Create a virtual environment
- Install all dependencies
- Create activation scripts
- Set up everything automatically

3. Activate the environment and run:
```bash
# Unix/Linux/macOS
source activate_env.sh
# or
./run_app_venv.sh

# Windows
activate_env.bat
# or
run_app_venv.bat
```

### Option 2: Manual Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd YouTube_Video_Intelligence_App
```

2. Create virtual environment:
```bash
python -m venv venv
```

3. Activate virtual environment:
```bash
# Unix/Linux/macOS
source venv/bin/activate

# Windows
venv\Scripts\activate.bat
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Set up environment variables:
```bash
cp env.example .env
# Add your YouTube API key to .env
```

6. Run the application:
```bash
streamlit run app/app.py
```

## Project Structure

```
project_root/
├── app/
│   └── app.py                 # Main Streamlit application
├── scripts/
│   ├── fetch_metadata.py      # YouTube metadata extraction
│   ├── preprocess.py          # Data preprocessing
│   ├── predict_virality.py    # Virality prediction
│   ├── predict_title_score.py # Title quality scoring
│   ├── forecast_engagement.py # Engagement forecasting
│   ├── recommend_tags.py      # Tag recommendations
│   └── summarize_description.py # Description summarization
├── models/                    # Pre-trained model files
├── data/                      # Dataset storage
├── utils/                     # Utility functions
├── requirements.txt
├── Dockerfile
└── README.md
```

## Usage

1. Enter a YouTube video URL in the input field
2. Click "Analyze Video" to process the video
3. View comprehensive insights including:
   - Virality prediction probability
   - Title quality score
   - Engagement forecasts
   - Recommended tags
   - Summarized description

## Model Training

Models are trained using the real YouTube Trending Video Dataset from the archive folder. The dataset contains over 400,000 videos from 10 countries (US, GB, CA, DE, FR, IN, JP, KR, MX, RU) with real engagement metrics, titles, descriptions, and tags.

## Deployment

The app is designed for deployment on Streamlit Cloud with automatic rebuilds on each commit.

## License

MIT License 