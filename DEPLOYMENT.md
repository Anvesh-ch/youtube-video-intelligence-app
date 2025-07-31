# YouTube Video Intelligence App - Deployment Guide

This guide covers how to deploy the YouTube Video Intelligence App both locally and on Streamlit Cloud.

## Local Development Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd YouTube_Video_Intelligence_App
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp env.example .env
   ```
   
   Edit `.env` file and add your YouTube API key:
   ```
   YOUTUBE_API_KEY=your_youtube_api_key_here
   ```

4. **Run the setup script**
   ```bash
   python setup.py
   ```
   
   This will:
   - Check dependencies
   - Create necessary directories
   - Train the ML models
   - Set up the environment

5. **Run the application**
   ```bash
   streamlit run app/app.py
   ```

6. **Access the app**
   Open your browser and go to `http://localhost:8501`

## YouTube API Setup

### Getting a YouTube API Key

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the YouTube Data API v3
4. Create credentials (API Key)
5. Copy the API key to your `.env` file

### API Quota Limits

- YouTube API has daily quota limits
- The app includes yt-dlp fallback for when API quota is exceeded
- Monitor your usage in Google Cloud Console

## Streamlit Cloud Deployment

### Prerequisites

- GitHub account
- Streamlit Cloud account

### Deployment Steps

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set the main file path to `app/app.py`
   - Add your YouTube API key as a secret

3. **Configure secrets**
   In Streamlit Cloud, add these secrets:
   ```
   YOUTUBE_API_KEY = your_youtube_api_key_here
   ```

4. **Deploy**
   - Click "Deploy"
   - Wait for the build to complete
   - Your app will be available at the provided URL

### Environment Variables for Streamlit Cloud

Add these to your Streamlit Cloud secrets:

```
YOUTUBE_API_KEY = your_youtube_api_key_here
VIRALITY_THRESHOLD = 100000
VIRALITY_TIME_WINDOW = 48
DEBUG = False
```

## Docker Deployment

### Build Docker Image

```bash
docker build -t youtube-video-intelligence .
```

### Run Docker Container

```bash
docker run -p 8501:8501 -e YOUTUBE_API_KEY=your_key youtube-video-intelligence
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - YOUTUBE_API_KEY=${YOUTUBE_API_KEY}
    volumes:
      - ./models:/app/models
```

Run with:
```bash
docker-compose up -d
```

## Model Training

### Training with Sample Data

The setup script automatically trains models with sample data. To retrain:

```bash
python scripts/train_models.py
```

### Training with Real Data

1. Download the YouTube Trending Video Dataset from Kaggle
2. Place CSV files in the `data/` directory
3. Modify `scripts/train_models.py` to use real data
4. Run training script

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python path: `export PYTHONPATH=$PYTHONPATH:/path/to/project`

2. **Model Loading Errors**
   - Run the setup script: `python setup.py`
   - Check if model files exist in `models/` directory

3. **YouTube API Errors**
   - Verify API key is correct
   - Check API quota limits
   - The app will fallback to yt-dlp if API fails

4. **Memory Issues**
   - Reduce model complexity in training scripts
   - Use smaller batch sizes
   - Consider using CPU-only models

### Performance Optimization

1. **Model Caching**
   - Models are cached after first load
   - Consider using model quantization for faster inference

2. **API Rate Limiting**
   - Implement request caching
   - Use batch processing for multiple videos

3. **Memory Management**
   - Clear model cache periodically
   - Use garbage collection for large models

## Monitoring and Logging

### Application Logs

Logs are stored in `logs/` directory. Monitor for:
- API errors
- Model loading issues
- Performance metrics

### Streamlit Cloud Monitoring

- Monitor app performance in Streamlit Cloud dashboard
- Check for deployment errors
- Monitor API usage

## Security Considerations

1. **API Key Security**
   - Never commit API keys to version control
   - Use environment variables or secrets
   - Rotate API keys regularly

2. **Input Validation**
   - Validate YouTube URLs
   - Sanitize user inputs
   - Implement rate limiting

3. **Model Security**
   - Validate model inputs
   - Implement input size limits
   - Monitor for adversarial inputs

## Scaling Considerations

1. **Horizontal Scaling**
   - Deploy multiple instances behind a load balancer
   - Use Redis for session management
   - Implement proper caching

2. **Model Optimization**
   - Use model quantization
   - Implement model versioning
   - Consider edge deployment

3. **Database Integration**
   - Store prediction results
   - Implement user management
   - Add analytics dashboard

## Support and Maintenance

### Regular Maintenance

1. **Update Dependencies**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

2. **Retrain Models**
   - Retrain models with new data periodically
   - Monitor model performance
   - Update model versions

3. **Backup and Recovery**
   - Backup model files
   - Version control configuration
   - Document deployment procedures

### Getting Help

- Check the README.md for basic usage
- Review error logs in `logs/` directory
- Open issues on GitHub for bugs
- Contact maintainers for support 