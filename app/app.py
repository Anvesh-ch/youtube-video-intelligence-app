import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.fetch_metadata import fetch_video_metadata, validate_youtube_url, get_video_summary
from scripts.predict_virality import predict_virality
from scripts.predict_title_score import predict_title_score
from scripts.forecast_engagement import forecast_engagement
from scripts.recommend_tags import recommend_tags
from scripts.summarize_description import summarize_description

# Page configuration
st.set_page_config(
    page_title="YouTube Video Intelligence App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .success-metric {
        color: #28a745;
        font-weight: bold;
    }
    .warning-metric {
        color: #ffc107;
        font-weight: bold;
    }
    .danger-metric {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">YouTube Video Intelligence App</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Input section
    st.header("Video Analysis")
    
    # URL input
    url = st.text_input(
        "Enter YouTube Video URL",
        placeholder="https://www.youtube.com/watch?v=...",
        help="Paste a valid YouTube video URL to analyze"
    )
    
    # Analysis button
    analyze_button = st.button("Analyze Video", type="primary")
    
    if analyze_button and url:
        if not validate_youtube_url(url):
            st.error("Please enter a valid YouTube URL")
            return
        
        # Show loading spinner
        with st.spinner("Fetching video metadata and running analysis..."):
            try:
                # Fetch metadata
                metadata = fetch_video_metadata(url)
                
                if not metadata:
                    st.error("Failed to fetch video metadata. Please check the URL and try again.")
                    return
                
                # Display video summary
                display_video_summary(metadata)
                
                # Run predictions
                run_predictions(metadata)
                
            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
    
    elif analyze_button and not url:
        st.warning("Please enter a YouTube URL to analyze")

def display_video_summary(metadata):
    """Display video metadata summary."""
    st.subheader("Video Information")
    
    # Create columns for video info
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Title:**")
        st.write(metadata.get('title', 'Unknown'))
        
        st.markdown("**Channel:**")
        st.write(metadata.get('channel_title', 'Unknown'))
        
        st.markdown("**Category:**")
        st.write(metadata.get('category', 'Unknown'))
    
    with col2:
        st.markdown("**Views:**")
        st.write(f"{metadata.get('views', 0):,}")
        
        st.markdown("**Likes:**")
        st.write(f"{metadata.get('likes', 0):,}")
        
        st.markdown("**Comments:**")
        st.write(f"{metadata.get('comments', 0):,}")

def run_predictions(metadata):
    """Run all prediction models and display results."""
    st.subheader("Analysis Results")
    
    # Create tabs for different predictions
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Virality Prediction", 
        "Title Quality", 
        "Engagement Forecast", 
        "Tag Recommendations", 
        "Description Summary"
    ])
    
    with tab1:
        display_virality_prediction(metadata)
    
    with tab2:
        display_title_quality(metadata)
    
    with tab3:
        display_engagement_forecast(metadata)
    
    with tab4:
        display_tag_recommendations(metadata)
    
    with tab5:
        display_description_summary(metadata)

def display_virality_prediction(metadata):
    """Display virality prediction results."""
    try:
        prediction = predict_virality(metadata)
        
        # Create metric cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                "Viral Probability",
                f"{prediction['viral_probability']:.1%}",
                help="Probability that the video will go viral within 48 hours"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            prediction_text = "Likely Viral" if prediction['prediction'] else "Not Likely Viral"
            prediction_color = "success-metric" if prediction['prediction'] else "danger-metric"
            st.markdown(f'<p class="{prediction_color}">{prediction_text}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                "Confidence",
                f"{prediction['confidence']:.1%}",
                help="Model confidence in the prediction"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Additional info
        st.info(f"Viral threshold: {prediction['threshold']:,} views within 48 hours")
        
    except Exception as e:
        st.error(f"Error in virality prediction: {str(e)}")

def display_title_quality(metadata):
    """Display title quality scoring results."""
    try:
        score_result = predict_title_score(metadata)
        
        # Create metric cards
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            score = score_result['score']
            if score >= 80:
                color_class = "success-metric"
            elif score >= 60:
                color_class = "warning-metric"
            else:
                color_class = "danger-metric"
            
            st.markdown(f'<p class="{color_class}">Title Score: {score}/100</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                "Confidence",
                f"{score_result['confidence']:.1%}",
                help="Model confidence in the score"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Feedback
        st.markdown("**Feedback:**")
        st.write(score_result['feedback'])
        
    except Exception as e:
        st.error(f"Error in title quality scoring: {str(e)}")

def display_engagement_forecast(metadata):
    """Display engagement forecasting results."""
    try:
        forecast = forecast_engagement(metadata)
        
        # Create metric cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                "Predicted Views",
                f"{forecast['predicted_views']:,}",
                help="Expected views after 48 hours"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                "Predicted Likes",
                f"{forecast['predicted_likes']:,}",
                help="Expected likes after 48 hours"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                "Predicted Comments",
                f"{forecast['predicted_comments']:,}",
                help="Expected comments after 48 hours"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Engagement rate
        st.markdown("**Engagement Rate:**")
        st.write(f"{forecast['engagement_rate']:.3f} (likes + comments / views)")
        
    except Exception as e:
        st.error(f"Error in engagement forecasting: {str(e)}")

def display_tag_recommendations(metadata):
    """Display tag recommendation results."""
    try:
        recommendations = recommend_tags(metadata, top_k=5)
        
        if recommendations:
            st.markdown("**Recommended Tags:**")
            
            for i, rec in enumerate(recommendations, 1):
                similarity = rec['similarity']
                if similarity > 0.7:
                    color_class = "success-metric"
                elif similarity > 0.5:
                    color_class = "warning-metric"
                else:
                    color_class = "danger-metric"
                
                st.markdown(f'{i}. <span class="{color_class}">{rec["tag"]}</span> (Similarity: {similarity:.2f})', unsafe_allow_html=True)
        else:
            st.warning("No tag recommendations available")
        
    except Exception as e:
        st.error(f"Error in tag recommendations: {str(e)}")

def display_description_summary(metadata):
    """Display description summarization results."""
    try:
        description = metadata.get('description', '')
        
        if description:
            summary = summarize_description(description)
            
            st.markdown("**Original Description:**")
            st.text_area("", value=description, height=150, disabled=True)
            
            st.markdown("**Summarized Description:**")
            st.text_area("", value=summary, height=100, disabled=True)
        else:
            st.warning("No description available for summarization")
        
    except Exception as e:
        st.error(f"Error in description summarization: {str(e)}")

if __name__ == "__main__":
    main() 