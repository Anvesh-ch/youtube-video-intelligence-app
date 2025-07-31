import re
import json
import requests
from urllib.parse import urlparse, parse_qs
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import yt_dlp
from utils.constants import YOUTUBE_API_KEY, YOUTUBE_CATEGORIES, DEFAULT_VALUES

def extract_video_id(url):
    """
    Extract video ID from YouTube URL.
    
    Args:
        url (str): YouTube URL
        
    Returns:
        str: Video ID or None if not found
    """
    # Handle different YouTube URL formats
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
        r'youtube\.com\/watch\?.*v=([^&\n?#]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None

def fetch_metadata_api(video_id):
    """
    Fetch video metadata using YouTube API.
    
    Args:
        video_id (str): YouTube video ID
        
    Returns:
        dict: Video metadata or None if failed
    """
    if not YOUTUBE_API_KEY:
        return None
    
    try:
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        
        # Get video details
        request = youtube.videos().list(
            part='snippet,statistics,contentDetails',
            id=video_id
        )
        response = request.execute()
        
        if not response['items']:
            return None
        
        video = response['items'][0]
        snippet = video['snippet']
        statistics = video.get('statistics', {})
        
        # Extract metadata
        metadata = {
            'video_id': video_id,
            'title': snippet.get('title', ''),
            'description': snippet.get('description', ''),
            'channel_title': snippet.get('channelTitle', ''),
            'publish_time': snippet.get('publishedAt', ''),
            'category_id': snippet.get('categoryId', ''),
            'tags': snippet.get('tags', []),
            'views': int(statistics.get('viewCount', 0)),
            'likes': int(statistics.get('likeCount', 0)),
            'comments': int(statistics.get('commentCount', 0)),
            'duration': video.get('contentDetails', {}).get('duration', '')
        }
        
        # Map category ID to name
        category_id = snippet.get('categoryId')
        if category_id and category_id.isdigit():
            metadata['category'] = YOUTUBE_CATEGORIES.get(int(category_id), 'Other')
        else:
            metadata['category'] = 'Other'
        
        return metadata
        
    except HttpError as e:
        print(f"YouTube API error: {e}")
        return None
    except Exception as e:
        print(f"Error fetching metadata via API: {e}")
        return None

def fetch_metadata_ytdlp(video_id):
    """
    Fetch video metadata using yt-dlp as fallback.
    
    Args:
        video_id (str): YouTube video ID
        
    Returns:
        dict: Video metadata or None if failed
    """
    try:
        url = f"https://www.youtube.com/watch?v={video_id}"
        
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            
            # Extract metadata
            metadata = {
                'video_id': video_id,
                'title': info.get('title', ''),
                'description': info.get('description', ''),
                'channel_title': info.get('uploader', ''),
                'publish_time': info.get('upload_date', ''),
                'category': info.get('categories', ['Other'])[0] if info.get('categories') else 'Other',
                'tags': info.get('tags', []),
                'views': info.get('view_count', 0),
                'likes': info.get('like_count', 0),
                'comments': info.get('comment_count', 0),
                'duration': info.get('duration', 0)
            }
            
            return metadata
            
    except Exception as e:
        print(f"Error fetching metadata via yt-dlp: {e}")
        return None

def fetch_video_metadata(url):
    """
    Fetch video metadata using YouTube API with yt-dlp fallback.
    
    Args:
        url (str): YouTube video URL
        
    Returns:
        dict: Video metadata or None if failed
    """
    # Extract video ID
    video_id = extract_video_id(url)
    if not video_id:
        print("Could not extract video ID from URL")
        return None
    
    # Try YouTube API first
    metadata = fetch_metadata_api(video_id)
    
    # Fallback to yt-dlp if API fails
    if not metadata:
        print("YouTube API failed, trying yt-dlp fallback...")
        metadata = fetch_metadata_ytdlp(video_id)
    
    if metadata:
        # Ensure all required fields exist
        for key, default_value in DEFAULT_VALUES.items():
            if key not in metadata:
                metadata[key] = default_value
        
        return metadata
    else:
        print("Failed to fetch metadata from both API and yt-dlp")
        return None

def validate_youtube_url(url):
    """
    Validate if the URL is a valid YouTube URL.
    
    Args:
        url (str): URL to validate
        
    Returns:
        bool: True if valid YouTube URL
    """
    if not url:
        return False
    
    # Check if it's a YouTube URL
    youtube_patterns = [
        r'youtube\.com',
        r'youtu\.be',
        r'youtube\.com\/embed'
    ]
    
    for pattern in youtube_patterns:
        if re.search(pattern, url):
            return True
    
    return False

def get_video_summary(metadata):
    """
    Create a summary of video metadata.
    
    Args:
        metadata (dict): Video metadata
        
    Returns:
        dict: Video summary
    """
    if not metadata:
        return None
    
    summary = {
        'title': metadata.get('title', 'Unknown'),
        'channel': metadata.get('channel_title', 'Unknown'),
        'category': metadata.get('category', 'Other'),
        'views': metadata.get('views', 0),
        'likes': metadata.get('likes', 0),
        'comments': metadata.get('comments', 0),
        'tags_count': len(metadata.get('tags', [])),
        'description_length': len(metadata.get('description', '')),
        'publish_date': metadata.get('publish_time', 'Unknown')
    }
    
    return summary 