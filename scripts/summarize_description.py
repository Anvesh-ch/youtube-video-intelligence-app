import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from utils.text_cleaning import clean_text
from utils.constants import MODEL_PATHS

class DescriptionSummarizer:
    """
    Description summarization model for YouTube videos.
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        
    def load_model(self, model_path=None):
        """
        Load the T5 summarization model.
        
        Args:
            model_path (str): Path to the model
        """
        try:
            if model_path is None:
                model_path = MODEL_PATHS['summary']
            
            # Load model and tokenizer
            self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
            self.model = T5ForConditionalGeneration.from_pretrained('t5-small')
            
            self.is_loaded = True
            print("T5 summarization model loaded successfully")
            
        except Exception as e:
            print(f"Error loading T5 model: {e}")
            print("Using fallback summarization method")
            self.is_loaded = False
    
    def summarize(self, description, max_length=150):
        """
        Summarize video description.
        
        Args:
            description (str): Video description
            max_length (int): Maximum length of summary
            
        Returns:
            str: Summarized description
        """
        if not description or len(description.strip()) == 0:
            return "No description available"
        
        # Clean the description
        cleaned_description = clean_text(description, remove_stopwords=False, max_length=500)
        
        if len(cleaned_description) < 50:
            return description[:max_length] + "..." if len(description) > max_length else description
        
        if self.is_loaded:
            return self._summarize_with_t5(cleaned_description, max_length)
        else:
            return self._summarize_fallback(cleaned_description, max_length)
    
    def _summarize_with_t5(self, text, max_length):
        """
        Summarize using T5 model.
        
        Args:
            text (str): Input text
            max_length (int): Maximum length of summary
            
        Returns:
            str: Summarized text
        """
        try:
            # Prepare input
            input_text = f"summarize: {text}"
            
            # Tokenize
            inputs = self.tokenizer.encode(
                input_text,
                max_length=512,
                truncation=True,
                return_tensors="pt"
            )
            
            # Generate summary
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs,
                    max_length=max_length,
                    min_length=30,
                    length_penalty=2.0,
                    num_beams=4,
                    early_stopping=True
                )
            
            # Decode
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            return summary.strip()
            
        except Exception as e:
            print(f"T5 summarization failed: {e}")
            return self._summarize_fallback(text, max_length)
    
    def _summarize_fallback(self, text, max_length):
        """
        Fallback summarization using extractive method.
        
        Args:
            text (str): Input text
            max_length (int): Maximum length of summary
            
        Returns:
            str: Summarized text
        """
        # Simple extractive summarization
        sentences = text.split('.')
        
        # Filter out very short sentences
        valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if not valid_sentences:
            return text[:max_length] + "..." if len(text) > max_length else text
        
        # Take first few sentences that fit within max_length
        summary = ""
        for sentence in valid_sentences[:3]:  # Take up to 3 sentences
            if len(summary + sentence) < max_length:
                summary += sentence + ". "
            else:
                break
        
        summary = summary.strip()
        
        if not summary:
            return text[:max_length] + "..." if len(text) > max_length else text
        
        return summary

def summarize_description(description, max_length=150):
    """
    Summarize video description.
    
    Args:
        description (str): Video description
        max_length (int): Maximum length of summary
        
    Returns:
        str: Summarized description
    """
    summarizer = DescriptionSummarizer()
    summarizer.load_model()
    
    return summarizer.summarize(description, max_length) 