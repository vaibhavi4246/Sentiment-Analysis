import streamlit as st
import re
import os

# Try importing with helpful error messages
try:
    import torch
except ImportError as e:
    st.error("⚠️ PyTorch not found. Installing dependencies...")
    st.stop()

try:
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
except ImportError as e:
    st.error("⚠️ Transformers library not found. Please check requirements.txt")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .sentiment-positive {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .sentiment-negative {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .confidence-score {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and tokenizer with caching
@st.cache_resource
def load_model():
    """Load the trained model and tokenizer"""
    import os
    
    # Try to load fine-tuned model first
    model_path = 'outputs/models/distilbert-sentiment'
    
    try:
        if os.path.exists(model_path):
            # Load fine-tuned model from outputs directory
            tokenizer = DistilBertTokenizer.from_pretrained(model_path)
            model = DistilBertForSequenceClassification.from_pretrained(model_path)
            st.sidebar.success("✅ Fine-tuned model loaded")
        else:
            raise FileNotFoundError("Fine-tuned model not found")
    except Exception as e:
        # Fallback to base model
        st.sidebar.warning(f"⚠️ Using base model (fine-tuned model not found)")
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    
    model.eval()
    return tokenizer, model

def clean_text(text):
    """Clean and preprocess text"""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_sentiment(text, tokenizer, model):
    """Predict sentiment of the given text"""
    cleaned_text = clean_text(text)
    
    inputs = tokenizer(
        cleaned_text,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicted_class].item()
    
    sentiment = "Positive" if predicted_class == 1 else "Negative"
    return sentiment, confidence, probabilities[0].tolist()

def main():
    st.title(" Sentiment Analysis with DistilBERT")
    st.markdown("""
    This app analyzes the sentiment of text using a fine-tuned DistilBERT model.
    Enter any text below to get instant sentiment predictions!
    """)
    
    with st.spinner("Loading model..."):
        tokenizer, model = load_model()
    
    st.sidebar.header("About")
    st.sidebar.info("""
    **Model**: DistilBERT Base Uncased
    
    **Sentiment Classes**:
    -  Positive
    -  Negative
    
    **Dataset**: SST-2 (Stanford Sentiment Treebank)
    """)
    
    st.sidebar.header("Examples")
    example_texts = [
        "This movie was absolutely fantastic! I loved every minute of it.",
        "The service was terrible and the food was cold.",
        "Best experience ever! Highly recommended!",
    ]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter Your Text")
        input_text = st.text_area(
            "Type or paste your text here:",
            height=150,
            placeholder="Enter the text you want to analyze..."
        )
        
        analyze_button = st.button(" Analyze Sentiment", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("Tips")
        st.markdown("""
        - Enter clear, complete sentences
        - Works best with English text
        - Can analyze reviews, comments, feedback
        - Longer texts get truncated to 128 tokens
        """)
    
    if analyze_button and input_text:
        with st.spinner("Analyzing..."):
            sentiment, confidence, probabilities = predict_sentiment(input_text, tokenizer, model)
            
            st.markdown("---")
            st.subheader(" Analysis Results")
            
            if sentiment == "Positive":
                st.markdown(f"""
                <div class="sentiment-positive">
                    <h2> Sentiment: {sentiment}</h2>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="sentiment-negative">
                    <h2> Sentiment: {sentiment}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="confidence-score">
                Confidence: {confidence * 100:.2f}%
            </div>
            """, unsafe_allow_html=True)
            
            st.subheader("Probability Distribution")
            col_neg, col_pos = st.columns(2)
            
            with col_neg:
                st.metric("Negative", f"{probabilities[0] * 100:.2f}%")
                st.progress(probabilities[0])
            
            with col_pos:
                st.metric("Positive", f"{probabilities[1] * 100:.2f}%")
                st.progress(probabilities[1])
            
            with st.expander("View Preprocessed Text"):
                st.text(clean_text(input_text))
    
    elif analyze_button and not input_text:
        st.warning(" Please enter some text to analyze!")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with Streamlit and DistilBERT | Powered by Hugging Face Transformers</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
