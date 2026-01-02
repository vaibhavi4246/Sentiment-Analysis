# Real-Time Social Media Sentiment Analyzer using DistilBERT

## Project Overview

This project implements a real-time sentiment analysis system using DistilBERT transformer model, trained on Twitter sentiment data. The system can classify social media text into three categories: Positive, Negative, and Neutral.

**Exam Project Alignment:**
- Module 1: Exploration & Analysis of Real-Time Data
- Module 2: Preprocessing
- Module 3: Model Building (Deep Learning)
- Module 4: Model Evaluation
- Module 5: Deployment & Real-Time Implementation
- Module 6: AI Exploration Experiment

---

## Features

- **Deep Learning Model**: DistilBERT-based transformer architecture
- **Large-Scale Training**: Trained on 4.7M+ tweets
- **Real-Time Inference**: <50ms prediction time
- **Web Interface**: Interactive Streamlit application
- **Comprehensive Analysis**: Complete EDA, training, and evaluation pipeline

---

## Project Structure

```
Sentiment-Analysis/
├── sentiment_analysis.ipynb           # Main notebook with all modules
├── sentiment_analysis-checkpoint.ipynb # Notebook checkpoint
├── app.py                             # Streamlit deployment application
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore configuration
├── README.md                          # This file
└── outputs/                           # Results and artifacts
    ├── classification_report.txt      # Model evaluation report
    ├── test_results.json              # Test metrics in JSON
    ├── training_metrics.json          # Training history
    ├── models/                        # Trained model storage
    │   └── distilbert-sentiment/      # Fine-tuned model
    │       ├── config.json
    │       ├── model.safetensors
    │       ├── tokenizer files
    │       └── checkpoints/           # Training checkpoints
    └── visualizations/                # Charts and plots
```

---

## Requirements

### Hardware
- **GPU**: NVIDIA H100 (or any CUDA-compatible GPU)
- **RAM**: 16GB minimum
- **Storage**: 10GB free space

### Software
- Python 3.8+
- CUDA 11.0+ (for GPU acceleration)

---

## Installation

### 1. Clone or Download Project

```bash
git clone <repository-url>
cd Sentiment-Analysis
```

Or if already downloaded:

```bash
cd "c:\Users\ASUS\Sentiment-Analysis"
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- `torch>=2.1.0` - Deep learning framework
- `transformers>=4.36.0` - Hugging Face transformers
- `datasets>=2.14.0` - Dataset loading and processing
- `streamlit>=1.29.0` - Web application framework
- `pandas`, `numpy`, `scikit-learn` - Data manipulation and ML utilities
- `matplotlib`, `seaborn`, `plotly`, `wordcloud` - Visualization
- `accelerate`, `evaluate` - Training acceleration and metrics
- `safetensors` - Model serialization

---

## Usage

### Step 1: Train the Model

Open and run the Jupyter notebook:

```bash
jupyter notebook sentiment_analysis.ipynb
```

**Execute all cells** in sequence. The notebook will:

1. Load the `bdstar/twitter-sentiment-analysis` dataset from Hugging Face
2. Perform exploratory data analysis (EDA)
3. Preprocess and tokenize text
4. Fine-tune DistilBERT model
5. Evaluate performance
6. Save trained model to `models/distilbert_sentiment/`

**Training Time:** 2-4 hours on H100 GPU (full dataset)

### Step 2: Run Streamlit App

After training is complete, launch the web application:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## Model Details

### Architecture

```
Input Text
    ↓
Text Preprocessing (cleaning, lowercasing)
    ↓
DistilBERT Tokenizer
    ↓
DistilBERT Encoder (6 transformer layers)
    ↓
Classification Head
    ↓
Output (Negative, Neutral, Positive)
```

### Training Configuration

- **Model**: `distilbert-base-uncased`
- **Dataset**: ~4.7M tweets (train/val/test splits)
- **Batch Size**: 128 (H100 optimized)
- **Learning Rate**: 2e-5
- **Epochs**: 3
- **Max Sequence Length**: 128 tokens
- **Optimizer**: AdamW
- **Mixed Precision**: FP16 enabled

### Why DistilBERT?

- 40% smaller than BERT
- 60% faster inference
- Retains 97% of BERT's performance
- Ideal for real-time applications

---

## Performance Metrics

After training, the model achieves:

- **Accuracy**: 91.4%
- **Precision**: 91.5% (macro avg)
- **Recall**: 91.4% (macro avg)
- **F1-Score**: 91.4% (macro avg)
- **Inference Time**: <50ms per prediction
- **Training Speed**: ~1412 samples/second

**Class-wise Performance:**
- **Negative (Class 0)**: Precision: 92.9%, Recall: 89.3%, F1: 91.1%
- **Positive (Class 1)**: Precision: 90.0%, Recall: 93.5%, F1: 91.7%

**Evaluation outputs:**
- Classification report: [outputs/classification_report.txt](outputs/classification_report.txt)
- Test results: [outputs/test_results.json](outputs/test_results.json)
- Training metrics: [outputs/training_metrics.json](outputs/training_metrics.json)
- Saved model: [outputs/models/distilbert-sentiment/](outputs/models/distilbert-sentiment/)

---

## Module Breakdown

### Module 1: Exploration & Analysis
- Dataset loading and inspection
- Label distribution analysis
- Word frequency and word clouds
- Sentence length distribution

### Module 2: Preprocessing
- Text cleaning (lowercasing, URL/mention removal)
- DistilBERT tokenization
- Before/after preprocessing examples

### Module 3: Model Building
- Model architecture explanation
- Training configuration
- Fine-tuning loop
- Model checkpointing

### Module 4: Model Evaluation
- Test set evaluation
- Confusion matrix
- Classification report
- Misclassification analysis

### Module 5: Real-Time Implementation
- Model loading for inference
- Prediction function with timing
- Real-time testing examples

### Module 6: AI Exploration
- Sarcasm detection testing
- Emoji and special character handling
- Slang and informal language
- Typo tolerance
- Mixed emotion analysis

---

## Streamlit App Features

### User Interface
- Clean, modern design
- Text input area
- Quick example buttons
- Real-time prediction display

### Results Display
- Sentiment label with color coding
- Confidence percentage
- Detailed scores for all classes
- Inference time metrics

### Sidebar Information
- Model details
- Usage instructions
- Performance stats

---

## Example Predictions

```python
Input: "I absolutely love this product! It's amazing!"
Output: POSITIVE (98.5% confidence)
Time: 24ms

Input: "This is the worst experience ever."
Output: NEGATIVE (97.2% confidence)
Time: 22ms

Input: "It's okay, nothing special."
Output: NEUTRAL (89.3% confidence)
Time: 23ms
```

---

## AI Exploration Findings

1. **Sarcasm Challenge**: Model struggles with sarcastic text
2. **Emoji Handling**: Removed during preprocessing
3. **Slang Robustness**: Good understanding of modern slang
4. **Typo Tolerance**: Resilient to minor spelling errors
5. **Mixed Emotions**: Classifies based on dominant sentiment

---

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution**: Reduce batch size in config:
```python
CONFIG['batch_size'] = 64  # or 32
```

### Issue: Model Not Found

**Solution**: Ensure training completed and model saved to:
```
models/distilbert_sentiment/
```

### Issue: Slow Inference

**Solution**: Check GPU availability:
```python
import torch
print(torch.cuda.is_available())  # Should be True
```

---

## Future Improvements

1. Add sarcasm detection module
2. Implement emoji sentiment analysis
3. Support for multilingual text
4. Real-time streaming from social media APIs
5. Batch prediction endpoint

---

## Submission Checklist

For exam/project submission, include:

- [ ] `sentiment_analysis.ipynb` (with all cells run)
- [ ] `app.py` (Streamlit application)
- [ ] `requirements.txt`
- [ ] Screenshots of:
  - EDA visualizations
  - Training progress
  - Confusion matrix
  - Streamlit app interface
  - Real-time predictions
- [ ] Model performance metrics
- [ ] AI exploration observations

---

## Technical Stack

**Deep Learning:**
- PyTorch 2.0+
- Hugging Face Transformers
- CUDA for GPU acceleration

**Data Processing:**
- Pandas, NumPy
- Scikit-learn
- HuggingFace Datasets

**Visualization:**
- Matplotlib, Seaborn
- Plotly
- WordCloud

**Deployment:**
- Streamlit
- Python 3.8+

---

## License

This is an academic project for educational purposes.

---

## Contact & Support

For questions or issues:
1. Check troubleshooting section above
2. Review notebook comments and markdown cells
3. Verify all dependencies are installed correctly

---

## Acknowledgments

- **Dataset**: bdstar/twitter-sentiment-analysis (Hugging Face)
- **Model**: DistilBERT (Hugging Face)
- **Framework**: PyTorch & Transformers library

---

**Project Status**: Ready for training and deployment
**Last Updated**: January 2026
