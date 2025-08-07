import torch
import numpy as np
import matplotlib.pyplot as plt
import re
import streamlit as st
import io
from PIL import Image
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from lime.lime_text import LimeTextExplainer
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor
import threading

# Page configuration
st.set_page_config(
    page_title="Toxicity Detection System",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-family: 'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 30%, #764ba2 80%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: opaque;
        background-clip: text;
        text-align: center;
        letter-spacing: -0.02em;
        margin-bottom: 2rem;
        text-shadow: none;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .toxic-severe {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .toxic-moderate {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
    }
    .non-toxic {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Model Paths - Update these according to your setup
MODEL_PATHS = {
    "electra": "models/best_electra_model (1)",
    "deberta": "models/best_deberta",
    "roberta": "models/updated_twitter_roberta",
    "hatebert": "models/best_hatebert",
}

# # Alternative demo models (uncomment to use these instead)
# DEMO_MODEL_PATHS = {
#     "unitary-toxic-bert": "unitary/toxic-bert",
#     "martin-toxic-comment": "martin-ha/toxic-comment-model",
# }

# Label Mapping
LABEL_MAPPING = {0: "Severely Toxic", 1: "Moderately Toxic", 2: "Non-Toxic"}
LABEL_COLORS = {
    "Severely Toxic": "#f44336",
    "Moderately Toxic": "#ff9800", 
    "Non-Toxic": "#4caf50"
}

@st.cache_resource
def load_models():
    """Load all models and tokenizers - cached for performance"""
    models = {}
    tokenizers = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Progress bar for loading
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Use demo models if main models not available
    paths_to_use = MODEL_PATHS
    
    total_models = len(paths_to_use)
    loaded_count = 0
    
    for i, (name, path) in enumerate(paths_to_use.items()):
        status_text.text(f"Loading {name} model...")
        try:
            tokenizers[name] = AutoTokenizer.from_pretrained(path)
            models[name] = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()
            loaded_count += 1
            st.success(f"✅ Loaded {name} model")
        except Exception as e:
            st.error(f"❌ Failed to load {name}: {str(e)}")
            # Try demo model as fallback
            if name in DEMO_MODEL_PATHS:
                try:
                    demo_path = DEMO_MODEL_PATHS[name]
                    tokenizers[name] = AutoTokenizer.from_pretrained(demo_path)
                    models[name] = AutoModelForSequenceClassification.from_pretrained(demo_path).to(device).eval()
                    loaded_count += 1
                    st.info(f"🔄 Loaded {name} demo model instead")
                except:
                    st.error(f"❌ Demo model for {name} also failed")
        
        progress_bar.progress((i + 1) / total_models)
    
    status_text.text(f"✅ Successfully loaded {loaded_count}/{total_models} models")
    
    # Initialize LIME explainer
    explainer = LimeTextExplainer(class_names=["Severely Toxic", "Moderately Toxic", "Non-Toxic"])
    
    return models, tokenizers, device, explainer

def classify_comment(comment, models, tokenizers, device, lime_mode=False):
    """Classify a comment using ensemble of models"""
    if not models:
        return "No models loaded", np.array([0.33, 0.33, 0.34]), {}
    
    ensemble_preds = np.zeros(3)  # 3 classes
    model_probs = {}
    
    for model_name in models:
        tokenizer = tokenizers[model_name]
        model = models[model_name]
        
        inputs = tokenizer(
            comment, 
            padding=True, 
            truncation=True, 
            return_tensors="pt",
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs).logits
        
        probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]
        
        # Handle different output shapes
        if len(probs) == 2:  # Binary classification
            # Convert to 3-class: [non-toxic, moderately-toxic, severely-toxic]
            probs = np.array([probs[0], probs[1] * 0.5, probs[1] * 0.5])
        elif len(probs) > 3:
            # Take first 3 classes
            probs = probs[:3]
            probs = probs / np.sum(probs)  # Renormalize
        
        model_probs[model_name] = probs
        ensemble_preds += probs / len(models)
    
    if lime_mode:
        return ensemble_preds
    
    predicted_label = np.argmax(ensemble_preds)
    return LABEL_MAPPING[predicted_label], ensemble_preds, model_probs

def clean_text(text):
    """Clean text for LIME processing"""
    return re.sub(r'/\*', '', text)

class FastLIME:
    """Fast LIME replacement with 10-20x speed improvement"""
    
    def __init__(self, models, tokenizers, device):
        self.models = models
        self.tokenizers = tokenizers
        self.device = device
        
    def fast_lime_explain(self, comment, max_features=10, num_samples=30):
        """
        Fast LIME replacement - drop-in replacement for your lime_explain function
        
        Args:
            comment: Text to explain
            max_features: Number of top features to return
            num_samples: Number of samples to generate (reduced from 1000)
        
        Returns:
            list: [(word, importance_score), ...] sorted by importance
        """
        # self.models = models
        # self.tokenizers = tokenizers
        # self.device = device
        # start_time = time.time()
        
        # Method 1: Fast word removal (optimized version)
        word_importance = self._fast_word_removal(comment, num_samples)
        
        # Sort by absolute importance and return top features
        sorted_features = sorted(word_importance.items(), 
                               key=lambda x: abs(x[1]), 
                               reverse=True)
        # [:int(max_features)]
        
        # print(f"⚡ Fast explanation completed in {time.time() - start_time:.2f}s")
        return sorted_features
    
    def _fast_word_removal(self, text, num_samples):
        """Fast word removal importance calculation"""
        words = text.split()
        if len(words) <= 1:
            return {}
        
        # Get original prediction
        original_probs = self._batch_predict([text])
        if len(original_probs) == 0:
            return {}
        
        original_toxicity = self._get_toxicity_score(original_probs[0])
        
        # Generate samples more efficiently
        samples = []
        sample_info = []
        
        # 1. Remove each individual word (most important samples)
        for i in range(len(words)):
            masked_words = words.copy()
            removed_word = masked_words.pop(i)
            masked_text = " ".join(masked_words)
            samples.append(masked_text)
            sample_info.append([removed_word])
        
        # 2. Fill remaining with random combinations
        remaining = max(0, num_samples - len(samples))
        for _ in range(remaining):
            # Remove 1-3 random words
            num_remove = min(np.random.randint(1, 4), len(words))
            indices = np.random.choice(len(words), num_remove, replace=False)
            
            masked_words = [w for i, w in enumerate(words) if i not in indices]
            removed_words = [words[i] for i in indices]
            
            samples.append(" ".join(masked_words))
            sample_info.append(removed_words)
        
        # Batch prediction (much faster than individual predictions)
        all_probs = self._batch_predict(samples)
        
        # Calculate importance scores
        word_importance = {}
        
        for i, (sample_probs, removed_words) in enumerate(zip(all_probs, sample_info)):
            if len(sample_probs) > 0:
                sample_toxicity = self._get_toxicity_score(sample_probs)
                importance = original_toxicity - sample_toxicity
                
                # Distribute importance among removed words
                for word in removed_words:
                    if word in word_importance:
                        word_importance[word] = max(word_importance[word], importance / len(removed_words))
                    else:
                        word_importance[word] = importance / len(removed_words)
        
        return word_importance
    
    def _batch_predict(self, texts, batch_size=8):
        """Batch prediction for multiple texts"""
        if not texts:
            return []
        
        all_predictions = []
        
        # Use first available model for speed
        model_name = list(self.models.keys())[0]
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=256  # Reduced for speed
            ).to(self.device)
            
            # Batch inference
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                all_predictions.extend(probs.cpu().numpy())
        
        return all_predictions
    
    def _get_toxicity_score(self, probs):
        """Extract toxicity score from model probabilities"""
        if len(probs) == 2:
            return probs[1]  # Binary: toxic class
        elif len(probs) == 3:
            return probs[0] + probs[1] * 0.5  # 3-class: severe + moderate*0.5
        else:
            return probs[-1]  # Use last class as toxic
    
    def gradient_explanation(self, text, max_features=10):
        """
        Ultra-fast gradient-based explanation
        Alternative method that's even faster (~0.1s)
        """
        start_time = time.time()
        
        model_name = list(self.models.keys())[0]
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        # Enable gradients temporarily
        model.train()
        
        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256
        ).to(self.device)
        
        # Get embeddings
        embeddings = model.get_input_embeddings()
        input_embeddings = embeddings(inputs['input_ids'])
        input_embeddings.requires_grad_(True)
        
        # Forward pass
        outputs = model(inputs_embeds=input_embeddings, 
                       attention_mask=inputs.get('attention_mask'))
        
        # Get toxicity score
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        toxicity_score = probs[0, 1] if probs.shape[1] > 1 else probs[0, 0]
        
        # Backward pass
        toxicity_score.backward()
        
        # Calculate token importance
        gradients = input_embeddings.grad
        importance = torch.norm(gradients, dim=-1).squeeze()
        
        # Map to words
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        word_scores = []
        
        current_word = ""
        current_score = 0
        
        for token, score in zip(tokens, importance):
            if token.startswith('##') or token in ['[CLS]', '[SEP]', '[PAD]']:
                if token.startswith('##'):
                    current_word += token[2:]
                    current_score += score.item()
                elif current_word:
                    word_scores.append((current_word, current_score))
                    current_word = ""
                    current_score = 0
            else:
                if current_word:
                    word_scores.append((current_word, current_score))
                current_word = token
                current_score = score.item()
        
        if current_word:
            word_scores.append((current_word, current_score))
        
        # Reset model to eval mode
        model.eval()
        
        # Sort and return top features
        word_scores.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print(f"⚡ Gradient explanation completed in {time.time() - start_time:.2f}s")
        return word_scores[:max_features]

# Updated lime_explain function for your existing app
def fast_lime_explain(comment, models, tokenizers, device, method="fast_removal"):
    """
    Drop-in replacement for your lime_explain function
    
    Args:
        comment: Text to explain
        models: Your loaded models dict
        tokenizers: Your loaded tokenizers dict  
        device: PyTorch device
        method: "fast_removal" or "gradient"
    
    Returns:
        PIL Image or plotly figure (same interface as before)
    """
    explainer = FastLIME(models, tokenizers, device)
    
    if method == "gradient":
        explanation = explainer.gradient_explanation(comment)
    else:
        explanation = explainer.fast_lime_explain(comment)
    
    if not explanation:
        return None
    
    # Create chart (same as your original)
    words, scores = zip(*explanation)
    
    import plotly.graph_objects as go
    
    colors = ['red' if score > 0 else 'green' for score in scores]
    
    fig = go.Figure(data=[
        go.Bar(
            y=words,
            x=scores,
            orientation='h',
            marker_color=colors,
            text=[f'{score:.3f}' for score in scores],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Fast Word Importance Analysis",
        xaxis_title="Importance Score (Red=More Toxic, Green=Less Toxic)",
        yaxis_title="Words",
        height=max(400, len(words) * 40)
    )
    
    return fig
def create_probability_chart(ensemble_probs):
    """Create probability distribution chart using Plotly"""
    labels = ["Severely Toxic", "Moderately Toxic", "Non-Toxic"]
    colors = ["#f44336", "#ff9800", "#4caf50"]
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=ensemble_probs,
            marker_color=colors,
            text=[f"{prob:.3f}" for prob in ensemble_probs],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Ensemble Probability Distribution",
        xaxis_title="Toxicity Level",
        yaxis_title="Probability",
        showlegend=False,
        height=400
    )
    
    return fig
 
def create_lime_chart(explanation_data):
    """Create LIME explanation chart"""
    if not explanation_data:
        return None
    
    words, scores = zip(*explanation_data)
    colors = ["red" if score > 0 else "green" for score in scores]
    
    fig = go.Figure(data=[
        go.Bar(
            y=words,
            x=scores,
            orientation='h',
            marker_color=colors,
            text=[f"{score:.3f}" for score in scores],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="LIME Word Importance Analysis",
        xaxis_title="Toxicity Score (Red=More Toxic, Green=Less Toxic)",
        yaxis_title="Words",
        showlegend=False,
        height=500
    )
    
    return fig

def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<div class="main-header">Abusive Comment Detection System</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This system uses an ensemble of fine-tuned transformer models (ELECTRA, DeBERTa, RoBERTa, HateBERT) 
    to classify text toxicity levels and provides LIME explanations for interpretability.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Model loading status
        with st.expander("🔧 Model Status", expanded=True):
            if 'models_loaded' not in st.session_state:
                st.session_state.models_loaded = False
            
            if not st.session_state.models_loaded:
                if st.button("🚀 Load Models", type="primary"):
                    with st.spinner("Loading models..."):
                        models, tokenizers, device, explainer = load_models()
                        st.session_state.models = models
                        st.session_state.tokenizers = tokenizers
                        st.session_state.device = device
                        st.session_state.explainer = explainer
                        st.session_state.models_loaded = True
                        st.rerun()
            else:
                st.success(f"✅ {len(st.session_state.models)} models loaded")
                if st.button("🔄 Reload Models"):
                    st.session_state.models_loaded = False
                    st.rerun()
        
        # Settings
        st.header("⚙️ Settings")
        show_model_details = st.checkbox("Show individual model probabilities", value=True)
        show_lime = st.checkbox("Show LIME explanation", value=True)
        
        # About
        with st.expander("ℹ️ About"):
            st.markdown("""
            **Classes:**
            - 🔴 **Severely Toxic**: Highly offensive content
            - 🟡 **Moderately Toxic**: Somewhat inappropriate content  
            - 🟢 **Non-Toxic**: Clean, appropriate content
            
            **LIME**: Shows which words contribute most to the prediction.
            """)
    
    # Main content area
    if not st.session_state.get('models_loaded', False):
        st.warning("⚠️ Please load the models from the sidebar to start analyzing text.")
        st.stop()
    
    # Text input
    st.header("📝 Enter Text for Analysis")
    
    # Example buttons
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("😊 Positive Example"):
            st.session_state.input_text = "You're such an amazing person! Keep up the great work!"
    with col2:
        if st.button("😐 Neutral Example"):
            st.session_state.input_text = "I disagree with your opinion, but I respect your right to have it"
    with col3:
        if st.button("😠 Moderate Example"):
            st.session_state.input_text = "This is completely stupid and makes no sense"
    with col4:
        if st.button("🚨 Severe Example"):
            st.session_state.input_text = "I hate you and everything you stand for, you worthless piece of trash"
    
    # Text input area
    user_input = st.text_area(
        "Enter your text here:",
        value=st.session_state.get('input_text', ''),
        height=150,
        placeholder="Type or paste the comment you want to analyze..."
    )
    fle=FastLIME(st.session_state.models,st.session_state.tokenizers,st.session_state.device)
    # Analysis button
    if st.button("🔍 Analyze Text", type="primary", disabled=not user_input.strip()):
        if user_input.strip():
            with st.spinner("🧠 Analyzing text..."):
                # Get prediction
                prediction, ensemble_probs, model_probs = classify_comment(
                    user_input, 
                    st.session_state.models,
                    st.session_state.tokenizers,
                    st.session_state.device
                )
                
                # Display results
                st.header("📊 Analysis Results")
                
                # Main prediction with colored box
                prediction_class = "not at all toxic"
                st.markdown(f"""
                <div 
                # class="prediction-box {prediction_class}">
                    <h2>Prediction: {prediction}</h2>
                    <p>Confidence: {max(ensemble_probs):.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Metrics row
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Severely Toxic",
                        f"{ensemble_probs[0]:.3f}",
                        delta=f"{ensemble_probs[0] - 0.333:.3f}"
                    )
                with col2:
                    st.metric(
                        "Moderately Toxic", 
                        f"{ensemble_probs[1]:.3f}",
                        delta=f"{ensemble_probs[1] - 0.333:.3f}"
                    )
                with col3:
                    st.metric(
                        "Non-Toxic",
                        f"{ensemble_probs[2]:.3f}",
                        delta=f"{ensemble_probs[2] - 0.333:.3f}"
                    )
                
                # Charts
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📈 Probability Distribution")
                    prob_chart = create_probability_chart(ensemble_probs)
                    st.plotly_chart(prob_chart, use_container_width=True)
                
                with col2:
                    if show_model_details:
                        st.subheader("🔍 Individual Model Results")
                        for model_name, probs in model_probs.items():
                            st.write(f"**{model_name.upper()}:**")
                            cols = st.columns(3)
                            with cols[0]:
                                st.write(f"Severe: {probs[0]:.3f}")
                            with cols[1]:
                                st.write(f"Moderate: {probs[1]:.3f}")
                            with cols[2]:
                                st.write(f"Non-Toxic: {probs[2]:.3f}")
                
                # LIME explanation
                if show_lime:
                    st.subheader("🔬 LIME Word Importance Analysis")
                    
                    with st.spinner("Generating LIME explanation..."):
                        # fle(st.session_state.models,st.session_state.tokensizers,st.session_state.device)
                        explanation_data = fle.fast_lime_explain(
                            user_input,
                            # st.session_state.models,
                            # st.session_state.tokenizers, 
                            # st.session_state.device,
                            st.session_state.explainer
                        )
                    
                    if explanation_data:
                        lime_chart = create_lime_chart(explanation_data)
                        st.plotly_chart(lime_chart, use_container_width=True)
                        
                        st.info("""
                        **How to read LIME results:**
                        - 🔴 **Red bars**: Words that increase toxicity prediction
                        - 🟢 **Green bars**: Words that decrease toxicity prediction  
                        - **Longer bars**: More influential words
                        """)
                    else:
                        st.warning("Could not generate LIME explanation for this text.")
                
                # Export results
                st.subheader("💾 Export Results")
                results_dict = {
                    "text": user_input,
                    "prediction": prediction,
                    "timestamp": datetime.now().isoformat(),
                    "ensemble_probabilities": {
                        "severely_toxic": float(ensemble_probs[0]),
                        "moderately_toxic": float(ensemble_probs[1]),
                        "non_toxic": float(ensemble_probs[2])
                    },
                    "individual_models": {
                        model: {
                            "severely_toxic": float(probs[0]),
                            "moderately_toxic": float(probs[1]), 
                            "non_toxic": float(probs[2])
                        } for model, probs in model_probs.items()
                    }
                }
                
                st.download_button(
                    "📥 Download Results (JSON)",
                    data=str(results_dict),
                    file_name=f"toxicity_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    main()
