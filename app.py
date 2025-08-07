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
        font-size: 3rem;
        color: #ff4b4b;
        text-align: center;
        margin-bottom: 2rem;
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

def create_lime_explanation(comment, models, tokenizers, device, explainer):
    """Generate LIME explanation"""
    def model_predict(texts):
        cleaned_texts = [clean_text(text) for text in texts]
        return np.array([
            classify_comment(text, models, tokenizers, device, lime_mode=True) 
            for text in cleaned_texts
        ])
    
    try:
        cleaned_comment = clean_text(comment)
        exp = explainer.explain_instance(cleaned_comment, model_predict, num_features=10)
        return exp.as_list()
    except Exception as e:
        st.error(f"LIME explanation failed: {str(e)}")
        return []

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
                prediction_class = prediction.lower().replace(" ", "-")
                st.markdown(f"""
                <div class="prediction-box {prediction_class}">
                    <h2>🚀 Prediction: {prediction}</h2>
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
                        explanation_data = create_lime_explanation(
                            user_input,
                            st.session_state.models,
                            st.session_state.tokenizers, 
                            st.session_state.device,
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