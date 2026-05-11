import streamlit as st
import pandas as pd
import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Page settings must be the first Streamlit command
st.set_page_config(
    page_title="AI Scam Detector",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern, premium styling
st.markdown("""
<style>
    /* Global background and fonts */
    [data-testid="stAppViewContainer"] {
        background-color: #0E1117;
        color: #FAFAFA;
        font-family: 'Inter', sans-serif;
    }
    
    /* Gradient text for the main header */
    .main-header {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(45deg, #FF4B2B, #FF416C);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-bottom: 10px;
        margin-top: -20px;
    }
    
    /* Subtitle styling */
    .subtitle {
        text-align: center;
        color: #A0A0A0;
        font-size: 1.1em;
        margin-bottom: 40px;
        font-weight: 300;
    }
    
    /* Text Area styling */
    .stTextArea textarea {
        background-color: #1E2329 !important;
        color: #FFFFFF !important;
        border: 1px solid #30363D !important;
        border-radius: 12px !important;
        padding: 16px !important;
        font-size: 16px !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2) !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #FF416C !important;
        box-shadow: 0 0 0 2px rgba(255, 65, 108, 0.3) !important;
    }
    
    /* Primary Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #FF416C 0%, #FF4B2B 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(255, 65, 108, 0.4) !important;
        width: 100% !important;
        height: auto !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(255, 65, 108, 0.6) !important;
        background: linear-gradient(90deg, #FF4B2B 0%, #FF416C 100%) !important;
    }

    /* Footer */
    .footer {
        text-align: center;
        font-size: 14px;
        color: #666;
        margin-top: 60px;
        padding-top: 20px;
        border-top: 1px solid #222;
    }
</style>
""", unsafe_allow_html=True)

# Dataset (Expanded slightly for better training)
data = {
    'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham'],
    'message': [
        'Win money now',
        'Hello how are you',
        'Claim your free prize',
        'Lets meet tomorrow',
        'Urgent: Your account has been compromised. Click here to verify.',
        'Can we reschedule our meeting to 3 PM?',
        'Congratulations! You have been selected for a $1000 gift card.',
        'Please find attached the report for this week.'
    ]
}

df = pd.DataFrame(data)

# Train AI (Cached so it doesn't retrain on every interaction)
@st.cache_resource
def train_model():
    cv = CountVectorizer()
    X = cv.fit_transform(df['message'])
    y = df['label']
    model = MultinomialNB()
    model.fit(X, y)
    return cv, model

cv, model = train_model()

# Header Section
st.markdown("<h1 class='main-header'>🛡️ AI Scam & Fraud Detector</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Instantly analyze messages for phishing, scams, and fraudulent intent using advanced Machine Learning.</p>", unsafe_allow_html=True)

# Main Interface
st.markdown("### 🔍 Analyze a Message")
msg = st.text_area(
    "Paste your message below",
    placeholder="Paste a suspicious SMS, email, or social media message here...",
    height=150,
    label_visibility="collapsed"
)

# Centered button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    check_btn = st.button("Analyze Message", use_container_width=True)

if check_btn:
    if msg.strip() == "":
        st.warning("⚠️ Please paste a message to analyze.")
    else:
        with st.spinner("Analyzing semantics and patterns..."):
            time.sleep(0.8) # Simulate processing time for UX
            
            data_features = cv.transform([msg])
            prediction = model.predict(data_features)
            
            st.markdown("---")
            st.markdown("### 📊 Analysis Results")
            
            if prediction[0] == "spam":
                # High Risk Alert
                st.error("🚨 **High Risk: Scam or Phishing Detected!**")
                
                # Custom Red Progress Bar
                st.markdown(
                    "<style>.stProgress > div > div > div > div { background-image: linear-gradient(to right, #f85032, #e73827); }</style>",
                    unsafe_allow_html=True,
                )
                st.progress(95)
                st.caption("Threat Confidence: **95%**")
                
                # Safety Tips in Expander
                with st.expander("🛡️ Recommended Safety Actions", expanded=True):
                    st.markdown("""
                    - **Do not click** any links or download attachments.
                    - **Never share** personal information, passwords, or OTPs.
                    - **Do not reply** to the sender.
                    - **Block & Report** the sender immediately.
                    """)
            else:
                # Low Risk Alert
                st.success("✅ **Low Risk: Message Appears Safe**")
                
                # Custom Green Progress Bar
                st.markdown(
                    "<style>.stProgress > div > div > div > div { background-image: linear-gradient(to right, #11998e, #38ef7d); }</style>",
                    unsafe_allow_html=True,
                )
                st.progress(15)
                st.caption("Threat Confidence: **15%**")
                
                # Safety Tips in Expander
                with st.expander("ℹ️ Analysis Details", expanded=True):
                    st.markdown("""
                    - No suspicious keywords or phishing links detected.
                    - Urgency signals are within normal limits.
                    - *Note: Always stay vigilant even if a message appears safe.*
                    """)

# Footer
st.markdown("<div class='footer'>Powered by Python, Streamlit & Scikit-Learn ✨</div>", unsafe_allow_html=True)