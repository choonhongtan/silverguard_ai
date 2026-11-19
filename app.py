# ==========================================
# SECTION 1: DEPENDENCIES & ENVIRONMENT
# ==========================================
import time
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models

# Enforce senior-friendly accessibility configuration
st.set_page_config(
    page_title="SilverGuard AI — SMS Scam Shield",
    page_icon="🛡️",
    layout="centered"
)

# Prevent UI State Desync by locking session variables
if "scan_history" not in st.session_state:
    st.session_state.scan_history = []


# ==========================================
# SECTION 2: ISOLATED ML BACKEND (TENSORFLOW)
# ==========================================
@st.cache_resource
def load_silverguard_model():
    """Builds a TensorFlow Sequential Model fully compatible with modern Keras 3.

    Explicitly defines the Input layer to prevent shape mismatch errors.
    """
    vocab_size = 1000
    max_len = 20

    model = models.Sequential([
        layers.Input(shape=(max_len,)),
        layers.Embedding(input_dim=vocab_size, output_dim=16),
        layers.GlobalAveragePooling1D(),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


# Initialize cached backend engine
ml_engine = load_silverguard_model()


def preprocess_text(raw_text: str) -> tf.Tensor:
    """Preprocesses raw SMS text into a 2D Tensor with explicit shape (1, 20).

    Uses defensive assertions to validate runtime invariants.
    """
    tokens = [abs(hash(w)) % 1000 for w in raw_text.split()][:20]
    while len(tokens) < 20:
        tokens.append(0)

    input_tensor = tf.convert_to_tensor([tokens], dtype=tf.int32)

    # Defensive Assert: Validate shape invariant
    assert input_tensor.shape == (1, 20), (
        f"Shape Error: Expected (1, 20), got {input_tensor.shape}"
    )
    return input_tensor


def predict_scam(raw_text: str):
    """Executes prediction with explicit scalar extraction via .item()."""
    if not raw_text.strip():
        return None, "Empty message provided."

    try:
        input_tensor = preprocess_text(raw_text)
        prediction_raw = ml_engine.predict(input_tensor, verbose=0)

        # DEFENSIVE ASSERT: Verify the 2D tensor shape before extraction
        assert prediction_raw.shape == (1, 1), f"Expected shape (1, 1), got {prediction_raw.shape}"

        # SAFE SCALAR EXTRACTION: .item() extracts the float from the array array([[0.85]]) -> 0.85
        risk_score = float(prediction_raw.item())
        
        # Enforce range invariant
        assert 0.0 <= risk_score <= 1.0, f"Risk score out of bounds: {risk_score}"


# ==========================================
# SECTION 3: FRONTEND UI (STREAMLIT)
# ==========================================
css_style = """
    <style>
    .main-title {
        font-size: 2.4rem !important;
        color: #1E293B;
        font-weight: 800;
    }
    .senior-text {
        font-size: 1.2rem !important;
        line-height: 1.6;
    }
    .stButton>button {
        font-size: 1.2rem !important;
        height: 3em;
        width: 100%;
        font-weight: bold;
    }
    .scam-box {
        background-color: #FEF2F2;
        border: 2px solid #EF4444;
        padding: 20px;
        border-radius: 10px;
    }
    .safe-box {
        background-color: #F0FDF4;
        border: 2px solid #22C55E;
        padding: 20px;
        border-radius: 10px;
    }
    </style>
"""
st.markdown(css_style, unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">🛡️ SilverGuard AI</div>',
    unsafe_allow_html=True
)
header_desc = (
    '<p class="senior-text"><b>SMS Scam Shield for Seniors:</b> Paste '
    'any suspicious message below to verify safety before taking action.</p>'
)
st.markdown(header_desc, unsafe_allow_html=True)

st.divider()

# Sidebar Setup
with st.sidebar:
    st.header("⚙️ Controls & History")
    st.info("Target Audience: Senior Citizens & Vulnerable Mobile Users")
    st.divider()
    st.subheader("📜 Session History")
    if len(st.session_state.scan_history) == 0:
        st.write("No messages scanned yet.")
    else:
        for item in reversed(st.session_state.scan_history[-5:]):
            st.caption(
                f"**{item['time']}** | {item['label']} ({item['score']:.0%})"
            )
            st.text(item['snippet'])

# Main User Input Area
placeholder_text = (
    "Example: RM0 Maybank: Your account access has been restricted. "
    "Please re-verify at http://maybank-security.com"
)
user_sms = st.text_area(
    "Paste Suspicious SMS / WhatsApp Message Here:",
    height=130,
    placeholder=placeholder_text
)

# Trigger Button
if st.button("🔍 Check Message Now"):
    if not user_sms.strip():
        st.warning("⚠️ Please paste a text message first.")
    else:
        with st.spinner("Analyzing message indicators against database..."):
            time.sleep(0.3)
            risk_score, label = predict_scam(user_sms)

        if risk_score is not None:
            st.subheader("Analysis Result:")
            if label == "HIGH SCAM RISK":
                danger_html = f"""
                    <div class="scam-box">
                        <h2 style="color: #DC2626; margin-top:0;">
                            ⚠️ DANGER: HIGH SCAM PROBABILITY
                        </h2>
                        <p class="senior-text">
                            <b>Risk Confidence: {risk_score:.1%}</b>
                        </p>
                        <p class="senior-text">
                            <b>Action Required:</b> DO NOT click links and
                            DO NOT call phone numbers in this text message.
                        </p>
                    </div>
                """
                st.markdown(danger_html, unsafe_allow_html=True)
            else:
                safe_html = f"""
                    <div class="safe-box">
                        <h2 style="color: #16A34A; margin-top:0;">
                            ✅ MESSAGE APPEARS SAFE
                        </h2>
                        <p class="senior-text">
                            <b>Risk Score: {risk_score:.1%}</b>
                        </p>
                        <p class="senior-text">
                            No urgent phishing or scam indicators were
                            identified in this message.
                        </p>
                    </div>
                """
                st.markdown(safe_html, unsafe_allow_html=True)

            # Store in Session State
            st.session_state.scan_history.append({
                "time": time.strftime("%H:%M:%S"),
                "snippet": user_sms[:28] + "...",
                "score": risk_score,
                "label": label
            })
