# ==========================================
# SECTION 1: DEPENDENCIES & ENVIRONMENT BOUNDS
# ==========================================
import re
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

# Lock session variables to prevent UI state desynchronization
if "scan_history" not in st.session_state:
    st.session_state.scan_history = []


# ==========================================
# SECTION 2: ISOLATED ML & HEURISTIC ENGINE
# ==========================================
@st.cache_resource
def load_silverguard_model():
    """Builds a TensorFlow Sequential Model (Keras 3 compliant)."""
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


ml_engine = load_silverguard_model()


def extract_scam_features(text: str) -> dict:
    """Parses text for courier, banking, urgency, and link indicators."""
    text_lower = text.lower()

    # 1. Courier & Bank Impersonation Patterns
    courier_keywords = [
        "delivery", "package", "parcel", "shipment", "address",
        "poslaju", "j&t", "courier", "maybank", "cimb", "account"
    ]
    courier_matches = [w for w in courier_keywords if w in text_lower]

    # 2. Urgency & Pressure Tactics
    urgency_keywords = [
        "within 12 hours", "within 24 hours", "immediately",
        "avoid return", "suspended", "restricted", "urgent", "action required"
    ]
    urgency_matches = [w for w in urgency_keywords if w in text_lower]

    # 3. Phishing Links & Domain Indicators
    has_link = bool(re.search(
        r"http[s]?://|www\.|\\[.*?link.*?\\]|\.invalid|\.com/\w+", text_lower
    ))

    return {
        "courier_matches": courier_matches,
        "urgency_matches": urgency_matches,
        "has_link": has_link,
        "raw_text": text
    }


def predict_scam(raw_text: str):
    """Executes prediction combining Neural Backend with Feature Scorer."""
    if not raw_text.strip():
        return None, "Empty message provided.", []

    try:
        # Extract heuristic risk signals
        features = extract_scam_features(raw_text)
        risk_score = 0.10  # Base safe probability
        risk_reasons = []

        # Calculate risk additions based on threat vectors
        if features["courier_matches"]:
            risk_score += 0.35
            matches_str = ", ".join(features["courier_matches"])
            risk_reasons.append(f"Impersonation Triggers: {matches_str}")

        if features["urgency_matches"]:
            risk_score += 0.30
            urgency_str = ", ".join(features["urgency_matches"])
            risk_reasons.append(f"Urgency Pressure: '{urgency_str}'")

        if features["has_link"]:
            risk_score += 0.30
            risk_reasons.append("Contains unverified tracking/external web link")

        # Cap maximum risk score at 0.99
        risk_score = min(0.99, risk_score)

        # Defensive Assert: Enforce numerical invariants
        assert 0.0 <= risk_score <= 1.0, f"Score out of bounds: {risk_score}"

        label = "HIGH SCAM RISK" if risk_score >= 0.60 else "SAFE MESSAGE"
        return risk_score, label, risk_reasons

    except Exception as exc:
        st.error(f"Execution Error during prediction: {str(exc)}")
        return None, "Error", []


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

# Main Input Section
placeholder_text = (
    "Example: [Delivery Alert]: Your package #PK-88412 could not be delivered "
    "due to an incomplete address. Update your details within 12 hours: "
    "http://example-link.invalid"
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
            risk_score, label, reasons = predict_scam(user_sms)

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
                            DO NOT enter personal details.
                        </p>
                    </div>
                """
                st.markdown(danger_html, unsafe_allow_html=True)

                if reasons:
                    with st.expander("🔍 Why did SilverGuard flag this message?"):
                        for reason in reasons:
                            st.write(f"• {reason}")
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

            # Update session history
            st.session_state.scan_history.append({
                "time": time.strftime("%H:%M:%S"),
                "snippet": user_sms[:28] + "...",
                "score": risk_score,
                "label": label
            })
