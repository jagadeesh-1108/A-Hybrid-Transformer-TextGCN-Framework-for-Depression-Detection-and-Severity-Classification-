import streamlit as st
import torch
from transformers import BertTokenizer
from model_binary import BinaryModel
from model_severity import SeverityModel

device = torch.device("cpu")

CRISIS_KEYWORDS = [
    "suicide",
    "kill myself",
    "end my life",
    "want to die",
    "no reason to live",
    "self harm",
    "hang myself",
    "cut myself"
]

def crisis_override(text):
    text = text.lower()
    return any(keyword in text for keyword in CRISIS_KEYWORDS)

@st.cache_resource
def load_models():
    binary_model = BinaryModel()
    binary_model.load_state_dict(torch.load("binary_model.pt", map_location=device))
    binary_model.eval()

    severity_model = SeverityModel()
    severity_model.load_state_dict(torch.load("severity_model.pt", map_location=device))
    severity_model.eval()

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    return binary_model, severity_model, tokenizer

binary_model, severity_model, tokenizer = load_models()

st.title("🧠 Mental Health Assessment & Depression Detection System")

user_text = st.text_area("Enter your thoughts:")

if st.button("Analyze"):

    if user_text.strip() == "":
        st.warning("Please enter some text.")
        st.stop()

    encoding = tokenizer(
        user_text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    # 🔵 Binary Model
    with torch.no_grad():
        binary_output = binary_model(
            encoding["input_ids"],
            encoding["attention_mask"]
        )

    binary_pred = torch.argmax(binary_output, dim=1).item()

    binary_prob = torch.softmax(binary_output, dim=1)
    depression_conf = binary_prob[0][1].item()

    st.write(f"Depression Confidence: {depression_conf*100:.2f}%")

    if binary_pred == 0:
        st.success("Non-Depressed")
        st.info("Maintain healthy lifestyle and emotional balance.")
        st.stop()

    # 🔴 Depressed Case
    st.error("Depressed")

    # 🚨 Crisis Override FIRST
    if crisis_override(user_text):
        st.warning("Severity Level: Severe")
        st.info("Immediate psychiatric evaluation advised.")
        st.stop()

    # 🧠 Severity Model
    with torch.no_grad():
        sev_output = severity_model(
            encoding["input_ids"],
            encoding["attention_mask"]
        )

    sev_pred = torch.argmax(sev_output, dim=1).item()

    severity_prob = torch.softmax(sev_output, dim=1)
    severity_conf = torch.max(severity_prob).item()

    st.write(f"Severity Confidence: {severity_conf*100:.2f}%")

    severity_dict = {
        0: "Minimum",
        1: "Mild",
        2: "Moderate",
        3: "Severe"
    }

    severity_label = severity_dict.get(sev_pred, "Unknown")

    st.warning(f"Severity Level: {severity_label}")

    if severity_label == "Minimum":
        st.info("Light mood changes. Monitor regularly.")
    elif severity_label == "Mild":
        st.info("Consider lifestyle improvements and social support.")
    elif severity_label == "Moderate":
        st.info("Consult a mental health professional.")
    elif severity_label == "Severe":
        st.info("Immediate psychiatric evaluation advised.")