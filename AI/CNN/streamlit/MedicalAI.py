import streamlit as st
import requests
import base64
import re

# ------------------------
# CONFIG
# ------------------------
GEMINI_API_KEY = "AIzaSyCkmBYTDwSKyGmtRMssExRlQCfvDk09ilg"  # Replace with your valid key
GEMINI_MODEL = "gemini-2.0-flash"

st.set_page_config(
    page_title="Team DDHVK Medical AI",
    layout="wide"
)

# ------------------------
# UTILITIES
# ------------------------
def format_medical_response(text: str) -> str:
    """Format AI response into bullet points + highlight keywords."""
    if not text:
        return "No response text found."

    # Convert line breaks into bullet points if appropriate
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    formatted_lines = []

    for line in lines:
        if line.startswith(("-", "*", "•", "1.", "2.", "3.")):
            formatted_lines.append(f"<li>{line.lstrip('-*•1234567890. ')}</li>")
        else:
            formatted_lines.append(f"<p>{line}</p>")

    formatted_text = "<h4>The Response is:</h4><ul>" + "".join(formatted_lines) + "</ul>"

    # Highlight medical keywords
    medical_keywords = [
        'symptoms', 'diagnosis', 'treatment', 'medication', 'therapy', 'disease', 'condition',
        'infection', 'inflammation', 'chronic', 'acute', 'syndrome', 'disorder', 'cancer',
        'diabetes', 'hypertension', 'cardiovascular', 'respiratory', 'neurological',
        'consultation', 'specialist', 'emergency', 'urgent care', 'prescription'
    ]
    for keyword in medical_keywords:
        regex = re.compile(rf"\b{keyword}\b", re.IGNORECASE)
        formatted_text = regex.sub(
            lambda m: f"<strong style='color:#38bdf8;'>{m.group(0)}</strong>",  # Cyan highlights
            formatted_text
        )
    return formatted_text

# ------------------------
# GEMINI QUERY
# ------------------------
def query_gemini_api(prompt, image_base64=None):
    """Send text+image query to Gemini API and get clean structured response."""
    try:
        request_body = {
            "contents": [{
                "parts": [
                    {"text": f"{prompt}\n\nReturn the answer in clear points or sections if applicable."}
                ]
            }]
        }
        if image_base64:
            request_body["contents"][0]["parts"].append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": image_base64.split(",")[1]
                }
            })

        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}",
            headers={"Content-Type": "application/json"},
            json=request_body
        )
        response.raise_for_status()
        data = response.json()

        if "candidates" in data and len(data["candidates"]) > 0:
            parts = data["candidates"][0].get("content", {}).get("parts", [])
            result_text = " ".join([p.get("text", "") for p in parts if "text" in p])
            return result_text.strip() if result_text else "No response text found."
        else:
            return "No valid response from API."

    except Exception as e:
        return f"Gemini API Error: {e}"

# ------------------------
# UI HEADER
# ------------------------
st.markdown(
    """
    <div style="text-align: center; padding: 20px;">
        <h1 style="color:#0f766e;">Medical AI</h1>
        <p style="color:gray;">Team DDHVK  • Samsung Prism • For educational purposes only</p>
    </div>
    """, unsafe_allow_html=True
)

# ------------------------
# MAIN INTERFACE
# ------------------------
tab1, tab2 = st.tabs(["💬 Textual Health Report", "🖼 Multi-Modal Health Analysis (Image + Text)"])

# --- TEXT MODE ---
with tab1:
    text_input = st.text_area("Describe your symptoms, diseases, or treatments...", height=150)
    if st.button("Consult AI", use_container_width=True, key="text_btn"):
        if not text_input.strip():
            st.warning("⚠ Please enter Patient's Condition and Diagnosis Data's.")
        else:
            with st.spinner(" Model Studying Your Reports...Please Wait For Sometime..."):
                gemini_response = query_gemini_api(
                    f'Based on this medical query: "{text_input}", provide a comprehensive and accurate response.'
                )
                st.markdown(
                    f"<div style='background:#001f3f; color:#f8fafc; padding:20px; border-radius:10px; border:1px solid #ddd;'>{format_medical_response(gemini_response)}</div>",
                    unsafe_allow_html=True
                )
                st.info("⚠ Disclaimer: This information is for educational purposes only and should not replace professional medical advice.")

# --- IMAGE + TEXT MODE ---
with tab2:
    text_input2 = st.text_area("post your symptoms or leave blank to analyze only the image...", height=150)
    uploaded_image = st.file_uploader("Upload Medical Image (Scan Reports) (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if st.button("Analyze Image + Text", use_container_width=True, key="img_btn"):
        if not uploaded_image and not text_input2.strip():
            st.warning("⚠ Please provide a Patient's Condition, Diagnosis or Scan Images.")
        else:
            with st.spinner("🖼 Model Analysing Your Data's..Please be patient..."):
                image_base64 = None
                if uploaded_image:
                    bytes_data = uploaded_image.getvalue()
                    image_base64 = "data:image/jpeg;base64," + base64.b64encode(bytes_data).decode()
                    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
                gemini_response = query_gemini_api(
                    text_input2 or "Analyze this medical image and provide relevant information.",
                    image_base64
                )
                st.markdown(
                    f"<div style='background:#001f3f; color:#f8fafc; padding:20px; border-radius:10px; border:1px solid #ddd;'>{format_medical_response(gemini_response)}</div>",
                    unsafe_allow_html=True
                )
                st.info("⚠ Disclaimer: This information is for educational purposes only and should not replace professional medical advice.")