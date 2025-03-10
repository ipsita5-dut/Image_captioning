import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pickle

st.set_page_config(page_title="AI Image Captioning", page_icon="📸", layout="centered")

st.markdown("""
    <style>
        .main-title {text-align: center; color: #007BFF; font-size: 40px; font-weight: bold;}
        .subtitle {text-align: center; color: #444; font-size: 18px;}
        .uploaded-image {border-radius: 10px; margin-top: 10px;}
        .caption-box {background-color: #f8f9fa; padding: 15px; border-radius: 10px; text-align: center; font-size: 20px; font-weight: bold; color: #333;}
        .footer {text-align: center; font-size: 12px; color: #888; margin-top: 20px;}
        .centered {display: flex; flex-direction: column; align-items: center;}
    </style>
""", unsafe_allow_html=True)

def generate_caption(image_path, model_path, tokenizer_path, feature_extractor_path, max_length=34, img_size=224):

    caption_model = load_model(model_path)
    feature_extractor = load_model(feature_extractor_path)

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    img = load_img(image_path, target_size=(img_size, img_size))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    image_features = feature_extractor.predict(img, verbose=0)

    in_text = "startseq"
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = caption_model.predict([image_features, sequence], verbose=0)
        yhat_index = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat_index, None)
        if word is None:
            break
        in_text += " " + word
        if word == "endseq":
            break
    return in_text.replace("startseq", "").replace("endseq", "").strip()

def main():
    
    st.markdown("<h1 class='main-title'>📸 AI Image Caption Generator</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Upload an image and let AI describe it!</p>", unsafe_allow_html=True)

    uploaded_image = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        with open("uploaded_image.jpg", "wb") as f:
            f.write(uploaded_image.getbuffer())

        model_path = "model.keras"
        tokenizer_path = "tokenizer.pkl"
        feature_extractor_path = "feature_extractor.keras"

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            generate_btn = st.button("✨ Generate Caption")

        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

        if generate_btn:
            with st.spinner("🤖 AI is analyzing the image..."):
                caption = generate_caption("uploaded_image.jpg", model_path, tokenizer_path, feature_extractor_path)

            st.markdown(f"<div class='caption-box'>📝 {caption}</div>", unsafe_allow_html=True)

    st.markdown("<p class='footer'>Built with ❤️ using AI</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
