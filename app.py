import streamlit as st
import google.generativeai as genai
from PIL import Image
import pandas as pd
from io import BytesIO

# Secure API configuration
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Initialize model
model = genai.GenerativeModel("models/gemini-1.5-flash-latest")


def image_to_text(img):
    try:
        response = model.generate_content(img)
        return response.text
    except Exception as e:
        st.error(f"Error in image processing: {str(e)}")
        return None


def image_and_query(img, query):
    try:
        response = model.generate_content([query, img])
        return response.text
    except Exception as e:
        st.error(f"Error in content generation: {str(e)}")
        return None


# Streamlit app interface
st.title("Image to Text Extractor & Generator")
st.write("Upload an image and get AI-powered insights")

upload_image = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'])
user_query = st.text_input("Enter your creative prompt (e.g., 'Write a story about this image')")

if st.button("Generate Content"):
    if upload_image is not None and user_query.strip():
        with st.spinner('Analyzing image...'):
            img = Image.open(upload_image)
            st.image(img, caption='Uploaded Image', use_column_width=True)

            # Process image and generate content
            extracted_details = image_to_text(img)
            generated_content = image_and_query(img, user_query)

            # Display results
            if extracted_details:
                st.subheader("Image Analysis")
                st.write(extracted_details)

            if generated_content:
                st.subheader("Generated Content")
                st.write(generated_content)

            # Create downloadable CSV
            if extracted_details or generated_content:
                df = pd.DataFrame({
                    "Extracted Details": [extracted_details or "N/A"],
                    "Generated Content": [generated_content or "N/A"]
                })

                csv_buffer = BytesIO()
                df.to_csv(csv_buffer, index=False)
                csv_buffer.seek(0)

                st.download_button(
                    label="Download Results as CSV",
                    data=csv_buffer,
                    file_name="image_analysis.csv",
                    mime="text/csv",
                    key='download-csv'
                )
    else:
        if not upload_image:
            st.warning("Please upload an image first")
        if not user_query.strip():
            st.warning("Please enter a creative prompt")
