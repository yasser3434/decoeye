import streamlit as st
import replicate
from dotenv import load_dotenv
from PIL import Image
import io

# Load API token from .env
load_dotenv()

# Page config
st.set_page_config(layout="wide")
st.title("DecoEye")

# Sidebar: room + style selection
with st.sidebar:
    st.header("Settings")
    
    # Room and Style dropdowns
    room = st.selectbox("Select Room", ["living room", "bedroom", "kitchen"])
    style = st.selectbox("Select Style", ["modern", "scandinavian", "coastal"])
    
    # Auto-generate prompt from selections
    prompt = f"A {style} style decorated {room}, 4k photo, highly detailed and realistic"
    # st.markdown(f"**Generated Prompt:**\n\n`{prompt}`")
    
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

# Layout
col1, col2 = st.columns(2)

if uploaded_file:
    # Show input image
    input_image = Image.open(uploaded_file)
    col1.subheader("Input Image")
    col1.image(input_image, use_container_width=True)

    if st.button("Generate Image"):
        with st.spinner("Generating image..."):
            # Call the model with the constructed prompt
            output = replicate.run(
                "black-forest-labs/flux-kontext-max",
                input={
                    "prompt": prompt,
                    "input_image": uploaded_file
                }
            )

            # Convert output bytes to image and show
            image_bytes = output.read()
            result_image = Image.open(io.BytesIO(image_bytes))

            col2.subheader("Generated Image")
            col2.image(result_image, use_container_width=True)
else:
    col1.info("Please upload an image to get started.")
