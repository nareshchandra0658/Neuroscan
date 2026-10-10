
from pathlib import Path

import streamlit as st
from PIL import Image

from predictor import load_model_and_classes, predict_image


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="NeuroScan | MRI Classifier",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 NeuroScan")
st.subheader("Brain MRI Image Classification")

st.write(
    "Upload an MRI image to obtain an experimental "
    "four-class model prediction."
)

st.info(
    "Research and educational use only. This model is "
    "not a medical diagnosis and must not replace "
    "assessment by a qualified healthcare professional."
)


# --------------------------------------------------
# LOAD MODEL ONCE
# --------------------------------------------------

@st.cache_resource
def get_model():
    return load_model_and_classes()


try:
    model, class_names = get_model()
except Exception as error:
    st.error("Could not load the NeuroScan model.")
    st.code(str(error))
    st.caption(
        "Check that the model and class names JSON file "
        "exist inside the project's models folder."
    )
    st.stop()


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

st.markdown("### Upload MRI Image")

uploaded_file = st.file_uploader(
    "Choose a JPG, JPEG or PNG image",
    type=["jpg", "jpeg", "png"],
    help="Upload a 2D brain MRI image."
)


if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        left, right = st.columns([1, 1])

        with left:
            st.markdown("#### Uploaded image")
            st.image(
                image,
                caption="Uploaded brain MRI",
                use_container_width=True
            )

            st.caption(
                f"Original image dimensions: "
                f"{image.width} × {image.height}"
            )

        with st.spinner("Analyzing image..."):
            result = predict_image(
                model,
                class_names,
                image
            )

        with right:
            st.markdown("#### Model output")

            st.metric(
                label="Predicted class",
                value=result["predicted_class"].replace(
                    "notumor", "No tumor"
                ).title()
            )

            st.metric(
                label="Model confidence",
                value=f"{result['confidence'] * 100:.2f}%"
            )

            st.progress(
                min(max(result["confidence"], 0.0), 1.0)
            )

            st.caption(
                "Confidence is the model's output score, "
                "not a measure of diagnostic certainty."
            )

        # ------------------------------------------
        # ALL CLASS SCORES
        # ------------------------------------------

        st.markdown("---")
        st.markdown("### Scores for all classes")

        sorted_results = sorted(
            result["probabilities"].items(),
            key=lambda item: item[1],
            reverse=True
        )

        for class_name, probability in sorted_results:
            display_name = class_name.replace(
                "notumor", "No tumor"
            ).title()

            st.write(
                f"**{display_name}** — "
                f"{probability * 100:.2f}%"
            )

            st.progress(
                min(max(probability, 0.0), 1.0)
            )

        st.warning(
            "A high model score does not establish whether "
            "a tumor is present. Do not use this result to "
            "make treatment or medical decisions."
        )

    except Exception as error:
        st.error("The uploaded image could not be processed.")
        st.code(str(error))

else:
    st.markdown("---")
    st.markdown("### Supported classes")

    col1, col2, col3, col4 = st.columns(4)

    col1.info("**Glioma**")
    col2.info("**Meningioma**")
    col3.info("**No tumor**")
    col4.info("**Pituitary**")

    st.caption(
        "Upload an image above to see the model's prediction."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")
st.caption(
    "NeuroScan | Deep learning project for brain MRI "
    "image classification"
)
