
from pathlib import Path

import streamlit as st
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image, ImageOps


# --------------------------------------------------
# 1. Application configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "resnet18_experiment.pth"

CLASS_NAMES = [
    "ADI",
    "BACK",
    "DEB",
    "LYM",
    "MUC",
    "MUS",
    "NORM",
    "STR",
    "TUM",
]

IMAGE_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

st.set_page_config(
    page_title="CRC Tissue Image Classifier",
    page_icon="🔬",
    layout="wide",
)


# --------------------------------------------------
# 2. Image preprocessing
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    ),
])


# --------------------------------------------------
# 3. Load the trained model
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    # Recreate the same ResNet18 architecture.
    model = resnet18(weights=None)

    # Replace the original classification layer
    # with the nine classes used in this project.
    model.fc = nn.Linear(
        model.fc.in_features,
        len(CLASS_NAMES),
    )

    # Load the weights learned during training.
    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    # The training checkpoint stores the state
    # dictionary under this key.
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()
    return model


# --------------------------------------------------
# 4. Predict the tissue class
# --------------------------------------------------

def predict_image(image, model):
    # Use the same RGB image format and normalization
    # as the project's evaluation pipeline.
    image = ImageOps.exif_transpose(image).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    with torch.inference_mode():
        logits = model(image_tensor)

        # Convert logits to scores that sum to 1.
        probabilities = torch.softmax(
            logits, dim=1
        )[0]

    top_scores, top_indices = torch.topk(
        probabilities,
        k=3,
    )

    predicted_index = int(top_indices[0])

    results = [
        {
            "class": CLASS_NAMES[int(index)],
            "score": float(score),
        }
        for score, index in zip(
            top_scores, top_indices
        )
    ]

    return (
        CLASS_NAMES[predicted_index],
        float(probabilities[predicted_index]),
        results,
    )


# --------------------------------------------------
# 5. Build the web page
# --------------------------------------------------

st.title("🔬 CRC Histopathology Image Classifier")

st.write(
    "An educational deep-learning prototype for "
    "classifying colorectal tissue image patches."
)

st.info(
    "Research demonstration only. Predictions are "
    "not medical diagnoses and must not be used to "
    "make treatment or clinical decisions."
)

with st.expander("About the model"):
    st.write(
        "This application uses a trained ResNet18 "
        "model to classify an image into one of "
        "nine tissue categories."
    )
    st.write(
        "The model was evaluated on an independent "
        "image-level test dataset. Its performance "
        "on a new uploaded image is not guaranteed."
    )

uploaded_file = st.file_uploader(
    "Upload a colorectal histopathology image",
    type=["jpg", "jpeg", "png"],
    help="Upload one tissue image patch at a time.",
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image).convert("RGB")

        left, right = st.columns(2)

        with left:
            st.subheader("Uploaded image")
            st.image(
                image,
                caption="Input tissue image",
                use_container_width=True,
            )

        with right:
            st.subheader("Classification result")

            if st.button(
                "Classify image",
                type="primary",
                use_container_width=True,
            ):
                try:
                    with st.spinner("Analyzing image..."):
                        model = load_model()
                        predicted_class, score, results = (
                            predict_image(image, model)
                        )

                    st.success(
                        f"Predicted class: {predicted_class}"
                    )

                    st.metric(
                        "Top-class softmax score",
                        f"{score:.2%}",
                    )

                    st.caption(
                        "This score is a model output, "
                        "not a clinically calibrated probability."
                    )

                    st.subheader("Top 3 predictions")

                    for result in results:
                        st.write(
                            f"**{result['class']}** — "
                            f"{result['score']:.2%}"
                        )
                        st.progress(result["score"])

                    st.caption(
                        "Class codes: ADI = adipose, "
                        "BACK = background, DEB = debris, "
                        "LYM = lymphocytes, MUC = mucus, "
                        "MUS = muscle, NORM = normal tissue, "
                        "STR = stroma, TUM = tumor epithelium."
                    )

                except FileNotFoundError as error:
                    st.error(str(error))

                except Exception as error:
                    st.error(
                        "The image could not be classified. "
                        "Check the model checkpoint and "
                        "application configuration."
                    )
                    st.exception(error)

    except Exception:
        st.error(
            "This file could not be opened as a valid image. "
            "Please try another JPG, JPEG, or PNG file."
        )

st.divider()

st.caption(
    "Educational AI project | ResNet18 | "
    "PyTorch | Streamlit"
)
