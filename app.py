from pathlib import Path

import streamlit as st
import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18


# ============================================================
# 1. Project configuration
# ============================================================
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "resnet18_experiment.pth"

CLASS_NAMES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLASS_DESCRIPTIONS = {
    "ADI": "Adipose tissue",
    "BACK": "Background",
    "DEB": "Debris",
    "LYM": "Lymphocytes",
    "MUC": "Mucus",
    "MUS": "Muscle",
    "NORM": "Normal tissue",
    "STR": "Stroma",
    "TUM": "Tumor epithelium",
}

# These are the reported ResNet18 results from the project's independent
# image-level test evaluation. They describe that evaluation, not a guarantee
# about any newly uploaded image.
TEST_METRICS = {
    "Accuracy": "87.65%",
    "Macro F1": "82.70%",
    "Weighted F1": "87.20%",
    "Test images": "7,180",
}

CLASS_ACCURACY = [
    {"Code": "ADI", "Tissue category": "Adipose tissue", "Images": 1338, "Accuracy": 95.89},
    {"Code": "BACK", "Tissue category": "Background", "Images": 847, "Accuracy": 100.00},
    {"Code": "DEB", "Tissue category": "Debris", "Images": 339, "Accuracy": 58.11},
    {"Code": "LYM", "Tissue category": "Lymphocytes", "Images": 634, "Accuracy": 99.37},
    {"Code": "MUC", "Tissue category": "Mucus", "Images": 1035, "Accuracy": 87.73},
    {"Code": "MUS", "Tissue category": "Muscle", "Images": 592, "Accuracy": 95.61},
    {"Code": "NORM", "Tissue category": "Normal tissue", "Images": 741, "Accuracy": 81.51},
    {"Code": "STR", "Tissue category": "Stroma", "Images": 421, "Accuracy": 36.34},
    {"Code": "TUM", "Tissue category": "Tumor epithelium", "Images": 1233, "Accuracy": 89.62},
]

st.set_page_config(
    page_title="CRC Tissue AI | Research Demo",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. Visual styling
# ============================================================
st.markdown(
    """
    <style>
    :root {
        --ink: #17233b;
        --muted: #65748b;
        --teal: #087f8c;
        --teal-light: #e7f5f5;
        --line: #e1e8ef;
        --panel: #f7f9fc;
    }
    .stApp {
        background: linear-gradient(180deg, #f8fbfd 0%, #ffffff 38%);
        color: var(--ink);
    }
    .block-container {
        max-width: 1220px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    [data-testid="stSidebar"] {
        background: #f2f7fa;
        border-right: 1px solid #e1e8ef;
    }
    .hero {
        padding: 1.7rem 1.9rem;
        border-radius: 22px;
        background: linear-gradient(120deg, #10243c 0%, #164d62 65%, #087f8c 100%);
        color: white;
        margin-bottom: 1.2rem;
        box-shadow: 0 12px 32px rgba(16, 36, 60, 0.12);
    }
    .hero-eyebrow {
        color: #bdecef;
        text-transform: uppercase;
        letter-spacing: .13em;
        font-size: .76rem;
        font-weight: 700;
        margin-bottom: .65rem;
    }
    .hero h1 {
        color: white !important;
        font-size: clamp(1.9rem, 4vw, 2.8rem) !important;
        line-height: 1.12;
        margin: 0 0 .75rem 0;
    }
    .hero p {
        color: #e0edf3;
        font-size: 1.02rem;
        max-width: 800px;
        margin-bottom: 0;
    }
    .section-kicker {
        color: #087f8c;
        font-size: .75rem;
        font-weight: 800;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin-bottom: .25rem;
    }
    .soft-card {
        border: 1px solid var(--line);
        background: white;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        height: 100%;
    }
    .small-muted { color: #65748b; font-size: .9rem; }
    .class-chip {
        display: inline-block;
        padding: .25rem .55rem;
        border-radius: 999px;
        background: #e7f5f5;
        color: #086b76;
        font-weight: 700;
        font-size: .78rem;
    }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e1e8ef;
        padding: .85rem 1rem;
        border-radius: 14px;
    }
    .stButton > button[kind="primary"] {
        background: #087f8c;
        border: 1px solid #087f8c;
        border-radius: 10px;
        font-weight: 700;
        min-height: 2.8rem;
    }
    .stButton > button[kind="primary"]:hover {
        background: #066a75;
        border-color: #066a75;
    }
    .notice {
        border-left: 4px solid #d97706;
        background: #fff8e8;
        padding: .85rem 1rem;
        border-radius: 8px;
        color: #6b4b12;
        margin: .6rem 0 1.1rem 0;
    }
    .footer-note {
        color: #65748b;
        font-size: .82rem;
        border-top: 1px solid #e1e8ef;
        padding-top: 1rem;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. Image preprocessing and model loading
# ============================================================
transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


@st.cache_resource(show_spinner=False)
def load_model():
    """Load the trained ResNet18 checkpoint once per app instance."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "The trained model checkpoint was not found. Expected it at: "
            f"{MODEL_PATH}. Ensure models/resnet18_experiment.pth is included "
            "in the deployed repository."
        )

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict_image(image, model):
    """Return top class, its softmax score, and the top-three class scores."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    with torch.inference_mode():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    top_scores, top_indices = torch.topk(probabilities, k=3)
    results = [
        {
            "class": CLASS_NAMES[int(index)],
            "description": CLASS_DESCRIPTIONS[CLASS_NAMES[int(index)]],
            "score": float(score),
        }
        for score, index in zip(top_scores, top_indices)
    ]
    best_class = results[0]["class"]
    best_score = results[0]["score"]
    return best_class, best_score, results, probabilities.tolist()


# ============================================================
# 4. Sidebar navigation and project information
# ============================================================
with st.sidebar:
    st.markdown("## 🔬 CRC Tissue AI")
    st.caption("Histopathology image classification · Research prototype")
    page = st.radio(
        "Navigate",
        ["Image analysis", "Model performance", "Methodology & limitations"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**Model**")
    st.write("ResNet18 · 9 tissue classes")
    st.markdown("**Input**")
    st.write("JPG, JPEG or PNG tissue image patch")
    st.markdown("**Processing**")
    st.write("CPU inference · 224 × 224 pixels")
    st.divider()
    st.caption("For research and educational demonstration only.")


# ============================================================
# 5. Shared header
# ============================================================
st.markdown(
    """
    <div class="hero">
      <div class="hero-eyebrow">Computational pathology · Research prototype</div>
      <h1>Colorectal Tissue Image Classifier</h1>
      <p>Explore how a deep-learning image classifier assigns a tissue category to a small
      histopathology image patch. Upload an image to view the model's leading predictions
      and explore the evaluation results behind this prototype.</p>
    </div>
    <div class="notice"><strong>Important:</strong> This tool is an experimental educational
    demonstration, not a medical device. Its output must not be used for diagnosis, treatment,
    screening, or other clinical decisions.</div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 6. Page: Image analysis
# ============================================================
if page == "Image analysis":
    st.markdown('<div class="section-kicker">01 / Analyze an image</div>', unsafe_allow_html=True)
    st.header("Upload a tissue image")
    st.write("Choose one JPG, JPEG or PNG image patch. The app resizes it to 224 × 224 pixels before inference.")

    uploaded_file = st.file_uploader(
        "Choose a histopathology image",
        type=["jpg", "jpeg", "png"],
        help="Use an image patch similar in format to the patches used to develop and evaluate this model.",
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            image = ImageOps.exif_transpose(image).convert("RGB")
            image_column, details_column = st.columns([1.1, 0.9], gap="large")

            with image_column:
                st.markdown("### Image preview")
                st.image(image, use_container_width=True)
                st.caption(f"Image dimensions: {image.width} × {image.height} pixels · RGB preview")

            with details_column:
                st.markdown("### Ready for analysis")
                st.markdown(
                    '<div class="soft-card"><span class="class-chip">Image loaded</span>'
                    '<p style="margin-top:.8rem"><strong>Model:</strong> ResNet18</p>'
                    '<p><strong>Model input:</strong> 224 × 224 pixels</p>'
                    '<p><strong>Output:</strong> Top class and top-three scores</p>'
                    '<p class="small-muted">The displayed scores are softmax outputs and are not calibrated clinical probabilities.</p></div>',
                    unsafe_allow_html=True,
                )
                analyze_clicked = st.button("Analyze image", type="primary", use_container_width=True)

            if analyze_clicked:
                try:
                    with st.spinner("Loading the model and analyzing the image…"):
                        model = load_model()
                        predicted_class, score, top_results, all_scores = predict_image(image, model)
                    st.session_state["prediction_result"] = {
                        "filename": uploaded_file.name,
                        "predicted_class": predicted_class,
                        "score": score,
                        "top_results": top_results,
                        "all_scores": all_scores,
                        "width": image.width,
                        "height": image.height,
                    }
                except FileNotFoundError as error:
                    st.error(str(error))
                except Exception as error:
                    st.error("The image could not be analyzed. Please check the model checkpoint and deployment logs.")
                    st.exception(error)

        except Exception:
            st.error("This file could not be opened as a valid image. Please try another JPG, JPEG or PNG file.")

    if st.session_state.get("prediction_result"):
        result = st.session_state["prediction_result"]
        if uploaded_file is None or result.get("filename") != uploaded_file.name:
            st.session_state.pop("prediction_result", None)
        else:
            st.divider()
            st.markdown('<div class="section-kicker">02 / Prediction output</div>', unsafe_allow_html=True)
            st.header("Model prediction")
            result_left, result_right = st.columns([0.8, 1.2], gap="large")
            with result_left:
                st.markdown("#### Leading predicted category")
                st.markdown(
                    f'<div class="soft-card"><div class="section-kicker">Predicted class</div>'
                    f'<h2 style="margin:.2rem 0;color:#087f8c">{result["predicted_class"]}</h2>'
                    f'<p style="margin-bottom:.4rem">{CLASS_DESCRIPTIONS[result["predicted_class"]]}</p>'
                    f'<p class="small-muted">Top-class softmax score: {result["score"]:.2%}</p></div>',
                    unsafe_allow_html=True,
                )
                st.caption("A higher score indicates the model assigned more of its output probability mass to that class. It does not establish diagnostic certainty.")
            with result_right:
                st.markdown("#### Top three model outputs")
                for item in result["top_results"]:
                    st.write(f"**{item['class']} — {item['description']}** · {item['score']:.2%}")
                    st.progress(item["score"])
            with st.expander("View scores for all nine classes"):
                for class_name, class_score in zip(CLASS_NAMES, result["all_scores"]):
                    st.write(f"**{class_name} — {CLASS_DESCRIPTIONS[class_name]}** · {class_score:.2%}")
                    st.progress(class_score)
            st.caption("Class codes: ADI = adipose, BACK = background, DEB = debris, LYM = lymphocytes, MUC = mucus, MUS = muscle, NORM = normal tissue, STR = stroma, TUM = tumor epithelium.")
            if st.button("Clear result", use_container_width=False):
                st.session_state.pop("prediction_result", None)
                st.rerun()
    else:
        st.markdown("### What happens after upload?")
        step1, step2, step3 = st.columns(3, gap="medium")
        with step1:
            st.markdown('<div class="soft-card"><div class="section-kicker">Step 1</div><strong>Upload</strong><p class="small-muted">Choose a tissue image in JPG, JPEG or PNG format.</p></div>', unsafe_allow_html=True)
        with step2:
            st.markdown('<div class="soft-card"><div class="section-kicker">Step 2</div><strong>Preprocess</strong><p class="small-muted">The image is resized and normalized for the model.</p></div>', unsafe_allow_html=True)
        with step3:
            st.markdown('<div class="soft-card"><div class="section-kicker">Step 3</div><strong>Review outputs</strong><p class="small-muted">Inspect the leading class scores as model outputs, not diagnoses.</p></div>', unsafe_allow_html=True)


# ============================================================
# 7. Page: Model performance
# ============================================================
elif page == "Model performance":
    st.markdown('<div class="section-kicker">03 / Evaluation</div>', unsafe_allow_html=True)
    st.header("ResNet18 evaluation results")
    st.write("These figures come from the project's independent image-level test evaluation of 7,180 images. They summarize performance on that test set; they do not guarantee performance on images uploaded to this website.")

    metric_columns = st.columns(4, gap="medium")
    for column, (label, value) in zip(metric_columns, TEST_METRICS.items()):
        column.metric(label, value)

    st.markdown("### Class-wise accuracy on the independent test set")
    st.write("Class accuracy is the percentage of test images belonging to each true class that ResNet18 classified correctly.")
    try:
        import pandas as pd
        class_frame = pd.DataFrame(CLASS_ACCURACY)
        st.dataframe(
            class_frame,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Accuracy": st.column_config.ProgressColumn(
                    "Accuracy (%)", min_value=0, max_value=100, format="%.2f%%"
                )
            },
        )
    except Exception:
        for row in CLASS_ACCURACY:
            st.write(f"{row['Code']} — {row['Tissue category']}: {row['Accuracy']:.2f}% ({row['Images']} images)")

    st.markdown("### Observations")
    observation_left, observation_right = st.columns(2, gap="large")
    with observation_left:
        st.markdown('<div class="soft-card"><strong>Relatively strong classes in this evaluation</strong><p class="small-muted">BACK (100.00%), LYM (99.37%), MUS (95.61%) and ADI (95.89%) had high class-wise accuracy on this test set.</p></div>', unsafe_allow_html=True)
    with observation_right:
        st.markdown('<div class="soft-card"><strong>Classes needing further investigation</strong><p class="small-muted">STR (36.34%) and DEB (58.11%) had substantially lower class-wise accuracy. These errors should be examined before considering any broader use.</p></div>', unsafe_allow_html=True)

    st.info("Accuracy can hide differences between classes. Macro F1 gives each class equal weight, while weighted F1 gives more influence to classes with more test examples. The macro F1 of 82.70% is lower than the overall accuracy of 87.65%, indicating that performance is not equally strong across all classes.")
    st.caption("The values shown here are the recorded ResNet18 independent-test results from this project. The app does not calculate new test metrics when a user uploads an image.")


# ============================================================
# 8. Page: Methodology and limitations
# ============================================================
else:
    st.markdown('<div class="section-kicker">04 / About the project</div>', unsafe_allow_html=True)
    st.header("Methodology and limitations")

    with st.expander("How does the classifier work?", expanded=True):
        st.markdown(
            """
            1. **Image upload:** the user supplies one image patch.
            2. **Image preparation:** the image is oriented correctly, converted to RGB, resized to 224 × 224 pixels, converted to a tensor, and normalized using ImageNet mean and standard deviation values.
            3. **ResNet18 inference:** the trained neural network calculates a score for each of the nine classes.
            4. **Softmax conversion:** the scores are converted into values that sum to 1. The app shows the highest-scoring class and the top three outputs.
            5. **Interpretation:** these are model outputs, not a medical diagnosis or calibrated clinical probabilities.
            """
        )

    with st.expander("The nine output classes"):
        for code in CLASS_NAMES:
            st.write(f"**{code}** — {CLASS_DESCRIPTIONS[code]}")

    with st.expander("Important limitations", expanded=True):
        st.markdown(
            """
            - **Not for clinical use.** This is a research and educational demonstration, not a validated medical device.
            - **Image-level evaluation only.** The project evaluation was performed at image level; patient-level independence has not been established.
            - **Uneven class performance.** The recorded test results show much lower accuracy for STR and DEB than for several other classes.
            - **Softmax scores are not calibrated confidence.** A high score should not be interpreted as the probability that a tissue diagnosis is correct.
            - **Input differences matter.** Image staining, scanner, magnification, compression, and image quality may differ from the data used during model development.
            - **Limited validation.** Results on the project's test set do not establish performance across hospitals, populations, laboratories, or acquisition systems.
            - **No human review or clinical context.** The app examines one image patch and does not consider patient history, whole-slide context, or a pathologist's assessment.
            - **Reproducibility requires the full setup.** Reproducing results depends on the model checkpoint, preprocessing, class order, software versions, dataset split, and training details.
            """
        )

    with st.expander("Responsible use"):
        st.write("Use the app to explore image classification and machine-learning evaluation. Do not use its predictions to diagnose a person, guide treatment, or replace a qualified pathologist. Uploaded images should not contain identifying or confidential information unless you have authorization to share them with this hosted service.")

    st.markdown("### Technology")
    tech1, tech2, tech3, tech4 = st.columns(4, gap="medium")
    for column, title, description in [
        (tech1, "PyTorch", "Model inference"),
        (tech2, "Torchvision", "ResNet18 and transforms"),
        (tech3, "Pillow", "Image handling"),
        (tech4, "Streamlit", "Interactive web interface"),
    ]:
        with column:
            st.markdown(f'<div class="soft-card"><strong>{title}</strong><p class="small-muted">{description}</p></div>', unsafe_allow_html=True)


# ============================================================
# 9. Footer
# ============================================================
st.markdown(
    '<div class="footer-note">CRC Tissue AI · Educational research prototype · ResNet18 · PyTorch · Streamlit<br>'
    'Predictions are experimental model outputs and must not be used for clinical decisions.</div>',
    unsafe_allow_html=True,
)
