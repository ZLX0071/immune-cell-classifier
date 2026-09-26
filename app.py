r"""
Immune Cell Classifier

Streamlit app for classifying H&E immune-cell images as:
    B Cell, Macrophage, or T Cell.

Colab setup:
    !pip install streamlit grad-cam pyngrok -q
    from google.colab import drive
    drive.mount("/content/drive")

    !streamlit run /content/drive/My\ Drive/ImmuneCellClassifier/app.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

try:
    import matplotlib.pyplot as plt
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DRIVE_BASE = "/content/drive/My Drive/ImmuneCellClassifier"
DRIVE_CHECKPOINT_DIR = os.path.join(DRIVE_BASE, "checkpoints")

EFFICIENTNET_CKPT = os.path.join(DRIVE_CHECKPOINT_DIR, "efficientnet_best.pt")
DENSENET_CKPT     = os.path.join(DRIVE_CHECKPOINT_DIR, "densenet_best.pt")
MOBILENET_CKPT    = os.path.join(DRIVE_CHECKPOINT_DIR, "mobilenet_best.pt")
RESNET_CKPT       = os.path.join(DRIVE_CHECKPOINT_DIR, "resnet_best.pt")
VIT_CKPT          = os.path.join(DRIVE_CHECKPOINT_DIR, "vit_best.pt")

APP_DIR = Path(__file__).resolve().parent
LOCAL_CHECKPOINT_DIR = APP_DIR / "model_checkpoints"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASS_NAMES  = ["B_Cells", "Macrophages", "T_Cells"]
CLASS_LABELS = ["B Cell", "Macrophage", "T Cell"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

BASE_MODEL_OPTIONS = [
    "EfficientNet-B0",
    "DenseNet121",
    "MobileNetV3-Large",
    "ResNet50",
    "ViT-B/16",
]

SUGGESTED_ENSEMBLE_OPTION = "Ensemble: (Default) MobileNetV3-Large + EfficientNet-B0 (suggested)"

ENSEMBLE_OPTIONS = {
    "Ensemble: EfficientNet-B0 + DenseNet121": ["EfficientNet-B0", "DenseNet121"],
    SUGGESTED_ENSEMBLE_OPTION: ["MobileNetV3-Large", "EfficientNet-B0"],
    "Ensemble: EfficientNet-B0 + ResNet50": ["EfficientNet-B0", "ResNet50"],
    "Ensemble: EfficientNet-B0 + ViT-B/16": ["EfficientNet-B0", "ViT-B/16"],
    "Ensemble: DenseNet121 + MobileNetV3-Large": ["DenseNet121", "MobileNetV3-Large"],
    "Ensemble: DenseNet121 + ResNet50": ["DenseNet121", "ResNet50"],
    "Ensemble: DenseNet121 + ViT-B/16": ["DenseNet121", "ViT-B/16"],
    "Ensemble: MobileNetV3-Large + ResNet50": ["MobileNetV3-Large", "ResNet50"],
    "Ensemble: MobileNetV3-Large + ViT-B/16": ["MobileNetV3-Large", "ViT-B/16"],
    "Ensemble: ResNet50 + ViT-B/16": ["ResNet50", "ViT-B/16"],
}

MODEL_OPTIONS = BASE_MODEL_OPTIONS + list(ENSEMBLE_OPTIONS.keys())

# Confidence threshold below which a low-confidence warning is shown
LOW_CONFIDENCE_THRESHOLD = 0.50


@dataclass(frozen=True)
class ModelSpec:
    display_name: str
    checkpoint_name: str
    drive_checkpoint: str
    builder: Callable[[], nn.Module]
    gradcam_target: Callable[[nn.Module], list[nn.Module]] | None
    reshape_transform: Callable[[torch.Tensor], torch.Tensor] | None = None


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
class CellResNet50(nn.Module):
    """ResNet50 wrapper matching the saved resnet_best.pt checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        base_model = models.resnet50(weights=None)
        self.features = nn.Sequential(
            base_model.conv1, base_model.bn1, base_model.relu,
            base_model.maxpool, base_model.layer1, base_model.layer2,
            base_model.layer3, base_model.layer4,
        )
        self.avgpool = base_model.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(2048, 256), nn.ReLU(),
            nn.BatchNorm1d(256), nn.Dropout(0.3), nn.Linear(256, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_efficientnet() -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, 3))
    return model


def build_densenet() -> nn.Module:
    model = models.densenet121(weights=None)
    model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1024, 3))
    return model


def build_mobilenet() -> nn.Module:
    model = models.mobilenet_v3_large(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, 3)
    return model


def build_resnet() -> nn.Module:
    return CellResNet50()


def build_vit() -> nn.Module:
    model = models.vit_b_16(weights=None)
    model.heads.head = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.heads.head.in_features, 3),
    )
    return model


def vit_reshape_transform(tensor: torch.Tensor, height: int = 14, width: int = 14) -> torch.Tensor:
    tensor = tensor[:, 1:, :]
    tensor = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
    return tensor.permute(0, 3, 1, 2)


MODEL_SPECS = {
    "EfficientNet-B0": ModelSpec(
        display_name="EfficientNet-B0",
        checkpoint_name="efficientnet_best.pt",
        drive_checkpoint=EFFICIENTNET_CKPT,
        builder=build_efficientnet,
        gradcam_target=lambda model: [model.features[-1]],
    ),
    "DenseNet121": ModelSpec(
        display_name="DenseNet121",
        checkpoint_name="densenet_best.pt",
        drive_checkpoint=DENSENET_CKPT,
        builder=build_densenet,
        gradcam_target=lambda model: [model.features[-1]],
    ),
    "MobileNetV3-Large": ModelSpec(
        display_name="MobileNetV3-Large",
        checkpoint_name="mobilenet_best.pt",
        drive_checkpoint=MOBILENET_CKPT,
        builder=build_mobilenet,
        gradcam_target=lambda model: [model.features[-1]],
    ),
    "ResNet50": ModelSpec(
        display_name="ResNet50",
        checkpoint_name="resnet_best.pt",
        drive_checkpoint=RESNET_CKPT,
        builder=build_resnet,
        gradcam_target=lambda model: [model.features[-1]],
    ),
    "ViT-B/16": ModelSpec(
        display_name="ViT-B/16",
        checkpoint_name="vit_best.pt",
        drive_checkpoint=VIT_CKPT,
        builder=build_vit,
        gradcam_target=lambda model: [model.encoder.layers[-1].ln_1],
        reshape_transform=vit_reshape_transform,
    ),
}

# ---------------------------------------------------------------------------
# Loading and inference
# ---------------------------------------------------------------------------
def resolve_checkpoint_path(spec: ModelSpec) -> Path:
    env_dir = os.environ.get("DATA3888_CHECKPOINT_DIR")
    candidates = []
    if env_dir:
        candidates.append(Path(env_dir) / spec.checkpoint_name)
    candidates.extend([
        Path(spec.drive_checkpoint),
        LOCAL_CHECKPOINT_DIR / spec.checkpoint_name,
        APP_DIR / spec.checkpoint_name,
    ])
    for path in candidates:
        if path.exists():
            return path
    checked = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(
        f"{spec.display_name} checkpoint was not found.\n\n"
        f"Checked paths:\n{checked}\n\n"
        "Fix: make sure Google Drive is mounted and the checkpoint file exists "
        "at the Drive path above. If the file is there but still not found, "
        "check that DRIVE_BASE matches your actual Drive folder name."
    )


def extract_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


@st.cache_resource(show_spinner=False)
def load_model(model_key: str) -> nn.Module:
    spec = MODEL_SPECS[model_key]
    checkpoint_path = resolve_checkpoint_path(spec)
    model = spec.builder()
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(extract_state_dict(checkpoint))
    model.to(DEVICE)
    model.eval()
    return model


def get_selected_model_keys(selection: str) -> list[str]:
    if selection in ENSEMBLE_OPTIONS:
        return ENSEMBLE_OPTIONS[selection]
    return [selection]


def predict(selection: str, tensor: torch.Tensor) -> tuple[int, np.ndarray, dict[str, np.ndarray]]:
    model_keys = get_selected_model_keys(selection)
    component_probs = {}
    with torch.no_grad():
        for key in model_keys:
            model = load_model(key)
            probs = F.softmax(model(tensor), dim=1).squeeze(0).cpu().numpy()
            component_probs[key] = probs
    avg_probs = np.mean(np.stack(list(component_probs.values()), axis=0), axis=0)
    pred_class = int(avg_probs.argmax())
    return pred_class, avg_probs, component_probs


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------
def build_gradcam(model_key: str, model: nn.Module) -> "GradCAM":
    spec = MODEL_SPECS[model_key]
    kwargs: dict = {
        "model": model,
        "target_layers": spec.gradcam_target(model),
    }
    if spec.reshape_transform is not None:
        kwargs["reshape_transform"] = spec.reshape_transform
    return GradCAM(**kwargs)


def make_gradcam_figure(
    image_array: np.ndarray,
    cam_map: np.ndarray,
    model_name: str,
) -> "plt.Figure":
    overlay = show_cam_on_image(image_array, cam_map, use_rgb=True)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    fig.suptitle(model_name, fontsize=12, fontweight="bold")

    axes[0].imshow(image_array)
    axes[0].set_title("Original", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title("Grad-CAM", fontsize=10)
    axes[1].axis("off")

    scale = plt.cm.ScalarMappable(
        cmap="jet",
        norm=plt.Normalize(vmin=0, vmax=1),
    )
    scale.set_array([])
    colorbar = fig.colorbar(scale, ax=axes[1], fraction=0.046, pad=0.04)
    colorbar.set_label("Relative influence", rotation=270, labelpad=14)
    colorbar.set_ticks([0, 0.5, 1])
    colorbar.set_ticklabels(["Low", "Medium", "High"])

    plt.tight_layout()
    return fig


def render_gradcam(
    selection: str,
    tensor: torch.Tensor,
    pil_img: Image.Image,
    pred_class: int,
) -> None:
    if not GRADCAM_AVAILABLE:
        st.info(
            "Grad-CAM is not installed. Run `pip install grad-cam` to enable heatmaps."
        )
        return

    st.subheader("Model explanation (Grad-CAM)")

    # ── Explanation of what Grad-CAM shows ───────────────────────────────────
    with st.expander("How to read these heatmaps", expanded=False):
        st.markdown(
            """
            **What is Grad-CAM?**
            Grad-CAM (Gradient-weighted Class Activation Mapping) highlights the
            regions of the image that most strongly influenced the model's prediction.

            **How to interpret the colours:**
            - **Red / warm** regions had the highest influence on the predicted class.
            - **Blue / cool** regions contributed little or nothing to the prediction.
            - The colour scale is relative within each image; it is not a probability.

            **What to look for:**
            In a well-behaved model, the highlighted regions should correspond to
            biologically meaningful structures — such as the cell nucleus, cytoplasm
            boundary, or characteristic staining patterns — rather than background
            tissue or slide artefacts.

            **Ensemble note:**
            When using the ensemble, two heatmaps are shown — one per component model.
            Comparing them reveals whether EfficientNet and MobileNet attend to the
            same or different cell features, which helps explain why combining them
            can improve robustness.
            """
        )

    st.caption("Warmer regions show image areas that most influenced the prediction.")

    image_array = np.asarray(pil_img.resize((224, 224))).astype(np.float32) / 255.0
    model_keys = get_selected_model_keys(selection)
    columns = st.columns(len(model_keys))

    for column, model_key in zip(columns, model_keys):
        with column:
            with st.spinner(f"Generating {model_key} heatmap..."):
                model = load_model(model_key)
                cam = build_gradcam(model_key, model)
                cam_map = cam(
                    input_tensor=tensor,
                    targets=[ClassifierOutputTarget(pred_class)],
                )[0]
                fig = make_gradcam_figure(image_array, cam_map, model_key)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Immune Cell Classifier",
    page_icon="🔬",
    layout="wide",
)

# ── Guard: Google Drive must be mounted in Colab ──────────────────────────────
_in_colab = os.path.exists("/content")
if _in_colab and not os.path.exists("/content/drive/My Drive"):
    st.error(
        "**Google Drive is not mounted.**  \n"
        "Run the following in a Colab cell, then reload this page:  \n"
        "```python\n"
        "from google.colab import drive\n"
        "drive.mount('/content/drive')\n"
        "```"
    )
    st.stop()

st.title("🔬 Immune Cell Classifier")
st.caption(
    "Upload an H&E-stained immune-cell image and classify it as "
    "**B Cell**, **Macrophage**, or **T Cell** using one of the trained DATA3888 models."
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] div[data-baseweb="select"] {
            font-size: 0.9rem;
        }
        section[data-testid="stSidebar"] .stSelectbox {
            margin-bottom: 0.35rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("**Model**")
    selected_model = st.selectbox(
        "Model",
        MODEL_OPTIONS,
        index=MODEL_OPTIONS.index(SUGGESTED_ENSEMBLE_OPTION),
        label_visibility="collapsed",
    )
    if selected_model == SUGGESTED_ENSEMBLE_OPTION:
        st.caption("Suggested final model from the ensemble comparison.")

    show_gradcam = st.checkbox("Show Grad-CAM heatmap(s)", value=True)

    st.divider()
    st.header("Model performance")
    st.markdown(
        """
        | Model | Accuracy | F1 | ROC-AUC |
        |---|---|---|---|
        | ResNet50 | 0.604 | 0.451 | 0.657 |
        | DenseNet121 | 0.545 | 0.448 | 0.689 |
        | EfficientNet-B0 | 0.606 | 0.453 | 0.707 |
        | ViT-B/16 | 0.574 | 0.449 | 0.690 |
        | MobileNetV3 | 0.621 | 0.455 | 0.644 |
        | **MobileNetV3 + EfficientNet-B0 (default ensemble)** | **0.612** | **0.488** | — |
        """
    )

    st.divider()
    st.header("Project context")
    st.markdown(
        """
        **Task:** 3-class H&E immune-cell classification

        **Classes:** B Cell, Macrophage, T Cell

        **Training:** pretrained CNN/transformer backbones, two-phase
        fine-tuning, weighted sampling for class imbalance.

        **Final ensemble:** MobileNetV3-Large + EfficientNet-B0 soft voting,
        chosen for strong validation performance and complementary errors
        (error correlation 0.388).
        """
    )
    st.caption(f"Running on: {DEVICE}")


# ── Checkpoint status panel ───────────────────────────────────────────────────
available_rows = []
for key, spec in MODEL_SPECS.items():
    try:
        checkpoint_path = resolve_checkpoint_path(spec)
        status = "✅ available"
    except FileNotFoundError:
        checkpoint_path = Path(spec.drive_checkpoint)
        status = "❌ missing"
    available_rows.append({
        "Model": key,
        "Checkpoint": spec.checkpoint_name,
        "Status": status,
    })

with st.expander("Checkpoint status"):
    st.dataframe(pd.DataFrame(available_rows), use_container_width=True, hide_index=True)


# ── File uploader ─────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a cell image",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is None:
    st.info("Upload a PNG or JPG image to run classification.")
    st.markdown(
        """
        The app expects a single cropped H&E-stained immune-cell image. It returns
        the predicted cell type, confidence, class probabilities, and optional
        Grad-CAM heatmaps for interpretable model output.
        """
    )
    st.stop()

pil_img = Image.open(uploaded_file).convert("RGB")
tensor = TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)

left, right = st.columns([1, 1.4])

with left:
    st.subheader("Uploaded image")
    st.image(pil_img, use_container_width=True)

with right:
    st.subheader("Prediction")
    with st.spinner("Loading checkpoint and running inference..."):
        try:
            pred_class, avg_probs, component_probs = predict(selected_model, tensor)
        except FileNotFoundError as exc:
            st.error(str(exc))
            st.stop()
        except RuntimeError as exc:
            st.error(
                "**Checkpoint architecture mismatch.**  \n"
                "The `.pt` file could not be loaded into the model definition in this app.  \n"
                "This usually means the classifier head in `app.py` does not match what was "
                "saved during training. Check the model definition in the project notebook "
                "and make sure `build_<model>()` in this file uses the same layer structure."
            )
            st.exception(exc)
            st.stop()

    predicted_label = CLASS_LABELS[pred_class]
    confidence = float(avg_probs[pred_class]) * 100

    st.success(f"Predicted cell type: **{predicted_label}**")
    st.metric("Confidence", f"{confidence:.1f}%")

    # ── Low confidence warning ────────────────────────────────────────────────
    if float(avg_probs[pred_class]) < LOW_CONFIDENCE_THRESHOLD:
        st.warning(
            f"**Low confidence ({confidence:.1f}%)** — the model is uncertain about this image.  \n"
            "This may occur with ambiguous cell morphology, unusual staining, or image artefacts.  \n"
            "**Recommend manual review by a pathologist.**"
        )

    probability_df = pd.DataFrame({
        "Cell type": CLASS_LABELS,
        "Probability": avg_probs,
    }).set_index("Cell type")
    st.bar_chart(probability_df, use_container_width=True)

    with st.expander("Probability values"):
        st.dataframe(
            probability_df.assign(
                Probability=lambda df: df["Probability"].map("{:.4f}".format)
            ),
            use_container_width=True,
        )

    if len(component_probs) > 1:
        with st.expander("Ensemble component probabilities"):
            component_df = pd.DataFrame(
                {key: probs for key, probs in component_probs.items()},
                index=CLASS_LABELS,
            )
            st.dataframe(component_df.style.format("{:.4f}"), use_container_width=True)

if show_gradcam:
    st.divider()
    render_gradcam(selected_model, tensor, pil_img, pred_class)
