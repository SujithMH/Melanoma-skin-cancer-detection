import streamlit as st
import torch
import cv2
import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.models.hybrid import HybridDeepNet

# --- CONFIGURATION & CACHING ---
st.set_page_config(page_title="Melanoma XAI DeepNet", layout="wide")

# ─── CLASS DEFINITIONS ────────────────────────────────────────────────────────
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

# Training-set class frequencies (from splits/train.csv):
#   nv=4718, mel=770, bkl=741, bcc=381, akiec=207, vasc=96, df=89  | total=7002
# Prior probability of each class in the training set (used for prior correction)
TRAIN_COUNTS = np.array([207, 381, 741, 89, 770, 4718, 96], dtype=np.float32)
TRAIN_PRIORS = TRAIN_COUNTS / TRAIN_COUNTS.sum()

# Asymmetric malignancy-detection thresholds.
# Because a missed melanoma is catastrophically worse than a false alarm,
# we use MUCH lower decision thresholds for high-risk classes.
# These are applied on the PRIOR-CORRECTED probabilities.
#   mel  -> flag if prior-corrected probability > 20%
#   bcc  -> flag if prior-corrected probability > 25%
#   akiec-> flag if prior-corrected probability > 30%  (pre-malignant)
MALIGNANT_THRESHOLDS = {
    'mel':   0.20,   # Melanoma: most dangerous — lowest threshold
    'bcc':   0.25,   # Basal Cell Carcinoma
    'akiec': 0.30,   # Actinic Keratosis (pre-malignant)
}


def prior_correct(softmax_probs: np.ndarray) -> np.ndarray:
    """
    Correct for training-set class imbalance using Bayes' theorem.

    The model's softmax output p(class|image) is biased by the training prior
    because nv makes up 67% of training data. We divide by training priors and
    re-normalize — equivalent to assuming a uniform prior at inference time.
    This is the standard approach for applying imbalanced classifiers to
    screening tasks where all classes are equally likely to appear.
    """
    corrected = softmax_probs / (TRAIN_PRIORS + 1e-8)
    corrected = corrected / corrected.sum()
    return corrected


@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Focal optimal config from Phase F
    config = {'fc_hidden': 448, 'fusion_type': 'concat', 'dropout': 0.357}

    model = HybridDeepNet(
        fusion_type=config['fusion_type'],
        fc_hidden=config['fc_hidden'],
        dropout=config['dropout']
    ).to(device)

    model.load_state_dict(torch.load('best_optimized_hybrid_focal.pth', map_location=device))
    model.eval()
    return model, device


# Wrapper for Grad-CAM
class HybridClassificationWrapper(torch.nn.Module):
    def __init__(self, hybrid_model):
        super().__init__()
        self.hybrid_model = hybrid_model

    def forward(self, x):
        cls_out, _ = self.hybrid_model(x)
        return cls_out


def get_transforms():
    # Match training val preprocessing: Resize to 256 first, then CenterCrop to 224
    return A.Compose([
        A.Resize(256, 256),
        A.CenterCrop(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def classify_lesion(raw_probs: np.ndarray, sev_tier_raw: int):
    """
    Full decision pipeline: prior correction + asymmetric threshold scan.

    We do NOT rely on argmax alone. Each malignant class is independently
    checked against a low threshold on the prior-corrected distribution.
    This prevents the nv-dominant model from suppressing melanoma signals.
    """
    corrected_probs = prior_correct(raw_probs)

    # Best class from corrected distribution
    pred_class_idx = int(np.argmax(corrected_probs))
    pred_class = CLASS_NAMES[pred_class_idx]
    pred_confidence = float(corrected_probs[pred_class_idx])

    sev_tier = sev_tier_raw
    override_reason = None
    malignancy_flags = {}

    # ── Asymmetric threshold scan ─────────────────────────────────────────────
    # Independently check each malignant class against its own threshold.
    # A class CAN trigger the flag even if it is not the top-1 prediction.
    # e.g. mel=28% corrected, nv=40% -> mel still triggers the malignancy flag.
    for cls, threshold in MALIGNANT_THRESHOLDS.items():
        cls_idx = CLASS_NAMES.index(cls)
        cls_prob = float(corrected_probs[cls_idx])
        if cls_prob >= threshold:
            malignancy_flags[cls] = cls_prob

    # ── Severity override logic ───────────────────────────────────────────────
    if malignancy_flags:
        top_malignant = max(malignancy_flags, key=malignancy_flags.get)
        top_prob = malignancy_flags[top_malignant]

        if top_malignant == 'mel' and sev_tier < 3:
            sev_tier = 3
            override_reason = f"mel flagged ({top_prob:.1%} > {MALIGNANT_THRESHOLDS['mel']:.0%} threshold)"
        elif top_malignant == 'bcc' and sev_tier < 2:
            sev_tier = 2
            override_reason = f"bcc flagged ({top_prob:.1%} > {MALIGNANT_THRESHOLDS['bcc']:.0%} threshold)"
        elif top_malignant == 'akiec' and sev_tier < 1:
            sev_tier = 1
            override_reason = f"akiec flagged ({top_prob:.1%} > {MALIGNANT_THRESHOLDS['akiec']:.0%} threshold)"

    is_malignant = sev_tier >= 2

    return pred_class, pred_confidence, corrected_probs, sev_tier, malignancy_flags, is_malignant, override_reason


# --- UI LAYOUT ---

tab_diag = st.tabs(["Diagnostics & XAI"])

# --- TAB 1: DIAGNOSTICS ---
with tab_diag[0]:
    st.header("Lesion Analysis Pipeline")

    col1, col2 = st.columns([1, 2])

    with col1:
        uploaded_file = st.file_uploader("Upload Dermoscopic Image (JPG/PNG)", type=['jpg', 'jpeg', 'png'])
        show_gradcam = st.toggle("Generate Grad-CAM XAI Heatmap")

    if uploaded_file is not None:
        # Read and preprocess
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transform = get_transforms()
        input_tensor = transform(image=image_rgb)['image'].unsqueeze(0)

        model, device = load_model()
        input_tensor = input_tensor.to(device)

        # Inference — only use autocast on CUDA to avoid CPU dtype issues
        with torch.no_grad():
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    cls_preds, sev_preds = model(input_tensor)
            else:
                cls_preds, sev_preds = model(input_tensor)

            # Raw softmax probabilities (nv-biased due to class imbalance)
            raw_probs = torch.nn.functional.softmax(cls_preds.float(), dim=1)[0].cpu().numpy()
            sev_tier_raw = torch.argmax(sev_preds.float(), dim=1).item()

        # --- FULL DECISION PIPELINE ---
        (pred_class, pred_confidence, corrected_probs,
         sev_tier, malignancy_flags, is_malignant, override_reason) = classify_lesion(raw_probs, sev_tier_raw)

        # Severity Labels
        sev_labels = {
            0: ("Benign / Low Risk", "green"),
            1: ("Pre-malignant (Actinic Keratosis)", "orange"),
            2: ("Non-Melanoma Skin Cancer", "red"),
            3: ("Malignant Melanoma", "darkred")
        }

        verdict_text = "MALIGNANT" if is_malignant else "BENIGN"
        verdict_color = "red" if is_malignant else "green"

        if override_reason:
            st.toast(f"Malignancy threshold triggered: {override_reason}", icon="⚠️")

        # UI Rendering
        with col2:
            st.markdown(
                f"### Binary Verdict: <span style='color:{verdict_color}'>{verdict_text}</span>",
                unsafe_allow_html=True
            )
            st.markdown(f"**Top Predicted Class:** `{pred_class}` ({pred_confidence:.1%} corrected confidence)")
            st.markdown(f"**Severity Tier {sev_tier}:** {sev_labels[sev_tier][0]}")

            if malignancy_flags:
                flagged_str = ", ".join(
                    f"`{cls}` ({prob:.1%})" for cls, prob in malignancy_flags.items()
                )
                st.warning(f"⚠️ Malignancy thresholds exceeded for: {flagged_str}")

            # Show both raw and corrected probabilities for transparency
            st.write("**Prior-Corrected Class Probabilities** *(bias-adjusted for class imbalance)*:")
            df_probs = pd.DataFrame({
                'Corrected (use this)': corrected_probs,
                'Raw model output': raw_probs
            }, index=CLASS_NAMES)
            st.bar_chart(df_probs)

        # ── XAI / Image Analysis ────────────────────────────────────────────
        st.subheader("Image Analysis")
        st.image(image_rgb, caption="Original Upload", use_container_width=True)

        if show_gradcam:
            with st.spinner("Generating Grad-CAM++ explanation..."):
                wrapped_model = HybridClassificationWrapper(model).to(device)
                wrapped_model.eval()

                # ── Layer choice ────────────────────────────────────────────
                # blocks[5][-1].conv_pwl → 14×14 spatial maps (2× finer than
                # conv_head's 7×7). Higher resolution = sharper heatmaps.
                target_layers = [model.cnn.blocks[5][-1].conv_pwl]

                # ── Determine which class to explain ────────────────────────
                # Always explain the clinically most relevant class:
                # flagged malignant class > prior-corrected top class
                if malignancy_flags:
                    gradcam_class = max(malignancy_flags, key=malignancy_flags.get)
                else:
                    gradcam_class = pred_class
                gradcam_class_idx = CLASS_NAMES.index(gradcam_class)

                # Also always generate nv map for direct comparison
                nv_idx = CLASS_NAMES.index('nv')

                # ── GradCAM++ with both smoothing modes ─────────────────────
                # aug_smooth   → averages over augmented versions (robust)
                # eigen_smooth → uses PCA on feature maps (denoises)
                cam = GradCAMPlusPlus(model=wrapped_model, target_layers=target_layers)

                img_resized = cv2.resize(image_rgb, (224, 224))
                img_norm = np.float32(img_resized) / 255.0

                # Primary heatmap (flagged/pred class)
                heatmap_primary = cam(
                    input_tensor=input_tensor,
                    targets=[ClassifierOutputTarget(gradcam_class_idx)],
                    aug_smooth=True,
                    eigen_smooth=True
                )[0]

                # Comparison heatmap (nv — the benign baseline)
                heatmap_nv = cam(
                    input_tensor=input_tensor,
                    targets=[ClassifierOutputTarget(nv_idx)],
                    aug_smooth=True,
                    eigen_smooth=True
                )[0]

                # ── Attention statistics ─────────────────────────────────────
                # Centroid: where is the model looking?
                h, w = heatmap_primary.shape
                ys, xs = np.mgrid[0:h, 0:w]
                total = heatmap_primary.sum() + 1e-8
                cx = int((xs * heatmap_primary).sum() / total)
                cy = int((ys * heatmap_primary).sum() / total)
                cx_pct = cx / w * 100
                cy_pct = cy / h * 100

                # Focus score: what fraction of attention is in the top 25% of pixels?
                threshold = np.percentile(heatmap_primary, 75)
                focus_score = float((heatmap_primary >= threshold).mean()) * 100
                # A focused model has a tight hotspot → low % area with high activation
                # (closer to 25% is good; much higher = diffuse, unfocused)
                focus_label = "Focused" if focus_score <= 32 else "Diffuse"

                # ── Build matplotlib figure: 4 panels ───────────────────────
                CLASS_LABELS = {
                    'mel': 'Melanoma (mel)', 'bcc': 'Basal Cell Carcinoma (bcc)',
                    'akiec': 'Actinic Keratosis (akiec)', 'bkl': 'Benign Keratosis (bkl)',
                    'df': 'Dermatofibroma (df)', 'nv': 'Melanocytic Nevi (nv)',
                    'vasc': 'Vascular Lesion (vasc)'
                }

                fig, axes = plt.subplots(1, 4, figsize=(18, 5),
                                         facecolor='#0e1117')
                fig.suptitle(
                    f"XAI Panel  |  Explaining '{CLASS_LABELS.get(gradcam_class, gradcam_class)}'  "
                    f"vs '{CLASS_LABELS.get('nv', 'nv')}'\n"
                    f"Attention centroid: ({cx_pct:.0f}%, {cy_pct:.0f}%)  |  "
                    f"Focus: {focus_label}",
                    color='white', fontsize=11, y=1.01
                )

                panel_titles = [
                    "① Original",
                    f"② GradCAM++\n→ '{gradcam_class}' features",
                    f"③ GradCAM++\n→ 'nv' (benign baseline)",
                    "④ Binary Attention Mask\n(Top 25% activations)"
                ]

                # Panel 1 — Original
                axes[0].imshow(img_resized)
                axes[0].plot(cx, cy, 'r+', markersize=14, markeredgewidth=2,
                             label=f'Centroid ({cx_pct:.0f}%, {cy_pct:.0f}%)')
                axes[0].legend(loc='lower left', fontsize=7,
                               facecolor='#333', labelcolor='white')

                # Panel 2 — Primary GradCAM++ overlay
                overlay_primary = show_cam_on_image(img_norm, heatmap_primary, use_rgb=True)
                axes[1].imshow(overlay_primary)
                # Add colorbar-style annotation
                im2 = axes[1].imshow(heatmap_primary, alpha=0, cmap='jet')
                plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04,
                             label='Activation').ax.yaxis.label.set_color('white')

                # Panel 3 — nv GradCAM++ overlay (for comparison)
                overlay_nv = show_cam_on_image(img_norm, heatmap_nv, use_rgb=True)
                axes[2].imshow(overlay_nv)
                im3 = axes[2].imshow(heatmap_nv, alpha=0, cmap='jet')
                plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04,
                             label='Activation').ax.yaxis.label.set_color('white')

                # Panel 4 — Binary mask of top activations
                mask = (heatmap_primary >= threshold).astype(np.float32)
                # Show original dimmed + bright mask
                axes[3].imshow(img_resized)
                axes[3].imshow(
                    np.stack([mask, np.zeros_like(mask), np.zeros_like(mask), mask * 0.6], axis=-1)
                )

                # Style all axes
                for ax, title in zip(axes, panel_titles):
                    ax.set_title(title, color='white', fontsize=9, pad=4)
                    ax.axis('off')
                    for spine in ax.spines.values():
                        spine.set_edgecolor('#444')

                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

                # ── Textual interpretation ───────────────────────────────────
                with st.expander("📖 How to read this XAI panel", expanded=False):
                    st.markdown("""
**① Original** — The uploaded image with the attention centroid marked as a red cross (+).

**② GradCAM++ → flagged class** — Shows *which pixels drive the model toward the suspected 
diagnosis*. Warm colors (red/yellow) = high importance. In a well-calibrated model, the hot 
spot should sit on the lesion itself. If it's on the background/watermark, the model may be 
using a spurious feature.

**③ GradCAM++ → nv (benign baseline)** — Shows what the model looks at when thinking 
*"this is a benign mole"*. Compare panels ② and ③: if ② highlights the lesion core and 
③ highlights background, the separation is clinically sensible.

**④ Binary Attention Mask** — The top 25% of activations from panel ② highlighted in red.
A tight cluster over the lesion = focused, reliable explanation. 
A scattered mask or one centered on watermarks/borders = unreliable explanation.
                    """)

                # ── Attention quality warning ────────────────────────────────
                # If centroid is in the bottom 30% or right 30% of image,
                # warn the user (watermark is bottom-right for VisualDx images)
                if cy_pct > 70 or cx_pct > 70:
                    st.warning(
                        "⚠️ **XAI Warning:** The model's attention centroid is near the "
                        "image border — it may be responding to a watermark or border artifact "
                        "rather than the lesion. Treat this explanation with caution."
                    )

# --- TAB 2: HPO METRICS ---
