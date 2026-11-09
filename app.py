# from pathlib import Path
# import json
# import numpy as np
# import streamlit as st
# import torch
# from PIL import Image
# from torchvision import transforms
# from coffee_leaf.model import build_classifier,UNet
# from coffee_leaf.data import MEAN,STD

# st.set_page_config(page_title='Coffee Leaf Disease Detection',page_icon='🌿')
# st.title('🌿 Coffee Leaf Disease Detection')
# st.write('Upload a coffee leaf photo for classification: Phoma, Cercospora, Rust, or Healthy.')
# st.caption('Research prototype. Predictions may be wrong and should not replace expert assessment.')
# art=Path('artifacts'); cp=art/'mobilenetv3_coffee.pth'; up=art/'unet_coffee.pth'
# dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# @st.cache_resource
# def classifier():
#     if not cp.exists(): return None,None,None
#     ck=torch.load(cp,map_location=dev); names=ck['class_names']
#     m=build_classifier(len(names),pretrained=False).to(dev); m.load_state_dict(ck['model_state_dict']); m.eval()
#     tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
#     return m,names,tf

# @st.cache_resource
# def segmenter():
#     if not up.exists(): return None,None
#     ck=torch.load(up,map_location=dev); m=UNet().to(dev); m.load_state_dict(ck['model_state_dict']); m.eval()
#     tf=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
#     return m,tf

# f=st.file_uploader('Choose leaf image',type=['jpg','jpeg','png','webp'])
# if f:
#     im=Image.open(f).convert('RGB'); st.image(im,caption='Uploaded image',use_container_width=True)
#     m,names,tf=classifier()
#     if m is None: st.warning('Classifier checkpoint missing. Train first with train_classifier.py.')
#     else:
#         with torch.no_grad(): probs=torch.softmax(m(tf(im).unsqueeze(0).to(dev)),1)[0].cpu().numpy()
#         ix=int(np.argmax(probs)); st.subheader('Prediction: '+names[ix].replace('_',' ').title())
#         st.metric('Model confidence',f'{probs[ix]*100:.1f}%')
#         st.write('Class scores'); st.bar_chart({n.title():float(v) for n,v in zip(names,probs)})
#     seg,stf=segmenter()
#     if seg is None:
#         st.info('Segmentation unavailable: U-Net requires annotated pixel masks. Class folders alone are not enough.')
#     else:
#         with torch.no_grad(): mask=(torch.sigmoid(seg(stf(im).unsqueeze(0).to(dev)))[0,0].cpu().numpy()>=.5)
#         small=im.resize((256,256)); arr=np.array(small).copy(); hsv=np.array(small.convert('HSV'))
#         leaf=(hsv[:,:,0]>20)&(hsv[:,:,0]<100)&(hsv[:,:,1]>25)&(hsv[:,:,2]>25)
#         affected=mask&leaf; pct=100*affected.sum()/max(leaf.sum(),1)
#         arr[affected]=[255,30,30]
#         st.subheader('Affected-area segmentation'); st.image(arr,caption='Red = predicted affected pixels (green-pixel leaf-region heuristic)',use_container_width=True)
#         st.metric('Estimated affected leaf area',f'{pct:.1f}%')
#         st.caption('Area estimate depends on the segmentation model and heuristic leaf mask; validate before using it.')
# else:
#     st.info('Upload an image to begin.')

# from pathlib import Path

# import numpy as np
# import streamlit as st
# import torch

# from PIL import Image
# from torchvision import transforms

# from coffee_leaf.model import (
#     build_classifier,
#     build_efficientnet_b0,
#     UNet
# )

# from coffee_leaf.data import MEAN, STD


# # =====================================================
# # 1. PAGE CONFIGURATION
# # =====================================================

# st.set_page_config(
#     page_title="Coffee Leaf Disease Detection",
#     page_icon="🌿",
#     layout="wide"
# )

# st.title("🌿 Coffee Leaf Disease Detection")
# st.write(
#     "Coffee Leaf Disease Detection, Segmentation "
#     "and Severity Estimation"
# )

# st.caption(
#     "Upload a coffee leaf image to identify diseases, "
#     "visualize affected regions and estimate disease severity."
# )


# # =====================================================
# # 2. PATHS AND DEVICE
# # =====================================================

# ARTIFACTS = Path("artifacts")

# CHECKPOINTS = {
#     "MobileNetV3": ARTIFACTS / "mobilenetv3_coffee.pth",
#     "EfficientNet-B0": ARTIFACTS / "efficientnet_b0_coffee.pth"
# }

# UNET_CHECKPOINT = ARTIFACTS / "unet_coffee.pth"

# DEVICE = torch.device(
#     "cuda" if torch.cuda.is_available() else "cpu"
# )


# # =====================================================
# # 3. IMAGE TRANSFORM
# # =====================================================

# CLASSIFIER_TRANSFORM = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(MEAN, STD)
# ])

# SEGMENTATION_TRANSFORM = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
#     transforms.Normalize(MEAN, STD)
# ])


# # =====================================================
# # 4. LOAD CLASSIFICATION MODELS
# # =====================================================

# @st.cache_resource
# def load_classifier(model_name):

#     checkpoint_path = CHECKPOINTS[model_name]

#     if not checkpoint_path.exists():
#         return None, None

#     checkpoint = torch.load(
#         checkpoint_path,
#         map_location=DEVICE,
#         weights_only=False
#     )

#     class_names = checkpoint["class_names"]

#     # Select the correct architecture

#     if model_name == "MobileNetV3":

#         model = build_classifier(
#             len(class_names),
#             pretrained=False
#         ).to(DEVICE)

#     elif model_name == "EfficientNet-B0":

#         model = build_efficientnet_b0(
#             len(class_names),
#             pretrained=False
#         ).to(DEVICE)

#     else:
#         return None, None

#     model.load_state_dict(
#         checkpoint["model_state_dict"]
#     )

#     model.eval()

#     return model, class_names


# # =====================================================
# # 5. LOAD U-NET MODEL
# # =====================================================

# @st.cache_resource
# def load_unet():

#     if not UNET_CHECKPOINT.exists():
#         return None

#     checkpoint = torch.load(
#         UNET_CHECKPOINT,
#         map_location=DEVICE,
#         weights_only=False
#     )

#     model = UNet().to(DEVICE)

#     model.load_state_dict(
#         checkpoint["model_state_dict"]
#     )

#     model.eval()

#     return model


# # =====================================================
# # 6. CLASSIFICATION FUNCTION
# # =====================================================

# def predict_disease(image, model_name):

#     model, class_names = load_classifier(model_name)

#     if model is None:
#         return None

#     image_tensor = CLASSIFIER_TRANSFORM(
#         image
#     ).unsqueeze(0).to(DEVICE)

#     with torch.no_grad():

#         outputs = model(image_tensor)

#         probabilities = torch.softmax(
#             outputs,
#             dim=1
#         )[0].cpu().numpy()

#     predicted_index = int(
#         np.argmax(probabilities)
#     )

#     predicted_disease = class_names[
#         predicted_index
#     ].replace("_", " ").title()

#     confidence = float(
#         probabilities[predicted_index] * 100
#     )

#     scores = {
#         name.replace("_", " ").title():
#         float(probability * 100)

#         for name, probability in zip(
#             class_names,
#             probabilities
#         )
#     }

#     return {
#         "disease": predicted_disease,
#         "confidence": confidence,
#         "scores": scores
#     }


# # =====================================================
# # 7. SEGMENTATION FUNCTION
# # =====================================================

# def predict_segmentation(image):

#     model = load_unet()

#     if model is None:
#         return None

#     image_tensor = SEGMENTATION_TRANSFORM(
#         image
#     ).unsqueeze(0).to(DEVICE)

#     with torch.no_grad():

#         output = model(image_tensor)

#         mask = (
#             torch.sigmoid(output)[0, 0]
#             .cpu()
#             .numpy() >= 0.5
#         )

#     return mask


# # =====================================================
# # 8. AFFECTED AREA CALCULATION
# # =====================================================

# def calculate_affected_area(image, mask):

#     small_image = image.resize((256, 256))

#     image_array = np.array(
#         small_image
#     ).copy()

#     hsv = np.array(
#         small_image.convert("HSV")
#     )

#     # Approximate leaf region using green pixels.
#     # This is a heuristic and needs validation.

#     leaf_mask = (
#         (hsv[:, :, 0] > 20) &
#         (hsv[:, :, 0] < 100) &
#         (hsv[:, :, 1] > 25) &
#         (hsv[:, :, 2] > 25)
#     )

#     affected_mask = mask & leaf_mask

#     total_leaf_pixels = leaf_mask.sum()

#     affected_pixels = affected_mask.sum()

#     if total_leaf_pixels == 0:
#         return None, image_array

#     affected_percentage = (
#         affected_pixels /
#         total_leaf_pixels
#     ) * 100

#     # Highlight affected regions in red.

#     image_array[affected_mask] = [
#         255, 30, 30
#     ]

#     return affected_percentage, image_array


# # =====================================================
# # 9. SEVERITY ESTIMATION
# # =====================================================

# def estimate_severity(percentage):

#     # Provisional thresholds only.
#     # Validate these before research use.

#     if percentage < 10:
#         return "Mild"

#     elif percentage < 30:
#         return "Moderate"

#     else:
#         return "Severe"


# # =====================================================
# # 10. SIDEBAR
# # =====================================================

# st.sidebar.title("Project Information")

# st.sidebar.write(
#     "**Project:** Coffee Leaf Disease Detection"
# )

# st.sidebar.write(
#     "**Classification Models:**"
# )

# st.sidebar.write("- MobileNetV3")
# st.sidebar.write("- EfficientNet-B0")

# st.sidebar.write(
#     "**Segmentation Model:** U-Net"
# )

# st.sidebar.divider()

# st.sidebar.write(
#     f"Running device: {DEVICE}"
# )

# st.sidebar.caption(
#     "Research prototype. Predictions require "
#     "validation before practical use."
# )


# # =====================================================
# # 11. MODEL SELECTION
# # =====================================================

# st.divider()

# st.header("Select Classification Model")

# if "selected_model" not in st.session_state:

#     st.session_state.selected_model = "MobileNetV3"

# if "compare_models" not in st.session_state:

#     st.session_state.compare_models = False


# col1, col2, col3 = st.columns(3)

# with col1:

#     if st.button(
#         "MobileNetV3",
#         use_container_width=True,
#         type=(
#             "primary"
#             if st.session_state.selected_model == "MobileNetV3"
#             else "secondary"
#         )
#     ):

#         st.session_state.selected_model = "MobileNetV3"
#         st.session_state.compare_models = False


# with col2:

#     if st.button(
#         "EfficientNet-B0",
#         use_container_width=True,
#         type=(
#             "primary"
#             if st.session_state.selected_model == "EfficientNet-B0"
#             else "secondary"
#         )
#     ):

#         st.session_state.selected_model = "EfficientNet-B0"
#         st.session_state.compare_models = False


# with col3:

#     if st.button(
#         "Compare Both Models",
#         use_container_width=True,
#         type="primary" if st.session_state.compare_models else "secondary"
#     ):

#         st.session_state.compare_models = True


# # =====================================================
# # 12. DISPLAY SELECTED MODEL
# # =====================================================

# if st.session_state.compare_models:

#     st.success(
#         "Comparison Mode: Both models will test the same image."
#     )

# else:

#     st.success(
#         f"Selected Model: {st.session_state.selected_model}"
#     )


# # =====================================================
# # 13. IMAGE UPLOAD
# # =====================================================

# st.divider()

# st.header("Upload Coffee Leaf Image")

# uploaded_file = st.file_uploader(
#     "Choose a coffee leaf image",
#     type=["jpg", "jpeg", "png", "webp"],
#     key="coffee_leaf_upload"
# )


# # =====================================================
# # 14. PROCESS UPLOADED IMAGE
# # =====================================================

# if uploaded_file is not None:

#     try:

#         image = Image.open(
#             uploaded_file
#         ).convert("RGB")

#     except Exception:

#         st.error(
#             "Unable to read this image. Please upload a valid image."
#         )

#         st.stop()

#     st.subheader("Uploaded Image")

#     st.image(
#         image,
#         caption="Coffee Leaf Image",
#         use_container_width=True
#     )

#     # -------------------------------------------------
#     # 15. CLASSIFICATION RESULTS
#     # -------------------------------------------------

#     st.divider()

#     st.header("Disease Classification Results")

#     if st.session_state.compare_models:

#         selected_models = [
#             "MobileNetV3",
#             "EfficientNet-B0"
#         ]

#     else:

#         selected_models = [
#             st.session_state.selected_model
#         ]

#     results = {}

#     for model_name in selected_models:

#         st.subheader(model_name)

#         result = predict_disease(
#             image,
#             model_name
#         )

#         if result is None:

#             st.error(
#                 f"{model_name} checkpoint is missing. "
#                 "Please train this model first."
#             )

#             continue

#         results[model_name] = result

#         col1, col2 = st.columns(2)

#         with col1:

#             st.metric(
#                 "Predicted Disease",
#                 result["disease"]
#             )

#         with col2:

#             st.metric(
#                 "Model Confidence",
#                 f'{result["confidence"]:.2f}%'
#             )

#         st.write("Class-wise Prediction Scores")

#         st.bar_chart(
#             result["scores"]
#         )

#         st.divider()

#     # -------------------------------------------------
#     # 16. MODEL COMPARISON TABLE
#     # -------------------------------------------------

#     if (
#         st.session_state.compare_models
#         and len(results) == 2
#     ):

#         st.header("Model Comparison")

#         comparison_data = {

#             "MobileNetV3": [
#                 results["MobileNetV3"]["disease"],
#                 f'{results["MobileNetV3"]["confidence"]:.2f}%'
#             ],

#             "EfficientNet-B0": [
#                 results["EfficientNet-B0"]["disease"],
#                 f'{results["EfficientNet-B0"]["confidence"]:.2f}%'
#             ]

#         }

#         st.table({

#             "Evaluation": [
#                 "Predicted Disease",
#                 "Prediction Confidence"
#             ],

#             **comparison_data

#         })

#         if (
#             results["MobileNetV3"]["disease"]
#             == results["EfficientNet-B0"]["disease"]
#         ):

#             st.success(
#                 "Both models predicted the same disease."
#             )

#         else:

#             st.warning(
#                 "The models predicted different diseases. "
#                 "Further expert verification is recommended."
#             )

#     # -------------------------------------------------
#     # 17. U-NET SEGMENTATION
#     # -------------------------------------------------

#     st.divider()

#     st.header("Disease Segmentation")

#     segmentation_mask = predict_segmentation(
#         image
#     )

#     if segmentation_mask is None:

#         st.warning(
#             "U-Net checkpoint is missing. "
#             "Segmentation is unavailable until U-Net is trained."
#         )

#     else:

#         percentage, highlighted_image = (
#             calculate_affected_area(
#                 image,
#                 segmentation_mask
#             )
#         )

#         if percentage is None:

#             st.warning(
#                 "Leaf region could not be detected. "
#                 "Please upload a clearer leaf image."
#             )

#         else:

#             col1, col2 = st.columns(2)

#             with col1:

#                 st.subheader("Original Image")

#                 st.image(
#                     image,
#                     use_container_width=True
#                 )

#             with col2:

#                 st.subheader("Segmented Image")

#                 st.image(
#                     highlighted_image,
#                     caption="Red = Predicted affected region",
#                     use_container_width=True
#                 )

#             st.metric(
#                 "Estimated Affected Leaf Area",
#                 f"{percentage:.2f}%"
#             )

#             # -----------------------------------------
#             # 18. SEVERITY ESTIMATION
#             # -----------------------------------------

#             st.divider()

#             st.header("Disease Severity Estimation")

#             severity = estimate_severity(
#                 percentage
#             )

#             st.subheader(
#                 f"Estimated Severity: {severity}"
#             )

#             if severity == "Mild":

#                 st.info(
#                     "Low estimated affected area. "
#                     "Monitor the leaf and validate the symptoms."
#                 )

#             elif severity == "Moderate":

#                 st.warning(
#                     "Moderate estimated affected area. "
#                     "Further inspection is recommended."
#                 )

#             else:

#                 st.error(
#                     "High estimated affected area. "
#                     "Expert assessment is recommended."
#                 )

#             st.caption(
#                 "Severity thresholds are provisional. "
#                 "Validate them using ground-truth annotations. "
#                 "Area estimation depends on U-Net predictions "
#                 "and the heuristic leaf mask."
#             )


# # =====================================================
# # 19. FOOTER
# # =====================================================

# else:

#     st.info(
#         "Select a classification model and upload a "
#         "coffee leaf image to begin."
#     )


# st.divider()

# st.caption(
#     "MCA Major Project | Coffee Leaf Disease Detection, "
#     "Segmentation and Severity Estimation"
# )

from pathlib import Path

import numpy as np
import streamlit as st
import torch

from PIL import Image
from torchvision import transforms

from coffee_leaf.model import (
    build_classifier,
    build_efficientnet_b0,
    UNet
)

from coffee_leaf.data import MEAN, STD


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Coffee Leaf Disease Detection",
    page_icon="🌿",
    layout="wide"
)

st.title("🌿 Coffee Leaf Disease Detection")

st.write(
    "Coffee Leaf Disease Detection, Segmentation "
    "and Severity Estimation"
)

st.caption(
    "Upload a coffee leaf image to identify diseases, "
    "visualize affected regions and estimate severity."
)


# =====================================================
# 2. PATHS AND DEVICE
# =====================================================

ARTIFACTS = Path("artifacts")

CHECKPOINTS = {
    "MobileNetV3": ARTIFACTS / "mobilenetv3_coffee.pth",
    "EfficientNet-B0": ARTIFACTS / "efficientnet_b0_coffee.pth"
}

UNET_CHECKPOINT = ARTIFACTS / "unet_coffee.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =====================================================
# 3. IMAGE TRANSFORMATIONS
# =====================================================

CLASSIFIER_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

SEGMENTATION_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])


# =====================================================
# 4. LOAD CLASSIFICATION MODELS
# =====================================================

@st.cache_resource
def load_classifier(model_name):

    checkpoint_path = CHECKPOINTS[model_name]

    if not checkpoint_path.exists():
        return None, None

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False
    )

    class_names = checkpoint["class_names"]

    if model_name == "MobileNetV3":

        model = build_classifier(
            len(class_names),
            pretrained=False
        ).to(DEVICE)

    elif model_name == "EfficientNet-B0":

        model = build_efficientnet_b0(
            len(class_names),
            pretrained=False
        ).to(DEVICE)

    else:
        return None, None

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, class_names


# =====================================================
# 5. LOAD U-NET SEGMENTATION MODEL
# =====================================================

@st.cache_resource
def load_unet():

    if not UNET_CHECKPOINT.exists():
        return None

    checkpoint = torch.load(
        UNET_CHECKPOINT,
        map_location=DEVICE,
        weights_only=False
    )

    model = UNet().to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


# =====================================================
# 6. DISEASE CLASSIFICATION FUNCTION
# =====================================================

def predict_disease(image, model_name):

    model, class_names = load_classifier(model_name)

    if model is None:
        return None

    image_tensor = CLASSIFIER_TRANSFORM(
        image
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0].cpu().numpy()

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_disease = class_names[
        predicted_index
    ].replace("_", " ").title()

    confidence = float(
        probabilities[predicted_index] * 100
    )

    scores = {
        name.replace("_", " ").title():
        float(probability * 100)

        for name, probability in zip(
            class_names,
            probabilities
        )
    }

    return {
        "disease": predicted_disease,
        "confidence": confidence,
        "scores": scores
    }


# =====================================================
# 7. SEGMENTATION FUNCTION
# =====================================================

def predict_segmentation(image):

    model = load_unet()

    if model is None:
        return None

    image_tensor = SEGMENTATION_TRANSFORM(
        image
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        output = model(image_tensor)

        mask = (
            torch.sigmoid(output)[0, 0]
            .cpu()
            .numpy() >= 0.5
        )

    return mask


# =====================================================
# 8. AFFECTED AREA CALCULATION
# =====================================================

def calculate_affected_area(image, mask):

    small_image = image.resize((256, 256))

    image_array = np.array(
        small_image
    ).copy()

    hsv = np.array(
        small_image.convert("HSV")
    )

    # Approximate green leaf region

    leaf_mask = (
        (hsv[:, :, 0] > 20) &
        (hsv[:, :, 0] < 100) &
        (hsv[:, :, 1] > 25) &
        (hsv[:, :, 2] > 25)
    )

    affected_mask = mask & leaf_mask

    total_leaf_pixels = leaf_mask.sum()

    affected_pixels = affected_mask.sum()

    if total_leaf_pixels == 0:

        return None, image_array

    percentage = (
        affected_pixels / total_leaf_pixels
    ) * 100

    # Highlight affected pixels in red

    image_array[affected_mask] = [
        255, 30, 30
    ]

    return percentage, image_array


# =====================================================
# 9. SEVERITY ESTIMATION
# =====================================================

def estimate_severity(percentage):

    # Provisional thresholds.
    # Validate using annotated research data.

    if percentage < 10:

        return "Mild"

    elif percentage < 30:

        return "Moderate"

    else:

        return "Severe"


# =====================================================
# 10. SIDEBAR
# =====================================================

st.sidebar.title("Project Information")

st.sidebar.write(
    "**Project:** Coffee Leaf Disease Detection"
)

st.sidebar.write(
    "**Classification Models:**"
)

st.sidebar.write("- MobileNetV3")
st.sidebar.write("- EfficientNet-B0")

st.sidebar.write(
    "**Segmentation Model:** U-Net"
)

st.sidebar.divider()

st.sidebar.write(
    f"Device: {DEVICE}"
)

st.sidebar.caption(
    "Research prototype. Predictions require validation."
)


# =====================================================
# 11. MODEL SELECTION BUTTONS
# =====================================================

st.divider()

st.header("Select Classification Model")


# Initialize session state

if "selected_model" not in st.session_state:

    st.session_state.selected_model = "MobileNetV3"

if "compare_models" not in st.session_state:

    st.session_state.compare_models = False


col1, col2, col3 = st.columns(3)


# MobileNetV3 button

with col1:

    if st.button(
        "MobileNetV3",
        use_container_width=True,
        type=(
            "primary"
            if (
                st.session_state.selected_model == "MobileNetV3"
                and not st.session_state.compare_models
            )
            else "secondary"
        )
    ):

        st.session_state.selected_model = "MobileNetV3"

        st.session_state.compare_models = False

        st.rerun()


# EfficientNet-B0 button

with col2:

    if st.button(
        "EfficientNet-B0",
        use_container_width=True,
        type=(
            "primary"
            if (
                st.session_state.selected_model == "EfficientNet-B0"
                and not st.session_state.compare_models
            )
            else "secondary"
        )
    ):

        st.session_state.selected_model = "EfficientNet-B0"

        st.session_state.compare_models = False

        st.rerun()


# Compare Both Models button

with col3:

    if st.button(
        "Compare Both Models",
        use_container_width=True,
        type=(
            "primary"
            if st.session_state.compare_models
            else "secondary"
        )
    ):

        st.session_state.compare_models = True

        st.rerun()


# =====================================================
# 12. DISPLAY SELECTED MODEL
# =====================================================

if st.session_state.compare_models:

    st.success(
        "Comparison Mode: Both models will test the same image."
    )

else:

    st.success(
        f"Selected Model: {st.session_state.selected_model}"
    )


# =====================================================
# 13. IMAGE UPLOAD
# =====================================================

st.divider()

st.header("Upload Coffee Leaf Image")

if st.session_state.compare_models:

    st.write(
        "Upload one image to test both classification models."
    )

else:

    st.write(
        f"Upload an image to test {st.session_state.selected_model}."
    )


uploaded_file = st.file_uploader(
    "Choose a coffee leaf image",
    type=["jpg", "jpeg", "png", "webp"],
    key="coffee_leaf_upload"
)


# =====================================================
# 14. PROCESS UPLOADED IMAGE
# =====================================================

if uploaded_file is not None:

    try:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

    except Exception:

        st.error(
            "Unable to read the image. Please upload a valid image."
        )

        st.stop()


    st.subheader("Uploaded Coffee Leaf Image")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )


    # =================================================
    # 15. CLASSIFICATION RESULTS
    # =================================================

    st.divider()

    st.header("Disease Classification Results")


    if st.session_state.compare_models:

        selected_models = [
            "MobileNetV3",
            "EfficientNet-B0"
        ]

    else:

        selected_models = [
            st.session_state.selected_model
        ]


    results = {}


    for model_name in selected_models:

        st.subheader(model_name)

        result = predict_disease(
            image,
            model_name
        )

        if result is None:

            st.error(
                f"{model_name} checkpoint is missing. "
                "Please train this model first."
            )

            continue


        results[model_name] = result


        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Predicted Disease",
                result["disease"]
            )

        with col2:

            st.metric(
                "Model Confidence",
                f'{result["confidence"]:.2f}%'
            )


        st.write("Class-wise Prediction Scores")

        st.bar_chart(
            result["scores"]
        )

        st.divider()


    # =================================================
    # 16. COMPARE BOTH MODELS
    # =================================================

    if (
        st.session_state.compare_models
        and len(results) == 2
    ):

        st.header("Model Comparison")

        st.table({

            "Evaluation": [
                "Predicted Disease",
                "Prediction Confidence"
            ],

            "MobileNetV3": [
                results["MobileNetV3"]["disease"],
                f'{results["MobileNetV3"]["confidence"]:.2f}%'
            ],

            "EfficientNet-B0": [
                results["EfficientNet-B0"]["disease"],
                f'{results["EfficientNet-B0"]["confidence"]:.2f}%'
            ]

        })


        if (
            results["MobileNetV3"]["disease"]
            == results["EfficientNet-B0"]["disease"]
        ):

            st.success(
                "Both models predicted the same disease."
            )

        else:

            st.warning(
                "Both models predicted different diseases. "
                "Expert verification is recommended."
            )


    # =================================================
    # 17. DISEASE SEGMENTATION
    # =================================================

    st.divider()

    st.header("Disease Segmentation")

    segmentation_mask = predict_segmentation(
        image
    )


    if segmentation_mask is None:

        st.warning(
            "U-Net checkpoint is missing. "
            "Train U-Net using annotated segmentation masks."
        )

    else:

        percentage, highlighted_image = (
            calculate_affected_area(
                image,
                segmentation_mask
            )
        )


        if percentage is None:

            st.warning(
                "Leaf region could not be detected. "
                "Please upload a clearer leaf image."
            )

        else:

            col1, col2 = st.columns(2)

            with col1:

                st.subheader("Original Image")

                st.image(
                    image,
                    use_container_width=True
                )


            with col2:

                st.subheader("Segmented Image")

                st.image(
                    highlighted_image,
                    caption="Red = Predicted affected region",
                    use_container_width=True
                )


            st.metric(
                "Estimated Affected Leaf Area",
                f"{percentage:.2f}%"
            )


            # =========================================
            # 18. SEVERITY ESTIMATION
            # =========================================

            st.divider()

            st.header("Disease Severity Estimation")

            severity = estimate_severity(
                percentage
            )

            st.subheader(
                f"Estimated Severity: {severity}"
            )


            if severity == "Mild":

                st.info(
                    "Low estimated affected area. "
                    "Monitor the leaf and validate the symptoms."
                )


            elif severity == "Moderate":

                st.warning(
                    "Moderate estimated affected area. "
                    "Further inspection is recommended."
                )


            else:

                st.error(
                    "High estimated affected area. "
                    "Expert assessment is recommended."
                )


            st.caption(
                "Severity thresholds are provisional and "
                "must be validated using ground-truth annotations. "
                "Area estimation depends on U-Net predictions "
                "and the heuristic leaf mask."
            )


# =====================================================
# 19. FOOTER
# =====================================================

else:

    st.info(
        "Select a classification model and upload "
        "a coffee leaf image to begin."
    )


st.divider()

st.caption(
    "MCA Major Project | Coffee Leaf Disease Detection, "
    "Segmentation and Severity Estimation"
)
