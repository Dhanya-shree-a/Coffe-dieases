# Coffee Leaf Disease Detection (PyTorch)

MobileNetV3 classification for `phoma`, `cercospora`, `rust`, and `healthy`; U-Net segmentation code; Streamlit website.

## Important limitation
Your current dataset has class-labeled folders only. It can train the classifier, but **cannot properly train U-Net**. U-Net needs paired images and pixel-level masks identifying affected pixels. The app only shows segmentation after you annotate masks and train U-Net.

## Dataset
If all images are currently in four class folders, use:
```
data/all/phoma/
data/all/cercospora/
data/all/rust/
data/all/healthy/
```
Or use split folders:
```
data/train/{phoma,cercospora,rust,healthy}/
data/val/{phoma,cercospora,rust,healthy}/
data/test/{phoma,cercospora,rust,healthy}/
```
The training script supports either structure. A separate test set is recommended. Keep images of the same physical leaf/capture session in only one split to avoid leakage.

## Setup
Python 3.10 or 3.11 recommended.
```
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Train MobileNetV3
For four class folders directly inside `data/all`:
```
python train_classifier.py --data-dir data/all --output-dir artifacts --epochs 15 --batch-size 32
```
For an existing `data/train` and optional `data/val`, use `--data-dir data`.
The best validation checkpoint is saved as `artifacts/mobilenetv3_coffee.pth`.

## Evaluate and launch
```
python evaluate_classifier.py --data-dir data/all --checkpoint artifacts/mobilenetv3_coffee.pth
streamlit run app.py
```

## U-Net annotations
Create paired files with matching basenames:
```
segmentation_data/images/leaf001.jpg
segmentation_data/masks/leaf001.png
```
Masks must match their source image. White pixels (>=128) mean diseased tissue; black pixels mean not affected. Annotate representative images with a tool such as CVAT or Label Studio, then run:
```
python train_unet.py --images-dir segmentation_data/images --masks-dir segmentation_data/masks --output artifacts/unet_coffee.pth --epochs 30
```
The supplied area percentage uses a simple green-pixel heuristic to estimate leaf pixels; inspect its overlays and do not treat it as reliable for every background/lighting condition. For robust area percentages, annotate or train a separate leaf/background mask too.

## Metrics
Classifier: accuracy, per-class precision/recall/F1 and confusion matrix. U-Net: Dice and IoU on held-out annotated masks. Model confidence is not a calibrated probability; validate on representative images. This is a prototype, not a definitive plant-health diagnosis.
