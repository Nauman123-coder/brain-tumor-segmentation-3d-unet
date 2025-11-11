import gradio as gr
import numpy as np
import nibabel as nib
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K 
import matplotlib.pyplot as plt
import io
import util  # your prediction & visualization utils

def load_case(image_nifty_file, label_nifty_file):
    # load the image and label file, get the image content and return a numpy array for each
    image = np.array(nib.load(image_nifty_file).get_fdata())
    label = np.array(nib.load(label_nifty_file).get_fdata())
    
    return image, label

def dice_coefficient(y_true, y_pred, axis=(1, 2, 3), epsilon=0.00001):
    dice_numerator = 2. * K.sum(y_true * y_pred, axis=axis) + epsilon
    dice_denominator = K.sum(y_true, axis=axis) + K.sum(y_pred, axis=axis) + epsilon
    dice_coefficient = K.mean((dice_numerator)/(dice_denominator))

    return dice_coefficient

def soft_dice_loss(y_true, y_pred, axis=(1, 2, 3), epsilon=0.00001):
    dice_numerator = 2. * K.sum(y_true * y_pred, axis=axis) + epsilon
    dice_denominator = K.sum(y_true**2, axis=axis) + K.sum(y_pred**2, axis=axis) + epsilon
    dice_loss = 1 - K.mean((dice_numerator)/(dice_denominator))

    ### END CODE HERE ###

    return dice_loss

# Load model
segment_model = load_model(
    r"E:\AI in Medical Diagnosis\MRI Image Segmentation\Final Model\Saved Model\unet3d_model.keras",
    custom_objects={
        "soft_dice_loss": soft_dice_loss,
        "dice_coefficient": dice_coefficient
    }
)
print("Model loaded successfully!")

from PIL import Image

def predict_mri(image_file, label_file):
    try:
        # Load image and label
        image, label = load_case(image_file.name, label_file.name)
    except Exception as e:
        raise ValueError(f"Error loading NIfTI files: {e}")
    
    # Ensure channel dimension
    if image.ndim == 3:
        image = np.expand_dims(image, axis=-1)
    if label.ndim == 3:
        label = np.expand_dims(label, axis=-1)
    
    # Predict segmentation
    pred = util.predict_and_viz(image, label, segment_model, threshold=0.5, loc=(130, 130, 77))
    
    # Overlay middle slice (axial)
    slice_idx = image.shape[2] // 2
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(image[:, :, slice_idx, 0], cmap="gray")
    ax.imshow(pred[:, :, slice_idx, 0], cmap="Reds", alpha=0.5)
    ax.axis("off")
    
    # Convert figure to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    pil_image = Image.open(buf)
    
    return pil_image


gr.Interface(
    fn=predict_mri,
    inputs=[
        gr.File(label="Upload MRI Scan (.nii.gz)"),
        gr.File(label="Upload Ground Truth Label (.nii.gz)")
    ],
    outputs=gr.Image(type="pil"),
    title="Brain MRI Segmentation",
    description="Upload a 3D MRI scan and its ground truth label (.nii.gz format) to visualize segmentation."
).launch()
