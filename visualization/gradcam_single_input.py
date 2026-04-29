# visualization/gradcam_single_input.py
import os
import argparse
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from gcvit import GCViTTiny
import sys

# Add project root to path to allow importing config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# --- Image Preprocessing ---
MEAN = tf.constant([0.485 * 255, 0.456 * 255, 0.406 * 255], dtype=tf.float32)
STD = tf.constant([0.229 * 255, 0.224 * 255, 0.225 * 255], dtype=tf.float32)
IMG_SIZE = 256

def normalize_image(img):
    return (img - MEAN) / STD

def load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img_orig = tf.cast(img, tf.uint8).numpy()
    img_norm = normalize_image(tf.cast(img, tf.float32))
    return img_orig, tf.expand_dims(img_norm, axis=0)

# --- Model and Grad-CAM Logic ---
def create_model(num_classes):
    base_model = GCViTTiny(input_shape=(IMG_SIZE, IMG_SIZE, 3), pretrain=False)
    base_model.reset_classifier(num_classes=0, head_act=None)
    
    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inp)
    x = tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU())(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    full_model = tf.keras.Model(inputs=inp, outputs=out)
    gradcam_model = tf.keras.Model(inputs=base_model.input, outputs=[base_model.layers[-3].output, base_model.output])
    return full_model, gradcam_model

def compute_gradcam(gradcam_model, img_tensor, pred_index):
    with tf.GradientTape() as tape:
        conv_outputs, predictions = gradcam_model(img_tensor)
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    heatmap = tf.matmul(conv_outputs[0], pooled_grads[..., tf.newaxis])
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_heatmap(img, heatmap, alpha=0.5):
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Grad-CAM for a single-input model.")
    parser.add_argument("--weights_path", required=True, help="Path to the trained model weights (.h5 file).")
    parser.add_argument("--image_path", required=True, help="Path to the input image for visualization.")
    parser.add_argument("--class_names", required=True, nargs='+', help="List of class names for the model.")
    parser.add_argument("--output_name", required=True, help="Name for the output visualization file (e.g., 'svi_material_wall_gradcam.png').")
    args = parser.parse_args()

    # --- Main Execution ---
    os.makedirs(config.GRADCAM_OUTPUT_DIR, exist_ok=True)
    num_classes = len(args.class_names)

    # 1. Create and load the model
    model, gradcam_model = create_model(num_classes)
    model.load_weights(args.weights_path)
    print(f"Successfully loaded weights from {args.weights_path}")

    # 2. Load and prepare the image
    original_img, img_tensor = load_image(args.image_path)

    # 3. Get prediction and compute Grad-CAM
    preds = model.predict(img_tensor)[0]
    pred_class_idx = np.argmax(preds)
    heatmap = compute_gradcam(gradcam_model, img_tensor, pred_class_idx)
    
    # 4. Create and save the visualization
    overlayed_img = overlay_heatmap(original_img, heatmap)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_img)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(overlayed_img)
    pred_text = "\n".join([f"{name}: {prob*100:.1f}%" for name, prob in zip(args.class_names, preds)])
    axes[1].set_title(f"Grad-CAM\n{pred_text}")
    axes[1].axis('off')
    
    save_path = os.path.join(config.GRADCAM_OUTPUT_DIR, args.output_name)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ Grad-CAM visualization saved to: {save_path}")