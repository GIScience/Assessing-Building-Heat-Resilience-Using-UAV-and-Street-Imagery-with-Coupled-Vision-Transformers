# visualization/gradcam_dual_input.py
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
    return img_orig, img_norm

# --- Model and Grad-CAM Logic ---
def create_model(num_classes):
    svi_backbone = GCViTTiny(input_shape=(IMG_SIZE, IMG_SIZE, 3), pretrain=False, name='svi_backbone')
    svi_backbone.reset_classifier(num_classes=0, head_act=None)
    
    uav_backbone = GCViTTiny(input_shape=(IMG_SIZE, IMG_SIZE, 3), pretrain=False, name='uav_backbone')
    uav_backbone.reset_classifier(num_classes=0, head_act=None)

    inp_svi = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='svi_input')
    inp_uav = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='uav_input')
    
    feat_svi = svi_backbone(inp_svi)
    feat_uav = uav_backbone(inp_uav)
    
    concat = tf.keras.layers.Concatenate()([feat_svi, feat_uav])
    x = tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU())(concat)
    x = tf.keras.layers.Dropout(0.2)(x)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return tf.keras.Model(inputs=[inp_svi, inp_uav], outputs=output)

def compute_gradcam_on_branch(full_model, branch_name, img_inputs, pred_index):
    branch_model = full_model.get_layer(branch_name)
    last_conv_layer = branch_model.layers[-3] # A reasonable guess for the last conv-like layer
    
    grad_model = tf.keras.models.Model(
        [full_model.inputs], [last_conv_layer.output, full_model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_inputs)
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
    parser = argparse.ArgumentParser(description="Generate Grad-CAM for a dual-input model.")
    parser.add_argument("--weights_path", required=True, help="Path to the trained model weights (.h5 file).")
    parser.add_argument("--svi_path", required=True, help="Path to the SVI input image.")
    parser.add_argument("--uav_path", required=True, help="Path to the UAV input image.")
    parser.add_argument("--class_names", required=True, nargs='+', help="List of class names for the model.")
    parser.add_argument("--output_name", required=True, help="Name for the output visualization file (e.g., 'dual_material_wall_gradcam.png').")
    args = parser.parse_args()

    # --- Main Execution ---
    os.makedirs(config.GRADCAM_OUTPUT_DIR, exist_ok=True)
    num_classes = len(args.class_names)

    # 1. Create and load model
    model = create_model(num_classes)
    model.load_weights(args.weights_path)
    print(f"Successfully loaded weights from {args.weights_path}")

    # 2. Load and prepare images
    svi_orig, svi_norm = load_image(args.svi_path)
    uav_orig, uav_norm = load_image(args.uav_path)
    img_tensors = [np.expand_dims(svi_norm, 0), np.expand_dims(uav_norm, 0)]

    # 3. Get prediction
    preds = model.predict(img_tensors)[0]
    pred_class_idx = np.argmax(preds)

    # 4. Compute Grad-CAM for each branch
    svi_heatmap = compute_gradcam_on_branch(model, 'svi_backbone', img_tensors, pred_class_idx)
    uav_heatmap = compute_gradcam_on_branch(model, 'uav_backbone', img_tensors, pred_class_idx)
    
    # 5. Create and save visualization
    svi_overlay = overlay_heatmap(svi_orig, svi_heatmap)
    uav_overlay = overlay_heatmap(uav_orig, uav_heatmap)
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(f"Prediction: {args.class_names[pred_class_idx]} ({preds[pred_class_idx]*100:.1f}%)", fontsize=16)
    
    axes[0, 0].imshow(svi_orig)
    axes[0, 0].set_title("SVI Input")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(svi_overlay)
    axes[0, 1].set_title("SVI Grad-CAM")
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(uav_orig)
    axes[1, 0].set_title("UAV Input")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(uav_overlay)
    axes[1, 1].set_title("UAV Grad-CAM")
    axes[1, 1].axis('off')
    
    save_path = os.path.join(config.GRADCAM_OUTPUT_DIR, args.output_name)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300)
    print(f"✅ Grad-CAM visualization saved to: {save_path}")