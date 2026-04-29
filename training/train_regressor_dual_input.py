# training/train_regressor_dual_input.py
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import uuid
from tensorflow.keras import mixed_precision
from gcvit import GCViTTiny
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# --- Path Setup ---
unique_id = str(uuid.uuid4())
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'output')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output', 'training_results', f'regressor_{unique_id}')
os.makedirs(OUTPUT_DIR, exist_ok=True)
# --- End Path Setup ---

# MODIFIED: Save log file to the new output directory
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, 'regression_training.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Setup mixed precision
mixed_precision.set_global_policy('mixed_float16')

# Parse arguments
parser = argparse.ArgumentParser(description='Dual Image Brightness Regression with K-Fold CV')
parser.add_argument('--target_column', type=str, required=True, choices=['wall_brightness', 'roof_brightness'])
parser.add_argument('--epochs', type=int, default=75)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--k_folds', type=int, default=5, help='Number of folds for cross-validation')
parser.add_argument('--augmentation_strength', type=float, default=0.3, help='Augmentation strength (0.0-1.0)')
parser.add_argument('--test_size', type=float, default=0.15, help='Test set size')
args = parser.parse_args()

# Distributed training setup
strategy = tf.distribute.get_strategy()
logger.info(f"Number of accelerators: {strategy.num_replicas_in_sync}")

# Configuration
IMG_SIZE = 224
BATCH_SIZE = args.batch_size * strategy.num_replicas_in_sync
INIT_LR = 1e-4 * strategy.num_replicas_in_sync
WARMUP_LR = 1e-5
SEED = 42

# Set seeds for reproducibility
tf.random.set_seed(SEED)
np.random.seed(SEED)

class WarmUpCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Custom learning rate schedule with warmup and cosine decay."""
    def __init__(self, lr_base, total_steps, warmup_lr, warmup_steps):
        super(WarmUpCosine, self).__init__()
        self.lr_base = tf.cast(lr_base, tf.float32)
        self.total_steps = tf.cast(total_steps, tf.float32)
        self.warmup_lr = tf.cast(warmup_lr, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.pi = tf.constant(np.pi, dtype=tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        cosine_decay = 0.5 * self.lr_base * (
            1 + tf.cos(self.pi * (step - self.warmup_steps) / tf.maximum(self.total_steps - self.warmup_steps, 1.0))
        )
        slope = tf.where(
            self.warmup_steps > 0,
            (self.lr_base - self.warmup_lr) / tf.maximum(self.warmup_steps, 1.0),
            0.0
        )
        warmup_rate = slope * step + self.warmup_lr
        lr = tf.where(
            tf.logical_and(step < self.warmup_steps, self.warmup_steps > 0),
            warmup_rate,
            cosine_decay
        )
        lr = tf.maximum(lr, 0.0)
        return tf.where(step > self.total_steps, 0.0, lr)

    def get_config(self):
        return {
            "lr_base": float(self.lr_base),
            "total_steps": float(self.total_steps),
            "warmup_lr": float(self.warmup_lr),
            "warmup_steps": float(self.warmup_steps)
        }

def get_augmentation_layer(strength=0.3):
    """Create augmentation layer for regression."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(strength * 0.1),
        tf.keras.layers.RandomZoom(height_factor=(-strength * 0.1, strength * 0.1),
                                  width_factor=(-strength * 0.1, strength * 0.1)),
        tf.keras.layers.RandomBrightness(strength * 0.2),
        tf.keras.layers.RandomContrast(strength * 0.2),
    ])

def load_image(path, h, w):
    """Load and resize image with error handling."""
    try:
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.cast(img, tf.float32) / 255.0
        return tf.image.resize(img, [h, w])
    except tf.errors.InvalidArgumentError:
        logger.warning(f"Failed to load image at {path}. Returning zero tensor.")
        return tf.zeros((h, w, 3), dtype=tf.float32)

def process_paths(svi_path, uav_path, label, augment=False):
    """Process image paths with identical augmentation for SVI and UAV."""
    svi_img = load_image(svi_path, IMG_SIZE, IMG_SIZE)
    uav_img = load_image(uav_path, IMG_SIZE, IMG_SIZE)

    if augment:
        aug_layer = get_augmentation_layer(args.augmentation_strength)
        stacked = tf.stack([svi_img, uav_img], axis=0)
        stacked = aug_layer(stacked, training=True)
        svi_img, uav_img = stacked[0], stacked[1]

    return (svi_img, uav_img), tf.cast(label, tf.float32)

def create_dataset(df, augment=False, shuffle=True):
    """Create TensorFlow dataset."""
    ds = tf.data.Dataset.from_tensor_slices((
        df.svi_path.values,
        df.uav_path.values,
        df.label.values
    ))
    ds = ds.map(
        lambda svi, uav, label: process_paths(svi, uav, label, augment),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=SEED)
    return ds.cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def load_and_preprocess_data():
    """Load and preprocess dataset with label normalization."""
    csv_path = os.path.join(DATA_DIR, "CV_regression.csv")
    if not os.path.exists(csv_path):
        logger.error(f"CSV file {csv_path} does not exist.")
        raise FileNotFoundError(f"CSV file {csv_path} does not exist.")

    logger.info(f"Loading data from {csv_path}...")
    master_df = pd.read_csv(csv_path)

    if args.target_column not in master_df.columns:
        logger.error(f"Target column {args.target_column} not found in CSV.")
        raise ValueError(f"Target column {args.target_column} not found in CSV.")

    master_df['label'] = master_df[args.target_column]
    task_df = master_df.dropna(subset=['svi_path', 'uav_path', 'label']).copy()

    Q1, Q3 = task_df['label'].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    task_df = task_df[(task_df['label'] >= Q1 - 1.5 * IQR) & (task_df['label'] <= Q3 + 1.5 * IQR)]

    label_mean = task_df['label'].mean()
    label_std = task_df['label'].std()
    task_df['label'] = (task_df['label'] - label_mean) / label_std

    logger.info(f"Dataset size after preprocessing: {len(task_df)} samples")
    logger.info(f"Label mean: {label_mean:.4f}, std: {label_std:.4f}")

    return task_df, label_mean, label_std


def create_model():
    """Create dual-input regression model."""
    with strategy.scope():
        try:
            subModelSVI = GCViTTiny(input_shape=(IMG_SIZE, IMG_SIZE, 3), pretrain=True, resize_query=True)
        except Exception as e:
            logger.warning(f"Pretrained weights unavailable for GCViTTiny. Using random initialization: {str(e)}")
            subModelSVI = GCViTTiny(input_shape=(IMG_SIZE, IMG_SIZE, 3), pretrain=False, resize_query=True)
        subModelSVI._name = 'svi'

        try:
            subModelUAV = GCViTTiny(input_shape=(IMG_SIZE, IMG_SIZE, 3), pretrain=True, resize_query=True)
        except Exception as e:
            logger.warning(f"Pretrained weights unavailable for GCViTTiny. Using random initialization: {str(e)}")
            subModelUAV = GCViTTiny(input_shape=(IMG_SIZE, IMG_SIZE, 3), pretrain=False, resize_query=True)
        subModelUAV._name = 'uav'

        inpSVI = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
        inpUAV = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
        featSVI = subModelSVI(inpSVI)
        featUAV = subModelUAV(inpUAV)

        concat = tf.keras.layers.Concatenate()([featSVI, featUAV])
        x = tf.keras.layers.Dense(256, activation=tf.keras.layers.LeakyReLU())(concat)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        output = tf.keras.layers.Dense(1, activation='linear', dtype='float32')(x)

        model = tf.keras.Model(inputs=[inpSVI, inpUAV], outputs=output)
        return model


def plot_training_history(history, fold_number=None):
    """Plot training and validation loss/MAE curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # --- Plot Training & Validation Loss (MSE) ---
    ax1.plot(history.history['loss'], label='Train Loss')
    ax1.plot(history.history['val_loss'], label='Val Loss')
    title_loss = 'Loss (MSE)'
    if fold_number:
        title_loss += f' - Fold {fold_number}'
    ax1.set_title(title_loss)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- Plot Training & Validation Mean Absolute Error ---
    ax2.plot(history.history['mean_absolute_error'], label='Train MAE')
    ax2.plot(history.history['val_mean_absolute_error'], label='Val MAE')
    title_mae = 'Mean Absolute Error'
    if fold_number:
        title_mae += f' - Fold {fold_number}'
    ax2.set_title(title_mae)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    # --- Save the Figure ---
    filename = os.path.join(OUTPUT_DIR, f"{args.target_column}_training_curves")
    if fold_number:
        filename += f"_fold_{fold_number}"
    filename += ".png"
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    logger.info(f"Training curves saved to {filename}")


def evaluate_model(model, ds, label_mean, label_std, dataset_name="dataset"):
    """Evaluate regression model with multiple metrics."""
    y_true_normalized = np.concatenate([label for _, label in ds], axis=0)
    y_pred_normalized = model.predict(ds).flatten()

    y_true = y_true_normalized * label_std + label_mean
    y_pred = y_pred_normalized * label_std + label_mean

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Predictions vs True Values - {dataset_name}")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{args.target_column}_{dataset_name}_predictions.png"))
    plt.close()

    return mae, rmse, r2
def main():
    """Main function to execute k-fold cross-validation training."""
    logger.info("Starting K-Fold regression training process...")
    logger.info(f"Target: {args.target_column}, Folds: {args.k_folds}, Epochs: {args.epochs}, Batch size: {args.batch_size}")

    task_df, label_mean, label_std = load_and_preprocess_data()

    # --- SPLIT DATA: Hold out a final test set that is NEVER used in CV ---
    train_val_df, test_df = train_test_split(task_df, test_size=args.test_size, random_state=SEED)
    test_ds = create_dataset(test_df, augment=False, shuffle=False)
    
    logger.info(f"Train/Validation set: {len(train_val_df)} samples")
    logger.info(f"Held-out Test set: {len(test_df)} samples")
    test_df.to_csv(os.path.join(OUTPUT_DIR, f"{args.target_column}_test_set.csv"), index=False)

    # --- K-FOLD CROSS-VALIDATION SETUP ---
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=SEED)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_df)):
        logger.info(f"\n{'='*60}")
        logger.info(f"STARTING FOLD {fold + 1}/{args.k_folds}")
        logger.info(f"{'='*60}")

        # --- Get fold-specific data ---
        train_df = train_val_df.iloc[train_idx]
        val_df = train_val_df.iloc[val_idx]
        train_ds = create_dataset(train_df, augment=True, shuffle=True)
        val_ds = create_dataset(val_df, augment=False, shuffle=False)

        # --- Create a new model for each fold ---
        model = create_model()
        
        total_steps = len(train_df) // BATCH_SIZE * args.epochs
        warmup_steps = int(total_steps * 0.1)
        lr_schedule = WarmUpCosine(
            lr_base=INIT_LR,
            total_steps=total_steps,
            warmup_lr=WARMUP_LR,
            warmup_steps=warmup_steps
        )
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss='mean_squared_error',
            metrics=['mean_absolute_error']
        )
        
        ckpt_path = os.path.join(OUTPUT_DIR, f"{args.target_column}_fold_{fold+1}_best.h5")
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=ckpt_path,
                save_best_only=True,
                save_weights_only=True,
                monitor='val_mean_absolute_error',
                mode='min'
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_mean_absolute_error',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.TensorBoard(log_dir=f'logs/{args.target_column}/fold_{fold+1}')
        ]
        
        logger.info(f"Fold {fold+1} - Train set: {len(train_df)}, Validation set: {len(val_df)}")
        history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks, verbose=2)

        # --- PLOT THE TRAINING HISTORY FOR THE CURRENT FOLD ---
        plot_training_history(history, fold_number=fold + 1)

        # --- Evaluate and store fold results ---
        val_mae, val_rmse, val_r2 = evaluate_model(model, val_ds, label_mean, label_std, f"validation_fold_{fold+1}")
        logger.info(f"Fold {fold+1} Validation MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
        fold_results.append({
            'fold': fold + 1,
            'val_mae': val_mae,
            'val_rmse': val_rmse,
            'val_r2': val_r2,
            'checkpoint_path': ckpt_path
        })

    # --- FINAL EVALUATION ---
    logger.info(f"\n{'='*60}")
    logger.info("CROSS-VALIDATION SUMMARY")
    logger.info(f"{'='*60}")
    
    val_maes = [r['val_mae'] for r in fold_results]
    mean_mae = np.mean(val_maes)
    std_mae = np.std(val_maes)
    logger.info(f"Average Validation MAE across {args.k_folds} folds: {mean_mae:.4f} (+/- {std_mae:.4f})")

    # --- Find the best model and evaluate it on the test set ---
    best_fold = min(fold_results, key=lambda x: x['val_mae'])
    logger.info(f"Best fold: {best_fold['fold']} (Val MAE: {best_fold['val_mae']:.4f})")

    best_model = create_model()
    best_model.load_weights(best_fold['checkpoint_path'])
    
    test_mae, test_rmse, test_r2 = evaluate_model(best_model, test_ds, label_mean, label_std, "final_test")
    logger.info(f"\nFinal Test Set Performance (from best fold):")
    logger.info(f"Test MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")

    # --- Save final results ---
    final_results = {
        'target_column': args.target_column,
        'k_folds': args.k_folds,
        'cv_mean_mae': float(mean_mae),
        'cv_std_mae': float(std_mae),
        'best_fold_val_mae': float(best_fold['val_mae']),
        'best_fold_val_r2': float(best_fold['val_r2']),
        'test_mae': float(test_mae),
        'test_rmse': float(test_rmse),
        'test_r2': float(test_r2),
        'label_mean': float(label_mean),
        'label_std': float(label_std)
    }
    
    results_file = os.path.join(OUTPUT_DIR, f"{args.target_column}_kfold_regression_results.json")
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    logger.info(f"K-Fold results saved to {results_file}")

if __name__ == "__main__":
    main()