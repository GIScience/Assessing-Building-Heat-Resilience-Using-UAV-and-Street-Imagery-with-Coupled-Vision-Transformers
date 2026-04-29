# -*- coding: utf-8 -*-
import argparse
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import mixed_precision
from gcvit import GCViTXXTiny, GCViTXTiny, GCViTTiny, GCViTSmall, GCViTBase, GCViTLarge
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from collections import Counter
from sklearn.utils import class_weight 
import json
import uuid
import warnings
warnings.filterwarnings('ignore')



# --- Path Setup ---
# defining absolute paths

unique_id = str(uuid.uuid4())
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'output')
LABELS_DATA_DIR = os.path.join(DATA_DIR, 'labels_data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output', 'training_results', f'single_input_{unique_id}')
os.makedirs(OUTPUT_DIR, exist_ok=True)
# --- End Path Setup ---


parser = argparse.ArgumentParser(description='Single Input Classification with K-Fold CV')
parser.add_argument('--task', type=str, required=True, help='Task folder (e.g., material_rooftop).')
parser.add_argument('--input_type', type=str, required=True, choices=['svi', 'uav'], help='Input type: svi or uav')
parser.add_argument('--backbone', type=str, default='Tiny', choices=['XXTiny', 'XTiny', 'Tiny', 'Small', 'Base', 'Large'])
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--k_folds', type=int, default=5)
parser.add_argument('--min_samples_per_class', type=int, default=5)
parser.add_argument('--augmentation_strength', type=float, default=0.3)
parser.add_argument('--test_size', type=float, default=0.15)
args = parser.parse_args()




strategy = tf.distribute.get_strategy()
print("Number of Accelerators:", strategy.num_replicas_in_sync)

IMG_SIZE = 256
BATCH_SIZE = args.batch_size * strategy.num_replicas_in_sync
INIT_LR, WARMUP_LR, SEED = 1e-4, 1e-5, 42

backbone_map = {
    'xxtiny': GCViTXXTiny, 'xtiny': GCViTXTiny, 'tiny': GCViTTiny,
    'small': GCViTSmall, 'base': GCViTBase, 'large': GCViTLarge
}
backbone = backbone_map[args.backbone.lower()]

# Set seeds
tf.random.set_seed(SEED)
np.random.seed(SEED)

class WarmUpCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
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
        def warmup_logic():
            slope = (self.lr_base - self.warmup_lr) / tf.maximum(self.warmup_steps, 1.0)
            return slope * step + self.warmup_lr
        return tf.cond(step < self.warmup_steps, warmup_logic, lambda: cosine_decay)

    def get_config(self):
        return {
            "lr_base": float(self.lr_base),
            "total_steps": float(self.total_steps),
            "warmup_lr": float(self.warmup_lr),
            "warmup_steps": float(self.warmup_steps)
        }

MEAN = tf.constant([0.485 * 255, 0.456 * 255, 0.406 * 255], dtype=tf.float32)
STD = tf.constant([0.229 * 255, 0.224 * 255, 0.225 * 255], dtype=tf.float32)

def normalize_image(img):
    return (img - MEAN) / STD

def load_image(path, h, w):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, [h, w])
    return normalize_image(img)

def get_augmentation_layer(strength=0.3):
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(strength * 0.1),
        tf.keras.layers.RandomZoom(height_factor=(-strength * 0.1, strength * 0.1),
                                   width_factor=(-strength * 0.1, strength * 0.1)),
        tf.keras.layers.RandomBrightness(strength * 0.2),
        tf.keras.layers.RandomContrast(strength * 0.2),
    ])


def load_and_preprocess_data():
    # Use the absolute DATA_DIR path
    images_task_dir = os.path.join(LABELS_DATA_DIR, args.input_type, args.task)
    csv_path = os.path.join(DATA_DIR, "CV_classdata.csv")
    print(f"Loading data from {csv_path}...")
    master_df = pd.read_csv(csv_path)
    task_df = master_df[master_df[f'{args.input_type}_path'].str.contains(f'/{args.task}/', na=False)].copy()
    class_names = sorted([d.name for d in os.scandir(images_task_dir)
                          if d.is_dir() and not d.name.startswith('.')])
    class_map = {name: i for i, name in enumerate(class_names)}
    task_df['label'] = task_df[f'{args.input_type}_path'].apply(lambda x: x.split('/')[-2]).map(class_map)
    task_df.dropna(subset=['label'], inplace=True)
    task_df['label'] = task_df['label'].astype(int)
    class_counts = Counter(task_df['label'])
    print(f"Original classes: {len(class_names)}")
    print(f"Class distribution: {class_counts}")
    valid_classes = [cls for cls, count in class_counts.items() if count >= args.min_samples_per_class]
    print(f"Classes after filtering (min {args.min_samples_per_class} samples): {len(valid_classes)}")
    task_df = task_df[task_df['label'].isin(valid_classes)].copy()
    valid_class_names = [class_names[i] for i in sorted(valid_classes)]
    new_class_map = {old_label: new_label for new_label, old_label in enumerate(sorted(valid_classes))}
    task_df['label'] = task_df['label'].map(new_class_map)
    print(f"Final dataset size: {len(task_df)} samples")
    print(f"Final class distribution: {Counter(task_df['label'])}")
    print(f"Class names: {valid_class_names}")
    return task_df, valid_class_names

def create_dataset(df, augment=False, shuffle=True, skip_aug_for_class=None):
    def process(path, label):
        img = load_image(path, IMG_SIZE, IMG_SIZE)
        if augment and (skip_aug_for_class is None or label != skip_aug_for_class):
            aug_layer = get_augmentation_layer(args.augmentation_strength)
            img = aug_layer(img, training=True)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((df[f'{args.input_type}_path'].values, df.label.values))
    ds = ds.map(lambda path, label: process(path, label), num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(len(df), seed=SEED)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def create_model(num_classes):
    with strategy.scope():
        base_model = backbone(input_shape=(IMG_SIZE, IMG_SIZE, 3), pretrain=True, resize_query=True)
        base_model.reset_classifier(num_classes=0, head_act=None)
        inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='image_input')
        x = base_model(inp)
        x = tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU())(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        out = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        return tf.keras.Model(inputs=inp, outputs=out)

def evaluate_model(model, test_ds, class_names):
    y_true = np.concatenate([label for _, label in test_ds], axis=0)
    y_pred = np.argmax(model.predict(test_ds), axis=1)
    unique_labels = np.unique(y_true)
    filtered_class_names = [class_names[i] for i in unique_labels]
    acc = np.mean(y_true == y_pred)
    report = classification_report(
        y_true,
        y_pred,
        labels=unique_labels,
        target_names=filtered_class_names,
        digits=4,
        output_dict=True,
        zero_division=0)
    return acc, report, y_true, y_pred

def train_fold(model, train_ds, val_ds, fold_num, total_steps, class_weights):
    warmup_steps = int(total_steps * 0.1)
    lr_schedule = WarmUpCosine(INIT_LR, total_steps, WARMUP_LR, warmup_steps)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    ckpt_path = os.path.join(OUTPUT_DIR, f"{args.task}_{args.input_type}_fold{fold_num+1}_best.h5")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor='val_accuracy', save_best_only=True,
                                          save_weights_only=True, mode='max', verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                        class_weight=class_weights, callbacks=callbacks, verbose=1)
    model.load_weights(ckpt_path)
    return history, ckpt_path



def main():
    print("Loading and preprocessing data...")
    task_df, class_names = load_and_preprocess_data()
    NUM_CLASSES = len(class_names)
    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Class names: {class_names}")
    
    train_val_df, test_df = train_test_split(task_df, test_size=args.test_size, stratify=task_df['label'], random_state=SEED)
    class_counts = Counter(train_val_df['label'])
    if class_counts:
        majority_class = class_counts.most_common(1)[0][0]
        print(f"Majority class (no augmentation applied): {majority_class}")
    else:
        majority_class = None
        print("No majority class found, proceeding without skip-augmentation logic.")
    print(f"Train+Val set: {len(train_val_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    test_df.to_csv(os.path.join(OUTPUT_DIR, f"{args.task}_{args.input_type}_test_set.csv"), index=False)
    print(f"Test set saved to {os.path.join(OUTPUT_DIR, f'{args.task}_{args.input_type}_test_set.csv')}")
    

    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=SEED)
    fold_results = []
    fold_histories = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_df, train_val_df['label'])):
        print(f"\n{'='*60}")
        print(f"FOLD {fold+1}/{args.k_folds}")
        print(f"{'='*60}")
        train_df, val_df = train_val_df.iloc[train_idx], train_val_df.iloc[val_idx]
        print(f"Train fold size: {len(train_df)}")
        print(f"Val fold size: {len(val_df)}")
        
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_df['label']), y=train_df['label'])
        class_weights_dict = dict(enumerate(class_weights))
        print(f"Class weights for this fold: {class_weights_dict}")
        
        train_ds = create_dataset(train_df, augment=True, shuffle=True, skip_aug_for_class=majority_class)
        val_ds = create_dataset(val_df, augment=False, shuffle=False)
        
        model = create_model(NUM_CLASSES)
        if fold == 0:
            model.summary()
        
        total_steps = len(train_df) // BATCH_SIZE * args.epochs
        history, ckpt_path = train_fold(model, train_ds, val_ds, fold, total_steps, class_weights_dict)
        val_accuracy, val_report, y_true, y_pred = evaluate_model(model, val_ds, class_names)
        
        fold_results.append({
            'fold': fold + 1,
            'val_accuracy': val_accuracy,
            'val_report': val_report,
            'checkpoint_path': ckpt_path
        })
        fold_histories.append(history.history)
        
        print(f"\nFold {fold+1} Validation Accuracy: {val_accuracy:.4f}")
        print(f"Fold {fold+1} Classification Report:")
        print(classification_report(
            y_true,
            y_pred,
            labels=np.unique(y_true),
            target_names=[class_names[i] for i in np.unique(y_true)],
            digits=4,
            zero_division=0))

    cv_accuracies = [result['val_accuracy'] for result in fold_results]
    mean_cv_accuracy = np.mean(cv_accuracies)
    std_cv_accuracy = np.std(cv_accuracies)
    
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Mean CV Accuracy: {mean_cv_accuracy:.4f} ± {std_cv_accuracy:.4f}")
    
    best_fold_idx = np.argmax(cv_accuracies)
    best_fold_result = fold_results[best_fold_idx]
    print(f"Best fold: {best_fold_result['fold']} (Accuracy: {best_fold_result['val_accuracy']:.4f})")
    
    print(f"\n{'='*60}")
    print("FINAL TEST SET EVALUATION")
    print(f"{'='*60}")
    
    final_model = create_model(NUM_CLASSES)
    final_model.load_weights(best_fold_result['checkpoint_path'])
    test_ds = create_dataset(test_df, augment=False, shuffle=False)
    test_accuracy, test_report, y_true_test, y_pred_test = evaluate_model(final_model, test_ds, class_names)
    
    print(f"Final Test Accuracy: {test_accuracy:.4f}")
    print("\nFinal Test Classification Report:")
    print(classification_report(
        y_true_test,
        y_pred_test,
        labels=np.unique(y_true_test),
        target_names=[class_names[i] for i in np.unique(y_true_test)],
        digits=4,
        zero_division=0))
    
    final_results = {
        'task': args.task,
        'input_type': args.input_type,
        'backbone': args.backbone,
        'k_folds': args.k_folds,
        'cv_mean_accuracy': float(mean_cv_accuracy),
        'cv_std_accuracy': float(std_cv_accuracy),
        'best_fold_accuracy': float(best_fold_result['val_accuracy']),
        'test_accuracy': float(test_accuracy),
        'test_report': test_report,
    }

    results_file = os.path.join(OUTPUT_DIR, f"{args.task}_{args.input_type}_cv_results.json")
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    main()