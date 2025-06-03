import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import argparse
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

def create_model(input_shape, num_filters=256):
    inputs = keras.Input(shape=input_shape)
    
    x = layers.Conv1D(num_filters, 19, activation='relu', padding='valid')(inputs)
    x = layers.GlobalMaxPooling1D()(x)
    
    attention_layer1 = layers.Dense(num_filters, activation='tanh')(x)
    attention_layer2 = layers.Dense(num_filters, activation='softmax')(attention_layer1)
    weighted_features = layers.Multiply()([x, attention_layer2])
    
    x = layers.Dense(64, activation='relu')(weighted_features)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    return keras.Model(inputs=inputs, outputs=outputs)

def train_model(model, X_train, y_train, X_test, y_test, model_output, batch_size=128):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(model_output, monitor='val_loss', save_best_only=True, 
                                      save_weights_only=False, mode='min', verbose=1)
    ]
    try:
        history = model.fit(
            X_train, y_train,
            epochs=300,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=callbacks
        )
    except tf.errors.ResourceExhaustedError:
        if batch_size > 8:
            tf.keras.backend.clear_session()
            model = create_model(X_train.shape[1:])
            history = train_model(model, X_train, y_train, X_test, y_test, model_output, batch_size=batch_size//2)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            tf.keras.backend.clear_session()
            model = create_model(X_train.shape[1:])
            history = model.fit(
                X_train, y_train,
                epochs=300,
                batch_size=8,
                validation_data=(X_test, y_test),
                callbacks=callbacks
            )
    return history

def plot_training_history(history, output_dir):
    plt.figure(figsize=(12, 5))
    plt.rcParams.update({'font.size': 12, 'font.family': 'Arial'})
    
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    
    plt.subplot(gs[0])
    plt.plot(history.history['loss'], color='#4c72b0', linewidth=2, label='Training Loss')
    plt.plot(history.history['val_loss'], color='#c44e52', linewidth=2, label='Validation Loss')
    plt.title('Model Loss', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(gs[1])
    plt.plot(history.history['accuracy'], color='#4c72b0', linewidth=2, label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], color='#c44e52', linewidth=2, label='Validation Accuracy')
    plt.title('Model Accuracy', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=600, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'training_history.pdf'), bbox_inches='tight')
    plt.close()
    
    pd.DataFrame(history.history).to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)

def plot_performance_metrics(model, X_test, y_test, output_dir):
    y_pred_proba = model.predict(X_test).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    fig = plt.figure(figsize=(18, 6))
    plt.rcParams.update({'font.size': 12, 'font.family': 'Arial'})
    
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    
    ax1 = plt.subplot(gs[0])
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    ax1.plot(fpr, tpr, color='#1f77b4', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='#7f7f7f', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    ax1.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    ax1.set_title('Receiver Operating Characteristic (ROC)', fontsize=16, fontweight='bold')
    ax1.legend(loc="lower right", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    ax2 = plt.subplot(gs[1])
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    average_precision = average_precision_score(y_test, y_pred_proba)
    
    ax2.step(recall, precision, color='#ff7f0e', where='post', lw=2,
             label=f'Precision-Recall curve (AP = {average_precision:.3f})')
    ax2.fill_between(recall, precision, step='post', alpha=0.2, color='#ff7f0e')
    ax2.set_xlabel('Recall', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Precision', fontsize=14, fontweight='bold')
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlim([0.0, 1.0])
    ax2.set_title('Precision-Recall Curve', fontsize=16, fontweight='bold')
    ax2.legend(loc="lower left", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_metrics.png'), dpi=600, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'performance_metrics.pdf'), bbox_inches='tight')
    plt.close()
    
    metrics_df = pd.DataFrame({
        'Metric': ['ROC AUC', 'PR AUC'],
        'Value': [roc_auc, average_precision]
    })
    
    metrics_df.to_csv(os.path.join(output_dir, 'performance_metrics.csv'), index=False)
    
    return roc_auc, average_precision

def main():
    parser = argparse.ArgumentParser(description="Train a TAD boundary prediction model")
    parser.add_argument("cell_line", help="Name of the cell line to train the model for (e.g., GM12878)")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    args = parser.parse_args()
    
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    cell_line = args.cell_line
    data_file = f"{cell_line}_tad_boundary_data.npz"
    output_dir = f"{cell_line}_output"
    os.makedirs(output_dir, exist_ok=True)
    model_output = os.path.join(output_dir, f"{cell_line}_tad_boundary_model.h5")
    
    print(f"Loading data from {data_file}...")
    data = np.load(data_file)
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    print(f"Data loaded: X_train shape: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Test data: X_test shape: {X_test.shape}, y_test: {y_test.shape}")
    
    model = create_model((10000, 4))
    print("Model created. Starting training...")
    
    history = train_model(model, X_train, y_train, X_test, y_test, 
                        model_output, batch_size=args.batch_size)
    
    print("Training complete. Loading best model...")
    best_model = keras.models.load_model(model_output)
    
    print("Plotting training history...")
    plot_training_history(history, output_dir)
    
    print("Evaluating model performance...")
    roc_auc, pr_auc = plot_performance_metrics(best_model, X_test, y_test, output_dir)
    
    print("\n=== Model Performance Summary ===")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    print(f"Model saved to: {model_output}")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
