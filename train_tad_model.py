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

def train_model(model, X_train, y_train, model_output, batch_size=128):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    callbacks = [
        keras.callbacks.ModelCheckpoint(model_output, monitor='loss', save_best_only=True,
                                      save_weights_only=False, mode='min', verbose=1)
    ]
    try:
        history = model.fit(
            X_train, y_train,
            epochs=300,
            batch_size=batch_size,
            callbacks=callbacks
        )
    except tf.errors.ResourceExhaustedError:
        if batch_size > 8:
            tf.keras.backend.clear_session()
            model = create_model(X_train.shape[1:])
            history = train_model(model, X_train, y_train, model_output, batch_size=batch_size//2)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            tf.keras.backend.clear_session()
            model = create_model(X_train.shape[1:])
            history = model.fit(
                X_train, y_train,
                epochs=300,
                batch_size=8,
                callbacks=callbacks
            )
    return history

def plot_training_history(history, output_dir):
    plt.figure(figsize=(12, 5))
    plt.rcParams.update({'font.size': 12, 'font.family': 'Arial'})

    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

    plt.subplot(gs[0])
    plt.plot(history.history['loss'], color='#4c72b0', linewidth=2, label='Training Loss')
    plt.title('Model Loss', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(gs[1])
    plt.plot(history.history['accuracy'], color='#4c72b0', linewidth=2, label='Training Accuracy')
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
    X_train = data['X_train']
    y_train = data['y_train']
    print(f"Data loaded: X_train shape: {X_train.shape}, y_train: {y_train.shape}")

    model = create_model((10000, 4))
    print("Model created. Starting training...")

    history = train_model(model, X_train, y_train, model_output, batch_size=args.batch_size)

    print("Training complete.")

    print("Plotting training history...")
    plot_training_history(history, output_dir)

    print(f"\nModel saved to: {model_output}")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
