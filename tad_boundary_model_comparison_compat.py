import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import os
from tqdm import tqdm
import warnings
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    plt.style.use('seaborn-v0_8-paper')
except OSError:
    plt.style.use('seaborn-paper')
matplotlib.rcParams.update({
    'font.size': 12,
    'axes.linewidth': 1.5,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'legend.frameon': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

@tf.keras.utils.register_keras_serializable(package='TransTAD')
class TransTADLoss(tf.keras.losses.Loss):
    def __init__(self, fixed_positions=None, weight=0.3, name='transtad_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.fixed_positions = fixed_positions if fixed_positions is not None else []
        self.weight = weight
    
    def call(self, y_true, y_pred):
        base_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        return base_loss
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'fixed_positions': self.fixed_positions,
            'weight': self.weight
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

def create_transtad_model(input_shape=(10000, 4)):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv1D(256, 19, activation='relu', padding='valid')(inputs)
    x = layers.GlobalMaxPooling1D()(x)
    attention = layers.Dense(256, activation='tanh')(x)
    attention = layers.Dense(256, activation='softmax')(attention)
    x = layers.Multiply()([x, attention])
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return keras.Model(inputs=inputs, outputs=outputs)

def load_transtad_model(cell_line):
    output_dir = f"{cell_line}/output"
    
    savedmodel_path = os.path.join(output_dir, f"{cell_line}_tad_boundary_savedmodel")
    if os.path.exists(savedmodel_path):
        try:
            model = keras.models.load_model(
                savedmodel_path,
                custom_objects={'TransTADLoss': TransTADLoss}
            )
            logger.info(f"Loaded {cell_line} model from SavedModel format")
            return model
        except Exception as e:
            logger.warning(f"Failed to load SavedModel: {e}")
    
    config_path = os.path.join(output_dir, f"{cell_line}_model_config.json")
    model_json_path = os.path.join(output_dir, f"{cell_line}_tad_boundary_architecture.json")
    weights_path = os.path.join(output_dir, f"{cell_line}_tad_boundary_weights.h5")
    
    if os.path.exists(config_path) and os.path.exists(model_json_path) and os.path.exists(weights_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            fixed_positions = config['fixed_positions']
            
            with open(model_json_path, 'r') as f:
                model_json = f.read()
            
            model = keras.models.model_from_json(model_json)
            model.load_weights(weights_path)
            
            loss_fn = TransTADLoss(fixed_positions=fixed_positions)
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss=loss_fn, metrics=['accuracy'])
            
            logger.info(f"Loaded {cell_line} model from JSON+weights format")
            return model
        except Exception as e:
            logger.warning(f"Failed to load JSON+weights: {e}")
    
    old_h5_path = os.path.join(output_dir, f"{cell_line}_tad_boundary_model.h5")
    if os.path.exists(old_h5_path):
        try:
            data_path = f"{cell_line}/{cell_line}_tad_boundary_data.npz"
            data = np.load(data_path, allow_pickle=True)
            input_shape = data['X_train'].shape[1:]
            
            model = create_transtad_model(input_shape)
            model.load_weights(old_h5_path)
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            logger.info(f"Loaded {cell_line} model from legacy h5 format (weights only)")
            return model
        except Exception as e:
            logger.warning(f"Failed to load legacy h5 with weights-only approach: {e}")
            
            try:
                model = keras.models.load_model(
                    old_h5_path,
                    compile=False
                )
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                logger.info(f"Loaded {cell_line} model from legacy h5 format (full model)")
                return model
            except Exception as e2:
                logger.error(f"All loading methods failed for {cell_line}: {e2}")
    
    raise FileNotFoundError(f"TransTAD model files not found for {cell_line}")

class ModelEvaluator:
    def __init__(self, cell_line):
        self.cell_line = cell_line
        self.results = {}
        self.load_data()
        
    def load_data(self):
        data_path = f"{self.cell_line}/{self.cell_line}_tad_boundary_data.npz"
        data = np.load(data_path, allow_pickle=True)
        self.X_train = data['X_train']
        self.X_test = data['X_test']
        self.y_train = data['y_train']
        self.y_test = data['y_test']
            
    def evaluate_transTAD(self):
        model = load_transtad_model(self.cell_line)
        y_pred = model.predict(self.X_test, verbose=0).flatten()
        return self._calculate_metrics(y_pred)
            
    def evaluate_pTADs(self):
        model = PTADsModel()
        model.train(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        return self._calculate_metrics(y_pred)
            
    def evaluate_preciseTAD(self):
        model = PreciseTADModel()
        model.train(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        return self._calculate_metrics(y_pred)

    def evaluate_TADBoundaryDetector(self):
        model_path = f"{self.cell_line}/output/TADBoundaryDetector_model.h5"
        model = keras.models.load_model(model_path)
        y_pred = model.predict(self.X_test, verbose=0).flatten()
        return self._calculate_metrics(y_pred)
            
    def _calculate_metrics(self, y_pred):
        y_pred_binary = (y_pred > 0.5).astype(int)
        fpr, tpr, _ = roc_curve(self.y_test, y_pred)
        precision, recall, _ = precision_recall_curve(self.y_test, y_pred)
        
        return {
            'Accuracy': accuracy_score(self.y_test, y_pred_binary),
            'Precision': precision_score(self.y_test, y_pred_binary),
            'Recall': recall_score(self.y_test, y_pred_binary),
            'AUC': auc(fpr, tpr),
            'AUPRC': average_precision_score(self.y_test, y_pred),
            'ROC': (fpr, tpr),
            'PR': (precision, recall)
        }
        
    def run_evaluation(self):
        self.results['TransTAD'] = self.evaluate_transTAD()
        self.results['pTADs'] = self.evaluate_pTADs()
        self.results['preciseTAD'] = self.evaluate_preciseTAD()
        self.results['TADBoundaryDetector'] = self.evaluate_TADBoundaryDetector()
        return {self.cell_line: self.results}

class BaseModel:
    def __init__(self, name):
        self.name = name
        self.scaler = StandardScaler()
        
    def preprocess_data(self, X):
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)
        return self.scaler.fit_transform(X) if not hasattr(self, 'is_fitted_') else self.scaler.transform(X)

class PTADsModel(BaseModel):
    def __init__(self):
        super().__init__("PTADs")
        self.model = RandomForestClassifier(
            n_estimators=500,
            max_features='sqrt',
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42
        )
        
    def train(self, X_train, y_train):
        X_train = self.preprocess_data(X_train)
        self.model.fit(X_train, y_train)
        self.is_fitted_ = True
        
    def predict(self, X_test):
        X_test = self.preprocess_data(X_test)
        return self.model.predict_proba(X_test)[:, 1]

class PreciseTADModel(PTADsModel):
    def __init__(self):
        super().__init__()
        self.name = "PreciseTAD"
        self.model = RandomForestClassifier(
            n_estimators=500,
            max_features='sqrt',
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=42
        )

def plot_metrics_boxplot(results, model_colors, output_dir):
    data = []
    for cell_line in results:
        for model_name, metrics in results[cell_line].items():
            for metric in ['Accuracy', 'Precision', 'Recall', 'AUC', 'AUPRC']:
                data.append({
                    'Cell Line': cell_line,
                    'Model': model_name,
                    'Metric': metric,
                    'Value': metrics[metric]
                })
    
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(15, 10))
    sns.set_style("whitegrid", {'axes.grid': True, 'grid.linestyle': '--', 'grid.alpha': 0.6})
    
    ax = sns.boxplot(data=df, x='Metric', y='Value', hue='Model',
                    palette=model_colors, 
                    width=0.7,
                    fliersize=5,
                    dodge=True,
                    linewidth=1.5)
    
    sns.swarmplot(data=df, x='Metric', y='Value', hue='Model',
                  palette=model_colors, 
                  dodge=True,
                  size=4, 
                  alpha=0.6)
    
    handles, labels = ax.get_legend_handles_labels()
    n_models = len(model_colors)
    ax.legend(handles[:n_models], labels[:n_models], title='Model', 
             bbox_to_anchor=(1.02, 0.5),
             loc='center left',
             frameon=True,
             edgecolor='black')
    
    plt.title('Performance Metrics Comparison', pad=50)
    plt.xlabel('Metric', labelpad=10)
    plt.ylabel('Value', labelpad=10)
    
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1.1, 0.1))
    
    mean_values = df.groupby(['Metric', 'Model'])['Value'].mean().round(3)
    
    for i, metric in enumerate(df['Metric'].unique()):
        y_start = 1.15
        for j, model in enumerate(['TransTAD', 'pTADs', 'preciseTAD', 'TADBoundaryDetector']):
            mean_val = mean_values[metric][model]
            plt.text(i, y_start - j*0.04, f"{model}: {mean_val:.3f}",
                    horizontalalignment='center',
                    transform=ax.get_xaxis_transform(),
                    fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/metrics_comparison.pdf', 
                dpi=300, 
                bbox_inches='tight',
                pad_inches=0.2)
    plt.close()

    summary = df.groupby(['Model', 'Metric'])['Value'].agg(['mean']).round(3)
    summary.to_csv(f'{output_dir}/metrics_summary.csv')

def plot_curves_by_model(results, model_colors, output_dir):
    plt.figure(figsize=(16, 7))
    
    plt.subplot(1, 2, 1)
    for model_name in model_colors.keys():
        fprs, tprs = [], []
        mean_auc = 0
        
        for cell_line in results:
            metrics = results[cell_line][model_name]
            fpr, tpr = metrics['ROC']
            fprs.append(fpr)
            tprs.append(tpr)
            mean_auc += metrics['AUC']
        
        mean_auc /= len(results)
        
        interp_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.mean([np.interp(interp_fpr, fpr, tpr) for fpr, tpr in zip(fprs, tprs)], axis=0)
        plt.plot(interp_fpr, mean_tpr, color=model_colors[model_name],
                label=f'{model_name} (Avg AUC={mean_auc:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], '--', color='gray', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.ylim(0, 1)
    plt.title('ROC Curves by Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    for model_name in model_colors.keys():
        precisions, recalls = [], []
        mean_auprc = 0
        
        for cell_line in results:
            metrics = results[cell_line][model_name]
            precision, recall = metrics['PR']
            precisions.append(precision)
            recalls.append(recall)
            mean_auprc += metrics['AUPRC']
        
        mean_auprc /= len(results)
        
        interp_recall = np.linspace(0, 1, 100)
        mean_precision = np.mean([np.interp(interp_recall, recall[::-1], precision[::-1]) 
                                for precision, recall in zip(precisions, recalls)], axis=0)
        plt.plot(interp_recall, mean_precision, color=model_colors[model_name],
                label=f'{model_name} (Avg AUPRC={mean_auprc:.3f})', linewidth=2)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim(0, 1)
    plt.title('PR Curves by Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/curves_by_model.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_model_curves(results, model_name, output_dir):
    plt.figure(figsize=(16, 7))
    cell_colors = dict(zip(sorted(results.keys()), sns.color_palette('husl', len(results))))
    
    plt.subplot(1, 2, 1)
    for cell_line in sorted(results.keys()):
        metrics = results[cell_line][model_name]
        fpr, tpr = metrics['ROC']
        auc_score = metrics['AUC']
        plt.plot(fpr, tpr, color=cell_colors[cell_line],
                label=f'{cell_line} (AUC={auc_score:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], '--', color='gray', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.ylim(0, 1)
    plt.title(f'{model_name} ROC Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    for cell_line in sorted(results.keys()):
        metrics = results[cell_line][model_name]
        precision, recall = metrics['PR']
        auprc_score = metrics['AUPRC']
        plt.plot(recall, precision, color=cell_colors[cell_line],
                label=f'{cell_line} (AUPRC={auprc_score:.3f})', linewidth=2)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim(0, 1)
    plt.title(f'{model_name} PR Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/curves_{model_name}.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_curves_with_celllines(results, model_colors, output_dir):
    plt.figure(figsize=(16, 7))
    
    plt.subplot(1, 2, 1)
    for cell_line in results:
        for model_name, metrics in results[cell_line].items():
            fpr, tpr = metrics['ROC']
            plt.plot(fpr, tpr, color=model_colors[model_name], alpha=0.15)
            
    for model_name in model_colors.keys():
        fprs, tprs = [], []
        mean_auc = 0
        
        for cell_line in results:
            metrics = results[cell_line][model_name]
            fpr, tpr = metrics['ROC']
            fprs.append(fpr)
            tprs.append(tpr)
            mean_auc += metrics['AUC']
        
        mean_auc /= len(results)
        
        interp_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.mean([np.interp(interp_fpr, fpr, tpr) for fpr, tpr in zip(fprs, tprs)], axis=0)
        plt.plot(interp_fpr, mean_tpr, color=model_colors[model_name],
                label=f'{model_name} (Avg AUC={mean_auc:.3f})', linewidth=2.5)
    
    plt.plot([0, 1], [0, 1], '--', color='gray', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.ylim(0, 1)
    plt.title('ROC Curves Across All Cell Lines')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    for cell_line in results:
        for model_name, metrics in results[cell_line].items():
            precision, recall = metrics['PR']
            plt.plot(recall, precision, color=model_colors[model_name], alpha=0.15)
    
    for model_name in model_colors.keys():
        precisions, recalls = [], []
        mean_auprc = 0
        
        for cell_line in results:
            metrics = results[cell_line][model_name]
            precision, recall = metrics['PR']
            precisions.append(precision)
            recalls.append(recall)
            mean_auprc += metrics['AUPRC']
        
        mean_auprc /= len(results)
        
        interp_recall = np.linspace(0, 1, 100)
        mean_precision = np.mean([np.interp(interp_recall, recall[::-1], precision[::-1]) 
                                for precision, recall in zip(precisions, recalls)], axis=0)
        plt.plot(interp_recall, mean_precision, color=model_colors[model_name],
                label=f'{model_name} (Avg AUPRC={mean_auprc:.3f})', linewidth=2.5)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim(0, 1)
    plt.title('PR Curves Across All Cell Lines')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/curves_with_celllines.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def save_results_summary(results, output_dir):
    metrics = ['Accuracy', 'Precision', 'Recall', 'AUC', 'AUPRC']
    data = []
    for cell_line in results:
        for model_name, model_metrics in results[cell_line].items():
            row = {'Cell Line': cell_line, 'Model': model_name}
            for metric in metrics:
                row[metric] = model_metrics[metric]
            data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(f'{output_dir}/all_results.csv', index=False)
    
    print("\nPerformance Summary:")
    summary = df.groupby('Model')[metrics].agg(['mean', 'std']).round(4)
    print(summary)
    summary.to_csv(f'{output_dir}/summary_statistics.csv')

def main():
    cell_lines = ['GM12878', 'HCT116', 'HMEC', 'HUVEC', 'K562', 'NHEK']
    output_dir = 'model_comparison_results_v2'
    os.makedirs(output_dir, exist_ok=True)
    
    model_colors = {
        'TransTAD': sns.color_palette('Set2')[0],
        'pTADs': sns.color_palette('Set2')[1],
        'preciseTAD': sns.color_palette('Set2')[2],
        'TADBoundaryDetector': sns.color_palette('Set2')[3]
    }
    
    all_results = {}
    for cell_line in tqdm(cell_lines, desc='Evaluating cell lines'):
        try:
            evaluator = ModelEvaluator(cell_line)
            results = evaluator.run_evaluation()
            all_results.update(results)
        except Exception as e:
            print(f"Error processing {cell_line}: {str(e)}")
    
    plot_metrics_boxplot(all_results, model_colors, output_dir)
    plot_curves_by_model(all_results, model_colors, output_dir)
    
    for model_name in model_colors.keys():
        plot_model_curves(all_results, model_name, output_dir)
    
    plot_curves_with_celllines(all_results, model_colors, output_dir)
    save_results_summary(all_results, output_dir)
    print(f"\nResults saved in: {output_dir}")

if __name__ == "__main__":
    main()
