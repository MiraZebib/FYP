import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import os


def evaluate_model(model, X_test, y_test, scaler=None, model_name='Model'):
    if scaler is not None:
        X_test_scaled = scaler.transform(X_test)
        X_eval = X_test_scaled
    else:
        X_eval = X_test
    
    y_pred = model.predict(X_eval)
    
    try:
        y_proba = model.predict_proba(X_eval)[:, 1]
    except:
        y_proba = None
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    roc_auc = None
    if y_proba is not None:
        try:
            roc_auc = roc_auc_score(y_test, y_proba)
        except:
            pass
    
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }
    
    return metrics, y_pred, y_proba


def plot_confusion_matrix(cm, model_name, save_path=None):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Malicious'],
                yticklabels=['Benign', 'Malicious'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    return plt.gcf()


def plot_roc_curve(y_test, y_proba_list, model_names, save_path=None):
    plt.figure(figsize=(10, 8))
    
    for y_proba, name in zip(y_proba_list, model_names):
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    
    return plt.gcf()


def evaluate_all_models(models_dict, scalers_dict, X_test, y_test, 
                       plots_dir='plots'):
    os.makedirs(plots_dir, exist_ok=True)
    
    all_metrics = []
    y_proba_list = []
    model_names = []
    
    rf_metrics, _, rf_proba = evaluate_model(
        models_dict['random_forest'], X_test, y_test, 
        scaler=None, model_name='Random Forest'
    )
    all_metrics.append(rf_metrics)
    y_proba_list.append(rf_proba)
    model_names.append('Random Forest')
    
    plot_confusion_matrix(
        rf_metrics['confusion_matrix'], 
        'Random Forest',
        os.path.join(plots_dir, 'confusion_matrix_rf.png')
    )
    plt.close()
    
    lr_metrics, _, lr_proba = evaluate_model(
        models_dict['logistic_regression'], X_test, y_test,
        scaler=scalers_dict.get('lr_scaler'), model_name='Logistic Regression'
    )
    all_metrics.append(lr_metrics)
    y_proba_list.append(lr_proba)
    model_names.append('Logistic Regression')
    
    plot_confusion_matrix(
        lr_metrics['confusion_matrix'],
        'Logistic Regression',
        os.path.join(plots_dir, 'confusion_matrix_lr.png')
    )
    plt.close()
    
    svm_metrics, _, svm_proba = evaluate_model(
        models_dict['svm'], X_test, y_test,
        scaler=scalers_dict.get('svm_scaler'), model_name='SVM'
    )
    all_metrics.append(svm_metrics)
    y_proba_list.append(svm_proba)
    model_names.append('SVM')
    
    plot_confusion_matrix(
        svm_metrics['confusion_matrix'],
        'SVM',
        os.path.join(plots_dir, 'confusion_matrix_svm.png')
    )
    plt.close()
    
    plot_roc_curve(
        y_test, y_proba_list, model_names,
        os.path.join(plots_dir, 'roc_curves.png')
    )
    plt.close()
    
    comparison_df = pd.DataFrame([
        {
            'Model': m['model_name'],
            'Accuracy': f"{m['accuracy']:.4f}",
            'Precision': f"{m['precision']:.4f}",
            'Recall': f"{m['recall']:.4f}",
            'F1-Score': f"{m['f1_score']:.4f}",
            'ROC-AUC': f"{m['roc_auc']:.4f}" if m['roc_auc'] else 'N/A'
        }
        for m in all_metrics
    ])
    
    comparison_path = os.path.join(plots_dir, 'model_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\nComparison table saved to {comparison_path}")
    
    print("\n" + "="*70)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*70)
    print(comparison_df.to_string(index=False))
    print("="*70)
    
    print("\nDetailed Classification Reports:")
    print("\n" + "-"*70)
    print("Random Forest:")
    print("-"*70)
    _, rf_pred, _ = evaluate_model(
        models_dict['random_forest'], X_test, y_test, model_name='RF'
    )
    print(classification_report(y_test, rf_pred, 
                              target_names=['Benign', 'Malicious']))
    
    print("\n" + "-"*70)
    print("Logistic Regression:")
    print("-"*70)
    _, lr_pred, _ = evaluate_model(
        models_dict['logistic_regression'], X_test, y_test,
        scaler=scalers_dict.get('lr_scaler'), model_name='LR'
    )
    print(classification_report(y_test, lr_pred,
                              target_names=['Benign', 'Malicious']))
    
    print("\n" + "-"*70)
    print("SVM:")
    print("-"*70)
    _, svm_pred, _ = evaluate_model(
        models_dict['svm'], X_test, y_test,
        scaler=scalers_dict.get('svm_scaler'), model_name='SVM'
    )
    print(classification_report(y_test, svm_pred,
                              target_names=['Benign', 'Malicious']))
    
    return comparison_df


if __name__ == "__main__":
    print("Evaluation module - Example usage")
    print("Use evaluate_all_models() to evaluate trained models")
