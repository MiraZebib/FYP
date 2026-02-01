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


def analyze_false_negatives(y_test, y_pred, model_name, X_test_urls=None):
    fn_indices = np.where((y_test == 1) & (y_pred == 0))[0]
    fn_count = len(fn_indices)
    total_malicious = np.sum(y_test == 1)
    fn_rate = fn_count / total_malicious if total_malicious > 0 else 0
    
    print(f"\n{'='*70}")
    print(f"False Negative Analysis - {model_name}")
    print(f"{'='*70}")
    print(f"Total Malicious URLs in Test Set: {total_malicious}")
    print(f"False Negatives (FN): {fn_count}")
    print(f"False Negative Rate: {fn_rate:.2%}")
    print(f"WARNING: {fn_count} malicious URLs were incorrectly classified as benign")
    
    if fn_count > 0 and X_test_urls is not None:
        print(f"\nSample False Negative URLs (first 5):")
        for i, idx in enumerate(fn_indices[:5]):
            print(f"  {i+1}. {X_test_urls[idx]}")
    
    return fn_indices, fn_count, fn_rate


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
    
    rf_metrics, rf_pred, rf_proba = evaluate_model(
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
    
    analyze_false_negatives(y_test, rf_pred, 'Random Forest')
    
    lr_metrics, lr_pred, lr_proba = evaluate_model(
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
    
    analyze_false_negatives(y_test, lr_pred, 'Logistic Regression')
    
    gb_metrics, gb_pred, gb_proba = evaluate_model(
        models_dict['gradient_boosting'], X_test, y_test,
        scaler=None, model_name='Gradient Boosting'
    )
    all_metrics.append(gb_metrics)
    y_proba_list.append(gb_proba)
    model_names.append('Gradient Boosting')
    
    plot_confusion_matrix(
        gb_metrics['confusion_matrix'],
        'Gradient Boosting',
        os.path.join(plots_dir, 'confusion_matrix_gb.png')
    )
    plt.close()
    
    analyze_false_negatives(y_test, gb_pred, 'Gradient Boosting')
    
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
    print("Gradient Boosting:")
    print("-"*70)
    _, gb_pred, _ = evaluate_model(
        models_dict['gradient_boosting'], X_test, y_test,
        scaler=None, model_name='GB'
    )
    print(classification_report(y_test, gb_pred,
                              target_names=['Benign', 'Malicious']))
    
    return comparison_df


if __name__ == "__main__":
    print("Evaluation module - Example usage")
    print("Use evaluate_all_models() to evaluate trained models")
