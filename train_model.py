import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, f1_score
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV
import joblib
import os
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import warnings
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import GridSearchCV
warnings.filterwarnings('ignore')

def load_and_preprocess_data(file_path):
    """Load and preprocess the dataset with enhanced feature engineering"""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Basic info about the dataset
    print("\nDataset Info:")
    print(f"Number of records: {len(df)}")
    print(f"Number of features: {len(df.columns)}")
    print("\nFirst few records:")
    print(df.head())
    
    # Check for missing values
    print("\nMissing values per column:")
    print(df.isnull().sum())
    
    # Handle missing values
    df = df.dropna()
    
    # Convert categorical variables
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    
    # Enhanced Feature Engineering
    # --------------------------
    
    # 1. Vital signs risk tiers with medical thresholds
    # SpO2 risk levels with finer granularity
    df['spo2_critical'] = (df['SpO2 (%)'] < 80).astype(int) * 4    # Critical hypoxemia
    df['spo2_severe'] = ((df['SpO2 (%)'] >= 80) & (df['SpO2 (%)'] < 85)).astype(int) * 3  # Severe hypoxemia
    df['spo2_moderate'] = ((df['SpO2 (%)'] >= 85) & (df['SpO2 (%)'] < 90)).astype(int) * 2  # Moderate hypoxemia
    df['spo2_mild'] = ((df['SpO2 (%)'] >= 90) & (df['SpO2 (%)'] < 95)).astype(int)  # Mild hypoxemia
    
    # Temperature risk levels with clinical thresholds
    df['temp_critical'] = (df['Temperature (°C)'] > 39.5).astype(int) * 4   # Critical fever
    df['temp_severe'] = ((df['Temperature (°C)'] > 38.5) & (df['Temperature (°C)'] <= 39.5)).astype(int) * 3   # Severe fever
    df['temp_moderate'] = ((df['Temperature (°C)'] > 38.0) & (df['Temperature (°C)'] <= 38.5)).astype(int) * 2   # Moderate fever
    df['temp_mild'] = ((df['Temperature (°C)'] > 37.5) & (df['Temperature (°C)'] <= 38.0)).astype(int)   # Mild fever
    df['temp_low'] = (df['Temperature (°C)'] < 36.0).astype(int) * 2   # Hypothermia risk
    
    # Dehydration risk
    df['dehydration_critical'] = (df['Dehydration_Label'] >= 2).astype(int) * 3
    df['dehydration_moderate'] = (df['Dehydration_Label'] == 1).astype(int) * 1.5
    
    # 2. Advanced transformations
    # Exponential transformations for critical values
    df['exp_temp'] = np.exp(df['Temperature (°C)'] - 37.0)  # Exponential deviation from normal
    df['exp_spo2'] = np.exp((100 - df['SpO2 (%)']) / 10)    # Exponential oxygen deficit
    
    # Log transformations for skewed variables
    if 'GSR Value' in df.columns:
        df['log_gsr'] = np.log1p(df['GSR Value'])
    
    # 3. Age-related risk factors
    df['is_child'] = (df['Age'] < 12).astype(int) * 2
    df['is_elderly'] = (df['Age'] > 65).astype(int) * 2
    df['age_risk'] = df['is_child'] + df['is_elderly']
    
    # Age buckets with medical relevance
    age_bins = [0, 2, 5, 12, 18, 40, 65, 75, 85, 100]
    age_labels = list(range(len(age_bins)-1))
    df['age_group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)
    
    # 4. Interaction features
    # Vital sign interactions
    df['temp_spo2_critical'] = df['temp_critical'] * df['spo2_critical']
    df['temp_spo2_interaction'] = df['Temperature (°C)'] * (100 - df['SpO2 (%)'])
    
    # Age-weighted risks
    df['age_temp_risk'] = (1 + 0.5 * df['age_risk']) * (1 + df['temp_critical'] + 0.5 * df['temp_severe'])
    df['age_spo2_risk'] = (1 + 0.5 * df['age_risk']) * (1 + df['spo2_critical'] + 0.5 * df['spo2_severe'])
    
    # 5. Clinical severity scores
    # Combined clinical risk score with medical weighting
    df['clinical_risk_score'] = (
        df['spo2_critical'] * 2 + df['spo2_severe'] * 1.5 + df['spo2_moderate'] + df['spo2_mild'] * 0.5 +
        df['temp_critical'] * 2 + df['temp_severe'] * 1.5 + df['temp_moderate'] + df['temp_mild'] * 0.5 +
        df['dehydration_critical'] * 1.5 + df['dehydration_moderate'] +
        df['age_risk']
    )
    
    # Medical risk score combining features using domain knowledge
    df['medical_risk_score'] = (
        ((100 - df['SpO2 (%)']) / 5) +              # Oxygen deficit
        ((df['Temperature (°C)'] - 37.0) * 3) +     # Temperature deviation
        (df['Dehydration_Label'] * 2) +             # Dehydration factor
        (df['age_risk'])                            # Age risk
    )
    
    # GSR clinical correlation if available
    if 'GSR Value' in df.columns:
        # Normalize GSR
        gsr_mean = df['GSR Value'].mean()
        gsr_std = df['GSR Value'].std()
        df['gsr_norm'] = (df['GSR Value'] - gsr_mean) / gsr_std
        df['gsr_norm'] = np.clip(df['gsr_norm'], -3, 3)  # Handle outliers
        
        # GSR clinical interactions
        df['gsr_clinical'] = df['gsr_norm'] * df['clinical_risk_score']
    
    # 6. Select features based on medical relevance
    features = [
        # Raw vital signs
        'SpO2 (%)', 'Temperature (°C)', 'Dehydration_Label',
        
        # Demographics
        'Age', 'Gender', 'age_group', 'is_child', 'is_elderly', 'age_risk',
        
        # Risk tiers
        'spo2_critical', 'spo2_severe', 'spo2_moderate', 'spo2_mild',
        'temp_critical', 'temp_severe', 'temp_moderate', 'temp_mild', 'temp_low',
        'dehydration_critical', 'dehydration_moderate',
        
        # Transformations
        'exp_temp', 'exp_spo2',
        
        # Interactions
        'temp_spo2_critical', 'temp_spo2_interaction',
        'age_temp_risk', 'age_spo2_risk',
        
        # Clinical scores
        'clinical_risk_score', 'medical_risk_score'
    ]
    
    # Add GSR features if available
    if 'GSR Value' in df.columns:
        gsr_features = ['GSR Value', 'log_gsr', 'gsr_norm', 'gsr_clinical']
        features.extend(gsr_features)
    
    # Ensure all selected features exist
    features = [f for f in features if f in df.columns]
    
    # Print selected features
    print("\nSelected features:")
    for feature in features:
        print(f"- {feature}")
    
    # Split the data
    X = df[features]
    y = df['Crisis_Flag']
    
    # Check feature correlation
    print("\nChecking feature correlations...")
    corr_matrix = X.corr().abs()
    
    # Identify highly correlated features (|r| > 0.9)
    high_corr_features = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] > 0.9:
                colname = corr_matrix.columns[i]
                high_corr_features.add(colname)
    
    if high_corr_features:
        print(f"\nFound {len(high_corr_features)} highly correlated features:")
        print(high_corr_features)
        
        # Plot correlation matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
        plt.title('Feature Correlation Matrix')
        plt.savefig('feature_correlation.png')
        plt.close()
        
        # Remove one of each highly correlated pair
        X = X.drop(columns=high_corr_features)
        print(f"Removed highly correlated features. Remaining: {X.shape[1]}")
    
    # Print class distribution
    print("\nClass distribution:")
    print(y.value_counts())
    print(y.value_counts(normalize=True))
    
    return X, y, df

def train_model(X, y):
    """Train multiple models with advanced techniques"""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Check for class imbalance and apply SMOTE if needed
    class_counts = np.bincount(y_train)
    if abs(class_counts[0] - class_counts[1]) > 1:  # More than 1 sample difference
        print("\nApplying SMOTE to handle class imbalance...")
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
        print(f"Original training class distribution: {np.bincount(y_train)}")
        print(f"Resampled training class distribution: {np.bincount(y_train_resampled)}")
    else:
        print("\nClasses are balanced, skipping SMOTE...")
        X_train_resampled, y_train_resampled = X_train_scaled, y_train
    
    # Calculate class weights
    n_samples = len(y_train)
    n_classes = len(np.unique(y_train))
    n_pos = np.sum(y_train == 1)
    n_neg = n_samples - n_pos
    pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    
    # Create base models with stability-focused parameters
    base_models = [
        ('lr', LogisticRegression(
            class_weight='balanced',
            penalty='l2',
            C=0.2,  # Balanced regularization
            max_iter=2000,
            random_state=42,
            solver='liblinear'
        )),
        ('rf', RandomForestClassifier(
            n_estimators=150,  # Balanced number of trees
            max_depth=2,  # Shallower trees
            min_samples_split=4,
            min_samples_leaf=4,  # More conservative splitting
            max_features='sqrt',
            class_weight='balanced',
            bootstrap=True,
            random_state=42
        )),
        ('xgb', XGBClassifier(
            n_estimators=75,  # Fewer trees
            max_depth=2,  # Shallower trees
            learning_rate=0.1,
            min_child_weight=4,  # More conservative splitting
            gamma=0.2,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            random_state=42,
            eval_metric='auc'
        ))
    ]
    
    # Create meta-learner with stability-focused parameters
    meta_learner = LogisticRegression(
        class_weight='balanced',
        penalty='l2',
        C=0.2,  # Balanced regularization
        max_iter=2000,
        random_state=42,
        solver='liblinear'
    )
    
    # Create stacking ensemble with stability focus
    print("\nTraining stacking ensemble...")
    ensemble = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_learner,
        cv=10,  # More folds for stability
        n_jobs=-1,
        passthrough=True,
        stack_method='predict_proba'
    )
    
    # Train ensemble model
    ensemble.fit(X_train_resampled, y_train_resampled)
    
    # Evaluate ensemble
    y_ensemble_proba = ensemble.predict_proba(X_test_scaled)[:, 1]
    ensemble_roc_auc = roc_auc_score(y_test, y_ensemble_proba)
    print(f"Stacking Ensemble - ROC AUC: {ensemble_roc_auc:.4f}")
    
    # Select final model based on performance
    if ensemble_roc_auc > 0.7:
        print("Stacking ensemble selected as final model")
        final_model = ensemble
    else:
        print(f"{base_models[0][0]} selected as final model")
        final_model = base_models[0][1]  # Use first base model
    
    # Find optimal threshold and evaluate final model
    print("\nFinding optimal threshold for final model...")
    optimal_threshold = find_optimal_threshold(final_model, X_test_scaled, y_test)
    
    # Comprehensive evaluation
    evaluate_model(final_model, X_test, y_test, scaler, optimal_threshold)
    
    # Feature importance analysis if possible
    if base_models[0][0] == 'Random Forest' or base_models[0][0] == 'XGBoost' or base_models[0][0] == 'CatBoost':
        print("\nAnalyzing feature importance...")
        model_for_importance = base_models[0][1]
        analyze_feature_importance(model_for_importance, X.columns)
    
    return final_model, scaler, optimal_threshold

def find_optimal_threshold(model, X_test_scaled, y_test):
    """Find the optimal threshold for classification"""
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    
    # Calculate F1 score for each threshold
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
    
    # Find threshold that maximizes F1 score
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    
    print(f"Optimal threshold: {best_threshold:.4f}")
    print(f"At optimal threshold - Precision: {precision[best_idx]:.4f}, Recall: {recall[best_idx]:.4f}")
    
    return best_threshold

def evaluate_model(model, X_test, y_test, scaler, threshold=0.5):
    """Comprehensive model evaluation"""
    X_test_scaled = scaler.transform(X_test)
    
    # Get predictions
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
    
    print(f"\nSensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Positive Predictive Value (Precision): {ppv:.4f}")
    print(f"Negative Predictive Value: {npv:.4f}")
    
    # ROC AUC
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # Average Precision Score (AP)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    print(f"Average Precision Score: {avg_precision:.4f}")
    
    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()
    
    # Plot Precision-Recall curve
    plt.figure(figsize=(10, 8))
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.plot(recall, precision, label=f'PR curve (AP = {avg_precision:.3f})')
    plt.axhline(y=np.sum(y_test) / len(y_test), color='r', linestyle='--', 
                label=f'Baseline (AP = {np.sum(y_test) / len(y_test):.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")
    plt.savefig('precision_recall_curve.png')
    plt.close()
    
    # Plot calibration curve if sklearn version supports it
    try:
        from sklearn.calibration import calibration_curve
        plt.figure(figsize=(10, 8))
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_pred_proba, n_bins=10)
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
        plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Fraction of positives")
        plt.title("Calibration Curve")
        plt.legend(loc="lower right")
        plt.savefig('calibration_curve.png')
        plt.close()
    except:
        print("Calibration curve plot skipped - sklearn version may not support it")


def analyze_feature_importance(model, feature_names):
    """Analyze and visualize feature importance"""
    # Get feature importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        print("Model doesn't provide feature importance. Skipping analysis.")
        return
    
    # Create DataFrame for better visualization
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance
    feature_importance = feature_importance.sort_values('Importance', ascending=False).reset_index(drop=True)
    
    # Display top 15 features
    print("\nTop 15 important features:")
    print(feature_importance.head(15))
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title('Top 15 Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    return feature_importance


def save_model(model, scaler, threshold, output_dir='models'):
    """Save the trained model and associated artifacts"""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate timestamp for model versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model
    model_path = os.path.join(output_dir, f'crisis_model_{timestamp}.joblib')
    joblib.dump(model, model_path)
    
    # Save scaler
    scaler_path = os.path.join(output_dir, f'scaler_{timestamp}.joblib')
    joblib.dump(scaler, scaler_path)
    
    # Save threshold
    with open(os.path.join(output_dir, f'threshold_{timestamp}.txt'), 'w') as f:
        f.write(str(threshold))
    
    print(f"\nModel saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    print(f"Optimal threshold: {threshold}")
    
    return model_path, scaler_path


def cross_validate_model(X, y, n_splits=5):
    """Perform cross-validation to assess model stability"""
    print("\nPerforming cross-validation...")
    
    # Create stratified k-fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Instantiate model
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        min_child_weight=2,
        gamma=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=np.sum(y==0) / np.sum(y==1),
        random_state=42,
        eval_metric='auc'
    )
    
    # Performance metrics
    roc_auc_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []
    
    # Perform cross-validation
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold+1}/{n_splits}")
        
        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Apply SMOTE to handle class imbalance
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
        
        # Train model
        model.fit(X_train_resampled, y_train_resampled)
        
        # Predict
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Find optimal threshold
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        f1_scores_th = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
        best_idx = np.argmax(f1_scores_th)
        best_threshold = thresholds[best_idx] if len(thresholds) > 0 else 0.5
        
        # Apply threshold
        y_pred = (y_pred_proba >= best_threshold).astype(int)
        
        # Calculate metrics
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        # Store metrics
        roc_auc_scores.append(roc_auc)
        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)
        
        print(f"Fold {fold+1} - ROC AUC: {roc_auc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    
    # Calculate mean and standard deviation
    print("\nCross-validation results:")
    print(f"ROC AUC: {np.mean(roc_auc_scores):.4f} (±{np.std(roc_auc_scores):.4f})")
    print(f"F1 Score: {np.mean(f1_scores):.4f} (±{np.std(f1_scores):.4f})")
    print(f"Precision: {np.mean(precision_scores):.4f} (±{np.std(precision_scores):.4f})")
    print(f"Recall: {np.mean(recall_scores):.4f} (±{np.std(recall_scores):.4f})")


def main(data_path, save_dir='models'):
    """Main function to run the complete pipeline"""
    print("="*80)
    print("MEDICAL CRISIS PREDICTION MODEL PIPELINE")
    print("="*80)
    
    # Load and process data
    X, y, df = load_and_preprocess_data(data_path)
    
    # Cross-validate to check model stability
    cross_validate_model(X, y)
    
    # Train and evaluate the model
    model, scaler, threshold = train_model(X, y)
    
    # Save model and artifacts
    save_model(model, scaler, threshold, save_dir)
    
    print("\nProcess completed successfully!")


if __name__ == "__main__":
    # Define data path
    data_file = "final_dataset_rescaled.csv"  # Using the rescaled data file
    
    # Run the pipeline
    main(data_file)