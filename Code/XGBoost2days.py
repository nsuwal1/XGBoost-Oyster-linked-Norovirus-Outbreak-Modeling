import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score,
    precision_recall_curve, confusion_matrix, precision_score, recall_score
)
from imblearn.over_sampling import SMOTE
import joblib
import logging
import matplotlib.pyplot as plt
from sklearn.utils.validation import check_array
import shap  # SHAP for model interpretation

# Directories for output
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("predictions", exist_ok=True)
os.makedirs("shap_outputs", exist_ok=True)

# Configure logging
logging.basicConfig(
    filename='logs/model_training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Validate dataset
def validate_data(df, required_columns):
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in dataset: {missing_columns}")
    if df.isnull().any().any():
        raise ValueError("Dataset contains missing values.")

# Prepare data
def prepare_data(file_path, sheet_name, target_column, id_column, date_column):
    logging.info("Loading dataset...")
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    validate_data(data, [target_column, id_column, date_column])
    dates = data[date_column]
    ids = data[id_column]
    X = data.drop(columns=[date_column, id_column, target_column])
    y = data[target_column]
    return X, y, dates, ids

# Split and balance
def split_and_balance_data(X, y, dates, ids, test_size=0.4, random_state=42):
    X_train, X_test, y_train, y_test, train_dates, test_dates, train_ids, test_ids = train_test_split(
        X, y, dates, ids, test_size=test_size, random_state=random_state, stratify=y
    )
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    return X_train_resampled, X_test, y_train_resampled, y_test, train_dates, test_dates, train_ids, test_ids

# Train and tune
def train_and_tune_model(X_train, y_train, param_grid, random_state=42):
    X_train = check_array(X_train)
    y_train = check_array(y_train, ensure_2d=False)

    xgb_model = xgb.XGBClassifier(
        eval_metric="logloss",
        enable_categorical=False,
        random_state=random_state
    )
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=StratifiedKFold(n_splits=5),
        verbose=2,
        n_jobs=-1
    )
    logging.info("Starting grid search for hyperparameter tuning...")
    grid_search.fit(X_train, y_train)
    logging.info(f"Best parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

# Evaluate model
def evaluate_model(model, X_test, y_test, ids, dates, output_path):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"AUC-ROC: {auc:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info("Classification Report:\n" + classification_rep)
    logging.info("Confusion Matrix:\n" + str(conf_matrix))

    predictions_df = pd.DataFrame({
        "ID": ids,
        "Date": dates,
        "True Target": y_test,
        "Predicted Probability": y_pred_proba,
        "Predicted Target": y_pred
    })
    predictions_df.to_excel(output_path, index=False)
    logging.info(f"Predictions saved to {output_path}")

# SHAP explanation
def explain_model_with_shap(model, X_train, feature_names, output_dir="shap_outputs"):
    os.makedirs(output_dir, exist_ok=True)
    logging.info("Starting SHAP explanation...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    X_train_df = pd.DataFrame(X_train, columns=feature_names)

    # Save SHAP values to CSV
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_df.to_csv(os.path.join(output_dir, "shap_values.csv"), index=False)
    logging.info("SHAP values saved to shap_outputs/shap_values.csv")

    # Beeswarm plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_train_df, feature_names=feature_names, show=False)
    plt.title("SHAP Beeswarm Plot: Feature Impact on Model Output")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_beeswarm_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("SHAP beeswarm plot saved to shap_outputs/shap_beeswarm_plot.png")

    # Bar plot for feature importance
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_train_df, feature_names=feature_names, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance (Mean Absolute SHAP Values)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_feature_importance_bar.png"), dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("SHAP feature importance bar plot saved to shap_outputs/shap_feature_importance_bar.png")

    # Dependence plots for top 5 features
    top_n = 5
    shap_importance = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(shap_importance)[-top_n:][::-1]

    for idx in top_indices:
        feature = feature_names[idx]
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(
            idx, shap_values, X_train_df, feature_names=feature_names, show=False,
            interaction_index="auto"  # Automatically select interaction feature
        )
        plt.title(f"SHAP Dependence Plot: {feature}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"shap_dependence_{feature}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Dependence plot saved for {feature}")

    # Generate text summary of feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Abs_SHAP': np.abs(shap_values).mean(axis=0)
    }).sort_values(by='Mean_Abs_SHAP', ascending=False)
    
    summary_text = "SHAP Feature Importance Summary:\n"
    summary_text += "Top features contributing to the model predictions:\n"
    for i, row in feature_importance.head(top_n).iterrows():
        summary_text += f"- {row['Feature']}: Mean |SHAP| = {row['Mean_Abs_SHAP']:.4f}\n"
    summary_text += "\nInterpretation:\n"
    summary_text += "- Features with higher mean absolute SHAP values have a greater impact on predictions.\n"
    summary_text += "- Positive SHAP values push predictions toward the positive class (1).\n"
    summary_text += "- Negative SHAP values push predictions toward the negative class (0).\n"
    summary_text += "- Beeswarm plot shows the distribution of SHAP values for each feature.\n"
    summary_text += "- Bar plot summarizes mean absolute SHAP values for feature importance.\n"
    summary_text += "- Dependence plots show how each feature's value affects predictions.\n"

    with open(os.path.join(output_dir, "shap_summary.txt"), "w") as f:
        f.write(summary_text)
    logging.info("SHAP summary text saved to shap_outputs/shap_summary.txt")

# Test on new data
def test_on_new_data(model, new_file_path, sheet_name, id_column, date_column, output_path):
    logging.info("Loading new test data...")
    new_data = pd.read_excel(new_file_path, sheet_name=sheet_name)
    validate_data(new_data, [id_column, date_column])
    new_dates = new_data[date_column]
    new_ids = new_data[id_column]
    X_new = new_data.drop(columns=[date_column, id_column])

    y_new_pred_proba = model.predict_proba(X_new)[:, 1]
    y_new_pred = model.predict(X_new)

    new_predictions_df = pd.DataFrame({
        "ID": new_ids,
        "Date": new_dates,
        "Predicted Probability": y_new_pred_proba,
        "Predicted Target": y_new_pred
    })
    new_predictions_df.to_excel(output_path, index=False)
    logging.info(f"New data predictions saved to {output_path}")
    print(f"New data predictions saved to {output_path}")

# Main script
if __name__ == "__main__":
    try:
        training_file_path = "Training.xlsx"
        sheet_name = "Sheet1"
        target_column = "Target"
        id_column = "ID"
        date_column = "Date"
        param_grid = {
            'n_estimators': [50, 100, 300],
            'learning_rate': [0.01,0.05, 0.1],
            'max_depth': [3, 5, 6],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'gamma': [0, 1, 5],
            'min_child_weight': [1, 5, 10],
        }

        # Step 1: Load and prepare data
        X, y, dates, ids = prepare_data(training_file_path, sheet_name, target_column, id_column, date_column)

        # Step 2: Split and balance data
        X_train, X_test, y_train, y_test, train_dates, test_dates, train_ids, test_ids = split_and_balance_data(
            X, y, dates, ids
        )

        # Step 3: Train and tune the model
        best_model = train_and_tune_model(X_train, y_train, param_grid)

        # Step 3.5: Explain with SHAP
        explain_model_with_shap(best_model, X_train, feature_names=X.columns.tolist())

        # Step 4: Evaluate the model
        evaluate_model(
            best_model,
            X_test,
            y_test,
            test_ids,
            test_dates,
            output_path="predictions/testing_predictions_with_id.xlsx"
        )

        # Step 5: Save the model
        joblib.dump(best_model, "models/xgboost_best_model.pkl")
        logging.info("Model training completed successfully.")

        # Step 6: Test on new data
        test_on_new_data(
            model=best_model,
            new_file_path="Testing.xlsx",
            sheet_name="Sheet1",
            id_column=id_column,
            date_column=date_column,
            output_path="predictions/new_test_predictions_with_id.xlsx"
        )

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        print(f"An error occurred: {str(e)}")