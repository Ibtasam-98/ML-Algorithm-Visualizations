import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import grad
import os

# Create directory for saving plots
output_dir = "SHAP_Results_Breast_Cancer"
os.makedirs(output_dir, exist_ok=True)

# Set high-quality style for research papers
plt.style.use('default')
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Load breast cancer dataset
print("Loading Breast Cancer Dataset...")
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print(f"Dataset shape: {X.shape}")
print(f"Target distribution: {pd.Series(y).value_counts()}")
print(f"Malignant (1): {pd.Series(y).value_counts()[1]}, Benign (0): {pd.Series(y).value_counts()[0]}")

# Preprocess the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Train a Random Forest classifier
print("\nTraining Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Calculate accuracy
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Initialize SHAP explainer
print("\nInitializing SHAP Explainer...")
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

print(f"SHAP values shape: {shap_values.shape}")

# Get the expected value and SHAP values for the positive class (malignant)
expected_value = explainer.expected_value
if isinstance(expected_value, np.ndarray) and len(expected_value) > 1:
    expected_value = expected_value[1]  # Use malignant class
    shap_values_pos = shap_values[..., 1]  # SHAP values for malignant class
else:
    shap_values_pos = shap_values.values

print(f"Expected value: {expected_value}")

# Create comprehensive SHAP plots with high-quality settings
print("\nGenerating and Saving SHAP Visualizations...")

# 1. Summary Plot (Bee Swarm Plot)
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values_pos, X_test, feature_names=data.feature_names, show=False)
plt.title("SHAP Summary Plot - Feature Importance for Breast Cancer Classification",
          fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f'{output_dir}/SHAP_Summary_Plot.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig(f'{output_dir}/SHAP_Summary_Plot.pdf', bbox_inches='tight')
plt.show()

# 2. Bar Plot (Global Feature Importance)
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values_pos, X_test, feature_names=data.feature_names,
                  plot_type="bar", show=False)
plt.title("SHAP Feature Importance (Global) - Breast Cancer Dataset",
          fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f'{output_dir}/SHAP_Bar_Plot.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig(f'{output_dir}/SHAP_Bar_Plot.pdf', bbox_inches='tight')
plt.show()

# 3. Waterfall Plot for individual predictions
print("\nGenerating individual prediction explanations...")

# Create waterfall plots for 3 different samples
for sample_idx in [0, 10, 20]:  # Multiple samples for demonstration
    plt.figure(figsize=(14, 8))

    # Create Explanation object for waterfall plot
    explanation = shap.Explanation(
        values=shap_values_pos[sample_idx],
        base_values=expected_value,
        data=X_test.iloc[sample_idx],
        feature_names=data.feature_names
    )

    shap.waterfall_plot(explanation, show=False)
    actual_class = "Malignant" if y_test[sample_idx] == 1 else "Benign"
    pred_proba = model.predict_proba(X_test.iloc[sample_idx:sample_idx + 1])[0][1]
    plt.title(
        f"SHAP Waterfall Plot - Sample {sample_idx} (Actual: {actual_class}, Malignant Probability: {pred_proba:.3f})",
        fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/SHAP_Waterfall_Plot_Sample_{sample_idx}.png',
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(f'{output_dir}/SHAP_Waterfall_Plot_Sample_{sample_idx}.pdf',
                bbox_inches='tight')
    plt.show()

# 4. Dependence Plot for top features
shap_importance = np.abs(shap_values_pos).mean(0)
top_features = np.argsort(shap_importance)[-5:][::-1]
top_feature_names = [data.feature_names[i] for i in top_features]

print(f"\nTop 5 most important features: {top_feature_names}")

# Create dependence plots for top 3 features
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, feature_idx in enumerate(top_features[:3]):
    shap.dependence_plot(
        feature_idx,
        shap_values_pos,
        X_test,
        feature_names=data.feature_names,
        ax=axes[i],
        show=False
    )
    axes[i].set_title(f'SHAP Dependence: {data.feature_names[feature_idx]}',
                      fontweight='bold', fontsize=12)
    axes[i].tick_params(axis='x', rotation=45)

plt.suptitle('SHAP Dependence Plots - Top 3 Features', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{output_dir}/SHAP_Dependence_Plots.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig(f'{output_dir}/SHAP_Dependence_Plots.pdf', bbox_inches='tight')
plt.show()

# 5. Enhanced Heatmap of SHAP values
plt.figure(figsize=(16, 12))
shap_values_df = pd.DataFrame(shap_values_pos, columns=data.feature_names)

# Select top 15 features for better visualization
top_15_features = shap_values_df.abs().mean().nlargest(15).index
shap_values_top = shap_values_df[top_15_features]

# Create enhanced heatmap
sns.heatmap(
    shap_values_top.iloc[:30],  # First 30 samples for clarity
    cmap='RdBu_r',
    center=0,
    cbar_kws={'label': 'SHAP Value (Impact on Prediction)'},
    annot=False,
    linewidths=0.5
)
plt.title(
    'SHAP Values Heatmap - Top 15 Features (First 30 Samples)\nRed: Increases Malignant Probability, Blue: Decreases Malignant Probability',
    fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Features', fontweight='bold')
plt.ylabel('Patient Samples', fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=9)
plt.tight_layout()
plt.savefig(f'{output_dir}/SHAP_Heatmap.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig(f'{output_dir}/SHAP_Heatmap.pdf', bbox_inches='tight')
plt.show()

# 6. Decision Plot for multiple samples
plt.figure(figsize=(14, 10))
shap.decision_plot(expected_value, shap_values_pos[:20],
                   feature_names=data.feature_names, show=False)
plt.title('SHAP Decision Plot - First 20 Samples\n(How Features Influence Individual Predictions)',
          fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f'{output_dir}/SHAP_Decision_Plot.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig(f'{output_dir}/SHAP_Decision_Plot.pdf', bbox_inches='tight')
plt.show()

# 7. Enhanced Feature Importance Comparison
print("\n" + "=" * 60)
print("COMPREHENSIVE FEATURE IMPORTANCE ANALYSIS")
print("=" * 60)

# Get feature importance from model and SHAP
model_importance = model.feature_importances_
shap_importance = np.abs(shap_values_pos).mean(0)

# Create comparison DataFrame
importance_df = pd.DataFrame({
    'Feature': data.feature_names,
    'RF_Importance': model_importance,
    'SHAP_Importance': shap_importance
}).sort_values('SHAP_Importance', ascending=False)

print("\nTop 10 Most Important Features (SHAP):")
print(importance_df.head(10).round(4))

# Enhanced comparison plot
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Model importance - enhanced
bars1 = axes[0].barh(range(10), importance_df.head(10)['RF_Importance'][::-1],
                     color='steelblue', alpha=0.8)
axes[0].set_yticks(range(10))
axes[0].set_yticklabels(importance_df.head(10)['Feature'][::-1], fontsize=10)
axes[0].set_xlabel('Feature Importance Score', fontweight='bold')
axes[0].set_title('Random Forest Feature Importance\n(Top 10 Features)',
                  fontweight='bold', fontsize=14)
axes[0].grid(axis='x', alpha=0.3)

# SHAP importance - enhanced
bars2 = axes[1].barh(range(10), importance_df.head(10)['SHAP_Importance'][::-1],
                     color='coral', alpha=0.8)
axes[1].set_yticks(range(10))
axes[1].set_yticklabels(importance_df.head(10)['Feature'][::-1], fontsize=10)
axes[1].set_xlabel('Mean |SHAP Value|', fontweight='bold')
axes[1].set_title('SHAP Feature Importance\n(Top 10 Features)',
                  fontweight='bold', fontsize=14)
axes[1].grid(axis='x', alpha=0.3)

plt.suptitle('Comparison of Feature Importance Metrics - Breast Cancer Classification',
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(f'{output_dir}/Feature_Importance_Comparison.png', dpi=300,
            bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig(f'{output_dir}/Feature_Importance_Comparison.pdf', bbox_inches='tight')
plt.show()

# 8. Save summary statistics and results
with open(f'{output_dir}/SHAP_Analysis_Summary.txt', 'w') as f:
    f.write("SHAP ANALYSIS SUMMARY - BREAST CANCER DATASET\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Dataset Information:\n")
    f.write(f"- Total samples: {X.shape[0]}\n")
    f.write(f"- Features: {X.shape[1]}\n")
    f.write(f"- Malignant cases: {pd.Series(y).value_counts()[1]}\n")
    f.write(f"- Benign cases: {pd.Series(y).value_counts()[0]}\n\n")

    f.write(f"Model Performance:\n")
    f.write(f"- Training Accuracy: {train_accuracy:.4f}\n")
    f.write(f"- Test Accuracy: {test_accuracy:.4f}\n\n")

    f.write("Top 10 Most Important Features (SHAP):\n")
    for i, row in importance_df.head(10).iterrows():
        f.write(f"{i + 1:2d}. {row['Feature']:25} SHAP Importance: {row['SHAP_Importance']:.4f}\n")

    f.write(f"\nExpected Value (Base Rate): {expected_value:.4f}\n")
    f.write("Interpretation: This is the model's output when no features are considered\n")

# Create a results summary DataFrame
results_summary = importance_df.head(10).copy()
results_summary['Rank'] = range(1, 11)
results_summary = results_summary[['Rank', 'Feature', 'SHAP_Importance', 'RF_Importance']]
results_summary.to_csv(f'{output_dir}/Top_Features_Summary.csv', index=False)

# Print detailed sample analysis
print(f"\nDetailed Sample Analysis:")
for sample_idx in [0, 10, 20]:
    actual_class = "Malignant" if y_test[sample_idx] == 1 else "Benign"
    pred_proba = model.predict_proba(X_test.iloc[sample_idx:sample_idx + 1])[0][1]
    pred_class = "Malignant" if pred_proba > 0.5 else "Benign"

    print(f"\nSample {sample_idx}:")
    print(f"  Actual: {actual_class}, Predicted: {pred_class}")
    print(f"  Malignant Probability: {pred_proba:.4f}")
    print(f"  Correct: {'✓' if actual_class == pred_class else '✗'}")

print(f"\n{'=' * 60}")
print("ANALYSIS COMPLETE - ALL VISUALIZATIONS SAVED")
print(f"{'=' * 60}")
print(f"✓ All plots saved in: {output_dir}/")
print(f"✓ Formats: PNG (300 DPI) + PDF (vector)")
print(f"✓ Total visualizations generated: 8 different plot types")
print(f"✓ Summary statistics saved: SHAP_Analysis_Summary.txt")
print(f"✓ Top features data: Top_Features_Summary.csv")

print("\nKey Clinical Insights:")
print("1. SHAP reveals which features drive malignant vs benign classifications")
print("2. Waterfall plots show individual patient risk factors")
print("3. Global patterns identify most important diagnostic markers")
print("4. Dependence plots show how feature values impact predictions")
print("5. Decision plots track prediction paths for multiple patients")

print(f"\nResearch Paper Ready Visualizations:")
print("✓ High-resolution plots (300 DPI)")
print("✓ Vector PDF versions for publication")
print("✓ Consistent styling and professional formatting")
print("✓ Comprehensive analysis summary")
print("✓ Multiple sample analyses for robustness")