import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lime
import lime.lime_tabular
import seaborn as sns

# Set HD style for research papers with LARGER FONT SIZES
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 14  # Increased from 12
plt.rcParams['axes.titlesize'] = 16  # Increased from 14
plt.rcParams['axes.labelsize'] = 14  # Increased from 12
plt.rcParams['xtick.labelsize'] = 12  # Increased from 10
plt.rcParams['ytick.labelsize'] = 12  # Increased from 10
plt.rcParams['legend.fontsize'] = 12  # Increased from 10
plt.rcParams['figure.titlesize'] = 18  # Increased from 16

sns.set_style("whitegrid")
sns.set_palette("husl")

# 1. Load and prepare the healthcare data
print("Loading Breast Cancer Dataset...")
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = [str(name) for name in data.feature_names]
target_names = ['Malignant', 'Benign']

print(f"Dataset shape: {X.shape}")
print(f"Class distribution: Malignant: {np.sum(y == 0)}, Benign: {np.sum(y == 1)}")

# 2. Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.3f}")

# 3. Create LIME Explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train,
    feature_names=feature_names,
    class_names=target_names,
    mode='classification',
    training_labels=y_train,
    random_state=42,
    verbose=False
)


def safe_lime_explanation(instance, model_predict_proba, explainer, num_features=8):
    """Safely generate LIME explanation handling different data types"""
    exp = explainer.explain_instance(
        instance,
        model_predict_proba,
        num_features=num_features,
        top_labels=2
    )

    available_labels = list(exp.local_exp.keys())
    pred_proba = model_predict_proba([instance])[0]
    predicted_class = int(np.argmax(pred_proba))

    if predicted_class in available_labels:
        explanation_label = predicted_class
    elif available_labels:
        explanation_label = available_labels[0]
    else:
        explanation_label = 0

    exp_list = exp.as_list(label=explanation_label)
    return exp, exp_list, predicted_class, pred_proba


# RESEARCH-READY VISUALIZATION 1: Comparative Case Studies
print(f"\n=== RESEARCH VISUALIZATION 1: COMPARATIVE CASE STUDIES ===")

fig, axes = plt.subplots(2, 2, figsize=(16, 14))  # Increased figure size
axes = axes.flatten()

# Select representative cases
cases_to_plot = []
case_descriptions = []

# Find clear malignant case
malignant_idx = np.where((y_test == 0) & (model.predict_proba(X_test)[:, 0] > 0.95))[0][0]
cases_to_plot.append(malignant_idx)
case_descriptions.append("High-Confidence\nMalignant")

# Find clear benign case
benign_idx = np.where((y_test == 1) & (model.predict_proba(X_test)[:, 1] > 0.95))[0][0]
cases_to_plot.append(benign_idx)
case_descriptions.append("High-Confidence\nBenign")

# Find borderline malignant case
borderline_malignant = np.where((y_test == 0) & (model.predict_proba(X_test)[:, 0] > 0.6) &
                                (model.predict_proba(X_test)[:, 0] < 0.8))[0]
if len(borderline_malignant) > 0:
    cases_to_plot.append(borderline_malignant[0])
    case_descriptions.append("Borderline\nMalignant")
else:
    cases_to_plot.append(np.where(y_test == 0)[0][1])
    case_descriptions.append("Malignant\nCase 2")

# Find borderline benign case
borderline_benign = np.where((y_test == 1) & (model.predict_proba(X_test)[:, 1] > 0.6) &
                             (model.predict_proba(X_test)[:, 1] < 0.8))[0]
if len(borderline_benign) > 0:
    cases_to_plot.append(borderline_benign[0])
    case_descriptions.append("Borderline\nBenign")
else:
    cases_to_plot.append(np.where(y_test == 1)[0][1])
    case_descriptions.append("Benign\nCase 2")

# Plot each case
for i, case_idx in enumerate(cases_to_plot):
    instance = X_test[case_idx]
    exp, exp_list, pred_class, pred_proba = safe_lime_explanation(
        instance, model.predict_proba, explainer, num_features=6
    )

    if exp_list:
        features, scores = zip(*exp_list)
        # Shorten feature names for better readability
        short_features = []
        for feat in features:
            if 'worst' in feat:
                short_feat = feat.replace('worst ', 'w.')
            elif 'mean' in feat:
                short_feat = feat.replace('mean ', 'm.')
            else:
                short_feat = feat
            short_features.append(short_feat)

        colors = ['#E74C3C' if score < 0 else '#27AE60' for score in scores]  # Red/Green colors

        y_pos = np.arange(len(features))
        bars = axes[i].barh(y_pos, scores, color=colors, alpha=0.8, height=0.7)
        axes[i].set_yticks(y_pos)
        axes[i].set_yticklabels(short_features, fontsize=11)  # Increased from 9
        axes[i].set_xlabel('Feature Impact Score', fontsize=13)  # Increased from 10
        axes[i].axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=0.8)

        # Add value annotations on bars
        for j, (bar, score) in enumerate(zip(bars, scores)):
            width = bar.get_width()
            ha = 'left' if width > 0 else 'right'
            x_pos = width + (0.01 if width > 0 else -0.01)
            axes[i].text(x_pos, bar.get_y() + bar.get_height() / 2,
                         f'{score:.2f}', ha=ha, va='center', fontsize=10, fontweight='bold')  # Increased from 8

        # Case info
        actual_label = 'Malignant' if y_test[case_idx] == 0 else 'Benign'
        pred_label = 'Malignant' if pred_class == 0 else 'Benign'
        prob_malignant = pred_proba[0]

        title = f"{case_descriptions[i]}\nActual: {actual_label}, Pred: {pred_label}\nP(Malignant) = {prob_malignant:.3f}"
        axes[i].set_title(title, fontsize=13, fontweight='bold', pad=10)  # Increased from 11

        axes[i].grid(axis='x', alpha=0.3, linestyle='--')
        axes[i].set_xlim(-1.2, 1.2)

plt.suptitle('LIME Explanations: Comparative Case Studies in Breast Cancer Diagnosis',
             fontsize=18, fontweight='bold', y=0.98)  # Increased from 16
plt.tight_layout()
plt.savefig('lime_comparative_cases.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# RESEARCH-READY VISUALIZATION 2: Aggregate Feature Importance
print(f"\n=== RESEARCH VISUALIZATION 2: AGGREGATE FEATURE IMPORTANCE ===")

feature_impact = {}
n_cases = min(50, len(X_test))
successful_explanations = 0

for i in range(n_cases):
    try:
        instance = X_test[i]
        exp, exp_list, pred_class, pred_proba = safe_lime_explanation(
            instance, model.predict_proba, explainer, num_features=8
        )

        if exp_list:
            successful_explanations += 1
            for feature, impact in exp_list:
                base_feature = feature.split(' ')[0] if ' <= ' in feature or ' > ' in feature else feature
                if base_feature not in feature_impact:
                    feature_impact[base_feature] = []
                feature_impact[base_feature].append(abs(impact))
    except:
        continue

print(f"Analyzed {successful_explanations} cases for feature importance")

if feature_impact:
    avg_impact = {feature: np.mean(impacts) for feature, impacts in feature_impact.items()}
    # Filter to top 12 features for clarity
    top_features = sorted(avg_impact.items(), key=lambda x: x[1], reverse=True)[:12]

    plt.figure(figsize=(16, 10))  # Increased figure size
    features, impacts = zip(*top_features)

    # Create horizontal bar plot
    y_pos = np.arange(len(features))
    bars = plt.barh(y_pos, impacts, color='#3498DB', alpha=0.8, height=0.7)

    plt.yticks(y_pos, features, fontsize=13)  # Increased from 11
    plt.xlabel('Average Absolute Impact Score\n(Mean |LIME Feature Importance|)',
               fontsize=14, fontweight='bold')  # Increased from 12

    # Add value annotations
    for i, (bar, impact) in enumerate(zip(bars, impacts)):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{impact:.3f}', ha='left', va='center', fontsize=12, fontweight='bold')  # Increased from 10

    plt.title('Aggregate Feature Importance in Breast Cancer Diagnosis\n'
              f'Based on {successful_explanations} Individual LIME Explanations',
              fontsize=18, fontweight='bold', pad=20)  # Increased from 16
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.xlim(0, max(impacts) * 1.15)
    plt.tight_layout()
    plt.savefig('lime_aggregate_importance.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

    print("\nTop 5 most influential features:")
    for feature, impact in top_features[:5]:
        print(f"  - {feature}: {impact:.4f}")

# RESEARCH-READY VISUALIZATION 3: Feature Impact Distribution
print(f"\n=== RESEARCH VISUALIZATION 3: FEATURE IMPACT DISTRIBUTION ===")

# Analyze impact directions for top features
top_feature_names = [feat for feat, _ in top_features[:6]]
feature_directions = {feat: [] for feat in top_feature_names}

for i in range(min(30, len(X_test))):
    try:
        instance = X_test[i]
        exp, exp_list, pred_class, pred_proba = safe_lime_explanation(
            instance, model.predict_proba, explainer, num_features=10
        )

        if exp_list:
            for feature, impact in exp_list:
                base_feature = feature.split(' ')[0] if ' <= ' in feature or ' > ' in feature else feature
                if base_feature in top_feature_names:
                    feature_directions[base_feature].append(impact)
    except:
        continue

plt.figure(figsize=(16, 10))  # Increased figure size
box_data = [feature_directions[feat] for feat in top_feature_names if feature_directions[feat]]

box_plot = plt.boxplot(box_data, vert=False, patch_artist=True, labels=top_feature_names[:len(box_data)])

# Color boxes based on median impact direction
for i, box in enumerate(box_plot['boxes']):
    medians = np.median(box_data[i])
    if medians > 0:
        box.set(facecolor='#27AE60', alpha=0.7)  # Green for positive impact
    else:
        box.set(facecolor='#E74C3C', alpha=0.7)  # Red for negative impact

plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
plt.xlabel('LIME Feature Impact Score Distribution', fontsize=14, fontweight='bold')  # Increased from 12
plt.ylabel('Top Features', fontsize=14, fontweight='bold')  # Increased from 12
plt.title('Distribution of Feature Impact Scores Across Multiple Explanations\n'
          '(Green: Typically supports Malignant, Red: Typically supports Benign)',
          fontsize=16, fontweight='bold', pad=20)  # Increased from 14
plt.grid(axis='x', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('lime_impact_distribution.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print("\n" + "=" * 60)
print("RESEARCH SUMMARY")
print("=" * 60)
print("✓ Comparative case studies show instance-level explanations")
print("✓ Aggregate analysis reveals consistently important features")
print("✓ Impact distribution shows feature reliability across patients")
print("✓ HD visualizations suitable for research publications")
print("✓ Provides both local and global model interpretability")