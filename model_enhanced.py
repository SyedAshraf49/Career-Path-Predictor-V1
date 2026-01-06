# model_enhanced.py - FIXED with correct paths
"""
Enhanced model training with ensemble methods
Saves models to models/ folder
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

# Create models folder if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')
    print("✅ Created 'models/' folder")

print("="*60)
print("Enhanced Career Prediction Model Training")
print("="*60)

# ==================== 1. LOAD AND PREPARE DATA ====================
print("\n📊 Loading data...")

# Load training data from datasets/ folder
df = pd.read_csv("datasets/training_data_comprehensive.csv")
print(f"Loaded {len(df)} training samples")
print(f"Number of unique careers: {df['career_title'].nunique()}")

# ==================== 2. FEATURE ENGINEERING ====================
print("\n🔧 Feature engineering...")

# Encode categorical variables
le_passion = LabelEncoder()
le_hobby = LabelEncoder()
le_edu = LabelEncoder()
le_career = LabelEncoder()

df["passion_enc"] = le_passion.fit_transform(df["passion"])
df["hobby_enc"] = le_hobby.fit_transform(df["hobby"])
df["edu_enc"] = le_edu.fit_transform(df["education_level"])
df["career_enc"] = le_career.fit_transform(df["career_title"])

# Process skills (one-hot encoding)
all_skills = set()
for skills_str in df["skills"]:
    all_skills.update(skills_str.split(","))

all_skills = sorted(list(all_skills))
print(f"Total unique skills: {len(all_skills)}")

# Create skill columns
for skill in all_skills:
    df[f"skill_{skill}"] = df["skills"].str.contains(skill, regex=False).astype(int)

# Create interaction features
df["passion_hobby_interaction"] = df["passion_enc"] * df["hobby_enc"]
df["age_education_ratio"] = df["age"] / (df["edu_enc"] + 1)
df["personality_creativity"] = df["openness"] * df["extraversion"] / 10
df["personality_analytical"] = df["conscientiousness"] * (11 - df["neuroticism"]) / 10
df["work_preference_score"] = (df["salary_importance"] + df["work_life_balance"]) / 2

# Calculate skill diversity
skill_cols = [col for col in df.columns if col.startswith("skill_")]
df["skill_diversity"] = df[skill_cols].sum(axis=1)

# ==================== 3. PREPARE FEATURES ====================
print("\n📋 Preparing feature matrix...")

feature_cols = (
    ["passion_enc", "hobby_enc", "edu_enc", "age"] +
    ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"] +
    ["salary_importance", "work_life_balance"] +
    ["passion_hobby_interaction", "age_education_ratio", 
     "personality_creativity", "personality_analytical", 
     "work_preference_score", "skill_diversity"] +
    skill_cols
)

X = df[feature_cols].values
y = df["career_enc"].values

print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# ==================== 4. SCALE FEATURES ====================
print("\n⚖️ Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==================== 5. SPLIT DATA ====================
print("\n✂️ Splitting data (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# ==================== 6. TRAIN ENSEMBLE MODEL ====================
print("\n🤖 Training ensemble model...")

# Model 1: Random Forest
print("  → Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_score = rf_model.score(X_test, y_test)
print(f"     Random Forest accuracy: {rf_score:.4f}")

# Model 2: Gradient Boosting
print("  → Training Gradient Boosting...")
gb_model = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=10,
    random_state=42
)
gb_model.fit(X_train, y_train)
gb_score = gb_model.score(X_test, y_test)
print(f"     Gradient Boosting accuracy: {gb_score:.4f}")

# Model 3: Neural Network
print("  → Training Neural Network...")
nn_model = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42,
    early_stopping=True
)
nn_model.fit(X_train, y_train)
nn_score = nn_model.score(X_test, y_test)
print(f"     Neural Network accuracy: {nn_score:.4f}")

# Ensemble Model (Voting Classifier)
print("  → Creating Ensemble...")
ensemble_model = VotingClassifier(
    estimators=[
        ('rf', rf_model),
        ('gb', gb_model),
        ('nn', nn_model)
    ],
    voting='soft',
    weights=[2, 2, 1]  # Give more weight to RF and GB
)
ensemble_model.fit(X_train, y_train)
ensemble_score = ensemble_model.score(X_test, y_test)
print(f"     Ensemble accuracy: {ensemble_score:.4f}")

# ==================== 7. CROSS-VALIDATION ====================
print("\n🔄 Cross-validation (5-fold)...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(ensemble_model, X_scaled, y, cv=cv, scoring='accuracy', n_jobs=-1)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# ==================== 8. DETAILED EVALUATION ====================
print("\n📊 Detailed evaluation on test set...")
y_pred = ensemble_model.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")

# Get top 10 most common careers for report
top_careers = df['career_title'].value_counts().head(10).index
top_career_indices = [le_career.transform([career])[0] for career in top_careers]

print("\nClassification Report (Top 10 careers):")
print(classification_report(
    y_test, 
    y_pred, 
    labels=top_career_indices,
    target_names=top_careers,
    zero_division=0
))

# ==================== 9. FEATURE IMPORTANCE ====================
print("\n⭐ Top 20 Most Important Features:")
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False).head(20)

for idx, row in feature_importance.iterrows():
    print(f"  {row['feature']:<30} {row['importance']:.4f}")

# ==================== 10. SAVE MODEL AND ENCODERS ====================
print("\n💾 Saving model and encoders to models/ folder...")

# Save ensemble model
joblib.dump(ensemble_model, "models/career_rf_model.pkl")
print("  ✅ Saved: models/career_rf_model.pkl")

# Save scaler
joblib.dump(scaler, "models/feature_scaler.pkl")
print("  ✅ Saved: models/feature_scaler.pkl")

# Save encoders
encoders = {
    'le_passion': le_passion,
    'le_hobby': le_hobby,
    'le_edu': le_edu,
    'le_career': le_career,
    'all_skills': all_skills,
    'feature_cols': feature_cols
}
joblib.dump(encoders, "models/sklearn_encoders.pkl")
print("  ✅ Saved: models/sklearn_encoders.pkl")

# ==================== 11. SUMMARY ====================
print("\n" + "="*60)
print("✅ MODEL TRAINING COMPLETE!")
print("="*60)
print(f"\n📈 Final Results:")
print(f"  • Ensemble Accuracy: {ensemble_score:.2%}")
print(f"  • Cross-validation: {cv_scores.mean():.2%} (+/- {cv_scores.std() * 2:.2%})")
print(f"  • Random Forest: {rf_score:.2%}")
print(f"  • Gradient Boosting: {gb_score:.2%}")
print(f"  • Neural Network: {nn_score:.2%}")
print(f"\n📦 Files Created:")
print(f"  • models/career_rf_model.pkl")
print(f"  • models/feature_scaler.pkl")
print(f"  • models/sklearn_encoders.pkl")
print("\n🎯 Ready to integrate with Flask app!")
print("Next step: Run 'python run.py' to start the application")
print("="*60)