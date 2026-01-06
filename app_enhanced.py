# app_enhanced.py
"""
Enhanced Flask app with:
- Top 5 career predictions with confidence scores
- Interactive personality quiz
- Detailed career information
- Learning path recommendations
"""

from flask import Flask, render_template, request, jsonify, session
import joblib
import numpy as np
import json
from datetime import datetime

app = Flask(__name__)
app.secret_key = "your-secret-key-here"  # Change in production

# Load models and encoders
print("Loading models...")
model = joblib.load("career_rf_model.pkl")
scaler = joblib.load("feature_scaler.pkl")
encoders = joblib.load("sklearn_encoders.pkl")

# Load career details
import pandas as pd
career_df = pd.read_csv("career.csv")
career_details = career_df.set_index("career_title").to_dict('index')

# ==================== ROUTES ====================

@app.route("/")
def index():
    """Home page with option to start quiz"""
    return render_template("index.html")

@app.route("/quiz")
def quiz():
    """Interactive personality quiz"""
    return render_template("quiz.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Predict top 5 careers with confidence scores"""
    try:
        data = request.json if request.is_json else request.form
        
        # Extract input data
        input_data = {
            "passion": data.get("passion"),
            "hobby": data.get("hobby"),
            "education": data.get("education"),
            "age": int(data.get("age")),
            "skills": data.getlist("skills") if hasattr(data, 'getlist') else data.get("skills", []),
            "openness": int(data.get("openness", 5)),
            "conscientiousness": int(data.get("conscientiousness", 5)),
            "extraversion": int(data.get("extraversion", 5)),
            "agreeableness": int(data.get("agreeableness", 5)),
            "neuroticism": int(data.get("neuroticism", 5)),
            "salary_importance": int(data.get("salary_importance", 5)),
            "work_life_balance": int(data.get("work_life_balance", 5)),
        }
        
        # Get predictions
        predictions = predict_career_with_confidence(input_data)
        
        # Enhance predictions with career details
        enhanced_predictions = []
        for career, confidence in predictions:
            career_info = career_details.get(career, {})
            enhanced_predictions.append({
                "career": career,
                "confidence": float(confidence),
                "confidence_percent": f"{confidence*100:.1f}%",
                "salary": career_info.get("median_salary_usd", "N/A"),
                "education": career_info.get("education_level", "N/A"),
                "growth": career_info.get("job_growth_outlook", "N/A"),
                "skills": career_info.get("skills_required", "N/A")
            })
        
        # Store in session for learning path
        session['last_prediction'] = enhanced_predictions[0]['career']
        session['user_profile'] = input_data
        
        if request.is_json:
            return jsonify({"predictions": enhanced_predictions})
        else:
            return render_template("results.html", predictions=enhanced_predictions)
            
    except Exception as e:
        print(f"Error in prediction: {e}")
        if request.is_json:
            return jsonify({"error": str(e)}), 400
        else:
            return f"Error: {e}", 400

@app.route("/career/<career_name>")
def career_details(career_name):
    """Detailed career information page"""
    career_info = career_details.get(career_name, {})
    
    # Generate learning path
    learning_path = generate_learning_path(career_name, session.get('user_profile', {}))
    
    return render_template("career_details.html", 
                         career=career_name, 
                         info=career_info,
                         learning_path=learning_path)

@app.route("/compare")
def compare_careers():
    """Compare multiple careers side by side"""
    career_names = request.args.getlist("careers")
    careers_info = [
        {"name": name, "info": career_details.get(name, {})}
        for name in career_names
    ]
    return render_template("compare.html", careers=careers_info)

@app.route("/learning-path/<career_name>")
def learning_path(career_name):
    """Generate personalized learning path"""
    user_profile = session.get('user_profile', {})
    path = generate_learning_path(career_name, user_profile)
    return jsonify(path)

# ==================== HELPER FUNCTIONS ====================

def predict_career_with_confidence(input_data):
    """Predict top 5 careers with confidence scores"""
    
    # Encode categorical inputs
    passion_enc = encoders['le_passion'].transform([input_data['passion']])[0]
    hobby_enc = encoders['le_hobby'].transform([input_data['hobby']])[0]
    edu_enc = encoders['le_edu'].transform([input_data['education']])[0]
    
    # Build feature vector
    features = [
        passion_enc, hobby_enc, edu_enc, input_data['age'],
        input_data.get('openness', 5),
        input_data.get('conscientiousness', 5),
        input_data.get('extraversion', 5),
        input_data.get('agreeableness', 5),
        input_data.get('neuroticism', 5),
        input_data.get('salary_importance', 5),
        input_data.get('work_life_balance', 5),
    ]
    
    # Add interaction features
    features.extend([
        passion_enc * hobby_enc,
        input_data['age'] / (edu_enc + 1),
        features[4] * features[6] / 10,
        features[5] * (11 - features[8]) / 10,
        (features[9] + features[10]) / 2,
    ])
    
    # Add skill features
    user_skills = set(input_data.get('skills', []))
    skill_features = [1 if skill in user_skills else 0 for skill in encoders['all_skills']]
    features.append(len(user_skills))
    features.extend(skill_features)
    
    # Ensure correct feature length
    expected_length = len(encoders['feature_cols'])
    if len(features) != expected_length:
        # Pad or trim features to match
        features = features[:expected_length] + [0] * max(0, expected_length - len(features))
    
    # Scale and predict
    features_scaled = scaler.transform([features])
    probabilities = model.predict_proba(features_scaled)[0]
    
    # Get top 5
    top_5_indices = np.argsort(probabilities)[-5:][::-1]
    top_5_careers = encoders['le_career'].inverse_transform(top_5_indices)
    top_5_probs = probabilities[top_5_indices]
    
    return list(zip(top_5_careers, top_5_probs))

def generate_learning_path(career_name, user_profile):
    """Generate personalized learning roadmap"""
    
    career_info = career_details.get(career_name, {})
    required_skills = career_info.get("skills_required", "").split(", ")
    required_education = career_info.get("education_level", "Bachelor's")
    
    user_skills = set(user_profile.get("skills", []))
    user_education = user_profile.get("education", "High School")
    
    # Skill gap analysis
    skill_gaps = [skill for skill in required_skills if skill not in user_skills]
    
    # Education gap
    education_levels = ["High School", "Vocational", "Associate's", "Bachelor's", "Master's", "Doctorate"]
    user_edu_level = education_levels.index(user_education) if user_education in education_levels else 0
    required_edu_level = education_levels.index(required_education) if required_education in education_levels else 0
    needs_more_education = user_edu_level < required_edu_level
    
    # Build learning path
    path = {
        "career": career_name,
        "current_match": f"{len(user_skills & set(required_skills))}/{len(required_skills)} skills",
        "timeline": "6-24 months" if len(skill_gaps) <= 3 else "24-48 months",
        "steps": []
    }
    
    step_num = 1
    
    # Education steps
    if needs_more_education:
        path["steps"].append({
            "step": step_num,
            "title": f"Complete {required_education}",
            "description": f"Upgrade from {user_education} to {required_education}",
            "duration": "2-4 years",
            "resources": ["University programs", "Online degrees", "Scholarships"]
        })
        step_num += 1
    
    # Skill acquisition steps
    for skill in skill_gaps[:5]:  # Top 5 skill gaps
        path["steps"].append({
            "step": step_num,
            "title": f"Learn {skill}",
            "description": f"Acquire proficiency in {skill}",
            "duration": "2-6 months",
            "resources": ["Online courses (Coursera, Udemy)", "Books", "Practice projects"]
        })
        step_num += 1
    
    # Experience step
    path["steps"].append({
        "step": step_num,
        "title": "Gain Practical Experience",
        "description": "Build portfolio and get real-world experience",
        "duration": "3-12 months",
        "resources": ["Internships", "Freelance projects", "Open source contributions"]
    })
    step_num += 1
    
    # Certification step
    path["steps"].append({
        "step": step_num,
        "title": "Earn Certifications",
        "description": "Get industry-recognized credentials",
        "duration": "1-6 months",
        "resources": ["Professional certifications", "Industry certificates", "Licenses"]
    })
    
    return path

# ==================== PERSONALITY QUIZ QUESTIONS ====================

QUIZ_QUESTIONS = [
    {
        "id": 1,
        "question": "I enjoy trying new and unusual activities",
        "trait": "openness"
    },
    {
        "id": 2,
        "question": "I am very organized and detail-oriented",
        "trait": "conscientiousness"
    },
    {
        "id": 3,
        "question": "I feel energized when around other people",
        "trait": "extraversion"
    },
    {
        "id": 4,
        "question": "I prioritize others' needs over my own",
        "trait": "agreeableness"
    },
    {
        "id": 5,
        "question": "I often worry about things",
        "trait": "neuroticism"
    },
    # Add more questions (20 total recommended)
]

@app.route("/quiz-questions")
def get_quiz_questions():
    """Return quiz questions"""
    return jsonify(QUIZ_QUESTIONS)

@app.route("/submit-quiz", methods=["POST"])
def submit_quiz():
    """Process quiz responses and calculate personality scores"""
    responses = request.json
    
    # Calculate trait scores (average responses per trait)
    trait_scores = {
        "openness": 5,
        "conscientiousness": 5,
        "extraversion": 5,
        "agreeableness": 5,
        "neuroticism": 5,
    }
    
    for response in responses:
        trait = QUIZ_QUESTIONS[response['id']-1]['trait']
        trait_scores[trait] = response['score']
    
    session['personality'] = trait_scores
    return jsonify(trait_scores)

if __name__ == "__main__":
    app.run(debug=True, port=5000)