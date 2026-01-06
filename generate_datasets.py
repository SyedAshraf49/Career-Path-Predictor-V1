# generate_datasets.py - FIXED with correct paths
"""
Generate missing datasets for Career Predictor:
1. personality.csv - Big Five personality traits
2. work_values.csv - Work preferences
3. training_data_comprehensive.csv - Training data
"""

import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Create datasets folder if it doesn't exist
if not os.path.exists('datasets'):
    os.makedirs('datasets')
    print("✅ Created 'datasets/' folder")

# ==================== 1. GENERATE PERSONALITY.CSV ====================
print("\n" + "="*60)
print("Generating personality.csv...")
print("="*60)

def generate_personality_data(n_samples=1000):
    """Generate realistic personality data based on Big Five traits"""
    
    personalities = []
    
    # Career archetypes with typical personality profiles
    archetypes = [
        {"name": "Analyst", "O": 8, "C": 8, "E": 4, "A": 6, "N": 5},
        {"name": "Creative", "O": 9, "C": 5, "E": 7, "A": 7, "N": 6},
        {"name": "Leader", "O": 7, "C": 8, "E": 9, "A": 6, "N": 4},
        {"name": "Helper", "O": 6, "C": 7, "E": 7, "A": 9, "N": 5},
        {"name": "Organizer", "O": 5, "C": 9, "E": 5, "A": 7, "N": 4},
        {"name": "Innovator", "O": 10, "C": 6, "E": 8, "A": 6, "N": 5},
        {"name": "Craftsperson", "O": 6, "C": 8, "E": 4, "A": 6, "N": 4},
    ]
    
    for i in range(n_samples):
        # Choose archetype with variation
        archetype = random.choice(archetypes)
        
        # Add noise to make it realistic
        noise = lambda base: max(1, min(10, int(np.random.normal(base, 1.5))))
        
        person = {
            "user_id": f"user_{i+1:04d}",
            "openness": noise(archetype["O"]),
            "conscientiousness": noise(archetype["C"]),
            "extraversion": noise(archetype["E"]),
            "agreeableness": noise(archetype["A"]),
            "neuroticism": noise(archetype["N"]),
            "risk_tolerance": random.randint(1, 10),
            "creativity_score": noise(archetype["O"]),
            "analytical_thinking": noise(archetype["C"]),
            "people_orientation": noise(archetype["E"]),
            "detail_oriented": noise(archetype["C"]),
            "stress_tolerance": 11 - noise(archetype["N"]),  # Inverse of neuroticism
        }
        
        personalities.append(person)
    
    return pd.DataFrame(personalities)

personality_df = generate_personality_data(1000)
personality_df.to_csv("datasets/personality.csv", index=False)
print(f"✅ Created datasets/personality.csv with {len(personality_df)} samples")
print(personality_df.head())

# ==================== 2. GENERATE WORK_VALUES.CSV ====================
print("\n" + "="*60)
print("Generating work_values.csv...")
print("="*60)

def generate_work_values(n_samples=1000):
    """Generate work value preferences"""
    
    values = []
    
    for i in range(n_samples):
        # Different career value profiles
        profile = random.choice([
            "wealth_focused", "balanced", "purpose_driven", 
            "security_focused", "growth_focused", "freedom_focused"
        ])
        
        if profile == "wealth_focused":
            base_values = {"salary": 9, "balance": 4, "security": 6, "growth": 7, "impact": 4, "autonomy": 6, "prestige": 8, "variety": 5}
        elif profile == "balanced":
            base_values = {"salary": 6, "balance": 8, "security": 7, "growth": 6, "impact": 6, "autonomy": 6, "prestige": 5, "variety": 6}
        elif profile == "purpose_driven":
            base_values = {"salary": 5, "balance": 7, "security": 6, "growth": 7, "impact": 9, "autonomy": 7, "prestige": 4, "variety": 7}
        elif profile == "security_focused":
            base_values = {"salary": 6, "balance": 8, "security": 9, "growth": 5, "impact": 6, "autonomy": 4, "prestige": 6, "variety": 4}
        elif profile == "growth_focused":
            base_values = {"salary": 6, "balance": 5, "security": 5, "growth": 9, "impact": 7, "autonomy": 7, "prestige": 7, "variety": 8}
        else:  # freedom_focused
            base_values = {"salary": 6, "balance": 9, "security": 4, "growth": 7, "impact": 6, "autonomy": 9, "prestige": 4, "variety": 8}
        
        noise = lambda x: max(1, min(10, int(np.random.normal(x, 1))))
        
        values.append({
            "user_id": f"user_{i+1:04d}",
            "salary_importance": noise(base_values["salary"]),
            "work_life_balance": noise(base_values["balance"]),
            "job_security": noise(base_values["security"]),
            "career_growth": noise(base_values["growth"]),
            "social_impact": noise(base_values["impact"]),
            "autonomy": noise(base_values["autonomy"]),
            "prestige": noise(base_values["prestige"]),
            "variety": noise(base_values["variety"]),
        })
    
    return pd.DataFrame(values)

work_values_df = generate_work_values(1000)
work_values_df.to_csv("datasets/work_values.csv", index=False)
print(f"✅ Created datasets/work_values.csv with {len(work_values_df)} samples")
print(work_values_df.head())

# ==================== 3. GENERATE TRAINING DATA ====================
print("\n" + "="*60)
print("Generating comprehensive training data...")
print("="*60)

def generate_training_data(n_samples=5000):
    """Generate synthetic training data for model"""
    
    # Load existing datasets
    passion_df = pd.read_csv("datasets/passion.csv")
    hobbies_df = pd.read_csv("datasets/hobbies.csv")
    career_df = pd.read_csv("datasets/career.csv")
    skills_df = pd.read_csv("datasets/skills.csv")
    
    passions = passion_df["passion"].tolist()
    hobbies = hobbies_df["hobby"].tolist()
    careers = career_df["career_title"].tolist()
    skills = skills_df["skill_name"].tolist()
    education_levels = ["High School", "Vocational", "Associate's", "Bachelor's", "Master's", "Doctorate"]
    
    # Career to likely passion/hobby/skills mappings
    career_profiles = {
        "Software Engineer": {
            "passions": ["Technology", "Gaming", "Entrepreneurship"],
            "hobbies": ["Video Gaming", "Chess", "Reading"],
            "skills": ["Programming", "Critical Thinking", "Data Analysis"],
            "education": ["Bachelor's", "Master's"],
            "age_range": (22, 45)
        },
        "Data Scientist": {
            "passions": ["Technology", "Reading", "Entrepreneurship"],
            "hobbies": ["Reading", "Chess", "Board Games"],
            "skills": ["Data Analysis", "Programming", "Critical Thinking"],
            "education": ["Master's", "Doctorate"],
            "age_range": (24, 50)
        },
        "Graphic Designer": {
            "passions": ["Art", "Photography", "Fashion"],
            "hobbies": ["Painting", "Photography", "Drawing"],
            "skills": ["Graphic Design", "Creativity", "UI/UX Design"],
            "education": ["Associate's", "Bachelor's"],
            "age_range": (20, 45)
        },
        "Elementary Teacher": {
            "passions": ["Reading", "Volunteering", "Animals"],
            "hobbies": ["Reading", "Volunteering", "Crafts"],
            "skills": ["Public Speaking", "Empathy", "Leadership"],
            "education": ["Bachelor's", "Master's"],
            "age_range": (22, 60)
        },
        "Registered Nurse": {
            "passions": ["Volunteering", "Animals", "Fitness"],
            "hobbies": ["Volunteering", "Reading", "Fitness"],
            "skills": ["Empathy", "Critical Thinking", "First Aid"],
            "education": ["Bachelor's", "Associate's"],
            "age_range": (22, 55)
        },
    }
    
    training_data = []
    
    for i in range(n_samples):
        # Select career
        career = random.choice(careers)
        
        # Get profile if exists, otherwise random
        if career in career_profiles:
            profile = career_profiles[career]
            passion = random.choice(profile["passions"]) if random.random() > 0.3 else random.choice(passions)
            hobby = random.choice(profile["hobbies"]) if random.random() > 0.3 else random.choice(hobbies)
            education = random.choice(profile["education"]) if random.random() > 0.4 else random.choice(education_levels)
            age = random.randint(*profile["age_range"])
            selected_skills = random.sample(profile["skills"], min(len(profile["skills"]), random.randint(2, 4)))
        else:
            passion = random.choice(passions)
            hobby = random.choice(hobbies)
            education = random.choice(education_levels)
            age = random.randint(18, 65)
            selected_skills = random.sample(skills, random.randint(2, 5))
        
        # Get personality traits for this user
        user_id = f"user_{i+1:04d}"
        if i < 1000:
            personality = personality_df[personality_df["user_id"] == user_id].iloc[0].to_dict()
            work_vals = work_values_df[work_values_df["user_id"] == user_id].iloc[0].to_dict()
        else:
            # Generate on the fly for additional samples
            personality = {
                "openness": random.randint(1, 10),
                "conscientiousness": random.randint(1, 10),
                "extraversion": random.randint(1, 10),
                "agreeableness": random.randint(1, 10),
                "neuroticism": random.randint(1, 10),
            }
            work_vals = {
                "salary_importance": random.randint(1, 10),
                "work_life_balance": random.randint(1, 10),
            }
        
        training_data.append({
            "user_id": user_id,
            "passion": passion,
            "hobby": hobby,
            "education_level": education,
            "age": age,
            "skills": ",".join(selected_skills),
            "openness": personality.get("openness", random.randint(1, 10)),
            "conscientiousness": personality.get("conscientiousness", random.randint(1, 10)),
            "extraversion": personality.get("extraversion", random.randint(1, 10)),
            "agreeableness": personality.get("agreeableness", random.randint(1, 10)),
            "neuroticism": personality.get("neuroticism", random.randint(1, 10)),
            "salary_importance": work_vals.get("salary_importance", random.randint(1, 10)),
            "work_life_balance": work_vals.get("work_life_balance", random.randint(1, 10)),
            "career_title": career
        })
    
    return pd.DataFrame(training_data)

training_df = generate_training_data(5000)
training_df.to_csv("datasets/training_data_comprehensive.csv", index=False)
print(f"✅ Created datasets/training_data_comprehensive.csv with {len(training_df)} samples")
print(training_df.head())

print("\n" + "="*60)
print("✅ All datasets generated successfully!")
print("="*60)
print("\nNext steps:")
print("1. Run: python model_enhanced.py")
print("2. This will train the model and save it to models/")
print("3. Then run: python run.py")
print("="*60)