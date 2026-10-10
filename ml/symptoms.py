"""
Symptom vocabulary and disease symptom profiles.

The profiles are derived from the chatbot's own knowledge base
(data/health_kb.txt). Each number is the rough probability that a patient
with that disease reports the symptom. They are used ONLY to generate a
starting training set when no real dataset (e.g. the Kaggle
"Disease Symptom Prediction" CSV) is supplied. Replace them with real data
as soon as you have it: `python -m ml.train --csv path/to/Training.csv`.
"""

SYMPTOM_LABELS = {
    "high_fever": "a high fever",
    "mild_fever": "a mild fever",
    "chills": "chills or shivering",
    "sweating": "heavy sweating",
    "night_sweats": "night sweats",
    "headache": "a headache",
    "pain_behind_eyes": "pain behind the eyes",
    "joint_pain": "joint pain",
    "muscle_pain": "muscle or body aches",
    "skin_rash": "a skin rash",
    "bleeding_gums_nose": "bleeding from the gums or nose",
    "nausea": "nausea",
    "vomiting": "vomiting",
    "fatigue": "unusual tiredness or weakness",
    "abdominal_pain": "stomach pain",
    "loss_of_appetite": "loss of appetite",
    "dry_cough": "a dry cough",
    "persistent_cough": "a cough lasting 3+ weeks",
    "cough_blood": "coughing up blood",
    "shortness_of_breath": "shortness of breath",
    "loss_taste_smell": "loss of taste or smell",
    "sore_throat": "a sore throat",
    "chest_pain": "chest pain",
    "weight_loss": "unexplained weight loss",
    "watery_diarrhea": "profuse watery diarrhea",
    "muscle_cramps": "muscle cramps",
    "frequent_urination": "frequent urination",
    "excessive_thirst": "excessive thirst",
    "blurred_vision": "blurred vision",
    "slow_healing_wounds": "wounds that heal slowly",
    "dizziness": "dizziness",
    "nosebleed": "nosebleeds",
    "confusion": "confusion",
}

SYMPTOMS = list(SYMPTOM_LABELS)

# disease -> {symptom: probability the symptom is present}
DISEASE_PROFILES = {
    "Dengue": {
        "high_fever": 0.92, "headache": 0.85, "pain_behind_eyes": 0.70,
        "joint_pain": 0.75, "muscle_pain": 0.75, "skin_rash": 0.50,
        "bleeding_gums_nose": 0.30, "nausea": 0.55, "vomiting": 0.45,
        "fatigue": 0.55, "abdominal_pain": 0.20,
    },
    "Malaria": {
        "high_fever": 0.85, "chills": 0.90, "sweating": 0.80, "headache": 0.70,
        "nausea": 0.55, "vomiting": 0.50, "muscle_pain": 0.60, "fatigue": 0.65,
        "confusion": 0.12,
    },
    "Typhoid": {
        "high_fever": 0.90, "fatigue": 0.75, "abdominal_pain": 0.65,
        "headache": 0.60, "loss_of_appetite": 0.70, "skin_rash": 0.20,
        "nausea": 0.35, "vomiting": 0.25,
    },
    "COVID-19": {
        "mild_fever": 0.65, "high_fever": 0.25, "dry_cough": 0.80,
        "shortness_of_breath": 0.40, "fatigue": 0.65, "loss_taste_smell": 0.55,
        "sore_throat": 0.50, "headache": 0.50, "muscle_pain": 0.50,
    },
    "Tuberculosis": {
        "persistent_cough": 0.90, "cough_blood": 0.30, "chest_pain": 0.40,
        "fatigue": 0.70, "weight_loss": 0.70, "mild_fever": 0.60,
        "night_sweats": 0.65, "loss_of_appetite": 0.40,
    },
    "Cholera": {
        "watery_diarrhea": 0.97, "vomiting": 0.80, "muscle_cramps": 0.65,
        "nausea": 0.50, "fatigue": 0.40, "abdominal_pain": 0.30,
        "dizziness": 0.30,
    },
    "Diabetes": {
        "frequent_urination": 0.85, "excessive_thirst": 0.85, "weight_loss": 0.45,
        "fatigue": 0.70, "blurred_vision": 0.50, "slow_healing_wounds": 0.50,
    },
    "Hypertension": {
        "headache": 0.45, "dizziness": 0.40, "nosebleed": 0.20,
        "blurred_vision": 0.15, "chest_pain": 0.10, "shortness_of_breath": 0.10,
    },
}
