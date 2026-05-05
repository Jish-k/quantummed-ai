# QuantumMed AI: Clinical Input Guide

This guide describes the 13 features required for heart disease prediction and provides sample data for testing the application.

## 📋 Feature Descriptions

| # | Feature | Technical Name | Description | Range / Values |
|---| :--- | :--- | :--- | :--- |
| 1 | **Age** | age | Patient age in years | 29 - 77 |
| 2 | **Sex** | sex | Biological sex | 1 = Male, 0 = Female |
| 3 | **Chest Pain** | cp | Type of chest pain experienced | 0: Typical Angina, 1: Atypical, 2: Non-anginal, 3: Asymptomatic |
| 4 | **Resting BP** | trestbps | Resting blood pressure (on admission) | 94 - 200 (mmHg) |
| 5 | **Cholesterol** | chol | Serum cholesterol | 126 - 564 (mg/dl) |
| 6 | **Fasting Sugar** | fbs | Fasting blood sugar > 120 mg/dl | 1 = True, 0 = False |
| 7 | **Resting ECG** | restecg | Resting electrocardiographic results | 0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy |
| 8 | **Max Heart Rate** | thalach | Maximum heart rate achieved | 71 - 202 |
| 9 | **Exercise Angina**| exang | Exercise induced angina | 1 = Yes, 0 = No |
| 10| **Oldpeak** | oldpeak | ST depression induced by exercise | 0.0 - 6.2 |
| 11| **Slope** | slope | Slope of peak exercise ST segment | 0: Upsloping, 1: Flat, 2: Downsloping |
| 12| **CA** | ca | Number of major vessels colored | 0 - 4 |
| 13| **Thal** | thal | Thalassemia type | 1: Normal, 2: Fixed Defect, 3: Reversible Defect |

---

## 🧪 Sample Input Cases

You can enter these values into the **QuantumMed AI** frontend to test the model comparison.

### Case 1: High Risk Profile (Likely "AT RISK")
*   **Input Sequence**: `[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]`
*   *Expected behavior*: Models should show high risk due to age, asymptomatic chest pain, and high ST depression (oldpeak).

### Case 2: Healthy Profile (Likely "HEALTHY")
*   **Input Sequence**: `[37, 1, 2, 130, 250, 0, 1, 187, 0, 3.5, 0, 0, 2]`
*   *Expected behavior*: Models should show healthy status due to high max heart rate and younger age.

### Case 3: Mixed Profile (Testing Model Sensitivity)
*   **Input Sequence**: `[55, 1, 1, 140, 220, 0, 0, 160, 0, 1.2, 1, 1, 3]`
*   *Expected behavior*: This case might show different confidence levels across ML, DL, and QML, highlighting the value of paradigm comparison.
