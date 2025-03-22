Heart Disease Detection System - Detailed Documentation
Overview
This project aims to develop a Heart Disease Detection System using two approaches:
1.	Rule-Based Expert System (Experta) - Uses predefined medical rules to assess heart disease risk.
2.	Machine Learning Model (Decision Tree - Scikit-Learn) - Learns from patient data to predict heart disease.
The system processes patient health indicators such as age, cholesterol, blood pressure, heart rate, and lifestyle factors to determine risk levels.
________________________________________
 Data Preprocessing
The first step is cleaning and preparing the dataset.
A. Load and Handle Missing Values
import pandas as pd

# Load dataset
df = pd.read_csv("heart.csv")

# Check for missing values
print(df.isnull().sum())

# Fill missing values with the column mean
df.fillna(df.mean(), inplace=True)
B. Normalize Numerical Features
To improve model performance, we scale numerical values using MinMaxScaler.
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
numerical_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
df[numerical_features] = scaler.fit_transform(df[numerical_features])
C. Encode Categorical Variables
Machine learning models require numerical inputs, so we convert categorical features using OneHotEncoding.
categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
df = pd.get_dummies(df, columns=categorical_features, drop_first=True)
D. Save the Cleaned Dataset
df.to_csv("cleaned_data.csv", index=False)
________________________________________
 Data Visualization
To better understand the dataset, we visualize feature relationships and distributions.
A. Correlation Heatmap
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()
B. Histograms and Boxplots
df.hist(figsize=(12, 10))
plt.show()

plt.figure(figsize=(12, 8))
sns.boxplot(data=df[numerical_features])
plt.title("Boxplots of Numerical Features")
plt.show()
________________________________________
 Rule-Based Expert System (Experta)
The expert system uses predefined medical rules to classify heart disease risk.
A. Define Rules in Experta
from experta import *

class HeartDiseaseRisk(KnowledgeEngine):
    @DefFacts()
    def _initial_facts(self):
        yield Fact(action="assess_risk")

    @Rule(Fact(action="assess_risk"), Fact(cholesterol=MATCH.chol), Fact(age=MATCH.age),
          TEST(lambda chol, age: chol > 0.6 and age > 50))
    def high_risk_cholesterol_age(self):
        self.declare(Fact(risk="high"))

    @Rule(Fact(action="assess_risk"), Fact(blood_pressure=MATCH.bp), Fact(smoking=MATCH.smoke),
          TEST(lambda bp, smoke: bp > 0.6 and smoke == "Yes"))
    def high_risk_bp_smoking(self):
        self.declare(Fact(risk="high"))

    @Rule(Fact(action="assess_risk"), Fact(exercise=MATCH.exercise), Fact(bmi=MATCH.bmi),
          TEST(lambda exercise, bmi: exercise == "Regular" and bmi < 25))
    def low_risk_exercise_bmi(self):
        self.declare(Fact(risk="low"))
________________________________________
 Machine Learning Model (Decision Tree)
A. Train the Decision Tree Model
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Split the dataset
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Precision: {precision_score(y_test, y_pred)}")
print(f"Recall: {recall_score(y_test, y_pred)}")
print(f"F1-Score: {f1_score(y_test, y_pred)}")

# Save the model
joblib.dump(model, "decision_tree_model.pkl")
________________________________________
Model vs. Expert System Comparison
Compares accuracy between the decision tree and rule-based system.
def evaluate_expert_system(data):
    engine = HeartDiseaseRisk()
    engine.reset()
    engine.declare(Fact(cholesterol=data["chol"]), Fact(age=data["age"]),
                   Fact(blood_pressure=data["trestbps"]), Fact(smoking="Yes"),
                   Fact(exercise="Regular"), Fact(bmi=24))
    engine.run()
    return engine.facts

expert_predictions = []
for _, row in X_test.iterrows():
    result = evaluate_expert_system(row)
    expert_predictions.append(1 if "high" in result else 0)

print(f"Expert System Accuracy: {accuracy_score(y_test, expert_predictions)}")
________________________________________
Conclusion
This project provides two different approaches for heart disease detection:
• Expert System: Based on predefined medical rules.
• Decision Tree Model: Learns from data to predict risk.

