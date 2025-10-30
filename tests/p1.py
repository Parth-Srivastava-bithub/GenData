from AutoEDA import DataRelationshipExtractor
import pandas as pd
import json
from rich import print
from Validator import SyntheticDataValidator
sample_df = pd.read_csv('Data/banking.csv')

target_diabetes_summary = {
  "univariate_analysis": {
    "title": "Univariate Analysis",
    "summary": "Mean age is 48.03 (std 12.15). Mean BMI is 27.10. Glucose ranges from 60 to 195.39 with potential outliers. Dataset has 768 observations with a few anomalies such as 0 blood pressure and insulin value of 846.",
    "key_findings": [
      "Mean age is 48.03 with standard deviation 12.15 (range 16–90)",
      "Mean BMI is 27.10, positively linked with diabetes risk",
      "Mean glucose level is 120.89 (std 31.97), range 60–195.39",
      "Potential outliers: glucose extremes, insulin max 846, blood pressure 0",
      "Outcome variable mean 0.35, std 0.48 (binary nature confirmed)",
      "Physical activity most common value: 'Moderate' (freq: 803)",
      "Dataset contains 768 observations"
    ]
  },
  "multivariate_correlation": {
    "title": "Correlation Matrix Analysis",
    "summary": "Strong positive correlations exist between glucose and outcome, age and pregnancies, BMI and diabetes risk. Weak correlations close to zero for several pairs. Potential multicollinearity observed between BMI and diabetes risk, and insulin and skin thickness.",
    "key_findings": [
      "Strong positive correlation between BMI and At_Risk_Diabetes (0.406105)",
      "Strong positive correlation between Glucose and Outcome (0.466581)",
      "Strong positive correlation between Age and Pregnancies (0.544341)",
      "Strong positive correlation between BMI and SkinThickness (0.392573)",
      "Moderate positive correlation between Age and At_Risk_Diabetes (0.277864)",
      "Moderate positive correlation between Glucose_Level and At_Risk_Diabetes (0.217897)",
      "Weak or near-zero correlations among several pairs (Age–BMI, Age–Glucose_Level, BMI–Glucose_Level, etc.)",
      "Strong negative correlation between Pregnancies and SkinThickness (-0.081672)",
      "Potential multicollinearity between BMI and At_Risk_Diabetes (0.406105)",
      "Potential multicollinearity between Insulin and SkinThickness (0.436783)"
    ]
  },
  "categorical_relationships": {
    "title": "Crosstab Analysis",
    "summary": "Polyuria and Polydipsia show strong conditional dependence, while Target variable distribution is imbalanced toward Positive outcomes. Some datasets lacked sufficient categorical pairs for deeper analysis.",
    "key_findings": [
      "If Polyuria = 'Yes', then Polydipsia = 'Yes' almost always (near-deterministic relationship)",
      "Target distribution is uneven: 'Positive' more frequent than 'Negative'",
      "Several uneven conditional distributions across categorical pairs",
      "Some datasets lacked two or more categorical columns for valid crosstab analysis"
    ]
  }
}

validator = SyntheticDataValidator()
validation_report = validator.validate(sample_df, target_diabetes_summary)
if validation_report:
    print("\n--- [bold]Final Validation Report[/bold] ---")
    # .json(indent=2) is a Pydantic method
    # print(validation_report.json(indent=2)) # <-- OLD (DEPRECATED)
    print(validation_report.model_dump_json(indent=2)) # <-- NEW (FIXED)


