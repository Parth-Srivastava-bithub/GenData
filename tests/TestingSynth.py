from repligauge import DataQualityComparator

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from rich import print



import pandas as pd

# CSV read kar le
df = pd.read_csv("Data/final_dataframe.csv")

# 'diabetes_diagnosis' ko 0 (Negative) aur 1 (Positive) me convert kar
df['Outcome'] = df['Outcome'].map({'Negative': 0, 'Positive': 1})

real_df = pd.read_csv("Data/diabetes.csv")

comparator = DataQualityComparator(real_df, df)
custom_models = [
    LogisticRegression(C=0.5, max_iter=1000, random_state=42),
    RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
]

full_report_results = comparator.generate_full_report(
    target_col='Outcome',
    models=custom_models,
    save_path='my_data_quality_report'
)



