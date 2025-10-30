
# if __name__ == "__main__":
#     # Ensure you have a .env file with your API keys
    
#     # --- 1. Define our TARGET (from the real banking data) ---
#     # This is the JSON output from our previous run
#     target_bank_summary = {
#       "univariate_analysis": {
#         "title": "Univariate Analysis",
#         "summary": "Mean age is 39.77, std 10.32. 'admin.' is top job.",
#         "key_findings": [
#           "Mean age is 39.77",
#           "Mean duration is 264.69",
#           "Job has 12 unique values with 'admin.' being the most common"
#         ]
#       },
#       "multivariate_correlation": {
#         "title": "Correlation Matrix Analysis",
#         "summary": "Strong positive correlation between emp_var_rate and euribor3m (0.97).",
#         "key_findings": [
#           "Strong positive correlation between emp_var_rate and euribor3m (0.972892)",
#           "Strong positive correlation between emp_var_rate and nr_employed (0.905244)",
#           "Potential multicollinearity between emp_var_rate, euribor3m, and nr_employed"
#         ]
#       },
#       "categorical_relationships": {
#         "title": "Crosstab Analysis",
#         "summary": "Strong relationship between job and marital status.",
#         "key_findings": [
#           "Strong relationship between job and marital status",
#           "Certain combinations never appear, e.g., 'unknown' job and 'unknown' education"
#         ]
#       }
#     }
    
#     # --- 2. Create our NEW SYNTHETIC DATA ---
#     # (We'll use our 5-row diabetes data as a "mismatch" example)
#     synthetic_diabetes_data = {
#         'Age': [32, 48, 65, 55, 40],
#         'BMI': [21.5, 29.8, 34.1, 24.5, 31],
#         'Fasting_Glucose_Level': [85, 108.5, 145.2, 98.1, 115.6],
#         'Genetic_Risk_Score': [2, 6, 9, 8, 4],
#         'Exercise_Frequency': ['High', 'Low', 'Sedentary', 'Moderate', 'Sedentary'],
#         'Diabetes_Diagnosis': ['Negative', 'Pre-Diabetic', 'Type 2 Diabetic', 'Negative', 'Pre-Diabetic']
#     }
#     synthetic_df = pd.DataFrame(synthetic_diabetes_data)

#     # --- 3. Run the Validation ---
#     try:
#         validator = SyntheticDataValidator()
        
#         print(f"\n[bold magenta]Validating (mismatched) Diabetes data against (target) Banking summary...[/bold magenta]")
        
#         validation_report = validator.validate(synthetic_df, target_bank_summary)
        
#         if validation_report:
#             print("\n--- [bold]Final Validation Report[/bold] ---")
#             # .json(indent=2) is a Pydantic method
#             # print(validation_report.json(indent=2)) # <-- OLD (DEPRECATED)
#             print(validation_report.model_dump_json(indent=2)) # <-- NEW (FIXED)
        
#         # --- 4. Example 2 (A better match) ---
#         # Let's create synthetic data that *should* match the target
#         synthetic_bank_data = {
#             'age': [35, 40, 38, 42],
#             'duration': [250, 260, 270, 265],
#             'emp_var_rate': [1.1, 1.1, 1.2, 1.1],
#             'euribor3m': [4.8, 4.8, 4.9, 4.8],
#             'nr_employed': [5191, 5191, 5192, 5191],
#             'job': ['admin.', 'blue-collar', 'admin.', 'management'],
#             'education': ['university.degree', 'high.school', 'university.degree', 'professional.course']
#         }
#         synthetic_df_good = pd.DataFrame(synthetic_bank_data)
        
#         print(f"\n[bold magenta]Validating (matched) Bank data against (target) Banking summary...[/bold magenta]")
        
#         validation_report_good = validator.validate(synthetic_df_good, target_bank_summary)
        
#         if validation_report_good:
#             print("\n--- [bold]Final Validation Report (Good Match)[/bold] ---")
#             # print(validation_report_good.json(indent=2)) # <-- OLD (DEPRECATED)
#             print(validation_report_good.model_dump_json(indent=2)) # <-- NEW (FIXED)

#     except Exception as e:
#         print(f"\n[bold red]An error occurred during initialization or execution:[/bold red]\n{e}")
#         print("Please check your API keys in the .env file and ensure Ollama is running.")

