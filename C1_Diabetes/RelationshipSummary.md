### 🧩 **Univariate Relationships**

1. **Age Relationships**

   * Mean age: 48.71 → 48.03 → ranges from 16 to 90.
   * Standard deviation of age: 12.15.
   * The min and max values are reasonable, no anomalies flagged.

2. **BMI Relationships**

   * Mean BMI: 27.10.
   * Positively correlated with diabetes risk and other features.
   * No reported anomalies in range, but might be linked with multicollinearity.

3. **Glucose Level Relationships**

   * Range: 60 → 195.39.
   * Mean glucose: 120.89 (std 31.97).
   * Low-end (60) and high-end (195.39) flagged as potential outliers.
   * Strong positive link with diabetes diagnosis (Outcome).

4. **Insulin Relationships**

   * Max insulin: 846 (way beyond 75th percentile of 127.25 → potential outlier).
   * Possible multicollinearity with SkinThickness.

5. **Blood Pressure Relationships**

   * Minimum recorded: 0 → likely an error or missing entry.
   * No strong correlations reported.

6. **Outcome (Target)**

   * Mean: 0.35, std: 0.48 → binary variable nature confirmed.
   * Distribution uneven (Positive > Negative).

7. **Physical Activity Level**

   * Most common value: “Moderate” (frequency: 803).
   * Unique values not fully listed.

8. **Dataset Stats**

   * Observations: 768.
   * Some features show zero or near-zero minimums.

---

### 🔗 **Multivariate Correlations**

1. **Strong Positive Correlations**

   * BMI ↔ At_Risk_Diabetes (0.406105)
   * Glucose ↔ Outcome (0.466581)
   * Age ↔ Pregnancies (0.544341)
   * BMI ↔ SkinThickness (0.392573)

2. **Moderate Positive Correlations**

   * Age ↔ At_Risk_Diabetes (0.277864)
   * Glucose_Level ↔ At_Risk_Diabetes (0.217897)

3. **Strong Negative Correlations**

   * Pregnancies ↔ SkinThickness (-0.081672) → mild but negative.

4. **Weak/No Correlations**

   * Age ↔ BMI (≈0)
   * Age ↔ Glucose_Level (≈0)
   * BMI ↔ Glucose_Level (≈0)
   * Pregnancies ↔ DiabetesPedigreeFunction (-0.033523)
   * BloodPressure ↔ DiabetesPedigreeFunction (0.041265)

5. **Perfect Self-Correlation**

   * Age ↔ Age = 1.0 (trivial, but recorded).

6. **Potential Multicollinearity**

   * BMI ↔ At_Risk_Diabetes (0.406105)
   * Insulin ↔ SkinThickness (0.436783)

---

### ⚙️ **Categorical / Conditional Relationships**

1. **Polyuria & Polydipsia**

   * If **Polyuria = 'Yes'**, then **Polydipsia = 'Yes'** almost always.
   * Indicates a near-deterministic relationship.

2. **Target Distribution**

   * Target “Positive” dominates over “Negative.”
   * Suggests class imbalance or bias in dataset.

3. **Others**

   * Several uneven conditional distributions across categorical pairs.
   * But some datasets lacked ≥2 categorical columns → no valid crosstabs.

---

### 🧠 **Meta Relationships (Structural Observations)**

* Repeated appearance of BMI, Glucose, and Age as dominant predictors.
* Multicollinearity indicates overlapping influence on diabetes risk.
* Categorical relationships mainly medical symptoms confirming diabetic tendencies (Polyuria-Polydipsia).
* Data quality flags include **0 blood pressure**, **846 insulin**, and **extreme glucose levels** → possible data entry errors or real outliers.

---

To summary in short:

👀 **Who’s teaming up:** BMI, Glucose, and Age are the trouble trio tagging along with diabetes risk.

💀 **Suspicious points:** Insulin 846, BP 0, Glucose extremes.

💬 **Categorial gossip:** Polyuria and Polydipsia are besties; Target’s biased toward Positive.

