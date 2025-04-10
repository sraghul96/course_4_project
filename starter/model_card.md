# Model Card

Github Link - https://github.com/sraghul96/course_4_project

## Model Details
### Model Type
- **Algorithm**: Random Forest Classifier
- **Library**: scikit-learn (`sklearn.ensemble.RandomForestClassifier`)
- **Use Case**: Binary classification

### Training & Optimization
- **Initial Training**: Model is trained using the `RandomForestClassifier` with `random_state=42` for reproducibility.
- **Hyperparameter Tuning Strategy**:
  1. **RandomizedSearchCV**:
     - Search space:
       - `n_estimators`: 50 to 190 (step 20)
       - `max_depth`: 3 to 19
       - `min_samples_split`: 2 to 47 (step 5)
       - `min_samples_leaf`: 5 to 45 (step 5)
       - `bootstrap`: [True, False]
       - `max_features`: ["sqrt", "log2"]
       - `criterion`: ["gini", "entropy"]
     - Cross-validation folds: 5
     - Number of iterations: 100
     - Goal: Identify a good region in the hyperparameter space.
  
  2. **GridSearchCV**:
     - Uses the best parameters found from the random search.
     - Refines search using narrower ranges around the best values.
     - Cross-validation folds: 5
     - Goal: Fine-tune the final model.

### Final Model Configuration
- `n_estimators`: (±5 around the best from RandomizedSearch)
- `max_depth`: (±2 around the best)
- `min_samples_split`: (±1 around the best)
- `min_samples_leaf`: (±1 around the best)
- `bootstrap`, `max_features`, and `criterion`: As selected by the best RandomizedSearchCV result

### Inference
- Predictions are generated using `model.predict(X)`.

### Reproducibility
- All random operations use `random_state=42`.
- Training and hyperparameter tuning are deterministic given the same data.

---

This model is designed to provide robust classification performance with a balanced tradeoff between precision and recall. The two-phase tuning (Randomized + Grid Search) ensures optimal hyperparameter selection.

## 🎯 Intended Use

The model is trained to predict whether an individual's income exceeds \$50K per year based on demographic and employment-related features. This can be used in:

- Sociodemographic analysis
- Public policy modeling
- Economic studies and research

⚠️ **Not intended for use in real-world high-stakes decision-making** such as credit scoring, hiring, or benefits eligibility due to the potential for bias and fairness concerns in the underlying data.

---

## 📚 Training Data

The model was trained on the **UCI Census Income dataset (also known as the Adult dataset)**, which contains demographic data derived from the 1994 U.S. Census database.

- **Total records**: ~26,000 (train split)
- **Features**:
  - Age
  - Work class
  - Education
  - Marital status
  - Occupation
  - Relationship
  - Race
  - Sex
  - Native country
  - Hours per week
  - Capital gain/loss
  - etc.
- **Label**: Binary classification – `>50K` vs `<=50K` annual income
- **Imbalance**: The dataset is imbalanced with more instances of `<=50K` labels.

---

## 🧪 Evaluation Data

Evaluation is performed on the **test split** of the UCI dataset (about 6,000 samples). The test set maintains the original class distribution.

Additionally, the model was evaluated across various **demographic slices**:
- Age groups
- Marital status
- Race
- Sex
- Country of origin

Slice-based performance was assessed using:
- **Precision**
- **Recall**
- **F1 Score (Fβ=1)**

This helps uncover potential biases and inconsistencies in model behavior across population subgroups.

---

## Metrics
Metrics computed using `compute_model_metrics()`:
- **Precision**: `precision_score(y_true, y_pred)`
- **Recall**: `recall_score(y_true, y_pred)`
- **F1 Score (F-beta with β=1)**: `fbeta_score(y_true, y_pred, beta=1)`
- 
### 🔍 Overall Evaluation

The model was evaluated using standard classification metrics:  
- **Precision**: 0.786  
- **Recall**: 0.574  
- **F1 Score (Fβ=1)**: 0.664

These scores reflect a moderately well-performing model, with good precision but relatively lower recall, suggesting that the model is more conservative in labeling positive instances.

## Ethical Considerations
### Analysis of Model Performance Metrics and Bias

The model's performance metrics indicate potential biases across various demographic slices. These biases are important to address to ensure fair and equitable outcomes for all users.

---

#### 📊 Key Findings

##### 🧓 Age
- **0–18 age group**: Perfect performance.
- **19–29 age group**: Significantly lower precision and recall.
- **Older age groups**: Performance improves but remains inconsistent.

##### 💍 Marital Status
- **Married individuals**: High precision but lower recall.
- **Divorced/Never-married**: Particularly low recall, indicating potential bias.

##### 🧑‍🤝‍🧑 Race
- **White individuals**: Best performance.
- **Other racial groups**: Lower precision and recall, suggesting racial bias.

##### ⚧ Sex
- **Males**: Higher precision and recall.
- **Females**: Lower performance, indicating gender-based disparity.

##### 🌍 Country
- **North America**: Best regional performance.
- **Other regions**: Lower and inconsistent performance.

---

## Caveats and Recommendations
These findings highlight the need for continuous monitoring and improvement of the model to mitigate biases. Suggested steps include:

- Re-evaluating the **training data**.
- Adjusting the **model architecture or training process**.
- Implementing **fairness techniques** (e.g., reweighting, adversarial debiasing).
- Performing **regular audits** across all demographic slices.

---

Ensuring fairness in model outcomes is crucial for building inclusive and trustworthy AI systems.
