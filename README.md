# Machine Learning Model Development

This document details the process of developing *machine learning* models, from data cleaning to model evaluation, including exploratory analysis, feature engineering, and metric generation. The project was implemented in two Jupyter notebooks:

* `001_EDA.ipynb`: exploratory analysis and data cleaning
* `002_model.ipynb`: model training and evaluation

---

## Project Objective

Develop and compare various binary classification models to predict the `condition` variable (whether a product is new or used) using a MercadoLibre dataset. Techniques applied included EDA, advanced cleaning, feature engineering, encoding, and training with various algorithms.

The final dataset, ready for modeling, was saved as `data_clean.csv`.

---

## Exploratory Data Analysis (EDA)

**Location:** `notebooks/001_EDA.ipynb`

### 1. Data Loading

* Source: `MLA_100k.jsonlines` (100,000 records, 48 columns), including product details like price, condition, shipping methods, seller location, among others
* Origin: The dataset comes from an e-commerce company and contains diverse information about products on its platform.

### 2. General Information

* Data types and null values reviewed (`df.info()`)
* Columns with nulls: `warranty`, `seller_contact`, `official_store_id`, etc.

### 3. Target Variable Distribution

* The `condition` distribution (new vs. used) was plotted using a bar chart, revealing a potential class imbalance that may need addressing during modeling.

### 4. Preliminary Feature Selection

* **Direct:** `price`, `base_price`, `sold_quantity`, `available_quantity` (numeric, no transformation needed)
* **Transformable:** `shipping`, `pictures`, `title`, `seller_address` (require processing to extract useful information)

---

## Data Cleaning

### 1. Initial Validations

* Date conversion: `start_time`, `stop_time` to `datetime` format

### 2. Handling Nulls

* Columns with too many nulls were discarded
* Binary variable `has_warranty` created from `warranty`
* `state` column imputed with the category `"Unknown"`

### 3. Irrelevant Column Removal

* Variables discarded based on:

  * High cardinality or low variance
  * Redundancy
  * Complete absence of useful data

### 4. Variable Transformation

* `picture_count`: calculated from the length of the list in `pictures`
* `title_length`: word count in the title
* Categorical variables like `state`, `buying_mode`, `listing_type_id` encoded using one-hot encoding
* Target variable `condition`: encoded as 0 (used) and 1 (new)

### 5. Consistency Validation

* Inconsistent records were removed
* Variables were normalized (clean names and well-defined categories)

### 6. Final Feature Selection

* Numerical: `price`, `base_price`, `sold_quantity`, `available_quantity`, `picture_count`, `title_length`
* Binary: `has_warranty`, `free_shipping`
* Encoded categoricals: `state`, `buying_mode`, `listing_type_id`
* Target: `condition`

### 7. Clean Dataset Export

* File: `data_clean.csv`
* Path: `/data/data_clean.csv`

---

## Key Improvements Implemented

* Smart Null Handling: Instead of blind imputation, binary variables were created or categories grouped
* Cardinality Reduction: Rare values grouped into "Other" to improve model generalization
* Justified Column Removal: Based on null ratio, cardinality, redundancy, or irrelevance
* Final Validations: Data checked for consistency, duplicates, and format

---

## Feature Engineering

* **Numeric Scaling:** `StandardScaler` without centering the mean (`with_mean=False`) to preserve sparsity
* **Categorical Encoding:** `OneHotEncoder` used for `state`, `buying_mode`, and `listing_type_id` (already encoded in `data_clean.csv`)
* **Data Split:** 80% training, 20% testing using `train_test_split(random_state=42)` for reproducibility

---

## Model Training

**Location:** `notebooks/002_model.ipynb`

Multiple models were evaluated using a pipeline (`modelPipeline`) integrating preprocessing and training. The following algorithms were trained:

* **Logistic Regression:** Linear model for binary classification
* **MLP (Multilayer Perceptron):** Neural network with hidden layers, capable of learning complex nonlinear patterns
* **Decision Tree:** Based on recursive splitting of feature space; prone to overfitting
* **Random Forest:** Ensemble of decision trees trained on random samples (bagging)
* **XGBoost:** Tree-based boosting algorithm that trains models sequentially to correct previous errors; efficient and accurate, ideal for structured data
* **SVC:** Classifier that maximizes class separation; effective with complex data
* **KNeighbors:** Classifies based on nearest neighbors; simple but sensitive to scaling and K choice
* **Gaussian Naive Bayes:** Fast probabilistic model assuming feature independence

---

## Model Evaluation

### Comparative Metrics

| Model               | F1 Score   | Precision  | Recall     | Accuracy   |
| ------------------- | ---------- | ---------- | ---------- | ---------- |
| Logistic Regression | 0.7919     | 0.6745     | 0.9589     | 0.7300     |
| MLP                 | 0.7959     | 0.9083     | 0.7083     | 0.8054     |
| Decision Tree       | 0.8071     | 0.8367     | 0.7796     | 0.8004     |
| Random Forest       | 0.8354     | 0.8501     | 0.8213     | 0.8266     |
| **XGBoost**         | **0.8452** | **0.8715** | **0.8204** | **0.8390** |
| SVC                 | 0.7936     | 0.6829     | 0.9471     | 0.7361     |
| KNeighbors          | 0.8209     | 0.8392     | 0.8034     | 0.8122     |
| GaussianNB          | 0.4458     | 0.8970     | 0.2966     | 0.6048     |

**XGBoost was the best model**, with the highest F1 score, good balance between precision and recall, and excellent accuracy.

---

## Final Evaluation of the XGBoost Model

### Confusion Matrix

![Confusion Matrix](model/xgb_Matriz_confucion.png)

* **True Positives (TP):** 8793
* **True Negatives (TN):** 7987
* **False Positives (FP):** 1296
* **False Negatives (FN):** 1924

Interpretation:

* The high number of TNs and TPs indicates that XGBoost effectively distinguishes between new and used products
* Errors (FP and FN) suggest the model could benefit from class imbalance techniques (e.g., SMOTE) or hyperparameter tuning

### ROC Curve

![ROC Curve](model/xgb_curva_ROC.png)

* **AUC = 0.9243**

The ROC curve shows the model has **strong discriminatory power**. The closer the AUC is to 1, the better the performance. In this case, **AUC = 0.9243**

---

## Conclusions

* The cleaning and transformation process allowed for converting complex and null fields into meaningful variables
* XGBoost outperformed other classifiers, with an excellent balance between precision (87.15%) and recall (82.04%)
* The confusion matrix shows robust performance with over 16,700 correct predictions and few false positives/negatives, supporting its use in real-world scenarios
* The ROC curve and AUC of 0.92 confirm the model’s strong ability to differentiate between new and used products
* The system can be deployed in production to automate product condition detection, optimizing alerts, recommendations, and quality validations
* XGBoost is resource-intensive, which may limit its use in low-resource environments due to high computational cost

---

## Project Future

* Fine-tune XGBoost hyperparameters using Bayesian search to optimize the model
* Apply SMOTE, class weighting, or oversampling to improve performance on minority classes
* Implement k-fold cross-validation for more robust estimates
* Use SHAP or LIME to analyze XGBoost predictions

---

## Contact

For questions, contact [isabellaperezcav@gmail.com](mailto:isabellaperezcav@gmail.com)

