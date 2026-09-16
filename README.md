# 🚢 Titanic Survival Prediction — Machine Learning Classification

## 📌 Project Overview

This project builds a **Machine Learning classification model** to predict whether a passenger survived the Titanic disaster based on passenger and travel-related information.

The project follows an end-to-end Machine Learning workflow including:

* Exploratory Data Analysis (EDA)
* Data cleaning
* Missing-value handling
* Duplicate detection
* Outlier treatment
* Feature engineering
* Categorical encoding
* Feature scaling
* Model training
* Model evaluation
* Model comparison
* Model serialization

The final pipeline uses **Logistic Regression** as the selected model based on the evaluation performed in the notebook.

---

## 🎯 Problem Statement

Predict whether a Titanic passenger **survived or did not survive** based on features such as passenger class, gender, age, family information, fare, and port of embarkation.

This is a **binary classification problem** where:

* `0` → Did not survive
* `1` → Survived

---

## 📊 Dataset

The dataset contains passenger information with the following columns:

| Feature       | Description                       |
| ------------- | --------------------------------- |
| `PassengerId` | Unique passenger identifier       |
| `Survived`    | Target variable                   |
| `Pclass`      | Passenger class                   |
| `Name`        | Passenger name                    |
| `Sex`         | Passenger gender                  |
| `Age`         | Passenger age                     |
| `SibSp`       | Number of siblings/spouses aboard |
| `Parch`       | Number of parents/children aboard |
| `Ticket`      | Ticket number                     |
| `Fare`        | Passenger fare                    |
| `Cabin`       | Cabin information                 |
| `Embarked`    | Port of embarkation               |

The notebook identifies missing values in `Age`, `Cabin`, and `Embarked`, with `Cabin` having the largest proportion of missing values.

---

## 🔍 Exploratory Data Analysis

The project performs initial data inspection and analysis, including:

* Dataset inspection
* Missing-value analysis
* Duplicate-row detection
* Numerical feature analysis
* Outlier analysis
* Visualization of numerical variables

The dataset contained **no duplicate rows** during the initial inspection.

---

## 🧹 Data Preprocessing

### 1. Duplicate Removal

Duplicate records were checked and removed using:

```python
df = df.drop_duplicates().copy()
```

### 2. Outlier Treatment

The project uses the **IQR (Interquartile Range)** method to calculate lower and upper bounds and cap numerical outliers.

### 3. Feature Engineering

Several additional features were created:

#### FamilySize

```text
FamilySize = SibSp + Parch + 1
```

#### IsAlone

Indicates whether the passenger was travelling alone.

```text
IsAlone = 1 → Passenger travelling alone
IsAlone = 0 → Passenger travelling with family
```

#### HasCabin

Indicates whether cabin information was available.

#### Title

Passenger titles such as `Mr`, `Mrs`, `Miss`, etc. were extracted from the passenger name.

---

## 🩹 Missing Value Treatment

Missing values were handled as part of the preprocessing workflow.

### Age

Age is imputed using the median age grouped by:

```text
Sex + Pclass
```

If any missing values remain, the overall median is used.

### Embarked

Missing values are replaced using the **mode**.

### Fare

Missing values are replaced using the **median**.

### Cabin

Instead of directly using the high-missingness `Cabin` column, a `HasCabin` feature is created and the original `Cabin` column is removed.

---

## 🗑️ Feature Removal

The following columns are removed before model training:

```python
Name
Ticket
Cabin
PassengerId
```

The target variable is:

```text
Survived
```

The remaining columns are used as model features.

---

## ✂️ Train-Test Split

The dataset is divided into training and testing sets using:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Therefore:

* **80%** → Training data
* **20%** → Testing data
* `random_state = 42`
* Stratified splitting is used to maintain the target-class distribution.

---

## ⚙️ Preprocessing Pipeline

A Scikit-learn `ColumnTransformer` is used to preprocess numerical and categorical features separately.

### Numerical Features

Numerical features are transformed using:

```python
StandardScaler()
```

### Categorical Features

Categorical features are transformed using:

```python
OneHotEncoder(handle_unknown="ignore")
```

The preprocessing and model are combined into a single Scikit-learn `Pipeline`.

---

## 🤖 Machine Learning Models

Two classification algorithms were trained and evaluated.

### 1. Logistic Regression

```python
LogisticRegression(max_iter=200)
```

### 2. Decision Tree Classifier

```python
DecisionTreeClassifier(random_state=42)
```

Both models use the same preprocessing pipeline.

---

## 📈 Model Evaluation

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

### Logistic Regression

| Metric    | Class 0 | Class 1 |
| --------- | ------: | ------: |
| Precision |    0.85 |    0.80 |
| Recall    |    0.88 |    0.75 |
| F1-score  |    0.87 |    0.78 |

**Accuracy: 83.24%**

Confusion Matrix:

```text
[[97 13]
 [17 52]]
```

### Decision Tree

| Metric    | Class 0 | Class 1 |
| --------- | ------: | ------: |
| Precision |    0.81 |    0.71 |
| Recall    |    0.83 |    0.68 |
| F1-score  |    0.82 |    0.70 |

**Accuracy: 77.09%**

Confusion Matrix:

```text
[[91 19]
 [22 47]]
```

These are the final evaluation results recorded in the notebook's reusable pipeline section.

---

## 🏆 Model Selection

The notebook compares the test-set accuracy of Logistic Regression and Decision Tree.

The selected model in the final pipeline is:

```text
Logistic Regression
```

with an accuracy of:

```text
0.8324
```

or approximately:

```text
83.24%
```

---

## 💾 Saved Artifacts

The project saves the processed dataset and trained model pipeline.

### Cleaned Dataset

```text
titanic_cleaned.csv
```

### Trained Model

```text
titanic_logreg_pipeline.pkl
```

The model is saved using `joblib`, allowing the complete preprocessing + Logistic Regression pipeline to be reused later.

---

## 📁 Project Structure

```text
Titanic-ML-Classification/
│
├── ML-Pipeline on Titanic.ipynb
├── titanic_cleaned.csv
├── titanic_logreg_pipeline.pkl
├── README.md
└── requirements.txt
```

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**
* **Joblib**
* **Jupyter Notebook**

The notebook imports Scikit-learn components for preprocessing, pipelines, Logistic Regression, Decision Trees, and model evaluation.

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Titanic-ML-Classification.git
cd Titanic-ML-Classification
```

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter
```

---

## ▶️ How to Run

### 1. Start Jupyter Notebook

```bash
jupyter notebook
```

### 2. Open

```text
ML-Pipeline on Titanic.ipynb
```

### 3. Run the notebook cells

The notebook performs the complete workflow:

```text
Load Dataset
      ↓
EDA
      ↓
Data Cleaning
      ↓
Missing Value Handling
      ↓
Outlier Treatment
      ↓
Feature Engineering
      ↓
Train-Test Split
      ↓
Preprocessing Pipeline
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Model Comparison
      ↓
Save Best Model
```

---

## 🔮 Making Predictions with the Saved Model

The trained pipeline can be loaded using `joblib`:

```python
import joblib

model = joblib.load("titanic_logreg_pipeline.pkl")
```

The saved pipeline contains the preprocessing steps together with the trained Logistic Regression model.

---

## 📌 Key Learnings

Through this project, the following Machine Learning concepts were implemented:

* Exploratory Data Analysis
* Missing-value analysis
* Duplicate detection
* IQR-based outlier treatment
* Feature engineering
* Numerical feature scaling
* One-hot encoding
* Train-test splitting
* Stratified sampling
* Scikit-learn pipelines
* Logistic Regression
* Decision Tree Classification
* Classification metrics
* Confusion matrices
* Model comparison
* Model serialization using Joblib

---

## 📊 Results Summary

| Model               |   Accuracy |
| ------------------- | ---------: |
| Logistic Regression | **83.24%** |
| Decision Tree       | **77.09%** |

The final notebook selected the Logistic Regression pipeline based on the recorded test accuracy.

---

## 🚀 Future Improvements

Possible extensions to this project include:

* Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
* Cross-validation
* Feature importance analysis
* ROC-AUC evaluation
* Precision-Recall analysis
* Additional classification algorithms
* Model deployment using Flask or FastAPI
* Creating a web interface for passenger survival prediction

---

## 👨‍💻 Author

**Vamsi Krishna**

Computer Science & Engineering Graduate
Interested in Machine Learning, Data Science, Python, SQL, and AI/ML.

---

## 📜 License

This project is intended for educational and learning purposes.
