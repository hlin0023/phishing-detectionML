# 🛡️ Phishing Website Detection using Machine Learning

## Overview

This project implements machine learning models to detect phishing websites based on extracted URL and webpage metadata features. Phishing websites mimic legitimate ones to steal sensitive user data such as credentials or financial information. As cybersecurity threats increase, automated detection systems become crucial.

The project leverages a rich dataset containing over 200,000 labeled URLs with 60+ extracted features to classify websites as either **phishing** or **legitimate**.

## 📊 Dataset

- **Name:** [PhiUSIIL Phishing URL Dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset)
- **Size:** 235,795 total instances
  - 134,850 legitimate
  - 100,945 phishing
- **Features:** 60+ features, including:
  - URL metadata (length, number of special characters, etc.)
  - Page structure (HTML/JS/CSS attributes)
  - Content sentiment (title polarity, subjectivity)
  - Network attributes (HTTPS usage, redirection count)

## 🔧 Requirements

- Python 3.8+
- Jupyter Notebook or JupyterLab
- Libraries (all imported in code):
  - pandas
  - numpy
  - sklearn
  - matplotlib
  - seaborn
  - textblob
  - word2vec (gensim)
  - scipy

> ⚠️ Note: You may need to download the dataset manually from the [UCI Repository](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset) and adjust the file path in `a2.ipynb`.

## ⚙️ Project Structure

- `ipynb` - The main notebook with all code and results
- `README.md` - Project documentation
- `pdf`  - Summarized findings and comparisons 

## 📌 Methodology Summary

### 🧹 Data Preprocessing

- Removed duplicates
- Handled categorical features (TLD, Title, Domain)
- Log transformations and z-score standardization for skewed numeric features
- Added new engineered features (e.g., `FinancialFlag` using XOR logic)

### 🛠️ Feature Engineering

- Word2Vec + SVD on webpage titles
- Sentiment analysis using TextBlob
- TLD risk encoding
- Suspicious token analysis in URLs

### 🧪 Feature Selection

Used three filter-based methods:
1. **Correlation Filtering** (remove redundancy, threshold > 0.7)
2. **Mutual Information** (capture nonlinear dependencies)
3. **Chi-Square Test** (statistical relevance)

Final feature count: **24**

### 🤖 Models & Training

- **Naive Bayes**
- **Decision Tree (max depth = 3)**
- **Support Vector Machine (SVC)**

Training used:
- 5-fold Cross-Validation
- Nested CV (3-fold) for hyperparameter tuning

### 🧮 Hyperparameter Tuning

- GNB: `var_smoothing`
- DT: `max_depth`, `min_samples_split`
- SVM: kernel = {`linear`, `rbf`}, `C` = [0.1, 1, 10]

### 📈 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score

Best results:
- **SVM (after tuning):** Accuracy = 0.996
- **Decision Tree:** High interpretability with accuracy ~0.985
- **Naive Bayes:** Strong precision (0.9856), fast and simple

## 💡 Key Findings

- Simpler models like Naive Bayes and Decision Trees performed surprisingly well after careful feature selection.
- SVM required more tuning but achieved the highest accuracy.
- Sentiment and title-based features contributed to model improvements.

## 📉 Performance Summary

| Model         | Accuracy | Precision | Recall | F1 Score |
|---------------|----------|-----------|--------|----------|
| Naive Bayes   | 0.9841   | 0.9856    | 0.9821 | 0.9837   |
| Decision Tree | 0.9845   | 0.9836    | 0.9848 | 0.9842   |
| SVM (tuned)   | 0.9960   | 0.9958    | 0.9963 | 0.9960   |

## 🧠 Future Improvements

- Use ensemble models like **Random Forest** or **XGBoost**
- Deploy the model as a REST API
- Interpretability with SHAP values
- Real-time URL scanning browser extension

## 📚 References

- Prasad, R., & Chandra, S. (2023). *PhiUSIIL: Phishing Website Detection Dataset*. UCI Machine Learning Repository.
- Prakash, P., et al. (2010). *PhishNet: Predictive Blacklisting*. NDSS Symposium.
- Jalil, Z., Usman, M., & Fong, S. (2022). *Feature Engineering for Phishing Detection*. Journal of Information Security.
