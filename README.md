# 🏟️ Premier League Player Data Analysis (2024–2025)

This project analyzes and models player performance and market value data for the 2024–2025 Premier League season using data collected from [FBref.com](https://fbref.com) and [FootballTransfers.com](https://www.footballtransfers.com).

---

## 📊 Project Goals

- Collect and clean player statistics and transfer values.
- Analyze performance using clustering (K-Means) and dimensionality reduction (PCA).
- Estimate transfer market value using machine learning models.
- Apply and compare feature selection techniques.
- Perform model tuning with RandomizedSearchCV and GridSearchCV.

---


## 📦 Technologies Used

- Python 3.10+
- `pandas`, `numpy`, `scikit-learn`
- `xgboost`, `matplotlib`, `seaborn`
- `BeautifulSoup`, `Selenium` (for scraping)
- `Google API` (to using that please get API key on [AIstudio](https://aistudio.google.com/))
---

## ⚙️ Models Used

- 📈 **Random Forest Regressor**
- 🚀 **XGBoost Regressor**
- 📉 **Linear Regression** (baseline)

All models were tuned using a two-step process:
1. RandomizedSearchCV for wide exploration
2. GridSearchCV for fine-tuning  
This hybrid approach balances performance and efficiency.

---

## 🧠 Feature Selection Techniques

- Random Forest Feature Importance
- Recursive Feature Elimination (RFE)
- SelectKBest with F-test

Combined results helped improve model performance and generalization.

---

## 🗂️ Data Sources

- [FBref](https://fbref.com) — player performance stats
- [FootballTransfers](https://www.footballtransfers.com) — player market values
- [Machine Learning Cơ Bản](https://machinelearningcoban.com) — feature selection reference
- [Hyperparameter Tuning Blog](https://datasciencedances.com/blog/2025/03/hyperparameter-tuning-RandomizedSearchCV/) — tuning strategy

---

## ✍️ Authors

- **Nguyen Pham Trung Hieu** – Student @ PTIT  
- **Instructor:** Kim Ngoc Bach

---

## 📄 License

This project is for academic and educational purposes only. No commercial use of scraped data is permitted.


