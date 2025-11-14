# TASK - 5: Advertising Sales Prediction using Machine Learning

## 📘 Objective
This project focuses on building a machine learning model that predicts **sales revenue** based on advertising spending across major channels such as **TV**, **Radio**, and **Newspaper**.  
The objective is to showcase how data-driven insights can help optimize marketing investments for better returns.

---

## 🧩 Steps Performed

### 1. **Data Loading**
- Loaded the dataset with `pandas`.
- Reviewed the dataset's dimensions, column types, and summary statistics.
- Checked for null entries and analyzed feature distributions.

### 2. **Exploratory Data Analysis (EDA)**
- Explored relationships between marketing spend and sales.
- Used **scatter plots**, **pair plots**, and **correlation heatmaps** for visualization.
- Observed that **TV advertising** has the highest influence on sales.

### 3. **Data Preprocessing**
- Divided the dataset into **training** and **test** sets using `train_test_split`.
- Normalized numerical variables using **StandardScaler**.
- Built preprocessing steps using `make_pipeline` and `make_column_transformer` for cleaner workflow.

### 4. **Model Building**
- Implemented a **Random Forest Regressor** to perform sales prediction.
- Trained the model using the training data.
- Tuned hyperparameters to enhance accuracy and overall model performance.

### 5. **Model Evaluation**
- Assessed the model using:
  - **R² Score**
  - **Mean Absolute Error (MAE)**
  - **Root Mean Squared Error (RMSE)**
- Visualized actual vs. predicted values to interpret model performance.

### 6. **Model Saving**
- Exported the trained model using `pickle`.
- Ensured the model can be reused in future applications without retraining.

---

## ⚙️ Tools & Libraries Used

| Category | Tools/Libraries |
|----------|------------------|
| Data Handling | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Machine Learning | `scikit-learn` (`RandomForestRegressor`, `Pipeline`, `StandardScaler`, `train_test_split`) |
| Model Saving | `pickle` |

---

## 🏁 Outcome
- Built an effective regression model for predicting sales based on advertising budgets.
- The **Random Forest Regressor** delivered a strong **R² score**, confirming high predictive accuracy.
- Gain insights into how different marketing channels influence sales outcomes.
- Completed a full ML pipeline — from EDA and preprocessing to training, evaluation, and saving the model.

---
