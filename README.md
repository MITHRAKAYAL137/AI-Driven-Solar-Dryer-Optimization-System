#  Solar Dryer AI Dashboard

An AI-powered web application to predict drying performance of agricultural products using environmental conditions.

Built with **Streamlit + Machine Learning**

---

##  Features

-  Predict Final Moisture Content
-  Estimate Drying Time
-  Calculate Drying Rate
-  Multiple ML Models:
  - Linear Regression
  - Support Vector Regression (SVR)
  - Random Forest
  - Gradient Boosting
  - Artificial Neural Network (ANN)
  - Stacking Ensemble
-  Auto Best Model Selection (based on performance)
-  Model comparison using R² and RMSE
-  Solar radiation estimation
-  Interactive dashboard (Streamlit)

##  Project Structure
solar_dryer_app/
│── app.py
│── models/
│ ├── linear_regression.pkl
│ ├── svr_model.pkl
│ ├── random_forest.pkl
│ ├── gradient_boosting.pkl
│ ├── stacking_model.pkl
│── requirements.txt
│── README.md