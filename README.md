# 🏠 House Price Prediction System

A Machine Learning web application that predicts house prices based on property details using a trained **Decision Tree Regressor** and **Streamlit**.

## ✨ Features

- 🏠 House price prediction
- 📊 20 property input features
- 🤖 Decision Tree Machine Learning model
- ⚡ Instant prediction
- 🎨 Modern dashboard UI
- 📋 Prediction input summary

## 🧠 ML Pipeline

```text
Property Data
     ↓
Ordinal Encoding
     ↓
DictVectorizer
     ↓
Decision Tree Regressor
     ↓
Predicted House Price
```

## 🗂️ Project Structure

```text
House-Price-Prediction/
│
├── app.py
├── house_price_model.pkl
├── vectorizer.pkl
├── encoder.pkl
├── features.pkl
├── Python_Implementation_for_house_price_prediction.ipynb
├── requirements.txt
└── README.md
```

## ⚙️ Technologies

- Python
- Pandas
- Scikit-learn
- Streamlit
- Decision Tree Regressor

## 🚀 Run Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
streamlit run app.py
```

## 🎯 Purpose

This project demonstrates an end-to-end Machine Learning solution for estimating house prices through an interactive web dashboard.

**For educational and project demonstration purposes.**
