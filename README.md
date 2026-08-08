# Automobile-price-prediction
https://automobile-price-prediction-181o.onrender.com/

# 🚗 AURA Motors — Executive Vehicle Intelligence & Analytics Portal

An enterprise-grade, AI-powered vehicle valuation, telemetry dashboard, and dynamic insurance recommendation platform built using **Flask**, **scikit-learn**, and **Chart.js**.

This project loads a trained **RandomForestRegressor** machine learning model (`RMR (1).pkl`) to predict vehicle market prices and renders real-time financial, telemetry, and risk analytics across customizable executive visual interfaces.

---

## ✨ Features

- **Machine Learning Valuation Engine**: Loads a pre-trained `RandomForestRegressor` pickle model to evaluate car prices based on 17 vehicle specifications.
- **Label Encoding Compatibility**: Renders human-readable strings (`Toyota`, `Automatic`, `Petrol`) in the UI while passing numerically encoded values to scikit-learn on the backend to avoid float conversion errors.
- **Interactive Visual Analytics Engine**:
  - **5-Year Price Depreciation Curve** (Line Chart)
  - **Valuation Component Drivers** (Doughnut Chart)
  - **Total Cost of Ownership (TCO)** Breakdown (Stacked Bar Chart)
  - **Vehicle Performance & Telemetry** (Radar Chart)
- **Real-Time Currency Converter**: Instant conversion across **USD ($)**, **EUR (€)**, **GBP (£)**, and **INR (₹)** on both main market prices and insurance proposals without extra API roundtrips.
- **Multilingual UI Support**: Real-time interface translation for **English**, **Spanish**, **French**, **German**, and **Japanese**.
- **Multi-Theme Studio**: 5 dynamic visual presets:
  - 🌙 **Premium Dark**
  - ☀️ **Premium Light**
  - ⚡ **Cyber Neon**
  - 👑 **Royal Gold**
  - 🌿 **Emerald Lux**
- **Dynamic Insurance Proposals**: Auto-generates tailored insurance coverage packages based on predicted market value and vehicle risk profile.

---

## 📁 Repository Structure

```text
├── app.py              # Main Flask server application & embedded single-page UI
├── RMR (1).pkl         # Trained Random Forest Regressor pickle model
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
🛠️ Installation & Setup1. PrerequisitesEnsure you have Python 3.9+ installed on your system.2. Clone / Setup WorkspacePlace app.py, RMR (1).pkl, and requirements.txt in the same root folder.3. Install DependenciesInstall the required packages using pip:Bashpip install -r requirements.txt
🚀 Running the ApplicationExecute the Flask server:Bashpython app.py
Once running, access the dashboard in your browser at:👉 http://127.0.0.1:5000📊 Feature Inputs SchemaThe underlying Random Forest model predicts vehicle value using the following 17 features:Categorical FeaturesNumerical FeaturesMakeYearModelEngine_SizeFuel_TypeMileageTransmissionHorsepowerAccident_HistoryTorqueService_HistoryOwnersColorFuel_EfficiencyBody_TypeDrivetrainLocation🧰 Tech StackBackend: Python, Flask, Pandas, NumPy, Scikit-learnFrontend: HTML5, CSS3 (Glassmorphism & CSS Variables), JavaScript (ES6+ AJAX)Charts & Visuals: Chart.jsIcons & Fonts: Font Awesome 6, Plus Jakarta Sans, Space Grotesk
