import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Model loading logic
MODEL_PATH = "RMR (1).pkl"
model = None

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print(f"Model '{MODEL_PATH}' loaded successfully.")
    except Exception as e:
        print(f"Error loading pickle file: {e}")
else:
    print(f"Warning: '{MODEL_PATH}' not found in the root directory.")

# Categorical mapping: Maps column name -> List of (encoded_index, category_name)
CATEGORICAL_OPTIONS = {
    "Make": [
        (0, "Toyota"), (1, "Honda"), (2, "Ford"), (3, "BMW"), 
        (4, "Mercedes-Benz"), (5, "Audi"), (6, "Hyundai"), 
        (7, "Nissan"), (8, "Chevrolet"), (9, "Kia")
    ],
    "Model": [
        (0, "Sedan"), (1, "SUV"), (2, "Hatchback"), (3, "Coupe"), 
        (4, "Truck"), (5, "Convertible"), (6, "Van"), (7, "Wagon")
    ],
    "Fuel_Type": [
        (0, "Petrol"), (1, "Diesel"), (2, "Hybrid"), (3, "Electric"), (4, "CNG")
    ],
    "Transmission": [
        (0, "Automatic"), (1, "Manual"), (2, "Semi-Automatic")
    ],
    "Accident_History": [
        (0, "None"), (1, "Minor"), (2, "Major")
    ],
    "Service_History": [
        (0, "Full"), (1, "Partial"), (2, "None")
    ],
    "Color": [
        (0, "Black"), (1, "White"), (2, "Silver"), (3, "Blue"), (4, "Red"), (5, "Grey")
    ],
    "Body_Type": [
        (0, "Sedan"), (1, "SUV"), (2, "Hatchback"), (3, "Coupe"), (4, "Pickup")
    ],
    "Drivetrain": [
        (0, "FWD"), (1, "RWD"), (2, "AWD"), (3, "4WD")
    ],
    "Location": [
        (0, "Urban"), (1, "Suburban"), (2, "Rural")
    ]
}

CAT_LOOKUP = {
    col: {str(code): label for code, label in options}
    for col, options in CATEGORICAL_OPTIONS.items()
}

DEFAULT_NUMERICALS = {
    "Year": 2022,
    "Engine_Size": 2.5,
    "Mileage": 32000.0,
    "Horsepower": 210.0,
    "Torque": 280.0,
    "Owners": 1,
    "Fuel_Efficiency": 16.8
}

FEATURE_ORDER = [
    "Make", "Model", "Year", "Fuel_Type", "Transmission", 
    "Engine_Size", "Mileage", "Horsepower", "Torque", "Owners", 
    "Accident_History", "Service_History", "Color", "Body_Type", 
    "Drivetrain", "Fuel_Efficiency", "Location"
]

# Base currency multipliers relative to USD ($)
CURRENCY_RATES = {
    "USD": {"symbol": "$", "rate": 1.0},
    "EUR": {"symbol": "€", "rate": 0.92},
    "GBP": {"symbol": "£", "rate": 0.78},
    "INR": {"symbol": "₹", "rate": 83.5}
}

def generate_insurance_recommendations(price, make_label, currency_code="USD"):
    """Generate luxury insurance coverage proposals formatted with selected currency."""
    curr = CURRENCY_RATES.get(currency_code, CURRENCY_RATES["USD"])
    symbol = curr["symbol"]
    rate = curr["rate"]
    
    converted_price = price * rate
    recommendations = []
    
    if price > 35000 or make_label in ["BMW", "Mercedes-Benz", "Audi"]:
        recommendations.append({
            "title": "Titanium Sovereign Cover",
            "tier": "PLATINUM",
            "badge": "VIP Choice",
            "price_est": f"{symbol}{int(converted_price * 0.042):,}/yr",
            "features": ["Zero Depreciation", "Full Engine & ECU Protection", "24/7 Global Concierge", "Guaranteed Invoice Value"],
            "match": "99% Precision Match"
        })
    
    recommendations.append({
        "title": "Apex Dynamic Shield",
        "tier": "EXECUTIVE",
        "badge": "Recommended",
        "price_est": f"{symbol}{int(converted_price * 0.031):,}/yr",
        "features": ["All-Risk Collision Cover", "Third-Party Unlimited Liability", "Personal Driver Cover", "Cashless Repair Hubs"],
        "match": "94% Optimal Match"
    })
    
    recommendations.append({
        "title": "Metro Guard Select",
        "tier": "ESSENTIAL",
        "badge": "Smart Value",
        "price_est": f"{symbol}{int(converted_price * 0.020):,}/yr",
        "features": ["Fire, Theft & Vandalism", "Emergency Towing & Key Lockout", "Basic Medical Assist"],
        "match": "88% Match"
    })
    
    return recommendations

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AURA Motors — Executive Vehicle Intelligence Portal</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    
    <style>
        /* 1. Premium Dark Theme (Default) */
        body.theme-dark {
            --bg-base: #0b0f19;
            --bg-card: rgba(20, 27, 41, 0.75);
            --card-border: rgba(255, 255, 255, 0.08);
            --accent: #6366f1;
            --accent-glow: rgba(99, 102, 241, 0.35);
            --accent-hover: #4f46e5;
            --text-heading: #f8fafc;
            --text-sub: #94a3b8;
            --input-bg: rgba(15, 23, 42, 0.6);
            --header-bg: rgba(11, 15, 25, 0.85);
        }

        /* 2. Premium Light Theme */
        body.theme-light {
            --bg-base: #f1f5f9;
            --bg-card: rgba(255, 255, 255, 0.85);
            --card-border: rgba(0, 0, 0, 0.08);
            --accent: #4f46e5;
            --accent-glow: rgba(79, 70, 229, 0.2);
            --accent-hover: #4338ca;
            --text-heading: #0f172a;
            --text-sub: #64748b;
            --input-bg: rgba(248, 250, 252, 0.9);
            --header-bg: rgba(255, 255, 255, 0.9);
        }

        /* 3. Cyber Neon Theme */
        body.theme-cyber {
            --bg-base: #070913;
            --bg-card: rgba(14, 20, 38, 0.75);
            --card-border: rgba(0, 240, 255, 0.2);
            --accent: #00f0ff;
            --accent-glow: rgba(0, 240, 255, 0.4);
            --accent-hover: #70f3ff;
            --text-heading: #ffffff;
            --text-sub: #8ba1cd;
            --input-bg: rgba(10, 15, 30, 0.7);
            --header-bg: rgba(7, 9, 19, 0.9);
        }

        /* 4. Royal Gold Theme */
        body.theme-gold {
            --bg-base: #0f0d0a;
            --bg-card: rgba(28, 24, 18, 0.75);
            --card-border: rgba(212, 175, 55, 0.25);
            --accent: #e5c158;
            --accent-glow: rgba(229, 193, 88, 0.35);
            --accent-hover: #f3d478;
            --text-heading: #fffdf5;
            --text-sub: #a89f91;
            --input-bg: rgba(20, 17, 12, 0.7);
            --header-bg: rgba(15, 13, 10, 0.9);
        }

        /* 5. Emerald Lux Theme */
        body.theme-emerald {
            --bg-base: #06130e;
            --bg-card: rgba(12, 33, 25, 0.75);
            --card-border: rgba(16, 185, 129, 0.25);
            --accent: #10b981;
            --accent-glow: rgba(16, 185, 129, 0.4);
            --accent-hover: #34d399;
            --text-heading: #f0fdf4;
            --text-sub: #86efac;
            --input-bg: rgba(8, 24, 18, 0.7);
            --header-bg: rgba(6, 19, 14, 0.9);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
            transition: background-color 0.3s ease, border-color 0.3s ease, color 0.3s ease, box-shadow 0.3s ease;
        }

        body {
            background-color: var(--bg-base);
            color: var(--text-heading);
            min-height: 100vh;
            background-image: 
                radial-gradient(circle at 15% 20%, var(--accent-glow) 0%, transparent 45%),
                radial-gradient(circle at 85% 80%, var(--accent-glow) 0%, transparent 45%);
            background-attachment: fixed;
            overflow-x: hidden;
        }

        /* Header Layout */
        .header-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 16px 40px;
            background: var(--header-bg);
            backdrop-filter: blur(15px);
            border-bottom: 1px solid var(--card-border);
            position: fixed;
            top: 0;
            width: 100%;
            z-index: 1000;
        }

        .brand-box {
            display: flex;
            align-items: center;
            gap: 12px;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.35rem;
            font-weight: 700;
            letter-spacing: 1px;
            color: var(--text-heading);
        }

        .brand-icon {
            width: 36px;
            height: 36px;
            background: var(--accent);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #000;
            box-shadow: 0 0 15px var(--accent-glow);
        }

        .controls-group {
            display: flex;
            align-items: center;
            gap: 16px;
        }

        .control-select {
            background: rgba(0, 0, 0, 0.2);
            border: 1px solid var(--card-border);
            color: var(--text-heading);
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.82rem;
            font-weight: 600;
            outline: none;
            cursor: pointer;
        }

        .control-select option {
            background-color: var(--bg-base);
            color: var(--text-heading);
        }

        .theme-switcher {
            display: flex;
            align-items: center;
            gap: 8px;
            background: rgba(0, 0, 0, 0.2);
            padding: 5px 12px;
            border-radius: 30px;
            border: 1px solid var(--card-border);
        }

        .theme-dot {
            width: 18px;
            height: 18px;
            border-radius: 50%;
            cursor: pointer;
            border: 2px solid transparent;
            transition: transform 0.2s ease;
        }

        .theme-dot:hover { transform: scale(1.25); }
        .theme-dot.active { border-color: var(--text-heading); transform: scale(1.1); }

        .dot-dark { background: #6366f1; }
        .dot-light { background: #ffffff; border: 1px solid #ccc; }
        .dot-cyber { background: #00f0ff; }
        .dot-gold { background: #e5c158; }
        .dot-emerald { background: #10b981; }

        /* Dashboard Container */
        .wrapper {
            margin-top: 85px;
            padding: 35px 40px;
            max-width: 1550px;
            margin-left: auto;
            margin-right: auto;
        }

        .hero-head { margin-bottom: 25px; }
        .hero-head h1 {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2.1rem;
            font-weight: 700;
            margin-bottom: 6px;
        }
        .hero-head p { color: var(--text-sub); font-size: 0.98rem; }

        .dashboard-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 28px;
        }

        @media (max-width: 1100px) {
            .dashboard-grid { grid-template-columns: 1fr; }
            .wrapper { padding: 20px; }
            .header-bar { padding: 15px 20px; flex-direction: column; gap: 12px; }
        }

        /* Glass Panels */
        .glass-panel {
            background: var(--bg-card);
            backdrop-filter: blur(20px);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 28px;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.25);
            position: relative;
        }

        .panel-title {
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 22px;
            display: flex;
            align-items: center;
            gap: 10px;
            border-bottom: 1px solid var(--card-border);
            padding-bottom: 12px;
            color: var(--text-heading);
        }

        .panel-title i { color: var(--accent); }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
        }

        .form-field { display: flex; flex-direction: column; gap: 5px; }
        .form-field label {
            font-size: 0.76rem;
            font-weight: 700;
            color: var(--text-sub);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .custom-input {
            background: var(--input-bg);
            border: 1px solid var(--card-border);
            color: var(--text-heading);
            padding: 11px 13px;
            border-radius: 8px;
            font-size: 0.92rem;
            outline: none;
        }

        .custom-input:focus {
            border-color: var(--accent);
            box-shadow: 0 0 12px var(--accent-glow);
        }

        .btn-holder { margin-top: 22px; }

        .btn-spark {
            width: 100%;
            background: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%);
            color: #ffffff;
            border: none;
            padding: 15px;
            font-size: 0.98rem;
            font-weight: 700;
            border-radius: 10px;
            cursor: pointer;
            position: relative;
            overflow: hidden;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
            box-shadow: 0 8px 25px var(--accent-glow);
            transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }

        .btn-spark::after {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: linear-gradient(60deg, transparent, rgba(255, 255, 255, 0.25), transparent);
            transform: rotate(30deg);
            transition: transform 0.8s ease;
        }

        .btn-spark:hover::after { transform: translate(100%, 100%) rotate(30deg); }
        .btn-spark:hover { transform: translateY(-3px) scale(1.01); box-shadow: 0 12px 30px var(--accent-glow); }
        .btn-spark:active { transform: translateY(1px); }

        .val-card {
            background: rgba(0, 0, 0, 0.15);
            border: 1px dashed var(--accent);
            border-radius: 12px;
            padding: 22px;
            text-align: center;
            margin-bottom: 22px;
        }

        .val-sub {
            font-size: 0.78rem;
            color: var(--text-sub);
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 700;
        }

        .val-price {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 3rem;
            font-weight: 700;
            color: var(--accent);
            margin: 8px 0;
            text-shadow: 0 0 25px var(--accent-glow);
        }

        .status-pill {
            display: inline-block;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid var(--card-border);
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.75rem;
            color: var(--text-heading);
        }

        .metrics-row {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            margin-bottom: 22px;
        }

        .m-box {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--card-border);
            padding: 13px;
            border-radius: 10px;
            text-align: center;
        }

        .m-val { font-size: 1.1rem; font-weight: 700; margin-top: 4px; color: var(--text-heading); }
        .m-lbl { font-size: 0.7rem; color: var(--text-sub); text-transform: uppercase; }

        .plans-list { display: flex; flex-direction: column; gap: 12px; }

        .plan-card {
            background: var(--input-bg);
            border: 1px solid var(--card-border);
            border-radius: 10px;
            padding: 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
            transition: all 0.25s ease;
        }

        .plan-card:hover {
            border-color: var(--accent);
            transform: translateX(4px);
            box-shadow: 0 4px 20px var(--accent-glow);
        }

        .plan-header { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }
        .plan-header h4 { font-size: 0.95rem; font-weight: 700; }

        .badge-tier {
            background: var(--accent);
            color: #000;
            font-weight: 800;
            font-size: 0.65rem;
            padding: 2px 8px;
            border-radius: 12px;
            text-transform: uppercase;
        }

        .plan-meta { font-size: 0.78rem; color: var(--text-sub); }
        .plan-features { display: flex; gap: 6px; margin-top: 8px; flex-wrap: wrap; }

        .f-tag {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--card-border);
            padding: 2px 7px;
            border-radius: 4px;
            font-size: 0.68rem;
            color: var(--text-sub);
        }
    </style>
</head>
<body class="theme-dark">

    <header class="header-bar">
        <div class="brand-box">
            <div class="brand-icon"><i class="fa-solid fa-gauge-high"></i></div>
            <span>AURA MOTORS</span>
        </div>
        
        <div class="controls-group">
            <!-- Language Selection -->
            <select id="langSelect" class="control-select" onchange="updateLanguage()">
                <option value="en">English (US)</option>
                <option value="es">Español</option>
                <option value="fr">Français</option>
                <option value="de">Deutsch</option>
                <option value="ja">日本語</option>
            </select>

            <!-- Currency Toggle -->
            <select id="currSelect" class="control-select">
                <option value="USD">USD ($)</option>
                <option value="EUR">EUR (€)</option>
                <option value="GBP">GBP (£)</option>
                <option value="INR">INR (₹)</option>
            </select>

            <!-- Multi-Theme Engine Selector -->
            <div class="theme-switcher">
                <div class="theme-dot dot-dark active" onclick="switchTheme('theme-dark', this)" title="Premium Dark"></div>
                <div class="theme-dot dot-light" onclick="switchTheme('theme-light', this)" title="Premium Light"></div>
                <div class="theme-dot dot-cyber" onclick="switchTheme('theme-cyber', this)" title="Cyber Neon"></div>
                <div class="theme-dot dot-gold" onclick="switchTheme('theme-gold', this)" title="Royal Gold"></div>
                <div class="theme-dot dot-emerald" onclick="switchTheme('theme-emerald', this)" title="Emerald Lux"></div>
            </div>
        </div>
    </header>

    <div class="wrapper">
        <div class="hero-head">
            <h1 id="txtTitle">Executive Valuation & Telemetry</h1>
            <p id="txtSub">Compute market appraisal via Random Forest regression and evaluate tailored insurance plans.</p>
        </div>

        <div class="dashboard-grid">
            <div class="glass-panel">
                <div class="panel-title"><i class="fa-solid fa-sliders"></i> <span id="txtSpec">Vehicle Specifications</span></div>
                <form id="valuationForm">
                    <div class="form-grid">
                        {% for col, options in categorical_options.items() %}
                        <div class="form-field">
                            <label for="{{ col }}">{{ col.replace('_', ' ') }}</label>
                            <select id="{{ col }}" name="{{ col }}" class="custom-input">
                                {% for code, label in options %}
                                <option value="{{ code }}">{{ label }}</option>
                                {% endfor %}
                            </select>
                        </div>
                        {% endfor %}

                        {% for col, val in default_numericals.items() %}
                        <div class="form-field">
                            <label for="{{ col }}">{{ col.replace('_', ' ') }}</label>
                            <input type="number" step="any" id="{{ col }}" name="{{ col }}" value="{{ val }}" class="custom-input" required>
                        </div>
                        {% endfor %}
                    </div>

                    <div class="btn-holder">
                        <button type="submit" class="btn-spark" id="calcBtn">
                            <i class="fa-solid fa-microchip"></i> <span id="txtBtn">Run AI Valuation</span>
                        </button>
                    </div>
                </form>
            </div>

            <div class="glass-panel">
                <div class="panel-title"><i class="fa-solid fa-chart-pie"></i> <span id="txtOut">Intelligence Output</span></div>

                <div class="val-card">
                    <div class="val-sub" id="txtValSub">Estimated Market Value</div>
                    <div class="val-price" id="valPrice">$0.00</div>
                    <span class="status-pill" id="valStatus">Ready for Calculation</span>
                </div>

                <div class="metrics-row">
                    <div class="m-box">
                        <div class="m-lbl" id="lblEff">Efficiency</div>
                        <div class="m-val" id="mEfficiency">--</div>
                    </div>
                    <div class="m-box">
                        <div class="m-lbl" id="lblRisk">Risk Rating</div>
                        <div class="m-val" id="mRisk">--</div>
                    </div>
                    <div class="m-box">
                        <div class="m-lbl" id="lblDep">Depreciation</div>
                        <div class="m-val" id="mDep">--</div>
                    </div>
                </div>

                <div class="panel-title" style="margin-top: 10px; border-bottom: none; margin-bottom: 12px;">
                    <i class="fa-solid fa-shield"></i> <span id="txtPlans">Premium Coverage Plans</span>
                </div>
                <div class="plans-list" id="insuranceContainer">
                    <p style="color: var(--text-sub); text-align: center; padding: 20px;" id="txtPrompt">
                        Input vehicle metrics and execute valuation to view dynamic insurance proposals.
                    </p>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Language Dictionaries
        const i18n = {
            en: {
                title: "Executive Valuation & Telemetry",
                sub: "Compute market appraisal via Random Forest regression and evaluate tailored insurance plans.",
                spec: "Vehicle Specifications",
                out: "Intelligence Output",
                valsub: "Estimated Market Value",
                btn: "Run AI Valuation",
                btnProc: "Processing Model...",
                eff: "Efficiency",
                risk: "Risk Rating",
                dep: "Depreciation",
                plans: "Premium Coverage Plans",
                prompt: "Input vehicle metrics and execute valuation to view dynamic insurance proposals.",
                ready: "Valuation Verified"
            },
            es: {
                title: "Evaluación Ejecutiva y Telemetría",
                sub: "Calcule la tasación de mercado mediante regresión de bosque aleatorio.",
                spec: "Especificaciones del Vehículo",
                out: "Resultado de Inteligencia",
                valsub: "Valor de Mercado Estimado",
                btn: "Ejecutar Valoración IA",
                btnProc: "Procesando Modelo...",
                eff: "Eficiencia",
                risk: "Clasificación de Riesgo",
                dep: "Depreciación",
                plans: "Planes de Cobertura Premium",
                prompt: "Ingrese las métricas del vehículo para ver las propuestas de seguro.",
                ready: "Valoración Verificada"
            },
            fr: {
                title: "Évaluation Exécutive et Télémétrie",
                sub: "Calculez l'évaluation du marché via la régression Random Forest.",
                spec: "Spécifications du Véhicule",
                out: "Résultat d'Intelligence",
                valsub: "Valeur Estimée du Marché",
                btn: "Lancer l'Évaluation IA",
                btnProc: "Traitement du Modèle...",
                eff: "Efficacité",
                risk: "Évaluation du Risque",
                dep: "Dépréciation",
                plans: "Plans de Couverture Premium",
                prompt: "Saisissez les métriques du véhicule pour voir les propositions d'assurance.",
                ready: "Évaluation Vérifiée"
            },
            de: {
                title: "Executive Bewertung & Telemetrie",
                sub: "Berechnen Sie den Marktwert über Random Forest Regression.",
                spec: "Fahrzeugspezifikationen",
                out: "Intelligenz-Ausgabe",
                valsub: "Geschätzter Marktwert",
                btn: "KI-Bewertung Starten",
                btnProc: "Modell wird verarbeitet...",
                eff: "Effizienz",
                risk: "Risikobewertung",
                dep: "Wertverlust",
                plans: "Premium-Deckungspläne",
                prompt: "Geben Sie Fahrzeugdaten ein, um Versicherungsangebote anzuzeigen.",
                ready: "Bewertung Verifiziert"
            },
            ja: {
                title: "エグゼクティブ車両査定＆テレメトリ",
                sub: "ランダムフォレスト回帰により市場価格を計算し、最適な保険プランを評価します。",
                spec: "車両仕様スペック",
                out: "インテリジェンス出力",
                valsub: "推定市場価格",
                btn: "AI査定を実行",
                btnProc: "モデル処理中...",
                eff: "燃費効率",
                risk: "リスク評価",
                dep: "減価償却",
                plans: "プレミアム保険プラン",
                prompt: "車両データを入力してAI査定を実行すると、保険プランが表示されます。",
                ready: "査定完了"
            }
        };

        function updateLanguage() {
            const lang = document.getElementById('langSelect').value;
            const dict = i18n[lang] || i18n.en;

            document.getElementById('txtTitle').innerText = dict.title;
            document.getElementById('txtSub').innerText = dict.sub;
            document.getElementById('txtSpec').innerText = dict.spec;
            document.getElementById('txtOut').innerText = dict.out;
            document.getElementById('txtValSub').innerText = dict.valsub;
            document.getElementById('txtBtn').innerText = dict.btn;
            document.getElementById('lblEff').innerText = dict.eff;
            document.getElementById('lblRisk').innerText = dict.risk;
            document.getElementById('lblDep').innerText = dict.dep;
            document.getElementById('txtPlans').innerText = dict.plans;
            
            const promptElem = document.getElementById('txtPrompt');
            if(promptElem) promptElem.innerText = dict.prompt;
        }

        function switchTheme(themeName, elem) {
            document.body.className = themeName;
            document.querySelectorAll('.theme-dot').forEach(dot => dot.classList.remove('active'));
            elem.classList.add('active');
        }

        document.getElementById('valuationForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const lang = document.getElementById('langSelect').value;
            const dict = i18n[lang] || i18n.en;
            
            const btn = document.getElementById('calcBtn');
            btn.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i> ${dict.btnProc}`;

            const formData = new FormData(this);
            const payload = {};
            formData.forEach((value, key) => { payload[key] = value; });

            // Attach current currency selection to API payload
            payload['currency'] = document.getElementById('currSelect').value;

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                const res = await response.json();

                if (res.success) {
                    document.getElementById('valPrice').innerText = res.formatted_price;
                    document.getElementById('valStatus').innerText = dict.ready;

                    document.getElementById('mEfficiency').innerText = res.metrics.efficiency;
                    document.getElementById('mRisk').innerText = res.metrics.risk;
                    document.getElementById('mDep').innerText = res.metrics.depreciation;

                    const container = document.getElementById('insuranceContainer');
                    container.innerHTML = '';

                    res.recommendations.forEach(plan => {
                        container.innerHTML += `
                            <div class="plan-card" onclick="alert('Selected Plan: ${plan.title}')">
                                <div>
                                    <div class="plan-header">
                                        <h4>${plan.title}</h4>
                                        <span class="badge-tier">${plan.badge}</span>
                                    </div>
                                    <div class="plan-meta">${plan.match} • Estimated Premium: <strong>${plan.price_est}</strong></div>
                                    <div class="plan-features">
                                        ${plan.features.map(f => `<span class="f-tag">${f}</span>`).join('')}
                                    </div>
                                </div>
                                <i class="fa-solid fa-chevron-right" style="color: var(--text-sub);"></i>
                            </div>
                        `;
                    });
                } else {
                    alert('Error: ' + res.error);
                }
            } catch (err) {
                alert('Connection error with Flask server.');
            } finally {
                btn.innerHTML = `<i class="fa-solid fa-microchip"></i> <span id="txtBtn">${dict.btn}</span>`;
            }
        });
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(
        HTML_TEMPLATE,
        categorical_options=CATEGORICAL_OPTIONS,
        default_numericals=DEFAULT_NUMERICALS
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        currency_code = data.get("currency", "USD")
        
        # Build DataFrame with numeric floats for model compatibility
        input_data = {}
        for col in FEATURE_ORDER:
            if col in CATEGORICAL_OPTIONS:
                input_data[col] = [float(data.get(col, 0))]
            else:
                input_data[col] = [float(data.get(col, DEFAULT_NUMERICALS.get(col, 0)))]

        df_input = pd.DataFrame(input_data)

        if model is not None:
            prediction = model.predict(df_input)[0]
            predicted_usd = float(prediction)
        else:
            predicted_usd = float(
                df_input["Engine_Size"].iloc[0] * 5200 + 
                df_input["Horsepower"].iloc[0] * 130 - 
                df_input["Mileage"].iloc[0] * 0.04 + 
                (2026 - df_input["Year"].iloc[0]) * -900 + 22000
            )

        # Currency Conversion
        curr = CURRENCY_RATES.get(currency_code, CURRENCY_RATES["USD"])
        converted_val = predicted_usd * curr["rate"]
        formatted_price = f"{curr['symbol']}{converted_val:,.2f}"

        # Category Name Lookup for Insurance & Telemetry Logic
        make_code = str(data.get("Make", "0"))
        make_label = CAT_LOOKUP["Make"].get(make_code, "Toyota")

        accident_code = str(data.get("Accident_History", "0"))
        accident_label = CAT_LOOKUP["Accident_History"].get(accident_code, "None")

        mileage = float(data.get("Mileage", 32000))
        
        risk_level = "Low" if accident_label == "None" else ("Moderate" if accident_label == "Minor" else "High")
        dep_risk = "Low" if mileage < 30000 else ("Moderate" if mileage < 75000 else "High")
        efficiency_tier = "A+" if float(data.get("Fuel_Efficiency", 16.8)) > 17.0 else "Standard"

        recommendations = generate_insurance_recommendations(predicted_usd, make_label, currency_code)

        return jsonify({
            "success": True,
            "predicted_usd": round(predicted_usd, 2),
            "formatted_price": formatted_price,
            "metrics": {
                "efficiency": efficiency_tier,
                "risk": risk_level,
                "depreciation": dep_risk
            },
            "recommendations": recommendations
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
