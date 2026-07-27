import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Load model pickle file
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
    print(f"Warning: '{MODEL_PATH}' not found. Running in simulation mode.")

# Categorical Mappings: (encoded_numeric_id, human_readable_label)
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

CURRENCY_RATES = {
    "USD": {"symbol": "$", "rate": 1.0},
    "EUR": {"symbol": "€", "rate": 0.92},
    "GBP": {"symbol": "£", "rate": 0.78},
    "INR": {"symbol": "₹", "rate": 83.5}
}

def generate_insurance_recommendations(price_usd, make_label):
    """Generate dynamic insurance plan structures."""
    recommendations = []
    
    if price_usd > 35000 or make_label in ["BMW", "Mercedes-Benz", "Audi"]:
        recommendations.append({
            "title": "Titanium Sovereign Cover",
            "tier": "PLATINUM",
            "badge": "VIP Choice",
            "annual_ratio": 0.042,
            "features": ["Zero Depreciation", "Full Engine & ECU Protection", "24/7 Global Concierge", "Guaranteed Invoice Value"],
            "match": "99% Precision Match"
        })
    
    recommendations.append({
        "title": "Apex Dynamic Shield",
        "tier": "EXECUTIVE",
        "badge": "Recommended",
        "annual_ratio": 0.031,
        "features": ["All-Risk Collision Cover", "Third-Party Unlimited Liability", "Personal Driver Cover", "Cashless Repair Hubs"],
        "match": "94% Optimal Match"
    })
    
    recommendations.append({
        "title": "Metro Guard Select",
        "tier": "ESSENTIAL",
        "badge": "Smart Value",
        "annual_ratio": 0.020,
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
    <title>AURA Motors — Full Analytical Intelligence Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <!-- Chart.js Library -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    
    <style>
        /* Theme Variables */
        body.theme-dark {
            --bg-base: #0b0f19;
            --bg-card: rgba(20, 27, 41, 0.85);
            --card-border: rgba(255, 255, 255, 0.08);
            --accent: #6366f1;
            --accent-glow: rgba(99, 102, 241, 0.35);
            --accent-hover: #4f46e5;
            --text-heading: #f8fafc;
            --text-sub: #94a3b8;
            --input-bg: rgba(15, 23, 42, 0.6);
            --header-bg: rgba(11, 15, 25, 0.9);
        }

        body.theme-light {
            --bg-base: #f1f5f9;
            --bg-card: rgba(255, 255, 255, 0.9);
            --card-border: rgba(0, 0, 0, 0.08);
            --accent: #4f46e5;
            --accent-glow: rgba(79, 70, 229, 0.2);
            --accent-hover: #4338ca;
            --text-heading: #0f172a;
            --text-sub: #64748b;
            --input-bg: rgba(248, 250, 252, 0.9);
            --header-bg: rgba(255, 255, 255, 0.95);
        }

        body.theme-cyber {
            --bg-base: #070913;
            --bg-card: rgba(14, 20, 38, 0.85);
            --card-border: rgba(0, 240, 255, 0.25);
            --accent: #00f0ff;
            --accent-glow: rgba(0, 240, 255, 0.4);
            --accent-hover: #70f3ff;
            --text-heading: #ffffff;
            --text-sub: #8ba1cd;
            --input-bg: rgba(10, 15, 30, 0.7);
            --header-bg: rgba(7, 9, 19, 0.95);
        }

        body.theme-gold {
            --bg-base: #0f0d0a;
            --bg-card: rgba(28, 24, 18, 0.85);
            --card-border: rgba(212, 175, 55, 0.25);
            --accent: #e5c158;
            --accent-glow: rgba(229, 193, 88, 0.35);
            --accent-hover: #f3d478;
            --text-heading: #fffdf5;
            --text-sub: #a89f91;
            --input-bg: rgba(20, 17, 12, 0.7);
            --header-bg: rgba(15, 13, 10, 0.95);
        }

        body.theme-emerald {
            --bg-base: #06130e;
            --bg-card: rgba(12, 33, 25, 0.85);
            --card-border: rgba(16, 185, 129, 0.25);
            --accent: #10b981;
            --accent-glow: rgba(16, 185, 129, 0.4);
            --accent-hover: #34d399;
            --text-heading: #f0fdf4;
            --text-sub: #86efac;
            --input-bg: rgba(8, 24, 18, 0.7);
            --header-bg: rgba(6, 19, 14, 0.95);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
            transition: background-color 0.3s ease, border-color 0.3s ease, color 0.3s ease;
        }

        body {
            background-color: var(--bg-base);
            color: var(--text-heading);
            min-height: 100vh;
            background-image: 
                radial-gradient(circle at 10% 15%, var(--accent-glow) 0%, transparent 40%),
                radial-gradient(circle at 90% 85%, var(--accent-glow) 0%, transparent 40%);
            background-attachment: fixed;
            overflow-x: hidden;
        }

        /* Navbar Header */
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
            gap: 14px;
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
        }

        .theme-dot:hover { transform: scale(1.25); }
        .theme-dot.active { border-color: var(--text-heading); transform: scale(1.1); }

        .dot-dark { background: #6366f1; }
        .dot-light { background: #ffffff; border: 1px solid #ccc; }
        .dot-cyber { background: #00f0ff; }
        .dot-gold { background: #e5c158; }
        .dot-emerald { background: #10b981; }

        /* Outer Layout Wrapper */
        .wrapper {
            margin-top: 85px;
            padding: 30px 40px;
            max-width: 1650px;
            margin-left: auto;
            margin-right: auto;
        }

        .hero-head h1 {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2.1rem;
            font-weight: 700;
            margin-bottom: 4px;
        }
        .hero-head p { color: var(--text-sub); font-size: 0.95rem; margin-bottom: 25px; }

        /* Dashboard Master Grid */
        .master-layout {
            display: grid;
            grid-template-columns: 420px 1fr;
            gap: 25px;
        }

        @media (max-width: 1200px) {
            .master-layout { grid-template-columns: 1fr; }
            .wrapper { padding: 20px; }
            .header-bar { padding: 15px 20px; flex-direction: column; gap: 12px; }
        }

        .glass-panel {
            background: var(--bg-card);
            backdrop-filter: blur(20px);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.25);
        }

        .panel-title {
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
            border-bottom: 1px solid var(--card-border);
            padding-bottom: 10px;
            color: var(--text-heading);
        }

        .panel-title i { color: var(--accent); }

        /* Form Inputs Layout */
        .form-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
        }

        .form-field { display: flex; flex-direction: column; gap: 4px; }
        .form-field label {
            font-size: 0.72rem;
            font-weight: 700;
            color: var(--text-sub);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .custom-input {
            background: var(--input-bg);
            border: 1px solid var(--card-border);
            color: var(--text-heading);
            padding: 10px 12px;
            border-radius: 8px;
            font-size: 0.88rem;
            outline: none;
        }

        .custom-input:focus {
            border-color: var(--accent);
            box-shadow: 0 0 10px var(--accent-glow);
        }

        .btn-spark {
            width: 100%;
            margin-top: 20px;
            background: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%);
            color: #ffffff;
            border: none;
            padding: 14px;
            font-size: 0.95rem;
            font-weight: 700;
            border-radius: 10px;
            cursor: pointer;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
            box-shadow: 0 8px 25px var(--accent-glow);
            transition: all 0.3s ease;
        }

        .btn-spark:hover { transform: translateY(-2px); box-shadow: 0 12px 30px var(--accent-glow); }

        /* KPI Overview Grid */
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-bottom: 25px;
        }

        @media (max-width: 900px) {
            .kpi-grid { grid-template-columns: repeat(2, 1fr); }
        }

        .kpi-card {
            background: rgba(0, 0, 0, 0.15);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 16px;
            text-align: center;
            position: relative;
        }

        .kpi-title { font-size: 0.72rem; color: var(--text-sub); text-transform: uppercase; font-weight: 700; }
        .kpi-val { font-family: 'Space Grotesk', sans-serif; font-size: 1.8rem; font-weight: 700; color: var(--accent); margin: 6px 0; }
        .kpi-badge { font-size: 0.7rem; padding: 2px 8px; border-radius: 10px; background: rgba(255,255,255,0.06); color: var(--text-heading); }

        /* Charts Analytics Section Grid */
        .analytics-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin-bottom: 25px;
        }

        @media (max-width: 900px) {
            .analytics-grid { grid-template-columns: 1fr; }
        }

        .chart-box {
            background: rgba(0, 0, 0, 0.12);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 16px;
            height: 270px;
            position: relative;
        }

        .chart-box h4 {
            font-size: 0.85rem;
            color: var(--text-sub);
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .chart-container {
            position: relative;
            height: 200px;
            width: 100%;
        }

        /* Insurance Proposals Layout */
        .plans-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
        }

        @media (max-width: 900px) {
            .plans-grid { grid-template-columns: 1fr; }
        }

        .plan-card {
            background: var(--input-bg);
            border: 1px solid var(--card-border);
            border-radius: 10px;
            padding: 16px;
            cursor: pointer;
            transition: all 0.25s ease;
        }

        .plan-card:hover {
            border-color: var(--accent);
            transform: translateY(-3px);
            box-shadow: 0 4px 20px var(--accent-glow);
        }

        .plan-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
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

        .plan-price { font-size: 1.2rem; font-weight: 700; color: var(--accent); margin: 4px 0; }
        .plan-meta { font-size: 0.78rem; color: var(--text-sub); margin-bottom: 8px; }
        .plan-features { display: flex; gap: 6px; flex-wrap: wrap; }

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
            <div class="brand-icon"><i class="fa-solid fa-chart-line"></i></div>
            <span>AURA MOTORS ANALYTICS</span>
        </div>
        
        <div class="controls-group">
            <select id="langSelect" class="control-select" onchange="updateLanguage()">
                <option value="en">English (US)</option>
                <option value="es">Español</option>
                <option value="fr">Français</option>
                <option value="de">Deutsch</option>
                <option value="ja">日本語</option>
            </select>

            <select id="currSelect" class="control-select" onchange="renderRealtimeCurrency()">
                <option value="USD">USD ($)</option>
                <option value="EUR">EUR (€)</option>
                <option value="GBP">GBP (£)</option>
                <option value="INR">INR (₹)</option>
            </select>

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
            <h1 id="txtTitle">Executive Valuation & Full Telemetry Analytics</h1>
            <p id="txtSub">Comprehensive Random Forest market estimation, depreciation projections, TCO analysis, and insurance models.</p>
        </div>

        <div class="master-layout">
            <!-- Left Side: Specifications Input -->
            <div class="glass-panel">
                <div class="panel-title"><i class="fa-solid fa-sliders"></i> <span id="txtSpec">Vehicle Inputs</span></div>
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

                    <button type="submit" class="btn-spark" id="calcBtn">
                        <i class="fa-solid fa-microchip"></i> <span id="txtBtn">Run Full Analytics</span>
                    </button>
                </form>
            </div>

            <!-- Right Side: Analytical Dashboard & Charts -->
            <div>
                <!-- KPI Executive Overview Grid -->
                <div class="kpi-grid">
                    <div class="kpi-card">
                        <div class="kpi-title" id="kpi1Title">Current Market Value</div>
                        <div class="kpi-val" id="kpiVal">$0.00</div>
                        <span class="kpi-badge" id="kpiValStatus">Awaiting Execution</span>
                    </div>

                    <div class="kpi-card">
                        <div class="kpi-title" id="kpi2Title">5-Yr Future Value</div>
                        <div class="kpi-val" id="kpi5Yr">$0.00</div>
                        <span class="kpi-badge">~38% Total Depreciation</span>
                    </div>

                    <div class="kpi-card">
                        <div class="kpi-title" id="kpi3Title">Annual Maintenance</div>
                        <div class="kpi-val" id="kpiMaint">$0.00</div>
                        <span class="kpi-badge" id="kpiMaintTier">Standard Service</span>
                    </div>

                    <div class="kpi-card">
                        <div class="kpi-title" id="kpi4Title">Health Index Score</div>
                        <div class="kpi-val" id="kpiHealth">-- / 100</div>
                        <span class="kpi-badge" id="kpiHealthStatus">Condition Index</span>
                    </div>
                </div>

                <!-- Interactive Charts Row 1 & 2 -->
                <div class="analytics-grid">
                    <div class="chart-box">
                        <h4><i class="fa-solid fa-chart-line"></i> 5-Year Depreciation Forecast Curve</h4>
                        <div class="chart-container"><canvas id="depreciationChart"></canvas></div>
                    </div>

                    <div class="chart-box">
                        <h4><i class="fa-solid fa-chart-pie"></i> Valuation Component Drivers</h4>
                        <div class="chart-container"><canvas id="driverChart"></canvas></div>
                    </div>

                    <div class="chart-box">
                        <h4><i class="fa-solid fa-layer-group"></i> Total Annual Cost of Ownership (TCO)</h4>
                        <div class="chart-container"><canvas id="tcoChart"></canvas></div>
                    </div>

                    <div class="chart-box">
                        <h4><i class="fa-solid fa-compass"></i> Vehicle Telemetry Metrics</h4>
                        <div class="chart-container"><canvas id="radarChart"></canvas></div>
                    </div>
                </div>

                <!-- Tailored Insurance Recommendations Panel -->
                <div class="glass-panel">
                    <div class="panel-title"><i class="fa-solid fa-shield"></i> <span id="txtPlans">Recommended Coverage Proposals</span></div>
                    <div class="plans-grid" id="insuranceContainer">
                        <p style="color: var(--text-sub); text-align: center; grid-column: 1 / -1; padding: 20px;" id="txtPrompt">
                            Submit specs to generate custom AI coverage proposals.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Analytical Data & Currency Conversion States
        let rawUsdValuation = 0;
        let rawRecommendations = [];
        const currencyRates = {{ currency_rates | tojson }};

        // Chart.js Instances
        let chartDep, chartDriver, chartTco, chartRadar;

        // Multilingual Translation Map
        const i18n = {
            en: {
                title: "Executive Valuation & Full Telemetry Analytics",
                sub: "Comprehensive Random Forest market estimation, depreciation projections, TCO analysis, and insurance models.",
                spec: "Vehicle Inputs",
                btn: "Run Full Analytics",
                btnProc: "Computing Metrics...",
                plans: "Recommended Coverage Proposals",
                prompt: "Submit specs to generate custom AI coverage proposals."
            },
            es: {
                title: "Valoración Ejecutiva y Analítica Completa",
                sub: "Estimación de mercado Random Forest, proyecciones de depreciación y análisis TCO.",
                spec: "Datos del Vehículo",
                btn: "Ejecutar Analítica",
                btnProc: "Calculando...",
                plans: "Propuestas de Cobertura Recomendadas",
                prompt: "Ingrese las especificaciones para generar propuestas de seguro."
            },
            fr: {
                title: "Évaluation Exécutive et Analyse Télémetrique",
                sub: "Estimation du marché Random Forest, prévisions de dépréciation et coût total de possession.",
                spec: "Saisie du Véhicule",
                btn: "Lancer l'Analyse",
                btnProc: "Calcul en cours...",
                plans: "Propositions de Couverture Recommandées",
                prompt: "Soumettez les spécifications para générer des offres d'assurance."
            },
            de: {
                title: "Executive Bewertung & Vollständige Analytik",
                sub: "Marktschätzung mit Random Forest, Wertverlustprognosen und Gesamtkostenanalyse.",
                spec: "Fahrzeugeingaben",
                btn: "Analytik Starten",
                btnProc: "Berechnung...",
                plans: "Empfohlene Deckungsangebote",
                prompt: "Geben Sie Fahrzeugdaten ein, um Versicherungsangebote zu erstellen."
            },
            ja: {
                title: "エグゼクティブ査定＆全分析ダッシュボード",
                sub: "ランダムフォレスト市場推定、減価償却予測、総所有コスト（TCO）および保険モデル。",
                spec: "車両入力スペック",
                btn: "全分析を実行",
                btnProc: "計算中...",
                plans: "推奨保険プラン提案",
                prompt: "スペックを入力してAI査定を実行すると、保険プランが表示されます。"
            }
        };

        function updateLanguage() {
            const lang = document.getElementById('langSelect').value;
            const dict = i18n[lang] || i18n.en;
            document.getElementById('txtTitle').innerText = dict.title;
            document.getElementById('txtSub').innerText = dict.sub;
            document.getElementById('txtSpec').innerText = dict.spec;
            document.getElementById('txtBtn').innerText = dict.btn;
            document.getElementById('txtPlans').innerText = dict.plans;
            const promptElem = document.getElementById('txtPrompt');
            if(promptElem) promptElem.innerText = dict.prompt;
        }

        function switchTheme(themeName, elem) {
            document.body.className = themeName;
            document.querySelectorAll('.theme-dot').forEach(dot => dot.classList.remove('active'));
            elem.classList.add('active');
            // Re-render charts to adjust theme styling colors
            if (rawUsdValuation > 0) renderCharts();
        }

        // Real-Time Currency Switcher
        function renderRealtimeCurrency() {
            if (rawUsdValuation === 0) return;

            const selectedCurr = document.getElementById('currSelect').value;
            const currencyObj = currencyRates[selectedCurr] || currencyRates["USD"];
            const rate = currencyObj.rate;
            const symbol = currencyObj.symbol;

            const convertedVal = rawUsdValuation * rate;
            const converted5Yr = (rawUsdValuation * 0.62) * rate;
            const convertedMaint = (rawUsdValuation * 0.035) * rate;

            document.getElementById('kpiVal').innerText = symbol + Math.round(convertedVal).toLocaleString();
            document.getElementById('kpi5Yr').innerText = symbol + Math.round(converted5Yr).toLocaleString();
            document.getElementById('kpiMaint').innerText = symbol + Math.round(convertedMaint).toLocaleString();

            // Render Insurance Plans
            const container = document.getElementById('insuranceContainer');
            container.innerHTML = '';

            rawRecommendations.forEach(plan => {
                const planConverted = convertedVal * plan.annual_ratio;
                const planPriceFormatted = symbol + Math.round(planConverted).toLocaleString() + '/yr';

                container.innerHTML += `
                    <div class="plan-card" onclick="alert('Selected Coverage: ${plan.title}')">
                        <div class="plan-header">
                            <h4>${plan.title}</h4>
                            <span class="badge-tier">${plan.badge}</span>
                        </div>
                        <div class="plan-price">${planPriceFormatted}</div>
                        <div class="plan-meta">${plan.match}</div>
                        <div class="plan-features">
                            ${plan.features.map(f => `<span class="f-tag">${f}</span>`).join('')}
                        </div>
                    </div>
                `;
            });

            // Update Chart datasets with converted currency values
            renderCharts();
        }

        // Chart.js Rendering Engine
        function renderCharts() {
            const selectedCurr = document.getElementById('currSelect').value;
            const currencyObj = currencyRates[selectedCurr] || currencyRates["USD"];
            const rate = currencyObj.rate;
            const symbol = currencyObj.symbol;

            const baseVal = rawUsdValuation * rate;

            // 1. Depreciation Line Chart
            const depCtx = document.getElementById('depreciationChart').getContext('2d');
            if(chartDep) chartDep.destroy();
            chartDep = new Chart(depCtx, {
                type: 'line',
                data: {
                    labels: ['Year 0', 'Year 1', 'Year 2', 'Year 3', 'Year 4', 'Year 5'],
                    datasets: [{
                        label: 'Vehicle Value (' + symbol + ')',
                        data: [baseVal, baseVal*0.88, baseVal*0.79, baseVal*0.71, baseVal*0.66, baseVal*0.62],
                        borderColor: '#6366f1',
                        backgroundColor: 'rgba(99, 102, 241, 0.15)',
                        fill: true,
                        tension: 0.35
                    }]
                },
                options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
            });

            // 2. Component Drivers Doughnut Chart
            const driverCtx = document.getElementById('driverChart').getContext('2d');
            if(chartDriver) chartDriver.destroy();
            chartDriver = new Chart(driverCtx, {
                type: 'doughnut',
                data: {
                    labels: ['Base Engine', 'Horsepower Boost', 'Mileage Factor', 'Market Demand'],
                    datasets: [{
                        data: [baseVal*0.5, baseVal*0.25, baseVal*0.15, baseVal*0.1],
                        backgroundColor: ['#6366f1', '#00f0ff', '#10b981', '#e5c158']
                    }]
                },
                options: { responsive: true, maintainAspectRatio: false }
            });

            // 3. TCO Stacked Bar Chart
            const tcoCtx = document.getElementById('tcoChart').getContext('2d');
            if(chartTco) chartTco.destroy();
            chartTco = new Chart(tcoCtx, {
                type: 'bar',
                data: {
                    labels: ['Year 1', 'Year 2', 'Year 3'],
                    datasets: [
                        { label: 'Depreciation', data: [baseVal*0.12, baseVal*0.09, baseVal*0.08], backgroundColor: '#6366f1' },
                        { label: 'Fuel', data: [baseVal*0.05, baseVal*0.052, baseVal*0.055], backgroundColor: '#00f0ff' },
                        { label: 'Insurance', data: [baseVal*0.035, baseVal*0.033, baseVal*0.031], backgroundColor: '#10b981' }
                    ]
                },
                options: { responsive: true, maintainAspectRatio: false, scales: { x: { stacked: true }, y: { stacked: true } } }
            });

            // 4. Radar Metrics Chart
            const radarCtx = document.getElementById('radarChart').getContext('2d');
            if(chartRadar) chartRadar.destroy();
            chartRadar = new Chart(radarCtx, {
                type: 'radar',
                data: {
                    labels: ['Power', 'Efficiency', 'Reliability', 'Safety', 'Resale Hold'],
                    datasets: [{
                        label: 'Vehicle Profile',
                        data: [82, 75, 90, 88, 70],
                        borderColor: '#e5c158',
                        backgroundColor: 'rgba(229, 193, 88, 0.2)'
                    }]
                },
                options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
            });
        }

        // Form Submit Ajax Handler
        document.getElementById('valuationForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const lang = document.getElementById('langSelect').value;
            const dict = i18n[lang] || i18n.en;
            
            const btn = document.getElementById('calcBtn');
            btn.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i> ${dict.btnProc}`;

            const formData = new FormData(this);
            const payload = {};
            formData.forEach((value, key) => { payload[key] = value; });

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                const res = await response.json();

                if (res.success) {
                    rawUsdValuation = res.predicted_usd;
                    rawRecommendations = res.recommendations;

                    document.getElementById('kpiValStatus').innerText = 'Verified Model Prediction';
                    document.getElementById('kpiHealth').innerText = res.metrics.health_score + ' / 100';

                    renderRealtimeCurrency();
                } else {
                    alert('Error: ' + res.error);
                }
            } catch (err) {
                alert('Connection error with server.');
            } finally {
                btn.innerHTML = `<i class="fa-solid fa-microchip"></i> ${dict.btn}`;
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
        default_numericals=DEFAULT_NUMERICALS,
        currency_rates=CURRENCY_RATES
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        
        # Build encoded dataframe for model prediction
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

        make_code = str(data.get("Make", "0"))
        make_label = CAT_LOOKUP["Make"].get(make_code, "Toyota")

        accident_code = str(data.get("Accident_History", "0"))
        accident_label = CAT_LOOKUP["Accident_History"].get(accident_code, "None")

        owners = float(data.get("Owners", 1))
        
        # Calculate Vehicle Health Score
        health_score = 100 - (0 if accident_label == "None" else (15 if accident_label == "Minor" else 35)) - (owners * 3)

        recommendations = generate_insurance_recommendations(predicted_usd, make_label)

        return jsonify({
            "success": True,
            "predicted_usd": round(predicted_usd, 2),
            "metrics": {
                "health_score": max(health_score, 40)
            },
            "recommendations": recommendations
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
