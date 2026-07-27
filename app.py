import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Configured to load your exact pickle file filename
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
    print(f"Warning: {MODEL_PATH} not found in the root directory.")

# Feature mapping structured natively with absolute human-readable strings
CATEGORICAL_OPTIONS = {
    "Make": ["Toyota", "Honda", "Ford", "BMW", "Mercedes-Benz", "Audi", "Hyundai", "Nissan", "Chevrolet", "Kia"],
    "Model": ["Sedan", "SUV", "Hatchback", "Coupe", "Truck", "Convertible", "Van", "Wagon"],
    "Fuel_Type": ["Petrol", "Diesel", "Hybrid", "Electric", "CNG"],
    "Transmission": ["Automatic", "Manual", "Semi-Automatic"],
    "Accident_History": ["None", "Minor", "Major"],
    "Service_History": ["Full", "Partial", "None"],
    "Color": ["Black", "White", "Silver", "Blue", "Red", "Grey"],
    "Body_Type": ["Sedan", "SUV", "Hatchback", "Coupe", "Pickup"],
    "Drivetrain": ["FWD", "RWD", "AWD", "4WD"],
    "Location": ["Urban", "Suburban", "Rural"]
}

DEFAULT_NUMERICALS = {
    "Year": 2020,
    "Engine_Size": 2.0,
    "Mileage": 45000.0,
    "Horsepower": 180.0,
    "Torque": 250.0,
    "Owners": 1,
    "Fuel_Efficiency": 15.5
}

FEATURE_ORDER = [
    "Make", "Model", "Year", "Fuel_Type", "Transmission", 
    "Engine_Size", "Mileage", "Horsepower", "Torque", "Owners", 
    "Accident_History", "Service_History", "Color", "Body_Type", 
    "Drivetrain", "Fuel_Efficiency", "Location"
]

def generate_insurance_recommendations(price, vehicle_data):
    """Generates a dynamic tier structure for premium auto cover match matching."""
    recommendations = []
    
    # Premium Luxury Tier
    if price > 30000 or vehicle_data.get("Make") in ["BMW", "Mercedes-Benz", "Audi"]:
        recommendations.append({
            "title": "Netflix Red Shield — Elite Cover",
            "tier": "PREMIUM",
            "badge": "Top Recommended",
            "price_est": f"${int(price * 0.045)}/year",
            "features": ["Zero Depreciation", "24/7 Roadside Assistance", "Engine & Gearbox Protect", "Global Return-to-Invoice"],
            "match": "98% Match for your vehicle class"
        })
    
    # Comprehensive Standard Tier
    recommendations.append({
        "title": "DriveGuard Comprehensive Plus",
        "tier": "POPULAR",
        "badge": "Best Value",
        "price_est": f"${int(price * 0.032)}/year",
        "features": ["Third-Party Liability", "Accident Damage Cover", "Personal Accident Protection", "Cashless Garage Network"],
        "match": "92% Match"
    })
    
    # Essential / Smart Saver Tier
    recommendations.append({
        "title": "SmartSaver Essential Auto",
        "tier": "ESSENTIAL",
        "badge": "Budget Friendly",
        "price_est": f"${int(price * 0.021)}/year",
        "features": ["Basic Liability", "Fire & Theft Cover", "Emergency Towing Support"],
        "match": "85% Match"
    })
    
    return recommendations

# Modern single-page web framework setup utilizing embedded styling layers
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AutoPredict Studio — Premium Vehicle Valuation</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    
    <style>
        :root {
            --bg-color: #141414;
            --card-bg: #1f1f1f;
            --card-border: #2b2b2b;
            --primary-accent: #E50914;
            --primary-hover: #f40612;
            --text-main: #FFFFFF;
            --text-secondary: #AAAAAA;
            --input-bg: #2b2b2b;
            --glow: rgba(229, 9, 20, 0.4);
        }

        /* UI Theme Profiles */
        body.theme-netflix {
            --bg-color: #141414;
            --card-bg: #181818;
            --primary-accent: #E50914;
            --primary-hover: #ff0f1a;
            --glow: rgba(229, 9, 20, 0.5);
        }

        body.theme-cyberpunk {
            --bg-color: #0d0e15;
            --card-bg: #151828;
            --card-border: #262b48;
            --primary-accent: #00F0FF;
            --primary-hover: #70F3FF;
            --text-main: #FFFFFF;
            --input-bg: #1c2038;
            --glow: rgba(0, 240, 255, 0.5);
        }

        body.theme-gold {
            --bg-color: #0d0d0d;
            --card-bg: #171717;
            --card-border: #2a2a2a;
            --primary-accent: #D4AF37;
            --primary-hover: #f3cf55;
            --input-bg: #222222;
            --glow: rgba(212, 175, 55, 0.4);
        }

        body.theme-emerald {
            --bg-color: #06120e;
            --card-bg: #0d211a;
            --card-border: #18382d;
            --primary-accent: #10B981;
            --primary-hover: #34D399;
            --input-bg: #122c23;
            --glow: rgba(16, 185, 129, 0.4);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
            transition: background-color 0.3s ease, border-color 0.3s ease, color 0.3s ease, box-shadow 0.3s ease;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            min-height: 100vh;
            overflow-x: hidden;
        }

        .navbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px 50px;
            background: linear-gradient(180deg, rgba(0,0,0,0.8) 0%, rgba(0,0,0,0) 100%);
            position: fixed;
            top: 0;
            width: 100%;
            z-index: 1000;
        }

        .brand-logo {
            font-size: 1.8rem;
            font-weight: 800;
            color: var(--primary-accent);
            text-decoration: none;
            letter-spacing: 1px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .theme-selector {
            display: flex;
            gap: 12px;
            background: rgba(0,0,0,0.5);
            padding: 8px 16px;
            border-radius: 30px;
            border: 1px solid var(--card-border);
            backdrop-filter: blur(10px);
        }

        .theme-btn {
            width: 22px;
            height: 22px;
            border-radius: 50%;
            cursor: pointer;
            border: 2px solid transparent;
            transition: transform 0.2s;
        }

        .theme-btn:hover { transform: scale(1.2); }
        .theme-btn.active { border-color: #FFFFFF; }

        .btn-netflix-red { background-color: #E50914; }
        .btn-cyberpunk { background-color: #00F0FF; }
        .btn-gold { background-color: #D4AF37; }
        .btn-emerald { background-color: #10B981; }

        .hero-container {
            margin-top: 100px;
            padding: 0 50px 50px;
            max-width: 1600px;
            margin-left: auto;
            margin-right: auto;
        }

        .hero-banner {
            margin-bottom: 30px;
            animation: fadeIn 1s ease-in-out;
        }

        .hero-banner h1 {
            font-size: 2.8rem;
            font-weight: 800;
            margin-bottom: 8px;
        }

        .hero-banner p {
            color: var(--text-secondary);
            font-size: 1.1rem;
        }

        .main-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }

        @media (max-width: 1024px) {
            .main-grid { grid-template-columns: 1fr; }
            .hero-container { padding: 0 20px 30px; }
            .navbar { padding: 15px 20px; }
        }

        .glass-card {
            background-color: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        }

        .card-header {
            font-size: 1.3rem;
            font-weight: 700;
            margin-bottom: 25px;
            display: flex;
            align-items: center;
            gap: 10px;
            color: var(--text-main);
            border-bottom: 1px solid var(--card-border);
            padding-bottom: 12px;
        }

        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 18px;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .form-group label {
            font-size: 0.85rem;
            color: var(--text-secondary);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .form-control {
            background-color: var(--input-bg);
            border: 1fr solid var(--card-border);
            color: var(--text-main);
            padding: 12px 14px;
            border-radius: 6px;
            font-size: 0.95rem;
            outline: none;
            border: 1px solid transparent;
        }

        .form-control:focus {
            border-color: var(--primary-accent);
            box-shadow: 0 0 10px var(--glow);
        }

        /* Shimmering Button Actions */
        .premium-btn {
            width: 100%;
            background: var(--primary-accent);
            color: #FFFFFF;
            border: none;
            padding: 16px;
            font-size: 1.1rem;
            font-weight: 700;
            border-radius: 8px;
            cursor: pointer;
            position: relative;
            overflow: hidden;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
            box-shadow: 0 4px 15px var(--glow);
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        }

        .premium-btn::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
            transition: left 0.6s ease;
        }

        .premium-btn:hover::before { left: 100%; }

        .premium-btn:hover {
            background: var(--primary-hover);
            transform: translateY(-2px);
            box-shadow: 0 8px 25px var(--glow);
        }

        .valuation-box {
            text-align: center;
            padding: 25px;
            background: rgba(0,0,0,0.3);
            border-radius: 10px;
            border: 1px dashed var(--primary-accent);
            margin-bottom: 25px;
        }

        .valuation-title {
            font-size: 0.9rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .valuation-amount {
            font-size: 3rem;
            font-weight: 800;
            color: var(--primary-accent);
            margin: 10px 0;
            text-shadow: 0 0 20px var(--glow);
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-bottom: 25px;
        }

        .metric-card {
            background-color: rgba(255,255,255,0.03);
            padding: 15px;
            border-radius: 8px;
            border: 1px solid var(--card-border);
            text-align: center;
        }

        .metric-value { font-size: 1.2rem; font-weight: 700; margin-top: 5px; }
        .metric-label { font-size: 0.75rem; color: var(--text-secondary); }

        .section-title {
            font-size: 1.2rem;
            font-weight: 700;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .recommendation-list { display: flex; flex-direction: column; gap: 15px; }

        .rec-card {
            background: var(--input-bg);
            border: 1px solid var(--card-border);
            border-radius: 8px;
            padding: 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
        }

        .rec-card:hover {
            transform: scale(1.02);
            border-color: var(--primary-accent);
        }

        .rec-info h4 { font-size: 1rem; margin-bottom: 4px; }
        .rec-info p { font-size: 0.8rem; color: var(--text-secondary); }

        .rec-features { display: flex; gap: 8px; margin-top: 8px; flex-wrap: wrap; }

        .tag {
            background: rgba(255,255,255,0.08);
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.7rem;
            color: var(--text-main);
        }

        .rec-badge {
            background: var(--primary-accent);
            color: #fff;
            padding: 4px 10px;
            font-size: 0.75rem;
            font-weight: 700;
            border-radius: 20px;
            text-transform: uppercase;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .pulse { animation: pulse-animation 2s infinite; }

        @keyframes pulse-animation {
            0% { box-shadow: 0 0 0 0px var(--glow); }
            70% { box-shadow: 0 0 0 15px rgba(0, 0, 0, 0); }
            100% { box-shadow: 0 0 0 0px rgba(0, 0, 0, 0); }
        }
    </style>
</head>
<body class="theme-netflix">

    <nav class="navbar">
        <a href="#" class="brand-logo"><i class="fa-solid fa-play"></i> AUTOPREDICT STUDIO</a>
        <div class="theme-selector">
            <div class="theme-btn btn-netflix-red active" onclick="setTheme('theme-netflix', this)"></div>
            <div class="theme-btn btn-cyberpunk" onclick="setTheme('theme-cyberpunk', this)"></div>
            <div class="theme-btn btn-gold" onclick="setTheme('theme-gold', this)"></div>
            <div class="theme-btn btn-emerald" onclick="setTheme('theme-emerald', this)"></div>
        </div>
    </nav>

    <div class="hero-container">
        <div class="hero-banner">
            <h1>Vehicle Valuation & Insurance Analytics</h1>
            <p>Enter parameters to compute real-time Random Forest predictions and receive tailored insurance plans.</p>
        </div>

        <div class="main-grid">
            <div class="glass-card">
                <div class="card-header"><i class="fa-solid fa-sliders"></i> Vehicle Specifications</div>
                <form id="predictionForm">
                    <div class="form-grid">
                        {% for col in categorical_keys %}
                        <div class="form-group">
                            <label for="{{ col }}">{{ col.replace('_', ' ') }}</label>
                            <select id="{{ col }}" name="{{ col }}" class="form-control">
                                {% for val in categorical_options[col] %}
                                <option value="{{ val }}">{{ val }}</option>
                                {% endfor %}
                            </select>
                        </div>
                        {% endfor %}

                        {% for col, val in default_numericals.items() %}
                        <div class="form-group">
                            <label for="{{ col }}">{{ col.replace('_', ' ') }}</label>
                            <input type="number" step="any" id="{{ col }}" name="{{ col }}" value="{{ val }}" class="form-control" required>
                        </div>
                        {% endfor %}
                    </div>
                    <div style="margin-top: 25px;">
                        <button type="submit" class="premium-btn pulse" id="submitBtn">
                            <i class="fa-solid fa-bolt"></i> Generate Valuation & Insights
                        </button>
                    </div>
                </form>
            </div>

            <div class="glass-card">
                <div class="card-header"><i class="fa-solid fa-chart-line"></i> Valuation Dashboard</div>
                <div class="valuation-box">
                    <div class="valuation-title">Predicted Market Price</div>
                    <div class="valuation-amount" id="predictedPrice">$0.00</div>
                    <span class="tag" id="statusBadge">Model Ready</span>
                </div>

                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-label">Efficiency Index</div>
                        <div class="metric-value" id="metricEfficiency">--</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Risk Rating</div>
                        <div class="metric-value" id="metricRisk">--</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Depreciation Risk</div>
                        <div class="metric-value" id="metricDep">--</div>
                    </div>
                </div>

                <div class="section-title"><i class="fa-solid fa-shield-halved"></i> Recommended Insurance Plans</div>
                <div class="recommendation-list" id="recommendationsList">
                    <p style="color: var(--text-secondary); text-align: center; padding: 20px;">
                        Complete vehicle parameters to generate tailored coverage options.
                    </p>
                </div>
            </div>
        </div>
    </div>

    <script>
        function setTheme(themeName, element) {
            document.body.className = themeName;
            document.querySelectorAll('.theme-btn').forEach(btn => btn.classList.remove('active'));
            element.classList.add('active');
        }

        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            const submitBtn = document.getElementById('submitBtn');
            submitBtn.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Analyzing...';
            
            const formData = new FormData(this);
            const data = {};
            formData.forEach((value, key) => { data[key] = value; });

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                const result = await response.json();

                if (result.success) {
                    document.getElementById('predictedPrice').innerText = '$' + result.predicted_price.toLocaleString(undefined, {minimumFractionDigits: 2});
                    document.getElementById('statusBadge').innerText = 'Valuation Complete';
                    document.getElementById('metricEfficiency').innerText = result.metrics.efficiency;
                    document.getElementById('metricRisk').innerText = result.metrics.risk;
                    document.getElementById('metricDep').innerText = result.metrics.depreciation;

                    const recContainer = document.getElementById('recommendationsList');
                    recContainer.innerHTML = '';
                    result.recommendations.forEach(plan => {
                        recContainer.innerHTML += `
                            <div class="rec-card" onclick="alert('Connecting to ${plan.title} security core...')">
                                <div class="rec-info">
                                    <div style="display: flex; align-items: center; gap: 10px;">
                                        <h4>${plan.title}</h4>
                                        <span class="rec-badge">${plan.badge}</span>
                                    </div>
                                    <p>${plan.match} • Estimated: <strong>${plan.price_est}</strong></p>
                                    <div class="rec-features">${plan.features.map(f => `<span class="tag">${f}</span>`).join('')}</div>
                                </div>
                                <div><i class="fa-solid fa-chevron-right" style="color: var(--text-secondary);"></i></div>
                            </div>`;
                    });
                }
            } catch (err) {
                alert('Connection failure to API infrastructure.');
            } finally {
                submitBtn.innerHTML = '<i class="fa-solid fa-bolt"></i> Generate Valuation & Insights';
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
        categorical_keys=list(CATEGORICAL_OPTIONS.keys()),
        categorical_options=CATEGORICAL_OPTIONS,
        default_numericals=DEFAULT_NUMERICALS
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        input_data = {}
        for col in FEATURE_ORDER:
            if col in CATEGORICAL_OPTIONS:
                input_data[col] = [str(data.get(col, CATEGORICAL_OPTIONS[col][0]))]
            else:
                input_data[col] = [float(data.get(col, DEFAULT_NUMERICALS.get(col, 0)))]

        df_input = pd.DataFrame(input_data)

        if model is not None:
            predicted_value = float(model.predict(df_input)[0])
        else:
            predicted_value = float(
                df_input["Engine_Size"].iloc[0] * 5000 + 
                df_input["Horsepower"].iloc[0] * 120 - 
                df_input["Mileage"].iloc[0] * 0.05 + 22000
            )

        mileage = float(data.get("Mileage", 45000))
        accident = data.get("Accident_History", "None")
        
        return jsonify({
            "success": True,
            "predicted_price": round(predicted_value, 2),
            "metrics": {
                "efficiency": "A+" if float(data.get("Fuel_Efficiency", 15)) > 18 else "Standard",
                "risk": "Low" if accident == "None" else ("Medium" if accident == "Minor" else "High"),
                "depreciation": "Low" if mileage < 30000 else ("Moderate" if mileage < 80000 else "High")
            },
            "recommendations": generate_insurance_recommendations(predicted_value, data)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
