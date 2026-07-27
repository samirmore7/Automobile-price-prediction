import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Model loading setup using the exact pickle filename
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
    print(f"Warning: '{MODEL_PATH}' not found in the root directory. Operating in preview mode.")

# Feature mapping with categorical options as readable strings
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

def generate_insurance_recommendations(price, vehicle_data):
    """Generate luxury insurance coverage proposals based on vehicle profile."""
    recommendations = []
    
    # Premium / Executive Tier
    if price > 35000 or vehicle_data.get("Make") in ["BMW", "Mercedes-Benz", "Audi"]:
        recommendations.append({
            "title": "Titanium Sovereign Cover",
            "tier": "PLATINUM",
            "badge": "VIP Choice",
            "price_est": f"${int(price * 0.042)}/yr",
            "features": ["Zero Depreciation", "Full Engine & ECU Protection", "24/7 Global Concierge", "Guaranteed Invoice Value"],
            "match": "99% Precision Match"
        })
    
    # Comprehensive Core Tier
    recommendations.append({
        "title": "Apex Dynamic Shield",
        "tier": "EXECUTIVE",
        "badge": "Recommended",
        "price_est": f"${int(price * 0.031)}/yr",
        "features": ["All-Risk Collision Cover", "Third-Party Unlimited Liability", "Personal Driver Cover", "Cashless Repair Hubs"],
        "match": "94% Optimal Match"
    })
    
    # Essential Urban Tier
    recommendations.append({
        "title": "Metro Guard Select",
        "tier": "ESSENTIAL",
        "badge": "Smart Value",
        "price_est": f"${int(price * 0.020)}/yr",
        "features": ["Fire, Theft & Vandalism", "Emergency Towing & Key Lockout", "Basic Medical Assist"],
        "match": "88% Match"
    })
    
    return recommendations

# Executive Dashboard Template with Glassmorphic Interface & Dynamic Themes
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
        /* Base Root Color Variables */
        :root {
            --bg-base: #0b0f19;
            --bg-card: rgba(20, 27, 41, 0.7);
            --card-border: rgba(255, 255, 255, 0.08);
            --accent: #6366f1;
            --accent-glow: rgba(99, 102, 241, 0.35);
            --accent-hover: #4f46e5;
            --text-heading: #f8fafc;
            --text-sub: #94a3b8;
            --input-bg: rgba(15, 23, 42, 0.6);
            --header-bg: rgba(11, 15, 25, 0.85);
        }

        /* Dynamic Preset Themes */
        body.theme-titanium {
            --bg-base: #0a0a0c;
            --bg-card: rgba(22, 22, 26, 0.75);
            --card-border: rgba(255, 255, 255, 0.12);
            --accent: #e2e8f0;
            --accent-glow: rgba(226, 232, 240, 0.25);
            --accent-hover: #ffffff;
            --text-heading: #ffffff;
            --text-sub: #94a3b8;
            --input-bg: rgba(30, 30, 36, 0.6);
        }

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
        }

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
        }

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

        /* Top Bar Navigation */
        .header-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 18px 40px;
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
            font-size: 1.4rem;
            font-weight: 700;
            letter-spacing: 1px;
            color: var(--text-heading);
        }

        .brand-icon {
            width: 38px;
            height: 38px;
            background: var(--accent);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #000;
            box-shadow: 0 0 15px var(--accent-glow);
        }

        .theme-switcher {
            display: flex;
            align-items: center;
            gap: 12px;
            background: rgba(0, 0, 0, 0.4);
            padding: 6px 14px;
            border-radius: 30px;
            border: 1px solid var(--card-border);
        }

        .theme-label {
            font-size: 0.75rem;
            color: var(--text-sub);
            text-transform: uppercase;
            font-weight: 700;
            letter-spacing: 0.5px;
        }

        .theme-dot {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            cursor: pointer;
            border: 2px solid transparent;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }

        .theme-dot:hover {
            transform: scale(1.25);
        }

        .theme-dot.active {
            border-color: var(--text-heading);
            transform: scale(1.1);
        }

        .dot-default { background: #6366f1; }
        .dot-titanium { background: #e2e8f0; }
        .dot-cyber { background: #00f0ff; }
        .dot-gold { background: #e5c158; }
        .dot-emerald { background: #10b981; }

        /* Dashboard Container */
        .wrapper {
            margin-top: 90px;
            padding: 40px;
            max-width: 1550px;
            margin-left: auto;
            margin-right: auto;
        }

        .hero-head {
            margin-bottom: 30px;
        }

        .hero-head h1 {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 6px;
        }

        .hero-head p {
            color: var(--text-sub);
            font-size: 1rem;
        }

        /* Two Column Grid Layout */
        .dashboard-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }

        @media (max-width: 1100px) {
            .dashboard-grid { grid-template-columns: 1fr; }
            .wrapper { padding: 20px; }
            .header-bar { padding: 15px 20px; }
        }

        /* Glassmorphic Panel Design */
        .glass-panel {
            background: var(--bg-card);
            backdrop-filter: blur(20px);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 30px;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4);
            position: relative;
        }

        .panel-title {
            font-size: 1.15rem;
            font-weight: 700;
            margin-bottom: 24px;
            display: flex;
            align-items: center;
            gap: 10px;
            border-bottom: 1px solid var(--card-border);
            padding-bottom: 14px;
            color: var(--text-heading);
        }

        .panel-title i {
            color: var(--accent);
        }

        /* Form Inputs Layout */
        .form-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 16px;
        }

        .form-field {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .form-field label {
            font-size: 0.78rem;
            font-weight: 700;
            color: var(--text-sub);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .custom-input {
            background: var(--input-bg);
            border: 1px solid var(--card-border);
            color: var(--text-heading);
            padding: 12px 14px;
            border-radius: 8px;
            font-size: 0.95rem;
            outline: none;
        }

        .custom-input:focus {
            border-color: var(--accent);
            box-shadow: 0 0 12px var(--accent-glow);
        }

        /* Premium Animation Action Button */
        .btn-holder {
            margin-top: 25px;
        }

        .btn-spark {
            width: 100%;
            background: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%);
            color: #ffffff;
            border: none;
            padding: 16px;
            font-size: 1rem;
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
            background: linear-gradient(
                60deg,
                transparent,
                rgba(255, 255, 255, 0.25),
                transparent
            );
            transform: rotate(30deg);
            transition: transform 0.8s ease;
        }

        .btn-spark:hover::after {
            transform: translate(100%, 100%) rotate(30deg);
        }

        .btn-spark:hover {
            transform: translateY(-3px) scale(1.01);
            box-shadow: 0 12px 30px var(--accent-glow);
        }

        .btn-spark:active {
            transform: translateY(1px);
        }

        /* Valuation Dashboard Card */
        .val-card {
            background: rgba(0, 0, 0, 0.3);
            border: 1px dashed var(--accent);
            border-radius: 12px;
            padding: 24px;
            text-align: center;
            margin-bottom: 24px;
        }

        .val-sub {
            font-size: 0.8rem;
            color: var(--text-sub);
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 700;
        }

        .val-price {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 3.2rem;
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

        /* Clean Analytics Telemetry Grid */
        .metrics-row {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            margin-bottom: 24px;
        }

        .m-box {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--card-border);
            padding: 14px;
            border-radius: 10px;
            text-align: center;
        }

        .m-val {
            font-size: 1.15rem;
            font-weight: 700;
            margin-top: 4px;
            color: var(--text-heading);
        }

        .m-lbl {
            font-size: 0.72rem;
            color: var(--text-sub);
            text-transform: uppercase;
        }

        /* Insurance Proposals Layout */
        .plans-list {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        .plan-card {
            background: var(--input-bg);
            border: 1px solid var(--card-border);
            border-radius: 10px;
            padding: 16px;
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

        .plan-header {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 4px;
        }

        .plan-header h4 {
            font-size: 0.98rem;
            font-weight: 700;
        }

        .badge-tier {
            background: var(--accent);
            color: #000;
            font-weight: 800;
            font-size: 0.65rem;
            padding: 2px 8px;
            border-radius: 12px;
            text-transform: uppercase;
        }

        .plan-meta {
            font-size: 0.8rem;
            color: var(--text-sub);
        }

        .plan-features {
            display: flex;
            gap: 6px;
            margin-top: 8px;
            flex-wrap: wrap;
        }

        .f-tag {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--card-border);
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.7rem;
            color: var(--text-sub);
        }
    </style>
</head>
<body class="theme-default">

    <!-- Header Navigation Bar -->
    <header class="header-bar">
        <div class="brand-box">
            <div class="brand-icon">
                <i class="fa-solid fa-gauge-high"></i>
            </div>
            <span>AURA MOTORS</span>
        </div>
        
        <div class="theme-switcher">
            <span class="theme-label">Theme Engine</span>
            <div class="theme-dot dot-default active" onclick="switchTheme('theme-default', this)" title="Indigo Core"></div>
            <div class="theme-dot dot-titanium" onclick="switchTheme('theme-titanium', this)" title="Titanium Dark"></div>
            <div class="theme-dot dot-cyber" onclick="switchTheme('theme-cyber', this)" title="Cyber Neon"></div>
            <div class="theme-dot dot-gold" onclick="switchTheme('theme-gold', this)" title="Royal Gold"></div>
            <div class="theme-dot dot-emerald" onclick="switchTheme('theme-emerald', this)" title="Emerald Lux"></div>
        </div>
    </header>

    <!-- Main Section -->
    <div class="wrapper">
        <div class="hero-head">
            <h1>Executive Valuation & Telemetry</h1>
            <p>Compute market appraisal via Random Forest regression and evaluate tailored insurance plans.</p>
        </div>

        <div class="dashboard-grid">
            <!-- Parameters Form -->
            <div class="glass-panel">
                <div class="panel-title">
                    <i class="fa-solid fa-sliders"></i> Vehicle Specifications
                </div>
                <form id="valuationForm">
                    <div class="form-grid">
                        {% for col in categorical_keys %}
                        <div class="form-field">
                            <label for="{{ col }}">{{ col.replace('_', ' ') }}</label>
                            <select id="{{ col }}" name="{{ col }}" class="custom-input">
                                {% for val in categorical_options[col] %}
                                <option value="{{ val }}">{{ val }}</option>
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
                            <i class="fa-solid fa-microchip"></i> Run AI Valuation
                        </button>
                    </div>
                </form>
            </div>

            <!-- Valuation Output & Insurance Analytics -->
            <div class="glass-panel">
                <div class="panel-title">
                    <i class="fa-solid fa-chart-pie"></i> Intelligence Output
                </div>

                <div class="val-card">
                    <div class="val-sub">Estimated Market Value</div>
                    <div class="val-price" id="valPrice">$0.00</div>
                    <span class="status-pill" id="valStatus">Ready for Calculation</span>
                </div>

                <!-- Clean Metric Dashboard -->
                <div class="metrics-row">
                    <div class="m-box">
                        <div class="m-lbl">Efficiency</div>
                        <div class="m-val" id="mEfficiency">--</div>
                    </div>
                    <div class="m-box">
                        <div class="m-lbl">Risk Rating</div>
                        <div class="m-val" id="mRisk">--</div>
                    </div>
                    <div class="m-box">
                        <div class="m-lbl">Depreciation</div>
                        <div class="m-val" id="mDep">--</div>
                    </div>
                </div>

                <!-- Insurance Plan Recommendations -->
                <div class="panel-title" style="margin-top: 10px; border-bottom: none; margin-bottom: 12px;">
                    <i class="fa-solid fa-shield"></i> Premium Coverage Plans
                </div>
                <div class="plans-list" id="insuranceContainer">
                    <p style="color: var(--text-sub); text-align: center; padding: 20px;">
                        Input vehicle metrics and execute valuation to view dynamic insurance proposals.
                    </p>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Switch dynamic CSS themes
        function switchTheme(themeName, elem) {
            document.body.className = themeName;
            document.querySelectorAll('.theme-dot').forEach(dot => dot.classList.remove('active'));
            elem.classList.add('active');
        }

        // AJAX Form Submission
        document.getElementById('valuationForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const btn = document.getElementById('calcBtn');
            btn.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Processing Model...';

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
                    // Update main valuation UI
                    document.getElementById('valPrice').innerText = '$' + res.predicted_price.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
                    document.getElementById('valStatus').innerText = 'Valuation Verified';

                    // Update metrics
                    document.getElementById('mEfficiency').innerText = res.metrics.efficiency;
                    document.getElementById('mRisk').innerText = res.metrics.risk;
                    document.getElementById('mDep').innerText = res.metrics.depreciation;

                    // Render tailored insurance proposals
                    const container = document.getElementById('insuranceContainer');
                    container.innerHTML = '';

                    res.recommendations.forEach(plan => {
                        container.innerHTML += `
                            <div class="plan-card" onclick="alert('Selected Coverage: ${plan.title}')">
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
                btn.innerHTML = '<i class="fa-solid fa-microchip"></i> Run AI Valuation';
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
        
        # Build single-row DataFrame with categorical columns retained as readable strings
        input_data = {}
        for col in FEATURE_ORDER:
            if col in CATEGORICAL_OPTIONS:
                input_data[col] = [str(data.get(col, CATEGORICAL_OPTIONS[col][0]))]
            else:
                input_data[col] = [float(data.get(col, DEFAULT_NUMERICALS.get(col, 0)))]

        df_input = pd.DataFrame(input_data)

        # Run Random Forest Model prediction
        if model is not None:
            prediction = model.predict(df_input)[0]
            predicted_value = float(prediction)
        else:
            # Fallback estimation formula if pkl file is missing
            predicted_value = float(
                df_input["Engine_Size"].iloc[0] * 5200 + 
                df_input["Horsepower"].iloc[0] * 130 - 
                df_input["Mileage"].iloc[0] * 0.04 + 
                (2026 - df_input["Year"].iloc[0]) * -900 + 22000
            )

        # Metrics evaluation
        mileage = float(data.get("Mileage", 32000))
        accident = data.get("Accident_History", "None")
        
        risk_level = "Low" if accident == "None" else ("Moderate" if accident == "Minor" else "High")
        dep_risk = "Low" if mileage < 30000 else ("Moderate" if mileage < 75000 else "High")
        efficiency_tier = "A+" if float(data.get("Fuel_Efficiency", 16.8)) > 17.0 else "Standard"

        recommendations = generate_insurance_recommendations(predicted_value, data)

        return jsonify({
            "success": True,
            "predicted_price": round(predicted_value, 2),
            "metrics": {
                "efficiency": efficiency_tier,
                "risk": risk_level,
                "depreciation": dep_risk
            },
            "recommendations": recommendations
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Model loading setup using the exact pickle filename
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
    print(f"Warning: '{MODEL_PATH}' not found in the root directory. Operating in preview mode.")

# Feature mapping with categorical options as readable strings
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

def generate_insurance_recommendations(price, vehicle_data):
    """Generate luxury insurance coverage proposals based on vehicle profile."""
    recommendations = []
    
    # Premium / Executive Tier
    if price > 35000 or vehicle_data.get("Make") in ["BMW", "Mercedes-Benz", "Audi"]:
        recommendations.append({
            "title": "Titanium Sovereign Cover",
            "tier": "PLATINUM",
            "badge": "VIP Choice",
            "price_est": f"${int(price * 0.042)}/yr",
            "features": ["Zero Depreciation", "Full Engine & ECU Protection", "24/7 Global Concierge", "Guaranteed Invoice Value"],
            "match": "99% Precision Match"
        })
    
    # Comprehensive Core Tier
    recommendations.append({
        "title": "Apex Dynamic Shield",
        "tier": "EXECUTIVE",
        "badge": "Recommended",
        "price_est": f"${int(price * 0.031)}/yr",
        "features": ["All-Risk Collision Cover", "Third-Party Unlimited Liability", "Personal Driver Cover", "Cashless Repair Hubs"],
        "match": "94% Optimal Match"
    })
    
    # Essential Urban Tier
    recommendations.append({
        "title": "Metro Guard Select",
        "tier": "ESSENTIAL",
        "badge": "Smart Value",
        "price_est": f"${int(price * 0.020)}/yr",
        "features": ["Fire, Theft & Vandalism", "Emergency Towing & Key Lockout", "Basic Medical Assist"],
        "match": "88% Match"
    })
    
    return recommendations

# Executive Dashboard Template with Glassmorphic Interface & Dynamic Themes
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
        /* Base Root Color Variables */
        :root {
            --bg-base: #0b0f19;
            --bg-card: rgba(20, 27, 41, 0.7);
            --card-border: rgba(255, 255, 255, 0.08);
            --accent: #6366f1;
            --accent-glow: rgba(99, 102, 241, 0.35);
            --accent-hover: #4f46e5;
            --text-heading: #f8fafc;
            --text-sub: #94a3b8;
            --input-bg: rgba(15, 23, 42, 0.6);
            --header-bg: rgba(11, 15, 25, 0.85);
        }

        /* Dynamic Preset Themes */
        body.theme-titanium {
            --bg-base: #0a0a0c;
            --bg-card: rgba(22, 22, 26, 0.75);
            --card-border: rgba(255, 255, 255, 0.12);
            --accent: #e2e8f0;
            --accent-glow: rgba(226, 232, 240, 0.25);
            --accent-hover: #ffffff;
            --text-heading: #ffffff;
            --text-sub: #94a3b8;
            --input-bg: rgba(30, 30, 36, 0.6);
        }

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
        }

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
        }

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

        /* Top Bar Navigation */
        .header-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 18px 40px;
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
            font-size: 1.4rem;
            font-weight: 700;
            letter-spacing: 1px;
            color: var(--text-heading);
        }

        .brand-icon {
            width: 38px;
            height: 38px;
            background: var(--accent);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #000;
            box-shadow: 0 0 15px var(--accent-glow);
        }

        .theme-switcher {
            display: flex;
            align-items: center;
            gap: 12px;
            background: rgba(0, 0, 0, 0.4);
            padding: 6px 14px;
            border-radius: 30px;
            border: 1px solid var(--card-border);
        }

        .theme-label {
            font-size: 0.75rem;
            color: var(--text-sub);
            text-transform: uppercase;
            font-weight: 700;
            letter-spacing: 0.5px;
        }

        .theme-dot {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            cursor: pointer;
            border: 2px solid transparent;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }

        .theme-dot:hover {
            transform: scale(1.25);
        }

        .theme-dot.active {
            border-color: var(--text-heading);
            transform: scale(1.1);
        }

        .dot-default { background: #6366f1; }
        .dot-titanium { background: #e2e8f0; }
        .dot-cyber { background: #00f0ff; }
        .dot-gold { background: #e5c158; }
        .dot-emerald { background: #10b981; }

        /* Dashboard Container */
        .wrapper {
            margin-top: 90px;
            padding: 40px;
            max-width: 1550px;
            margin-left: auto;
            margin-right: auto;
        }

        .hero-head {
            margin-bottom: 30px;
        }

        .hero-head h1 {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 6px;
        }

        .hero-head p {
            color: var(--text-sub);
            font-size: 1rem;
        }

        /* Two Column Grid Layout */
        .dashboard-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }

        @media (max-width: 1100px) {
            .dashboard-grid { grid-template-columns: 1fr; }
            .wrapper { padding: 20px; }
            .header-bar { padding: 15px 20px; }
        }

        /* Glassmorphic Panel Design */
        .glass-panel {
            background: var(--bg-card);
            backdrop-filter: blur(20px);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 30px;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4);
            position: relative;
        }

        .panel-title {
            font-size: 1.15rem;
            font-weight: 700;
            margin-bottom: 24px;
            display: flex;
            align-items: center;
            gap: 10px;
            border-bottom: 1px solid var(--card-border);
            padding-bottom: 14px;
            color: var(--text-heading);
        }

        .panel-title i {
            color: var(--accent);
        }

        /* Form Inputs Layout */
        .form-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 16px;
        }

        .form-field {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .form-field label {
            font-size: 0.78rem;
            font-weight: 700;
            color: var(--text-sub);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .custom-input {
            background: var(--input-bg);
            border: 1px solid var(--card-border);
            color: var(--text-heading);
            padding: 12px 14px;
            border-radius: 8px;
            font-size: 0.95rem;
            outline: none;
        }

        .custom-input:focus {
            border-color: var(--accent);
            box-shadow: 0 0 12px var(--accent-glow);
        }

        /* Premium Animation Action Button */
        .btn-holder {
            margin-top: 25px;
        }

        .btn-spark {
            width: 100%;
            background: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%);
            color: #ffffff;
            border: none;
            padding: 16px;
            font-size: 1rem;
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
            background: linear-gradient(
                60deg,
                transparent,
                rgba(255, 255, 255, 0.25),
                transparent
            );
            transform: rotate(30deg);
            transition: transform 0.8s ease;
        }

        .btn-spark:hover::after {
            transform: translate(100%, 100%) rotate(30deg);
        }

        .btn-spark:hover {
            transform: translateY(-3px) scale(1.01);
            box-shadow: 0 12px 30px var(--accent-glow);
        }

        .btn-spark:active {
            transform: translateY(1px);
        }

        /* Valuation Dashboard Card */
        .val-card {
            background: rgba(0, 0, 0, 0.3);
            border: 1px dashed var(--accent);
            border-radius: 12px;
            padding: 24px;
            text-align: center;
            margin-bottom: 24px;
        }

        .val-sub {
            font-size: 0.8rem;
            color: var(--text-sub);
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 700;
        }

        .val-price {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 3.2rem;
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

        /* Clean Analytics Telemetry Grid */
        .metrics-row {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            margin-bottom: 24px;
        }

        .m-box {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--card-border);
            padding: 14px;
            border-radius: 10px;
            text-align: center;
        }

        .m-val {
            font-size: 1.15rem;
            font-weight: 700;
            margin-top: 4px;
            color: var(--text-heading);
        }

        .m-lbl {
            font-size: 0.72rem;
            color: var(--text-sub);
            text-transform: uppercase;
        }

        /* Insurance Proposals Layout */
        .plans-list {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        .plan-card {
            background: var(--input-bg);
            border: 1px solid var(--card-border);
            border-radius: 10px;
            padding: 16px;
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

        .plan-header {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 4px;
        }

        .plan-header h4 {
            font-size: 0.98rem;
            font-weight: 700;
        }

        .badge-tier {
            background: var(--accent);
            color: #000;
            font-weight: 800;
            font-size: 0.65rem;
            padding: 2px 8px;
            border-radius: 12px;
            text-transform: uppercase;
        }

        .plan-meta {
            font-size: 0.8rem;
            color: var(--text-sub);
        }

        .plan-features {
            display: flex;
            gap: 6px;
            margin-top: 8px;
            flex-wrap: wrap;
        }

        .f-tag {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--card-border);
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.7rem;
            color: var(--text-sub);
        }
    </style>
</head>
<body class="theme-default">

    <!-- Header Navigation Bar -->
    <header class="header-bar">
        <div class="brand-box">
            <div class="brand-icon">
                <i class="fa-solid fa-gauge-high"></i>
            </div>
            <span>AURA MOTORS</span>
        </div>
        
        <div class="theme-switcher">
            <span class="theme-label">Theme Engine</span>
            <div class="theme-dot dot-default active" onclick="switchTheme('theme-default', this)" title="Indigo Core"></div>
            <div class="theme-dot dot-titanium" onclick="switchTheme('theme-titanium', this)" title="Titanium Dark"></div>
            <div class="theme-dot dot-cyber" onclick="switchTheme('theme-cyber', this)" title="Cyber Neon"></div>
            <div class="theme-dot dot-gold" onclick="switchTheme('theme-gold', this)" title="Royal Gold"></div>
            <div class="theme-dot dot-emerald" onclick="switchTheme('theme-emerald', this)" title="Emerald Lux"></div>
        </div>
    </header>

    <!-- Main Section -->
    <div class="wrapper">
        <div class="hero-head">
            <h1>Executive Valuation & Telemetry</h1>
            <p>Compute market appraisal via Random Forest regression and evaluate tailored insurance plans.</p>
        </div>

        <div class="dashboard-grid">
            <!-- Parameters Form -->
            <div class="glass-panel">
                <div class="panel-title">
                    <i class="fa-solid fa-sliders"></i> Vehicle Specifications
                </div>
                <form id="valuationForm">
                    <div class="form-grid">
                        {% for col in categorical_keys %}
                        <div class="form-field">
                            <label for="{{ col }}">{{ col.replace('_', ' ') }}</label>
                            <select id="{{ col }}" name="{{ col }}" class="custom-input">
                                {% for val in categorical_options[col] %}
                                <option value="{{ val }}">{{ val }}</option>
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
                            <i class="fa-solid fa-microchip"></i> Run AI Valuation
                        </button>
                    </div>
                </form>
            </div>

            <!-- Valuation Output & Insurance Analytics -->
            <div class="glass-panel">
                <div class="panel-title">
                    <i class="fa-solid fa-chart-pie"></i> Intelligence Output
                </div>

                <div class="val-card">
                    <div class="val-sub">Estimated Market Value</div>
                    <div class="val-price" id="valPrice">$0.00</div>
                    <span class="status-pill" id="valStatus">Ready for Calculation</span>
                </div>

                <!-- Clean Metric Dashboard -->
                <div class="metrics-row">
                    <div class="m-box">
                        <div class="m-lbl">Efficiency</div>
                        <div class="m-val" id="mEfficiency">--</div>
                    </div>
                    <div class="m-box">
                        <div class="m-lbl">Risk Rating</div>
                        <div class="m-val" id="mRisk">--</div>
                    </div>
                    <div class="m-box">
                        <div class="m-lbl">Depreciation</div>
                        <div class="m-val" id="mDep">--</div>
                    </div>
                </div>

                <!-- Insurance Plan Recommendations -->
                <div class="panel-title" style="margin-top: 10px; border-bottom: none; margin-bottom: 12px;">
                    <i class="fa-solid fa-shield"></i> Premium Coverage Plans
                </div>
                <div class="plans-list" id="insuranceContainer">
                    <p style="color: var(--text-sub); text-align: center; padding: 20px;">
                        Input vehicle metrics and execute valuation to view dynamic insurance proposals.
                    </p>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Switch dynamic CSS themes
        function switchTheme(themeName, elem) {
            document.body.className = themeName;
            document.querySelectorAll('.theme-dot').forEach(dot => dot.classList.remove('active'));
            elem.classList.add('active');
        }

        // AJAX Form Submission
        document.getElementById('valuationForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const btn = document.getElementById('calcBtn');
            btn.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Processing Model...';

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
                    // Update main valuation UI
                    document.getElementById('valPrice').innerText = '$' + res.predicted_price.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
                    document.getElementById('valStatus').innerText = 'Valuation Verified';

                    // Update metrics
                    document.getElementById('mEfficiency').innerText = res.metrics.efficiency;
                    document.getElementById('mRisk').innerText = res.metrics.risk;
                    document.getElementById('mDep').innerText = res.metrics.depreciation;

                    // Render tailored insurance proposals
                    const container = document.getElementById('insuranceContainer');
                    container.innerHTML = '';

                    res.recommendations.forEach(plan => {
                        container.innerHTML += `
                            <div class="plan-card" onclick="alert('Selected Coverage: ${plan.title}')">
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
                btn.innerHTML = '<i class="fa-solid fa-microchip"></i> Run AI Valuation';
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
        
        # Build single-row DataFrame with categorical columns retained as readable strings
        input_data = {}
        for col in FEATURE_ORDER:
            if col in CATEGORICAL_OPTIONS:
                input_data[col] = [str(data.get(col, CATEGORICAL_OPTIONS[col][0]))]
            else:
                input_data[col] = [float(data.get(col, DEFAULT_NUMERICALS.get(col, 0)))]

        df_input = pd.DataFrame(input_data)

        # Run Random Forest Model prediction
        if model is not None:
            prediction = model.predict(df_input)[0]
            predicted_value = float(prediction)
        else:
            # Fallback estimation formula if pkl file is missing
            predicted_value = float(
                df_input["Engine_Size"].iloc[0] * 5200 + 
                df_input["Horsepower"].iloc[0] * 130 - 
                df_input["Mileage"].iloc[0] * 0.04 + 
                (2026 - df_input["Year"].iloc[0]) * -900 + 22000
            )

        # Metrics evaluation
        mileage = float(data.get("Mileage", 32000))
        accident = data.get("Accident_History", "None")
        
        risk_level = "Low" if accident == "None" else ("Moderate" if accident == "Minor" else "High")
        dep_risk = "Low" if mileage < 30000 else ("Moderate" if mileage < 75000 else "High")
        efficiency_tier = "A+" if float(data.get("Fuel_Efficiency", 16.8)) > 17.0 else "Standard"

        recommendations = generate_insurance_recommendations(predicted_value, data)

        return jsonify({
            "success": True,
            "predicted_price": round(predicted_value, 2),
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
