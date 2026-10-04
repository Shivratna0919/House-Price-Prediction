
import streamlit as st
import pandas as pd
import pickle

# =====================================================
# Page Configuration
# =====================================================
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# Load Model Files
# =====================================================
@st.cache_resource
def load_artifacts():
    with open("house_price_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("vectorizer.pkl", "rb") as f:
        dv = pickle.load(f)

    with open("encoder.pkl", "rb") as f:
        encoder = pickle.load(f)

    with open("features.pkl", "rb") as f:
        features = pickle.load(f)

    return model, dv, encoder, features


model, dv, encoder, features = load_artifacts()

# =====================================================
# Modern UI Styling
# =====================================================
st.markdown(
    """
    <style>
        .stApp {
            background: #f5f7fb;
        }

        #MainMenu, footer, header {
            visibility: hidden;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #172554 0%, #1e3a8a 55%, #312e81 100%);
        }

        [data-testid="stSidebar"] * {
            color: #ffffff !important;
        }

        .hero {
            background: linear-gradient(115deg, #2563eb 0%, #4f46e5 42%, #7c3aed 72%, #db2777 100%);
            border-radius: 22px;
            padding: 30px 34px;
            color: white;
            box-shadow: 0 12px 30px rgba(37, 99, 235, .18);
            margin-bottom: 22px;
        }

        .hero-grid {
            display: grid;
            grid-template-columns: 1.15fr .85fr;
            gap: 25px;
            align-items: center;
        }

        .hero h1 {
            margin: 0;
            font-size: 38px;
            line-height: 1.1;
            font-weight: 850;
        }

        .hero h1 span {
            color: #fde047;
        }

        .hero p {
            margin: 10px 0 0;
            color: #eef2ff;
            font-size: 15px;
        }

        .hero-stats {
            display: flex;
            justify-content: flex-end;
            gap: 12px;
            flex-wrap: wrap;
        }

        .hero-stat {
            min-width: 125px;
            padding: 14px;
            border: 1px solid rgba(255,255,255,.28);
            background: rgba(255,255,255,.11);
            border-radius: 14px;
        }

        .hero-stat .icon {
            font-size: 22px;
        }

        .hero-stat b {
            display: block;
            margin-top: 5px;
            font-size: 13px;
        }

        .hero-stat small {
            color: #e0e7ff;
            font-size: 10px;
        }

        .section-card {
            background: #ffffff;
            border: 1px solid #e4e9f2;
            border-radius: 17px;
            padding: 20px;
            box-shadow: 0 6px 18px rgba(15, 23, 42, .055);
            margin-bottom: 15px;
        }

        .section-title {
            color: #17356f;
            font-size: 19px;
            font-weight: 850;
            margin-bottom: 3px;
        }

        .section-subtitle {
            color: #64748b;
            font-size: 12px;
            margin-bottom: 13px;
        }

        .accent-line {
            width: 90px;
            height: 3px;
            border-radius: 10px;
            background: linear-gradient(90deg, #2563eb, #7c3aed);
            margin-bottom: 15px;
        }

        /* Streamlit labels */
        div[data-testid="stTextInput"] label,
        div[data-testid="stNumberInput"] label,
        div[data-testid="stSelectbox"] label {
            color: #075fd6 !important;
            font-weight: 750 !important;
            font-size: 13px !important;
        }

        div[data-testid="stTextInput"] input,
        div[data-testid="stNumberInput"] input {
            color: #172033 !important;
            background: #ffffff !important;
            border: 1px solid #d7e0ee !important;
            border-radius: 9px !important;
        }

        div[data-baseweb="select"] > div {
            background: #ffffff !important;
            color: #172033 !important;
            border-color: #d7e0ee !important;
            border-radius: 9px !important;
        }

        .prediction-placeholder {
            text-align: center;
            padding: 38px 15px;
            background: linear-gradient(135deg, #eff6ff, #f5f3ff);
            border: 1px solid #dbeafe;
            border-radius: 15px;
        }

        .prediction-placeholder .big {
            font-size: 36px;
        }

        .prediction-placeholder h3 {
            color: #243b72;
            margin: 7px 0 4px;
        }

        .prediction-placeholder p {
            color: #64748b;
            font-size: 12px;
        }

        .price-card {
            background: linear-gradient(135deg, #ecfdf5, #f0fdf4);
            border: 1px solid #bbf7d0;
            border-radius: 17px;
            padding: 24px;
            text-align: center;
            box-shadow: 0 8px 22px rgba(16, 185, 129, .08);
        }

        .price-card .label {
            color: #047857;
            font-size: 13px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: .05em;
        }

        .price-card .price {
            color: #059669;
            font-size: 34px;
            font-weight: 900;
            margin-top: 6px;
        }

        .price-card .desc {
            color: #64748b;
            font-size: 12px;
            margin-top: 3px;
        }

        .result-mini-card {
            border-radius: 14px;
            padding: 15px 17px;
            min-height: 72px;
            border: 1px solid;
            box-shadow: 0 5px 15px rgba(15, 23, 42, .05);
        }

        .blue-card { background: #eff6ff; border-color: #bfdbfe; }
        .purple-card { background: #f5f3ff; border-color: #ddd6fe; }
        .result-mini-card .mini-label {
            font-size: 10px; font-weight: 850; letter-spacing: .06em;
            color: #475569; margin-bottom: 6px;
        }
        .result-mini-card .mini-value {
            font-size: 18px; font-weight: 850; color: #17356f;
        }
        div[data-testid="stExpander"] summary,
        div[data-testid="stExpander"] summary p {
            color: #17356f !important; font-weight: 750 !important;
        }

        .tip-card {
            background: linear-gradient(135deg, #fff7ed, #fff1f2);
            border: 1px solid #fed7aa;
            border-radius: 13px;
            padding: 14px;
            color: #7c2d12;
            font-size: 12px;
            margin-top: 13px;
        }

        .feature-card {
            background: #ffffff;
            border: 1px solid #e4e9f2;
            border-radius: 14px;
            padding: 16px;
            height: 100%;
            box-shadow: 0 4px 14px rgba(15, 23, 42, .045);
        }

        .feature-card .emoji {
            font-size: 23px;
        }

        .feature-card h4 {
            color: #17356f;
            margin: 7px 0 5px;
        }

        .feature-card p {
            color: #64748b;
            font-size: 12px;
            margin: 0;
        }

        .footer {
            text-align: center;
            color: #64748b;
            font-size: 12px;
            padding: 18px;
        }

        div.stButton > button {
            min-height: 46px;
            border-radius: 10px;
            font-weight: 800;
            font-size: 14px;
        }

        @media (max-width: 900px) {
            .hero-grid {
                grid-template-columns: 1fr;
            }

            .hero-stats {
                justify-content: flex-start;
            }
        }
    </style>
    """,
    unsafe_allow_html=True
)

# =====================================================
# Hero Header
# =====================================================
st.markdown(
    """
    <div class="hero">
        <div class="hero-grid">
            <div>
                <h1>🏠 House Price<br><span>Prediction System</span></h1>
                <p>AI-powered property price estimation using Machine Learning.</p>
            </div>
            <div class="hero-stats">
                <div class="hero-stat">
                    <div class="icon">🤖</div>
                    <b>Machine Learning</b>
                    <small>Decision Tree Model</small>
                </div>
                <div class="hero-stat">
                    <div class="icon">📊</div>
                    <b>Smart Analysis</b>
                    <small>20 Property Features</small>
                </div>
                <div class="hero-stat">
                    <div class="icon">⚡</div>
                    <b>Instant Result</b>
                    <small>Real-time Prediction</small>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# =====================================================
# Sidebar
# =====================================================
with st.sidebar:
    st.markdown("## 🏠 House Price AI")
    st.caption("Property price estimation dashboard")
    st.divider()

    st.markdown("### 🤖 Model")
    st.markdown("**Decision Tree Regressor**")

    st.markdown("### 📊 Pipeline")
    st.markdown(
        "- 20 Input Features\n"
        "- DictVectorizer\n"
        "- Ordinal Encoding\n"
        "- Machine Learning Prediction"
    )

    st.divider()

    st.info(
        "Enter the property details and click **Predict House Price** "
        "to generate an estimated price."
    )

# =====================================================
# Main Layout
# =====================================================
left, right = st.columns([1.35, 1], gap="large")

# =====================================================
# Input Section
# =====================================================
with left:
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">🏡 Property Information</div>
            <div class="section-subtitle">
                Enter the details of the property for price estimation.
            </div>
            <div class="accent-line"></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:
        state = st.selectbox(
            "📍 State",
            ["maharashtra", "karnataka", "gujarat", "delhi", "tamil nadu"]
        )

        city = st.text_input("🏙️ City", placeholder="e.g. Pune")

        property_type = st.selectbox(
            "🏢 Property Type",
            ["Apartment", "Independent House", "Villa"]
        )

        bhk = st.number_input(
            "🛏️ BHK", min_value=1, max_value=10, value=2
        )

        size_sqft = st.number_input(
            "📐 Size (Sq.Ft)", min_value=300, max_value=10000, value=1200
        )

        price_sqft = st.number_input(
            "💰 Price Per Sq.Ft", min_value=1000, value=5000
        )

        year = st.number_input(
            "📅 Year Built", min_value=1980, max_value=2025, value=2018
        )

        furnished = st.selectbox(
            "🛋️ Furnished Status",
            ["Unfurnished", "Semi-furnished", "Furnished"]
        )

        floor = st.number_input(
            "🏢 Floor Number", min_value=0, value=2
        )

        total_floor = st.number_input(
            "🏙️ Total Floors", min_value=1, value=10
        )

    with c2:
        age = st.number_input(
            "⌛ Age of Property", min_value=0, value=5
        )

        school = st.number_input(
            "🏫 Nearby Schools", min_value=0, value=5
        )

        hospital = st.number_input(
            "🏥 Nearby Hospitals", min_value=0, value=3
        )

        transport = st.selectbox(
            "🚌 Public Transport", ["Low", "Medium", "High"]
        )

        parking = st.selectbox(
            "🚗 Parking Space", ["Yes", "No"]
        )

        security = st.selectbox(
            "🔐 Security", ["No", "Yes"]
        )

        amenities = st.text_input(
            "✨ Amenities",
            placeholder="e.g. Gym, Lift, Garden"
        )

        facing = st.selectbox(
            "🧭 Facing",
            ["South", "East", "West", "North"]
        )

        owner = st.text_input(
            "👤 Owner Type",
            placeholder="e.g. Owner, Dealer"
        )

        availability = st.text_input(
            "📅 Availability Status",
            placeholder="e.g. Ready to Move"
        )

    # Encode ordinal features exactly as in the original application.
    ordinal_df = pd.DataFrame({
        "Property_Type": [property_type],
        "Furnished_Status": [furnished],
        "Public_Transport_Accessibility": [transport],
        "Facing": [facing],
        "Security": [security]
    })

    ordinal_encoded = encoder.transform(ordinal_df)

    property_type_encoded = ordinal_encoded[0][0]
    furnished_encoded = ordinal_encoded[0][1]
    transport_encoded = ordinal_encoded[0][2]
    facing_encoded = ordinal_encoded[0][3]
    security_encoded = ordinal_encoded[0][4]

    st.markdown("<br>", unsafe_allow_html=True)

    predict_clicked = st.button(
        "🚀 Predict House Price",
        type="primary",
        use_container_width=True
    )

# =====================================================
# Result Section
# =====================================================
with right:
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">🎯 Prediction Result</div>
            <div class="section-subtitle">
                Your estimated property price will appear here.
            </div>
            <div class="accent-line"></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if predict_clicked:
        input_data = {
            "State": state.lower(),
            "City": city.lower(),
            "Property_Type": property_type_encoded,
            "BHK": bhk,
            "Size_in_SqFt": size_sqft,
            "Price_per_SqFt": price_sqft,
            "Year_Built": year,
            "Furnished_Status": furnished_encoded,
            "Floor_No": floor,
            "Total_Floors": total_floor,
            "Age_of_Property": age,
            "Nearby_Schools": school,
            "Nearby_Hospitals": hospital,
            "Public_Transport_Accessibility": transport_encoded,
            "Parking_Space": parking.lower(),
            "Security": security_encoded,
            "Amenities": amenities.lower(),
            "Facing": facing_encoded,
            "Owner_Type": owner.lower(),
            "Availability_Status": availability.lower()
        }

        try:
            X = dv.transform([input_data])
            prediction = model.predict(X)[0]

            st.success("✅ Prediction Successful")

            st.markdown(
                f"""
                <div class="price-card">
                    <div class="label">Estimated House Price</div>
                    <div class="price">₹ {prediction:.2f} Lakhs</div>
                    <div class="desc">Predicted using the trained Machine Learning model</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                """
                <div class="tip-card">
                    💡 <b>Note:</b> This is a Machine Learning-based estimate.
                    Actual market prices may vary depending on location, demand,
                    property condition and other factors.
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("<br>", unsafe_allow_html=True)

            r1, r2 = st.columns(2)

            with r1:
                st.markdown(
                    f"""
                    <div class="result-mini-card blue-card">
                        <div class="mini-label">🏠 PROPERTY TYPE</div>
                        <div class="mini-value">{property_type}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with r2:
                st.markdown(
                    f"""
                    <div class="result-mini-card purple-card">
                        <div class="mini-label">📐 PROPERTY SIZE</div>
                        <div class="mini-value">{size_sqft:,} Sq.Ft</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with st.expander("📋 View Input Data"):
                st.json(input_data)

        except Exception as e:
            st.error("❌ Prediction Failed")
            st.exception(e)

    else:
        st.markdown(
            """
            <div class="prediction-placeholder">
                <div class="big">🏠</div>
                <h3>Ready for Prediction</h3>
                <p>Enter property details on the left and click<br>
                <b>Predict House Price</b> to get the estimated value.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        f1, f2 = st.columns(2)

        with f1:
            st.markdown(
                """
                <div class="feature-card">
                    <div class="emoji">🤖</div>
                    <h4>Machine Learning</h4>
                    <p>Uses a trained Decision Tree Regressor for price estimation.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with f2:
            st.markdown(
                """
                <div class="feature-card">
                    <div class="emoji">⚡</div>
                    <h4>Instant Prediction</h4>
                    <p>Get an estimated property price instantly from your inputs.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

# =====================================================
# Footer
# =====================================================
st.markdown(
    """
    <div class="footer">
        🏠 <b>House Price Prediction System</b>
        &nbsp;•&nbsp; Streamlit & Scikit-Learn
        &nbsp;•&nbsp; Decision Tree Regressor
    </div>
    """,
    unsafe_allow_html=True
)
