import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Big Mart Sales Prediction", page_icon="🛒", layout="centered")

# -------------------- STYLE --------------------
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg,#ff7f7f 0%, #ffe5e5 100%); }
h1 { text-align:center; color:white; font-weight:700; }
label, .stNumberInput label, .stSelectbox label { color:white !important; }
.stButton>button {
  background:#2563eb !important; color:white !important;
  border:0; border-radius:8px; height:2.5rem; width:120px; margin:0 6px;
  font-weight:600;
}
header, footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>Big Mart Sales Prediction</h1>", unsafe_allow_html=True)

# -------------------- LOAD SCALER + MODEL --------------------
BASE = Path(__file__).resolve().parent
scaler_path = BASE / "models" / "sc.sav"
model_path  = BASE / "models" / "rf.sav"   # نستخدم rf.sav الموجود عندك

if not scaler_path.exists():
    st.error(f"Scaler not found: {scaler_path}")
    st.stop()
if not model_path.exists():
    st.error(f"Model not found: {model_path}")
    st.stop()

sc = joblib.load(scaler_path)
model = joblib.load(model_path)

# -------------------- ENCODING MAPS --------------------
fat_map = {"High Fat":0, "Low Fat":1, "Regular":2}
type_map = {
    "Baking Goods":0,"Breads":1,"Breakfast":2,"Canned":3,"Dairy":4,"Frozen Foods":5,
    "Fruits and Vegetables":6,"Hard Drinks":7,"Health and Hygiene":8,"Household":9,
    "Meat":10,"Others":11,"Seafood":12,"Snack Foods":13,"Soft Drinks":14,"Starchy Foods":15
}
size_map = {"High":0,"Medium":1,"Small":2}
loc_map  = {"Tier 1":0,"Tier 2":1,"Tier 3":2}
otype_map= {"Grocery Store":0,"Supermarket Type1":1,"Supermarket Type2":2,"Supermarket Type3":3}

# نفس ترتيب الميزات الذي يذهب للـ scaler/model
feature_names = [
    "Item_Weight", "Item_Fat_Content(code)", "Item_Visibility", "Item_Type(code)",
    "Item_MRP", "Outlet_Establishment_Year", "Outlet_Size(code)",
    "Outlet_Location_Type(code)", "Outlet_Type(code)"
]

# -------------------- FORM --------------------
with st.form("predict_form"):
    item_weight = st.number_input("Enter Item Weight", min_value=0.0, step=0.1)
    item_fat_content = st.selectbox("Item Fat Content", list(fat_map.keys()))
    item_visibility = st.number_input("Enter Item Visibility", min_value=0.0, max_value=1.0, step=0.01)
    item_type = st.selectbox("Item Type", list(type_map.keys()))
    item_mrp = st.number_input("Enter Item MRP", min_value=0.0, step=0.25)
    outlet_year = st.number_input("Outlet Establishment Year (YYYY)", min_value=1950, max_value=2025, step=1)
    outlet_size = st.selectbox("Outlet Size", list(size_map.keys()))
    outlet_location = st.selectbox("Outlet Location Type", list(loc_map.keys()))
    outlet_type = st.selectbox("Outlet Type", list(otype_map.keys()))

    c1, c2 = st.columns([1,1])
    with c1:
        submitted = st.form_submit_button("Submit")
    with c2:
        _ = st.form_submit_button("Reset")  # يعيد النموذج افتراضيًا عند إعادة التحميل

# -------------------- PREDICT + PLOTS --------------------
if submitted:
    # بناء متجه الإدخال (RAW → codes)
    X_raw = np.array([[
        item_weight,
        fat_map[item_fat_content],
        item_visibility,
        type_map[item_type],
        item_mrp,
        outlet_year,
        size_map[outlet_size],
        loc_map[outlet_location],
        otype_map[outlet_type]
    ]], dtype=float)

    # Transform ثم predict
    X_std = sc.transform(X_raw)
    y_pred = float(model.predict(X_std)[0])
    st.success(f"Predicted Sales: **{y_pred:.2f}**")

    # ---------- Plot 1: Input feature snapshot (bar) ----------
    st.subheader("Input snapshot")
    fig1 = plt.figure()
    plt.bar(range(len(feature_names)), X_raw.flatten())
    plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha="right")
    plt.ylabel("Raw value")
    plt.title("Entered features (raw numeric/coded)")
    st.pyplot(fig1)

    # ---------- Plot 2: Feature Importances (if available) ----------
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        # تأكد أن الطول يطابق الميزات (لأنه موديلك يستخدم 9 أبعاد بعد الـ scaler)
        if len(importances) == len(feature_names):
            st.subheader("Model feature importances")
            order = np.argsort(importances)[::-1]
            fig2 = plt.figure()
            plt.bar(range(len(importances)), importances[order])
            plt.xticks(range(len(importances)), np.array(feature_names)[order], rotation=45, ha="right")
            plt.ylabel("Importance")
            plt.title("RandomForest Feature Importances")
            st.pyplot(fig2)

    # ---------- Plot 3: Sensitivity curve for Item MRP ----------
    st.subheader("Sensitivity: effect of Item MRP on prediction")
    mrp_min = max(0.0, item_mrp * 0.5)
    mrp_max = item_mrp * 1.5 if item_mrp > 0 else 300.0
    mrp_grid = np.linspace(mrp_min, mrp_max, 30)

    X_sweep = np.repeat(X_raw, len(mrp_grid), axis=0)
    X_sweep[:, 4] = mrp_grid  # العمود 4 هو Item_MRP في ترتيبنا
    X_sweep_std = sc.transform(X_sweep)
    y_sweep = model.predict(X_sweep_std)

    fig3 = plt.figure()
    plt.plot(mrp_grid, y_sweep)
    plt.xlabel("Item MRP")
    plt.ylabel("Predicted Sales")
    plt.title("Prediction vs. MRP (other inputs fixed)")
    st.pyplot(fig3)

# -------------------- Batch CSV + Plots --------------------
st.markdown("---")
st.subheader("Batch prediction (CSV) + plots")
csv = st.file_uploader("Upload CSV with the same training feature columns (coded where needed)", type=["csv"])

if csv is not None:
    try:
        df = pd.read_csv(csv)
        st.write("Preview", df.head())

        # توقعات
        df_std = sc.transform(df.values)   # إذا أعمدتك بنفس ترتيب التدرّيب (9 أعمدة بالأكواد)
        preds = model.predict(df_std)
        df_out = df.copy()
        df_out["prediction"] = preds
        st.dataframe(df_out.head(50), use_container_width=True)
        st.download_button("Download predictions.csv", df_out.to_csv(index=False).encode("utf-8"), "predictions.csv")

        # Histogram للتوقعات
        st.subheader("Predictions distribution")
        fig4 = plt.figure()
        plt.hist(preds, bins=20)
        plt.xlabel("Predicted Sales")
        plt.ylabel("Frequency")
        plt.title("Histogram of predictions")
        st.pyplot(fig4)

        # Scatter بين MRP والتوقع (إذا العمود موجود)
        if "Item_MRP" in df.columns:
            st.subheader("Prediction vs. Item MRP (scatter)")
            fig5 = plt.figure()
            plt.scatter(df["Item_MRP"], preds, s=12)
            plt.xlabel("Item MRP")
            plt.ylabel("Predicted Sales")
            plt.title("Scatter: MRP vs Prediction")
            st.pyplot(fig5)

    except Exception as e:
        st.error(f"Failed to process CSV. Details: {e}")
