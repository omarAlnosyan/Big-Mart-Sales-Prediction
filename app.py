import streamlit as st
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

st.set_page_config(page_title="Big Mart Sales Prediction", page_icon="🛒", layout="wide")

st.markdown("""
<style>
.stApp { background: linear-gradient(135deg,#ff7f7f 0%, #ffe5e5 100%); }
h1 { text-align:center; color:white; font-weight:800; font-size:44px; margin-top:4px; }
label, .stNumberInput label, .stSelectbox label { color:white !important; }
.stButton>button {
  background:#2563eb !important; color:white !important;
  border:0; border-radius:10px; height:2.6rem; padding:0 14px; font-weight:600;
}
.pred-pill{
  display:inline-block; padding:8px 14px; border-radius:999px;
  background:#ffffffcc; color:#111827; font-weight:700; border:1px solid #e5e7eb;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>Big Mart Sales Prediction</h1>", unsafe_allow_html=True)

BASE = Path(__file__).resolve().parent
scaler_path = BASE / "models" / "sc.sav"
model_path  = BASE / "models" / "rf.sav"

if not scaler_path.exists(): st.error(f"Scaler not found: {scaler_path}"); st.stop()
if not model_path.exists():  st.error(f"Model not found: {model_path}"); st.stop()

sc = joblib.load(scaler_path)
model = joblib.load(model_path)

fat_map = {"High Fat":0, "Low Fat":1, "Regular":2}
type_map = {
    "Baking Goods":0,"Breads":1,"Breakfast":2,"Canned":3,"Dairy":4,"Frozen Foods":5,
    "Fruits and Vegetables":6,"Hard Drinks":7,"Health and Hygiene":8,"Household":9,
    "Meat":10,"Others":11,"Seafood":12,"Snack Foods":13,"Soft Drinks":14,"Starchy Foods":15
}
size_map = {"High":0,"Medium":1,"Small":2}
loc_map  = {"Tier 1":0,"Tier 2":1,"Tier 3":2}
otype_map= {"Grocery Store":0,"Supermarket Type1":1,"Supermarket Type2":2,"Supermarket Type3":3}

col_left, col_right = st.columns([2, 2], gap="large")

with col_left:
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
        submit = st.form_submit_button("Submit")

if submit:
    X = np.array([[
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

    X_std = sc.transform(X)
    y_pred = float(model.predict(X_std)[0])

    with col_left:
        st.markdown(f'<div class="pred-pill">Predicted Sales: {y_pred:.2f}</div>', unsafe_allow_html=True)

    text_color = "#111827"
    tick_color = "#111827"

    with col_right:
        c1, c2 = st.columns(2)
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            names = [
                "Item_Weight","Item_Fat","Visibility","Item_Type","Item_MRP",
                "Outlet_Year","Outlet_Size","Outlet_Loc","Outlet_Type"
            ]
            order = np.argsort(importances)[::-1]
            fig1, ax1 = plt.subplots(figsize=(3.2, 3.2), facecolor="none")
            ax1.bar(range(len(importances)), importances[order], color="#1e3a8a")
            ax1.set_xticks(range(len(importances)))
            ax1.set_xticklabels(np.array(names)[order], rotation=60, ha="right", fontsize=7, color=tick_color)
            ax1.set_ylabel("Importance", fontsize=8, color=text_color)
            ax1.set_title("Feature Importances", fontsize=10, color=text_color, pad=6)
            ax1.set_facecolor("none")
            ax1.tick_params(axis='y', colors=tick_color, labelsize=8)
            c1.pyplot(fig1)

        mrp_grid = np.linspace(max(0.0, item_mrp*0.5), item_mrp*1.5 if item_mrp>0 else 300, 28)
        X_sweep = np.repeat(X, len(mrp_grid), axis=0)
        X_sweep[:, 4] = mrp_grid
        y_sweep = model.predict(sc.transform(X_sweep))

        fig2, ax2 = plt.subplots(figsize=(3.2, 3.2), facecolor="none")
        ax2.plot(mrp_grid, y_sweep, color="#2563eb", linewidth=2)
        ax2.set_xlabel("Item MRP", fontsize=8, color=text_color)
        ax2.set_ylabel("Predicted Sales", fontsize=8, color=text_color)
        ax2.set_title("Prediction vs. MRP", fontsize=10, color=text_color, pad=6)
        ax2.tick_params(axis='x', colors=tick_color, labelsize=8)
        ax2.tick_params(axis='y', colors=tick_color, labelsize=8)
        ax2.set_facecolor("none")
        c2.pyplot(fig2)
