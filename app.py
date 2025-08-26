import streamlit as st
import numpy as np
import joblib
from pathlib import Path

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
  border:0; border-radius:8px; height:2.5rem; width:100px; margin:0 5px;
  font-weight:600;
}
.reset-btn > button { background:#ef4444 !important; }
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
        reset = st.form_submit_button("Reset")

# -------------------- PREDICT --------------------
if submitted:
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

    st.success(f"Predicted Sales: **{y_pred:.2f}**")
