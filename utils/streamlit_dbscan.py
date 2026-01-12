import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import json
import sys
from pathlib import Path

# 1. SETUP & DATA LOADING
st.set_page_config(page_title="STL Quality Control", layout="wide")
st.title("STL Segmentation Outlier Check")

# Argument Handling: Check if a path was passed after the "--" separator
if len(sys.argv) > 1:
    # We take the last argument in case Streamlit injects its own
    json_path_input = sys.argv[-1]
else:
    # Default fallback path
    json_path_input = r"C:\Users\schum\Downloads\stl_metadata.json"

@st.cache_data
def load_data(path_str):
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
        
    with open(path, 'r') as f:
        data_dict = json.load(f)
        
    df = pd.DataFrame.from_dict(data_dict, orient='index').reset_index()
    df.rename(columns={'index': 'obj_id'}, inplace=True)

    # Filtering and Unit Conversion
    features = ['obj_id', 'Mesh_volume_mm3', 'Surface_Area_mm2']
    df = df[features].copy()
    df['volume'] = df['Mesh_volume_mm3'] / 1000  # mm3 to cm3
    df['surface_area'] = df['Surface_Area_mm2'] / 100  # mm2 to cm2
    return df

try:
    df = load_data(json_path_input)
    st.sidebar.success(f"Loaded: {Path(json_path_input).name}")
except Exception as e:
    st.error(f"Error loading JSON: {e}")
    st.info("Usage: streamlit run script.py -- \"path/to/data.json\"")
    st.stop()

# 2. SIDEBAR CONTROLS
st.sidebar.header("Clustering Settings")
eps = st.sidebar.slider("Epsilon (Strictness)", 0.001, 0.3, 0.07, step=0.001, help="Lower = stricter", format="%.3f") 
min_samples = st.sidebar.number_input("Min Cluster Size", value=10)

# 3. PROCESSING
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['volume', 'surface_area']])
db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scaled)

df['status'] = ['Outlier' if l == -1 else f'Cluster {l}' for l in db.labels_]

# 4. SEARCH & AUTO-ZOOM LOGIC
st.sidebar.header("Search & Inspect")
search_id = st.sidebar.selectbox("Find Object ID:", options=[""] + list(df['obj_id']))

# Toggle for the big yellow circle
show_highlight = st.sidebar.checkbox("Show Highlight Circle", value=True)

if search_id:
    target = df[df['obj_id'] == search_id].iloc[0]
    x_range = [target['volume'] * 0.8, target['volume'] * 1.2]
    y_range = [target['surface_area'] * 0.8, target['surface_area'] * 1.2]
    st.sidebar.info(f"Focused on {search_id}")
else:
    x_range = [df['volume'].min() * 0.9, df['volume'].max() * 1.1]
    y_range = [df['surface_area'].min() * 0.9, df['surface_area'].max() * 1.1]

# 5. VISUALIZATION
fig = px.scatter(
    df, x="volume", y="surface_area", color="status",
    hover_name="obj_id",
    hover_data={"status": True, "volume": ":.2f", "surface_area": ":.2f"},
    template="plotly_dark",
    color_discrete_map={'Outlier': "#AC0D0D"}
)

fig.update_layout(
    xaxis_range=x_range,
    yaxis_range=y_range,
    transition_duration=500,
    height=700, 
    xaxis_title="Volume (cm³)",
    yaxis_title="Surface Area (cm²)"
)

# Add highlight only if searched AND toggle is ON
if search_id and show_highlight:
    fig.add_scatter(
        x=[target['volume']], y=[target['surface_area']],
        mode='markers',
        marker=dict(size=20, color='yellow', symbol='circle-open', line=dict(width=3)),
        name="Selected Part",
        showlegend=False
    )

st.plotly_chart(fig, width="stretch")

# 6. DATA TABLE
st.subheader("Detected Outliers")
outliers = df[df['status'] == 'Outlier']
st.dataframe(outliers, width="stretch")