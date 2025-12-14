# ui/app.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
from core.sdt import Payoffs, compute_all_ev

st.set_page_config(page_title="SDT Human + DSS Simulation", layout="wide")
st.title("SDT Human + DSS Simulation")

st.markdown("""
This app simulates a human decision-maker interacting with one or two Decision Support Systems (DSS) using Signal Detection Theory (SDT).  
You can adjust **priors**, **sensitivities**, **payoffs**, and **DSS costs**, then compute expected values (EVs).
""")

# -----------------------------
# Sidebar / Inputs
# -----------------------------
st.sidebar.header("Simulation Parameters")

Ps = st.sidebar.slider("Prior probability of signal (Ps)", 0.0, 1.0, 0.2)
human_sensitivity = st.sidebar.slider("Human sensitivity (d')", 0.1, 5.0, 1.5)
DSS1_sensitivity = st.sidebar.slider("DSS1 sensitivity (d')", 0.0, 5.0, 1.5)
DSS2_sensitivity = st.sidebar.slider("DSS2 sensitivity (d')", 0.0, 5.0, 1.5)

# Optional settings
with st.sidebar.expander("Payoffs"):
    V_TP = st.number_input("V_TP", value=1)
    V_FP = st.number_input("V_FP", value=-1)
    V_FN = st.number_input("V_FN", value=-2)
    V_TN = st.number_input("V_TN", value=1)

payoffs = Payoffs(V_TP=V_TP, V_FP=V_FP, V_FN=V_FN, V_TN=V_TN)

with st.sidebar.expander("DSS Costs"):
    DSS1_cost = st.number_input("DSS1 cost", value=0.0)
    DSS2_cost = st.number_input("DSS2 cost", value=0.0)

# -----------------------------
# Compute EV
# -----------------------------
if st.button("Compute Expected Values"):

    results = compute_all_ev(
        Ps=Ps,
        human_sensitivity=human_sensitivity,
        DSS1_sensitivity=DSS1_sensitivity,
        DSS2_sensitivity=DSS2_sensitivity,
        payoffs=payoffs,
        DSS1_cost=DSS1_cost,
        DSS2_cost=DSS2_cost
    )

    # Display results in a table
    st.subheader("Expected Values")
    ev_df = pd.DataFrame([results]).T.reset_index()
    ev_df.columns = ["Scenario", "Expected Value"]
    st.table(ev_df)