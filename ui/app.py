# ui/app.py
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np

from core.sdt import Payoffs, compute_all_ev

# -------------------------------------------------
# Page configuration
# -------------------------------------------------
st.set_page_config(page_title="SDT Human + DSS Simulation", layout="wide")

st.title("Alarm Flood Simulation")

st.markdown("""
Simulate an alarm flood control-room scenario involving a human operator and one or two alarm systems.  
The simulation is based on **Signal Detection Theory (SDT)**.

You can:
- **Compute expected values (EVs)** for a single configuration, or
- **Plot EVs** as a function of one model parameter while keeping all others fixed.
""")

# -------------------------------------------------
# Sidebar — Mode
# -------------------------------------------------
st.sidebar.header("Simulation Mode")

mode = st.sidebar.radio(
    "Select functionality",
    ["Compute Expected Values", "Plot EV vs Parameter"]
)

# -------------------------------------------------
# Sidebar — Base parameters
# -------------------------------------------------
st.sidebar.header("Simulation Parameters")

Ps = st.sidebar.slider("Prior probability to failure (Ps)", 0.0, 1.0, 0.2, 0.01)

source_1_sensitivity = st.sidebar.slider(
    "Temperature sensitivity", 0.1, 5.0, 1.5, 0.1
)

source_2_sensitivity = st.sidebar.slider(
    "Humidity sensitivity", 0.1, 5.0, 1.5, 0.1
)

DSS1_sensitivity = st.sidebar.slider(
    "System 1 sensitivity", 0.0, 5.0, 1.5, 0.1
)

DSS2_sensitivity = st.sidebar.slider(
    "System 2 sensitivity", 0.0, 5.0, 1.5, 0.1
)

# -------------------------------------------------
# Payoffs
# -------------------------------------------------
with st.sidebar.expander("💰 Payoffs", expanded=False):
    V_TP = st.number_input("True Positive", value=1)
    V_FP = st.number_input("False Positive", value=-1)
    V_FN = st.number_input("False Negative", value=-2)
    V_TN = st.number_input("True Negative", value=1)

payoffs = Payoffs(V_TP=V_TP, V_FP=V_FP, V_FN=V_FN, V_TN=V_TN)

# -------------------------------------------------
# Costs
# -------------------------------------------------
with st.sidebar.expander("💸 System Costs", expanded=False):
    DSS1_cost = st.number_input("System 1 cost", value=0.0, step=0.1)
    DSS2_cost = st.number_input("System 2 cost", value=0.0, step=0.1)

# =================================================
# MODE 1 — Single-point EV
# =================================================
if mode == "Compute Expected Values":

    if st.sidebar.button("Compute Expected Values"):

        results = compute_all_ev(
            Ps=Ps,
            source_1_sensitivity=source_1_sensitivity,
            source_2_sensitivity=source_2_sensitivity,
            DSS1_sensitivity=DSS1_sensitivity,
            DSS2_sensitivity=DSS2_sensitivity,
            payoffs=payoffs,
            DSS1_cost=DSS1_cost,
            DSS2_cost=DSS2_cost
        )

        st.subheader("📊 Expected Values")

        ev_df = (
            pd.DataFrame([results])
            .T
            .reset_index()
            .rename(columns={"index": "Scenario", 0: "Expected Value"})
        )

        cols = st.columns(len(ev_df))
        for col, (_, row) in zip(cols, ev_df.iterrows()):
            col.metric(row["Scenario"], f"{row['Expected Value']:.2f}")

        st.markdown("### Full Results Table")
        st.dataframe(ev_df, use_container_width=True)

# =================================================
# MODE 2 — Plot EV vs selected parameter
# =================================================
else:
    st.subheader("📈 Expected Value as a Function of a Model Parameter")

    parameter = st.selectbox(
        "Select parameter to vary",
        [
            "Prior probability (Ps)",
            "Temperature sensitivity",
            "Humidity sensitivity",
            "System 1 sensitivity",
            "System 2 sensitivity",
        ]
    )

    x_min, x_max = st.slider(
        "Parameter range",
        0.0, 5.0, (0.1, 3.0), 0.05
    )

    n_points = st.slider("Number of points", 10, 200, 50)

    if st.sidebar.button("Generate Plot"):

        grid = np.linspace(x_min, x_max, n_points)
        records = []

        for x in grid:

            params = dict(
                Ps=Ps,
                source_1_sensitivity=source_1_sensitivity,
                source_2_sensitivity=source_2_sensitivity,
                DSS1_sensitivity=DSS1_sensitivity,
                DSS2_sensitivity=DSS2_sensitivity,
                payoffs=payoffs,
                DSS1_cost=DSS1_cost,
                DSS2_cost=DSS2_cost
            )

            # Override selected parameter
            if parameter == "Prior probability (Ps)":
                params["Ps"] = x
            elif parameter == "Temperature sensitivity":
                params["source_1_sensitivity"] = x
            elif parameter == "Humidity sensitivity":
                params["source_2_sensitivity"] = x
            elif parameter == "System 1 sensitivity":
                params["DSS1_sensitivity"] = x
            elif parameter == "System 2 sensitivity":
                params["DSS2_sensitivity"] = x

            results = compute_all_ev(**params)

            for scenario, ev in results.items():
                records.append({
                    "Parameter value": x,
                    "Scenario": scenario,
                    "EV": ev
                })

        plot_df = pd.DataFrame(records)

        st.line_chart(
            plot_df,
            x="Parameter value",
            y="EV",
            color="Scenario",
            use_container_width=True
        )

        st.markdown("""
        **Interpretation:**  
        Each curve shows how the expected value of a decision strategy changes
        as the selected parameter varies, while all other parameters are held constant.
        """)
