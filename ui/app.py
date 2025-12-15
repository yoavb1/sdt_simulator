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
    ["Compute Expected Values", "Plot EV vs Parameter", "Plot EV vs 2 Parameters"]
)

# -------------------------------------------------
# Sidebar — Base parameters
# -------------------------------------------------
st.sidebar.header("Simulation Parameters")

Ps = st.sidebar.slider("Prior probability to failure (Ps)", 0.01, 1.0, 0.2, 0.01)

source_1_sensitivity = st.sidebar.slider(
    "Temperature sensitivity", 0.5, 5.0, 1.5, 0.1
)

source_2_sensitivity = st.sidebar.slider(
    "Humidity sensitivity", 0.5, 5.0, 1.5, 0.1
)

DSS1_sensitivity = st.sidebar.slider(
    "System 1 sensitivity", 0.5, 5.0, 1.5, 0.1
)

DSS2_sensitivity = st.sidebar.slider(
    "System 2 sensitivity", 0.5, 5.0, 1.5, 0.1
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
invalid_payoff = (V_FN - V_TP) == 0

if invalid_payoff:
    st.sidebar.error("⚠️ Invalid payoffs: V_FN - V_TP cannot be 0.")
    compute_disabled = True
else:
    compute_disabled = False

# -------------------------------------------------
# Costs
# -------------------------------------------------
with st.sidebar.expander("💸 System Costs", expanded=False):
    DSS1_cost = st.number_input("System 1 cost", value=0.0, step=0.1)
    DSS2_cost = st.number_input("System 2 cost", value=0.0, step=0.1)

# =================================================
# MODE 1 — Single-point EV
# =================================================
if compute_disabled:
    st.markdown("""
    **Payoffs ratio denominator can not be 0**  
    """)
elif mode == "Compute Expected Values":

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
elif mode == "Plot EV vs Parameter":
    st.subheader("📈 Expected Value as a Function of a Model Parameter")

    PARAM_SPECS = {
        "Prior probability (Ps)": {
            "key": "Ps",
            "min": 0.01,
            "max": 0.99,
            "default": (0.05, 0.95),
            "step_default": 0.01
        },
        "Temperature sensitivity": {
            "key": "source_1_sensitivity",
            "min": 0.5,
            "max": 5.0,
            "default": (0.5, 3.0),
            "step_default": 0.1
        },
        "Humidity sensitivity": {
            "key": "source_2_sensitivity",
            "min": 0.5,
            "max": 5.0,
            "default": (0.5, 3.0),
            "step_default": 0.1
        },
        "System 1 sensitivity": {
            "key": "DSS1_sensitivity",
            "min": 0.5,
            "max": 5.0,
            "default": (0.5, 3.0),
            "step_default": 0.1
        },
        "System 2 sensitivity": {
            "key": "DSS2_sensitivity",
            "min": 0.5,
            "max": 5.0,
            "default": (0.5, 3.0),
            "step_default": 0.1
        },
    }

    parameter = st.selectbox(
        "Select parameter to vary",
        list(PARAM_SPECS.keys())
    )

    spec = PARAM_SPECS[parameter]

    x_min, x_max = st.slider(
        "Parameter range",
        min_value=spec["min"],
        max_value=spec["max"],
        value=spec["default"]
    )

    step = st.slider(
        "Step size",
        min_value=spec["step_default"] / 10,
        max_value=spec["step_default"] * 5,
        value=spec["step_default"]
    )

    # Step-based grid (correct)
    grid = np.arange(x_min, x_max + step, step)

    if st.button("Generate Plot"):

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

            # Override the selected parameter (clean & generic)
            params[spec["key"]] = float(x)

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


# =================================================
# MODE 3 — 3D Surface EV vs two parameters
# =================================================
elif mode == "Plot EV vs 2 Parameters":
    st.subheader("📊 3D Expected Value Surface")

    param_options = [
        "Prior probability (Ps)",
        "Temperature sensitivity",
        "Humidity sensitivity",
        "System 1 sensitivity",
        "System 2 sensitivity"
    ]

    param_x = st.selectbox("Select X-axis parameter", param_options, key="x3d")
    param_y = st.selectbox("Select Y-axis parameter", [p for p in param_options if p != param_x], key="y3d")

    # Define parameter specs as before
    PARAM_SPECS = {
        "Prior probability (Ps)": {"key": "Ps", "min": 0.01, "max": 0.99, "step": 0.01},
        "Temperature sensitivity": {"key": "source_1_sensitivity", "min": 0.5, "max": 5.0, "step": 0.1},
        "Humidity sensitivity": {"key": "source_2_sensitivity", "min": 0.5, "max": 5.0, "step": 0.1},
        "System 1 sensitivity": {"key": "DSS1_sensitivity", "min": 0.5, "max": 5.0, "step": 0.1},
        "System 2 sensitivity": {"key": "DSS2_sensitivity", "min": 0.5, "max": 5.0, "step": 0.1},
    }

    # Ranges for X and Y
    x_min, x_max = st.slider(
        f"Range for {param_x}",
        PARAM_SPECS[param_x]["min"],
        PARAM_SPECS[param_x]["max"],
        (PARAM_SPECS[param_x]["min"], PARAM_SPECS[param_x]["max"]),
        step=PARAM_SPECS[param_x]["step"]
    )

    y_min, y_max = st.slider(
        f"Range for {param_y}",
        PARAM_SPECS[param_y]["min"],
        PARAM_SPECS[param_y]["max"],
        (PARAM_SPECS[param_y]["min"], PARAM_SPECS[param_y]["max"]),
        step=PARAM_SPECS[param_y]["step"]
    )

    step_x = PARAM_SPECS[param_x]["step"]
    step_y = PARAM_SPECS[param_y]["step"]

    if st.button("Generate 3D Surface"):

        x_grid = np.arange(x_min, x_max + step_x, step_x)
        y_grid = np.arange(y_min, y_max + step_y, step_y)
        X, Y = np.meshgrid(x_grid, y_grid)

        # Initialize Z grid for a selected scenario (let's pick "Human only" as default)
        Z = np.zeros_like(X)

        for i in range(len(y_grid)):
            for j in range(len(x_grid)):
                # Current parameter values
                params = {
                    "Ps": Ps,
                    "source_1_sensitivity": source_1_sensitivity,
                    "source_2_sensitivity": source_2_sensitivity,
                    "DSS1_sensitivity": DSS1_sensitivity,
                    "DSS2_sensitivity": DSS2_sensitivity,
                    "payoffs": payoffs,
                    "DSS1_cost": DSS1_cost,
                    "DSS2_cost": DSS2_cost
                }

                # Override selected parameters
                params[PARAM_SPECS[param_x]["key"]] = x_grid[j]
                params[PARAM_SPECS[param_y]["key"]] = y_grid[i]

                result = compute_all_ev(**params)
                # Example: plot first scenario (you can add selection if needed)
                Z[i, j] = list(result.values())[0]

        # Plot
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")

        ax.set_xlabel(param_x)
        ax.set_ylabel(param_y)
        ax.set_zlabel("Expected Value")
        ax.set_title("3D Expected Value Surface")
        fig.colorbar(surf, shrink=0.5, aspect=10)

        st.pyplot(fig)
