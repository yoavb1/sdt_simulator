import numpy as np
import matplotlib.pyplot as plt
from core.sdt import compute_all_ev, Payoffs


def plot_ev_over_ps(
    ps_grid: np.ndarray,
    source_1_sensitivity: float,
    source_2_sensitivity: float,
    DSS1_sensitivity: float,
    DSS2_sensitivity: float,
    payoffs: Payoffs,
    DSS1_cost: float = 0.0,
    DSS2_cost: float = 0.0,
):
    """
    Plot Expected Value (EV) as a function of Ps for each scenario.
    """

    # Collect EV curves
    ev_curves = {}

    for Ps in ps_grid:
        results = compute_all_ev(
            Ps=Ps,
            source_1_sensitivity=source_1_sensitivity,
            source_2_sensitivity=source_2_sensitivity,
            DSS1_sensitivity=DSS1_sensitivity,
            DSS2_sensitivity=DSS2_sensitivity,
            payoffs=payoffs,
            DSS1_cost=DSS1_cost,
            DSS2_cost=DSS2_cost,
        )

        for scenario, ev in results.items():
            ev_curves.setdefault(scenario, []).append(ev)

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 5))

    for scenario, evs in ev_curves.items():
        ax.plot(ps_grid, evs, label=scenario)

    ax.set_xlabel("Prior probability of signal (Ps)")
    ax.set_ylabel("Expected Value")
    ax.set_title("Expected Value vs Prior Probability (Ps)")
    ax.legend()
    ax.grid(True)

    return fig
