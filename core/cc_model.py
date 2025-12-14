import numpy as np

def simulate_trial(signal, human_d, dss_d, human_criterion):
    """
    signal: 0 (noise) or 1 (signal)
    """
    # Internal responses
    human_x = np.random.normal(dss_d if signal else 0, 1)
    dss_x = np.random.normal(dss_d if signal else 0, 1)

    human_decision = human_x > human_criterion
    dss_decision = dss_x > 0

    return human_decision, dss_decision
