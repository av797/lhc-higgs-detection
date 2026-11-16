import math

def calculate_ams(labels, predictions, weights, threshold):

    # Select events whose predicted signal probability meets the threshold
    selected = predictions >= threshold

    # Calculate weighted signal and background
    signal = weights[selected & (labels == 1)].sum().item()
    background = weights[selected & (labels == 0)].sum().item()

    # AMS regularisation term
    background_regularisation = 10.0

    # Calculate AMS
    radicand = 2 * (
        (signal + background + background_regularisation)
        * math.log1p(signal / (background + background_regularisation))
        - signal
    )

    return math.sqrt(max(radicand, 0.0))