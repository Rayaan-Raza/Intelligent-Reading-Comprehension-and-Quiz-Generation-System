def evaluate(y_true, y_pred):
    """Simple accuracy metric placeholder."""
    if not y_true:
        return 0.0
    matches = sum(int(a == b) for a, b in zip(y_true, y_pred))
    return matches / len(y_true)
