from ml.drift_detector import DriftDetector

REFERENCE = [
    "The president signed the bill into law today",
    "Scientists discover new species in Amazon rainforest",
    "Stock markets rise after positive earnings reports",
    "Government announces new infrastructure spending plan",
    "University researchers publish climate change findings",
] * 20


def test_no_drift_on_same_distribution():
    detector = DriftDetector(REFERENCE, threshold=0.1)
    result = detector.check_drift(REFERENCE)
    assert result["drift_score"] < 0.05, "Same distribution should have near-zero drift"
    assert result["drift_detected"] is False


def test_drift_detected_on_shifted_distribution():
    detector = DriftDetector(REFERENCE, threshold=0.1)
    shifted = [
        "COVID vaccine causes autism according to leaked documents",
        "Aliens landed in Nevada government hiding truth from public",
        "5G towers spreading mind control chemicals proven by experts",
        "Banks secretly stealing money from ordinary citizens accounts",
        "Hollywood elites running secret underground child trafficking",
    ] * 20
    result = detector.check_drift(shifted)
    assert result["drift_detected"] is True, "Shifted distribution should trigger drift"


def test_update_reference_resets_drift():
    detector = DriftDetector(REFERENCE, threshold=0.1)
    shifted = ["fake news conspiracy theory virus hoax government"] * 30
    result = detector.check_drift(shifted)
    assert result["drift_detected"] is True
    detector.update_reference(shifted)
    result2 = detector.check_drift(shifted)
    assert result2["drift_score"] < 0.05


def test_drift_score_increases_with_shift():
    detector = DriftDetector(REFERENCE, threshold=0.1)
    severe_shift = [
        "Chemtrails poison population illuminati deep state fake election"
    ] * 20
    severe_result = detector.check_drift(severe_shift)
    same_result = detector.check_drift(REFERENCE)
    assert severe_result["drift_score"] > same_result["drift_score"]
