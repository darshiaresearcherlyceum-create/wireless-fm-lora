"""
src/metrics package
Physical-layer evaluation metrics across tasks (CE, DET, PREC, DEC, LOC).
"""
from src.eval.metrics import (
    evaluate_channel_estimation_nmse,
    evaluate_mimo_detection_ber,
    evaluate_precoding_sum_rate,
    compute_classical_rzf_precoder,
    compute_classical_zf_precoder,
    evaluate_channel_decoding,
    evaluate_user_localization_rmse,
    evaluate_user_localization_error_distance
)

__all__ = [
    "evaluate_channel_estimation_nmse",
    "evaluate_mimo_detection_ber",
    "evaluate_precoding_sum_rate",
    "compute_classical_rzf_precoder",
    "compute_classical_zf_precoder",
    "evaluate_channel_decoding",
    "evaluate_user_localization_rmse",
    "evaluate_user_localization_error_distance"
]
