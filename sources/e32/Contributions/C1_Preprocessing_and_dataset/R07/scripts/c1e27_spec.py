"""C1-E27 frozen scientific scope, migrated from R03; no execution on import."""
from __future__ import annotations

import hashlib
import c1e25 as base
import c1e26 as extension

EXPERIMENT = "C1-E27"
PROTOCOL = "C1-E27-R03-completion-v1"
FS = 40000
SAMPLES = 8000
NOISES = ("awgn", "lowfreq", "impulsive", "gain_plus_awgn", "mixed_gain_additive")
CONDITIONS = ("clean",) + tuple(f"{noise}__{snr}" for noise in NOISES for snr in base.SNR_LEVELS)
PREPROCESSORS = ("none", "savgol", "median", "wavelet_soft", "wavelet_hard")
MODELS = base.MODELS + extension.MODELS
MODEL_CONTRACTS = {**base.MODEL_CONTRACTS, **extension.MODEL_CONTRACTS}
NORMALIZERS = ("none", "standard", "minmax_0_1", "robust", "power_yeo_johnson", "quantile_normal")
SCALED = ("logistic_regression",) + extension.SCALED
EXTRA_MODELS = ("logistic_regression", "random_forest", "naive_bayes", "svm")
EXTRA_CONDITIONS = ("clean",) + tuple(f"{noise}__{snr}" for noise in
    ("awgn", "impulsive", "mixed_gain_additive") for snr in (10, 20))
FEATURES = base.FEATURES + [channel + "_" + statistic for channel in base.CHANNELS
                           for statistic in ("skew", "kurtosis", "crest_factor")]
REPRESENTATIONS = {**base.REPRESENTATIONS, "RAW_PLUS": list(range(6, 36)), "MULTI_PLUS": list(range(36))}
LIMITS = base.LIMITS
VERSIONS = {**base.VERSIONS, "PyWavelets": "1.8.0"}
WHEEL_SHA256 = "810a23a631da596fef7196ddec49b345b1aab13525bb58547eeebe1769edbbc1"
SIGNAL_CONTRACT = {
    "sampling_rate_hz": FS, "samples_per_window": SAMPLES,
    "axis": "time_by_six_channels", "seed": 42,
    "lowfreq_cutoff_hz": 5.0, "lowfreq_order": 4,
    "lowfreq_filter": "Butterworth_SOS_sosfiltfilt_odd_pad15",
    "impulse_start_rate_per_second": 100.0, "impulse_width_samples": 4,
    "at_least_one_impulse_per_window_channel": True, "gain_std": 0.05,
    "mixed_component_rms_weights": [1.0, 1.0, 1.0],
    "composite_snr_reference": "additive_error_relative_to_gain_perturbed_base_not_total_error",
    "shared_noise_direction_across_snr": True, "shared_corruption_across_filters": True,
    "awgn_identity": "EXACT_C1_E25_IDENTITY_AND_IMPLEMENTATION",
    "median_kernel_samples": 17, "median_boundary": "zero_padding_R03",
    "savgol_window_samples": 41, "savgol_polyorder": 3, "savgol_boundary": "mirror_C1_E25",
    "wavelet": "db4", "wavelet_level": 5, "wavelet_mode": "symmetric",
    "wavelet_threshold": "per_channel_finest_detail_centered_MAD_div_0.6745_times_sqrt_2logN",
    "tpi_primary": "db4_periodization_D3_nominal_2500_5000Hz",
    "tpi_sensitivities": ["db4_periodization_D1_nominal_10000_20000Hz", "FFT_band_2500_5000Hz"],
    "tpi_denominator_epsilon": 1e-12, "tpi_clipping": False,
    "std_ddof": 1, "rms_epsilon": 1e-12, "skew_bias": False,
    "kurtosis_fisher": False, "kurtosis_bias": False,
    "constant_channel_std_threshold": 1e-12,
    "constant_channel_skew_kurtosis": [0.0, 3.0], "zero_channel_crest": 0.0,
}
STATISTICS = {"seed": 42, "cluster": "acquisition_id_paired_windows",
    "bootstrap_resamples": 10000, "paired_swaps": 50000, "alpha": 0.05,
    "primary_metric": "macro_f1", "secondary_metrics": ["balanced_accuracy", "mcc"],
    "families": ["early_vs_RAW", "early_vs_RMS", "late_simple_vs_RAW", "late_simple_vs_RMS",
                 "denoiser_vs_none", "RAW_PLUS_vs_RAW", "MULTI_PLUS_vs_MULTI18", "normalizer_vs_none"],
    "multiplicity": "Holm_within_declared_family", "prior_E25_E26_inspection": "exploratory_extensions",
    "tpi_association": "exploratory_condition_level_not_causal", "oracle": "descriptive_only"}
STATISTICS.update(master_seed_derivation="SHA256_of_master_seed42_and_case_or_comparison_identity",
    paired_swap_absolute_tie_tolerance=1e-14, mc_interval_confidence=0.99,
    mc_interval_scope="marginal_not_simultaneous",
    nonconverged_family_members="retained_as_p1_for_Holm_but_not_reported_as_evaluated_tests")


def digest(value):
    return hashlib.sha256(base.canonical(value)).hexdigest()


def condition(value):
    if value not in CONDITIONS:
        raise ValueError("Condition outside frozen C1-E27 scope")
    if value == "clean":
        return "clean", None
    noise, snr = value.rsplit("__", 1)
    return noise, int(snr)


def default_normalizer(model):
    if model not in MODELS:
        raise ValueError("Unreviewed model")
    return "standard" if model in SCALED else "none"


def jobs(preprocessor, fold, model):
    if preprocessor not in PREPROCESSORS or fold not in range(1, 6) or model not in MODELS:
        raise ValueError("Invalid learning unit identity")
    result = []
    def add(study, representation, training_condition, normalizer):
        reused = (study == "robustness" and training_condition == "clean" and
                  preprocessor in ("none", "savgol"))
        identity = {"study": study, "representation": representation, "training_condition": training_condition,
                    "normalizer": normalizer, "preprocessor": preprocessor, "fold": fold, "classifier": model}
        result.append({**identity, "job_id": digest(identity), "reuse_E25_E26": reused})
    for name in base.REPRESENTATIONS:
        for training_condition in CONDITIONS:
            add("robustness", name, training_condition, default_normalizer(model))
    if preprocessor == "none":
        if model in EXTRA_MODELS:
            for name in ("RAW_PLUS", "MULTI_PLUS"):
                for training_condition in EXTRA_CONDITIONS:
                    add("raw_plus", name, training_condition, default_normalizer(model))
        for name in base.REPRESENTATIONS:
            for normalizer in NORMALIZERS:
                if normalizer != default_normalizer(model):
                    add("normalization", name, "clean", normalizer)
    return result


def evaluation_cases(job):
    if job["study"] == "normalization":
        return [("normalization", "clean")]
    if job["training_condition"] != "clean":
        return [("matched", job["training_condition"])]
    cases = CONDITIONS if job["study"] == "robustness" else EXTRA_CONDITIONS
    return [("deployment", c) for c in cases] + [("matched", "clean")]


def learning_plan():
    result = []
    for p, preprocessor in enumerate(PREPROCESSORS):
        for fold in range(1, 6):
            for m, model in enumerate(MODELS):
                unit_jobs = jobs(preprocessor, fold, model)
                result.append({"stage": "B", "preprocessor": preprocessor, "fold": fold, "classifier": model,
                    "role": "PC1" if (p + fold + m) % 2 == 0 else "PC2",
                    "expected_fits": sum(not j["reuse_E25_E26"] for j in unit_jobs),
                    "reused_fits": sum(j["reuse_E25_E26"] for j in unit_jobs)})
    return result


def scope():
    plan = learning_plan()
    return {"experiment_id": EXPERIMENT, "protocol": PROTOCOL, "signal_contract": SIGNAL_CONTRACT,
        "statistics": STATISTICS, "model_contracts": MODEL_CONTRACTS, "normalizers": list(NORMALIZERS),
        "conditions": list(CONDITIONS), "preprocessors": list(PREPROCESSORS),
        "base_representations": list(base.REPRESENTATIONS), "extra_representations": ["RAW_PLUS", "MULTI_PLUS"],
        "extra_models": list(EXTRA_MODELS), "extra_conditions": list(EXTRA_CONDITIONS),
        "labels": base.LABELS, "feature_columns": FEATURES, "scientific_limits": LIMITS,
        "versions": VERSIONS, "new_fit_budget": sum(u["expected_fits"] for u in plan),
        "convergence_policy": "R03_fixed_grid_record_warning_exclude_affected_pooled_cases_and_comparisons_no_retry",
        "integrity_or_fit_exception_policy": "STOP_PRESERVE_NO_AUTOMATIC_RETRY",
        "reused_fit_budget": sum(u["reused_fits"] for u in plan),
        "new_fits_per_role": {r: sum(u["expected_fits"] for u in plan if u["role"] == r) for r in ("PC1", "PC2")},
        "preprocessing_units": len(CONDITIONS), "learning_units": len(plan),
        "main_pooled_cells_including_fusions": 2 * len(CONDITIONS) * 5 * 8 * 5,
        "raw_plus_extra_pooled_cells": 2 * len(EXTRA_CONDITIONS) * 4 * 2,
        "normalization_extra_pooled_cells": 5 * 8 * 3,
        "oracle_policy": "R03_weights_0.7_RMS_at_SNR_le20_else_0.3_ORACLE_ONLY",
        "no_new_DL_baseline_authorized": True, "further_snr_refinement": "DEFERRED"}


def configuration(stage, inputs, **identity):
    value = {"experiment_id": EXPERIMENT, "protocol": PROTOCOL, "scope_sha256": digest(scope()),
             "stage": stage, "inputs": inputs, "scientific_limits": LIMITS, **identity}
    if stage == "A":
        condition(value.get("condition"))
        value["expected_fits"] = 0
    elif stage == "B":
        value["expected_fits"] = sum(not j["reuse_E25_E26"] for j in
            jobs(value["preprocessor"], value["fold"], value["classifier"]))
    else:
        raise ValueError("Unknown scientific stage")
    return value


def validate_configuration(config, expected_fits):
    expected = configuration(config["stage"], config["inputs"], **{k: config[k] for k in
        (("condition",) if config["stage"] == "A" else ("preprocessor", "fold", "classifier"))})
    if config != expected or expected_fits != expected["expected_fits"]:
        raise ValueError("Scientific configuration or fit budget differs from frozen scope")
