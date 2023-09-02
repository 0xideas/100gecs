import json
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import pytest
from numpy import float64, ndarray


@pytest.fixture(scope="session")
def seed():
    return 102


@pytest.fixture(scope="session")
def X(seed):
    np.random.seed(seed)
    return np.random.randn(300, 3)


@pytest.fixture(scope="session")
def y_real(X, seed):
    return X.sum(1) ** 2 + np.random.uniform(0, 1, X.shape[0])


@pytest.fixture(scope="session")
def y_class(y_real, seed):
    np.random.seed(seed)
    return np.array([min(4, yy.astype(int)) for yy in y_real])


@pytest.fixture(scope="session")
def lightgecs_params():
    return {
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "max_depth": -1,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "subsample_for_bin": 200000,
        "objective": None,
        "class_weight": None,
        "min_split_gain": 0.0,
        "min_child_weight": 1e-3,
        "min_child_samples": 20,
        "subsample": 1.0,
        "subsample_freq": 0,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "random_state": None,
        "n_jobs": -1,
        "silent": "warn",
        "importance_type": "split",
        "frozen": False,
    }

@pytest.fixture(scope="session")
def catgec_params():
    return {
        "iterations": None,
        "learning_rate": None,
        "depth": None,
        "l2_leaf_reg": None,
        "model_size_reg": None,
        "rsm": None,
        "loss_function": None,
        "border_count": None,
        "feature_border_type": None,
        "per_float_feature_quantization": None,
        "input_borders": None,
        "output_borders": None,
        "fold_permutation_block": None,
        "od_pval": None,
        "od_wait": None,
        "od_type": None,
        "nan_mode": None,
        "counter_calc_method": None,
        "leaf_estimation_iterations": None,
        "leaf_estimation_method": None,
        "thread_count": None,
        "random_seed": None,
        "use_best_model": None,
        "best_model_min_trees": None,
        "verbose": None,
        "silent": None,
        "logging_level": None,
        "metric_period": None,
        "ctr_leaf_count_limit": None,
        "store_all_simple_ctr": None,
        "max_ctr_complexity": None,
        "has_time": None,
        "allow_const_label": None,
        "target_border": None,
        "classes_count": None,
        "class_weights": None,
        "auto_class_weights": None,
        "class_names": None,
        "one_hot_max_size": None,
        "random_strength": None,
        "name": None,
        "ignored_features": None,
        "train_dir": None,
        "custom_loss": None,
        "custom_metric": None,
        "eval_metric": None,
        "bagging_temperature": None,
        "save_snapshot": None,
        "snapshot_file": None,
        "snapshot_interval": None,
        "fold_len_multiplier": None,
        "used_ram_limit": None,
        "gpu_ram_part": None,
        "pinned_memory_size": None,
        "allow_writing_files": None,
        "final_ctr_computation_mode": None,
        "approx_on_full_history": None,
        "boosting_type": None,
        "simple_ctr": None,
        "combinations_ctr": None,
        "per_feature_ctr": None,
        "ctr_description": None,
        "ctr_target_border_count": None,
        "task_type": None,
        "device_config": None,
        "devices": None,
        "bootstrap_type": None,
        "subsample": None,
        "mvs_reg": None,
        "sampling_unit": None,
        "sampling_frequency": None,
        "dev_score_calc_obj_block_size": None,
        "dev_efb_max_buckets": None,
        "sparse_features_conflict_fraction": None,
        "max_depth": None,
        "n_estimators": None,
        "num_boost_round": None,
        "num_trees": None,
        "colsample_bylevel": None,
        "random_state": None,
        "reg_lambda": None,
        "objective": None,
        "eta": None,
        "max_bin": None,
        "scale_pos_weight": None,
        "gpu_cat_features_storage": None,
        "data_partition": None,
        "metadata": None,
        "early_stopping_rounds": None,
        "cat_features": None,
        "grow_policy": None,
        "min_data_in_leaf": None,
        "min_child_samples": None,
        "max_leaves": None,
        "num_leaves": None,
        "score_function": None,
        "leaf_estimation_backtracking": None,
        "ctr_history_unit": None,
        "monotone_constraints": None,
        "feature_weights": None,
        "penalties_coefficient": None,
        "first_feature_use_penalties": None,
        "per_object_feature_penalties": None,
        "model_shrink_rate": None,
        "model_shrink_mode": None,
        "langevin": None,
        "diffusion_temperature": None,
        "posterior_sampling": None,
        "boost_from_average": None,
        "text_features": None,
        "tokenizers": None,
        "dictionaries": None,
        "feature_calcers": None,
        "text_processing": None,
        "embedding_features": None,
        "callback": None,
        "eval_fraction": None,
        "fixed_binary_splits": None,
        "frozen": False,
    }

@pytest.fixture(scope="session")
def catger_params():
    return {
        "iterations": None,
        "learning_rate": None,
        "depth": None,
        "l2_leaf_reg": None,
        "model_size_reg": None,
        "rsm": None,
        "loss_function": "RMSE",
        "border_count": None,
        "feature_border_type": None,
        "per_float_feature_quantization": None,
        "input_borders": None,
        "output_borders": None,
        "fold_permutation_block": None,
        "od_pval": None,
        "od_wait": None,
        "od_type": None,
        "nan_mode": None,
        "counter_calc_method": None,
        "leaf_estimation_iterations": None,
        "leaf_estimation_method": None,
        "thread_count": None,
        "random_seed": None,
        "use_best_model": None,
        "best_model_min_trees": None,
        "verbose": None,
        "silent": None,
        "logging_level": None,
        "metric_period": None,
        "ctr_leaf_count_limit": None,
        "store_all_simple_ctr": None,
        "max_ctr_complexity": None,
        "has_time": None,
        "allow_const_label": None,
        "target_border": None,
        "one_hot_max_size": None,
        "random_strength": None,
        "name": None,
        "ignored_features": None,
        "train_dir": None,
        "custom_metric": None,
        "eval_metric": None,
        "bagging_temperature": None,
        "save_snapshot": None,
        "snapshot_file": None,
        "snapshot_interval": None,
        "fold_len_multiplier": None,
        "used_ram_limit": None,
        "gpu_ram_part": None,
        "pinned_memory_size": None,
        "allow_writing_files": None,
        "final_ctr_computation_mode": None,
        "approx_on_full_history": None,
        "boosting_type": None,
        "simple_ctr": None,
        "combinations_ctr": None,
        "per_feature_ctr": None,
        "ctr_description": None,
        "ctr_target_border_count": None,
        "task_type": None,
        "device_config": None,
        "devices": None,
        "bootstrap_type": None,
        "subsample": None,
        "mvs_reg": None,
        "sampling_frequency": None,
        "sampling_unit": None,
        "dev_score_calc_obj_block_size": None,
        "dev_efb_max_buckets": None,
        "sparse_features_conflict_fraction": None,
        "max_depth": None,
        "n_estimators": None,
        "num_boost_round": None,
        "num_trees": None,
        "colsample_bylevel": None,
        "random_state": None,
        "reg_lambda": None,
        "objective": None,
        "eta": None,
        "max_bin": None,
        "gpu_cat_features_storage": None,
        "data_partition": None,
        "metadata": None,
        "early_stopping_rounds": None,
        "cat_features": None,
        "grow_policy": None,
        "min_data_in_leaf": None,
        "min_child_samples": None,
        "max_leaves": None,
        "num_leaves": None,
        "score_function": None,
        "leaf_estimation_backtracking": None,
        "ctr_history_unit": None,
        "monotone_constraints": None,
        "feature_weights": None,
        "penalties_coefficient": None,
        "first_feature_use_penalties": None,
        "per_object_feature_penalties": None,
        "model_shrink_rate": None,
        "model_shrink_mode": None,
        "langevin": None,
        "diffusion_temperature": None,
        "posterior_sampling": None,
        "boost_from_average": None,
        "text_features": None,
        "tokenizers": None,
        "dictionaries": None,
        "feature_calcers": None,
        "text_processing": None,
        "embedding_features": None,
        "eval_fraction": None,
        "fixed_binary_splits": None,
        "frozen": False,
    }
def monkey_patch_gecs_class(class_):
    def single_peak_real_hps_fn(
        params: Dict[str, Optional[Union[str, float, int, float64]]]
    ):
        score_constituents = {}

        score_constituents["learning_rate_loss"] = abs(params["learning_rate"] - 0.5)

        score_constituents["reg_alpha_loss"] = abs(params["reg_alpha"] - 2.5)

        score_constituents["reg_lambda_loss"] = abs(params["reg_lambda"] - 0.2)

        score_constituents["min_child_weight_loss"] = (
            abs(params["min_child_weight"] - 0.1) * 10
        )

        score_constituents["min_child_samples_loss"] = abs(
            params["min_child_samples"] - 5
        )

        score_constituents["colsample_bytree_loss"] = abs(
            params["colsample_bytree"] - 0.7
        )

        score = (20.0 - np.sum(list(score_constituents.values()))) / 20.0
        return (score, [score_constituents])

    def _calculate_cv_score_monkeypatch(
        X: ndarray,
        y: ndarray,
        params: Dict[str, Optional[Union[str, float, int, float64]]],
        class_: Any,
    ) -> float64:
        score, data = single_peak_real_hps_fn(params)

        return score

    class_._calculate_cv_score = _calculate_cv_score_monkeypatch
    class_._fit_best_params = lambda X, y: None

    return class_


@pytest.fixture(scope="session")
def return_monkeypatch_gecs_class():
    return monkey_patch_gecs_class
