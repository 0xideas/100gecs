import copy
import sys
import inspect
from typing import Callable, Dict, List, Optional, Union

import numpy as np
from catboost import CatBoostRegressor
from numpy import float64, ndarray
from six import iteritems, string_types, integer_types

#from .gec_base import GECBase

class GECar(CatBoostRegressor, GECBase):
    def __init__(
        self,
        iterations=None,
        learning_rate=None,
        depth=None,
        l2_leaf_reg=None,
        model_size_reg=None,
        rsm=None,
        loss_function='RMSE',
        border_count=None,
        feature_border_type=None,
        per_float_feature_quantization=None,
        input_borders=None,
        output_borders=None,
        fold_permutation_block=None,
        od_pval=None,
        od_wait=None,
        od_type=None,
        nan_mode=None,
        counter_calc_method=None,
        leaf_estimation_iterations=None,
        leaf_estimation_method=None,
        thread_count=None,
        random_seed=None,
        use_best_model=None,
        best_model_min_trees=None,
        verbose=None,
        silent=None,
        logging_level=None,
        metric_period=None,
        ctr_leaf_count_limit=None,
        store_all_simple_ctr=None,
        max_ctr_complexity=None,
        has_time=None,
        allow_const_label=None,
        target_border=None,
        one_hot_max_size=None,
        random_strength=None,
        name=None,
        ignored_features=None,
        train_dir=None,
        custom_metric=None,
        eval_metric=None,
        bagging_temperature=None,
        save_snapshot=None,
        snapshot_file=None,
        snapshot_interval=None,
        fold_len_multiplier=None,
        used_ram_limit=None,
        gpu_ram_part=None,
        pinned_memory_size=None,
        allow_writing_files=None,
        final_ctr_computation_mode=None,
        approx_on_full_history=None,
        boosting_type=None,
        simple_ctr=None,
        combinations_ctr=None,
        per_feature_ctr=None,
        ctr_description=None,
        ctr_target_border_count=None,
        task_type=None,
        device_config=None,
        devices=None,
        bootstrap_type=None,
        subsample=None,
        mvs_reg=None,
        sampling_frequency=None,
        sampling_unit=None,
        dev_score_calc_obj_block_size=None,
        dev_efb_max_buckets=None,
        sparse_features_conflict_fraction=None,
        max_depth=None,
        n_estimators=None,
        num_boost_round=None,
        num_trees=None,
        colsample_bylevel=None,
        random_state=None,
        reg_lambda=None,
        objective=None,
        eta=None,
        max_bin=None,
        gpu_cat_features_storage=None,
        data_partition=None,
        metadata=None,
        early_stopping_rounds=None,
        cat_features=None,
        grow_policy=None,
        min_data_in_leaf=None,
        min_child_samples=None,
        max_leaves=None,
        num_leaves=None,
        score_function=None,
        leaf_estimation_backtracking=None,
        ctr_history_unit=None,
        monotone_constraints=None,
        feature_weights=None,
        penalties_coefficient=None,
        first_feature_use_penalties=None,
        per_object_feature_penalties=None,
        model_shrink_rate=None,
        model_shrink_mode=None,
        langevin=None,
        diffusion_temperature=None,
        posterior_sampling=None,
        boost_from_average=None,
        text_features=None,
        tokenizers=None,
        dictionaries=None,
        feature_calcers=None,
        text_processing=None,
        embedding_features=None,
        eval_fraction=None,
        fixed_binary_splits=None,
        frozen=False
    ):
        adapted_cat_params = (
            str(inspect.signature(CatBoostRegressor.__init__))
            .replace(
                "fixed_binary_splits=None)",
                "fixed_binary_splits=None, frozen=False)",
            )
        )
        gecat_params = str(inspect.signature(GECar.__init__))
        assert (
            adapted_cat_params == gecat_params
        ), f"{gecat_params = } \n not equal to \n {adapted_cat_params = }"

        params = {}
        not_params = ["not_params", "self", "params", "__class__"]
        for key, value in iteritems(locals().copy()):
            if key not in not_params and value is not None:
                params[key] = value

        super(CatBoostRegressor, self).__init__(params)

        non_optimized_init_args = [
            "depth",
            "model_size_reg",
            "loss_function",
            "border_count",
            "feature_border_type",
            "per_float_feature_quantization",
            "input_borders",
            "output_borders",
            "fold_permutation_block",
            "od_pval",
            "od_wait",
            "od_type",
            "nan_mode",
            "counter_calc_method",
            "leaf_estimation_iterations",
            "leaf_estimation_method",
            "thread_count",
            "random_seed",
            "use_best_model",
            "best_model_min_trees",
            "verbose",
            "logging_level",
            "metric_period",
            "ctr_leaf_count_limit",
            "store_all_simple_ctr",
            "max_ctr_complexity",
            "has_time",
            "allow_const_label",
            "target_border",
            "one_hot_max_size",
            "random_strength",
            "name",
            "ignored_features",
            "train_dir",
            "custom_metric",
            "eval_metric",
            "bagging_temperature",
            "save_snapshot",
            "snapshot_file",
            "snapshot_interval",
            "fold_len_multiplier",
            "used_ram_limit",
            "gpu_ram_part",
            "pinned_memory_size",
            "allow_writing_files",
            "final_ctr_computation_mode",
            "approx_on_full_history",
            "boosting_type",
            "simple_ctr",
            "combinations_ctr",
            "per_feature_ctr",
            "ctr_description",
            "ctr_target_border_count",
            "task_type",
            "device_config",
            "devices",
            "bootstrap_type",
            "subsample",
            "mvs_reg",
            "sampling_frequency",
            "sampling_unit",
            "dev_score_calc_obj_block_size",
            "dev_efb_max_buckets",
            "sparse_features_conflict_fraction",
            "max_depth",
            "random_state",
            "objective",
            "max_bin",
            "gpu_cat_features_storage",
            "data_partition",
            "metadata",
            "early_stopping_rounds",
            "cat_features",
            "grow_policy",
            "score_function",
            "leaf_estimation_backtracking",
            "ctr_history_unit",
            "monotone_constraints",
            "feature_weights",
            "penalties_coefficient",
            "first_feature_use_penalties",
            "per_object_feature_penalties",
            "model_shrink_rate",
            "model_shrink_mode",
            "langevin",
            "diffusion_temperature",
            "posterior_sampling",
            "boost_from_average",
            "text_features",
            "tokenizers",
            "dictionaries",
            "feature_calcers",
            "text_processing",
            "embedding_features",
            "eval_fraction",
            "fixed_binary_splits"
        ]
        optimization_candidate_init_args = [
            "learning_rate",
            "n_estimators",
            "num_leaves",
            "reg_lambda",
            "min_child_samples",
            "colsample_bylevel",  # feature_fraction
        ]
        self._gec_init({}, frozen, non_optimized_init_args, optimization_candidate_init_args)



    def fit(self,
            X,
            y=None,
            n_iter=50,
            fixed_hyperparameters=["n_estimators", "num_leaves"],
            cat_features=None,
            text_features=None,
            embedding_features=None,
            sample_weight=None,
            baseline=None,
            use_best_model=None,
            eval_set=None,
            verbose=None,
            logging_level=None,
            plot=False,
            plot_file=None,
            column_description=None,
            verbose_eval=None,
            metric_period=None,
            silent=None,
            early_stopping_rounds=None,
            save_snapshot=None,
            snapshot_file=None,
            snapshot_interval=None,
            init_model=None,
            callbacks=None,
            log_cout=sys.stdout,
            log_cerr=sys.stderr
        ):
        self.gec_fit_params_ = {
            "cat_features": cat_features,
            "text_features": text_features,
            "embedding_features": embedding_features,
            "sample_weight": sample_weight,
            "baseline": baseline,
            "use_best_model": use_best_model,
            "eval_set": eval_set,
            "verbose": verbose,
            "logging_level": logging_level,
            "plot": plot,
            "plot_file": plot_file,
            "column_description": column_description,
            "verbose_eval": verbose_eval,
            "metric_period": metric_period,
            "silent": silent,
            "early_stopping_rounds": early_stopping_rounds,
            "save_snapshot": save_snapshot,
            "snapshot_file": snapshot_file,
            "snapshot_interval": snapshot_interval,
            "init_model": init_model,
            "callbacks": callbacks,
            "log_cout": log_cout,
            "log_cerr": log_cerr
            }
        self._fit_inner(X, y, n_iter, fixed_hyperparameters)


    def __sklearn_clone__(self):
        class_ = GECar()

        for k, v in self.__dict__.items():
            class_.__dict__[k] = copy.deepcopy(v)

        return class_

    def set_params(self, **kwargs) -> None:
        if "frozen" in kwargs:
            setattr(self, "frozen", kwargs.pop("frozen"))
        super().set_params(**kwargs)

    def get_params(
        self, deep: bool = True
    ) -> Dict[str, Optional[Union[str, float, int, bool]]]:
        if hasattr(self, "best_params_") and self.best_params_ is not None:
            params = copy.deepcopy(self.best_params_)
        else:
            params = super().get_params(deep)
        params["frozen"] = self.frozen

        return params

    def _fit_best_params(self, X: ndarray, y: ndarray) -> None:

        if hasattr(self, "best_params") and self.best_params_ is not None:
            for k, v in self.best_params_.items():
                setattr(self, k, v)
            setattr(self, "random_state", 101)

        super().fit(X, y, **self.gec_fit_params_)

    def score_single_iteration(
        self,
        X: ndarray,
        y: ndarray,
        params: Dict[str, Optional[Union[str, float, int, float64]]],
    ):
        
        if "subsample_freq" in params:
            del params["subsample_freq"]


        params["silent"] = True

        return self._calculate_cv_score(X, y, params, CatBoostRegressor)

    
    def retrieve_hyperparameter(self, hyperparameter):
        return(self._init_params.get(hyperparameter, None))
    
