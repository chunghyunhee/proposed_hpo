# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.com
# Powered by Seculayer © 2020 Solution Development 2 Team, R&D Center.
#
import random
import time
import numpy as np
import tensorflow as tf
import tensorflow_estimator as tfe
import bayes_opt
# import tensorflow_probability as tp

from hps.algorithms.HPOptimizationAbstract import HPOptimizationAbstract

class BayesianOptimization(HPOptimizationAbstract):

    def __init__(self, **kwargs):
        # inheritance init
        super(BayesianOptimization, self).__init__(**kwargs)
        self._check_hpo_params()
        self.DUP_CHECK = False

    ## genetic algorithm main function
    def _generate(self, param_list, score_list):
        result_param_list = list()

        # TODO: 1. initial generating -> eval fitness ->score_list
        #       2. update target function with score_list : Gaussian Processing
        #       3. Get new Params with acquisition function and target function : -> 1-param

        if self._current_steps < self._n_iter :
            gen_param_list = self._population(param_list)
            self._update_function(param_list, score_list)
            self._current_steps += 1
            return gen_param_list
        else :
            self._update_function(param_list, score_list)
            generate_param_dict = self._acquisition_function(param_list)
            result_param_list.append(generate_param_dict)
            return result_param_list

    #############################################################################
    ### Bayesian Optimization private functions
    def _population(self, param_list):
        if len(param_list) == 0:
            return self._generate_param_dict_list(self._n_pop)
        else :
            return param_list

    def _update_function(self,param_list, score_list):
        if len(score_list) ==0:
            return param_list
        else :
            raise NotImplementedError
    def _acquisition_function(self, param_list):

        raise NotImplementedError

if __name__ == '__main__':
    hprs_info = {
        "hpo_params" : {
            "mut_prob" : 0.5,
            "cx_prob" : 0.5,
            "sel_ratio" : 0.5,
            "mut_ratio" : 0.25,
            "cx_ratio" : 0.25,
            "n_steps" : 10,
            "n_params" : 10,
            "k_val" : 5,
            "eval_key" : "accuracy"
        },
        "ml_params":{
            "model_param":{
                "input_units" : "100",
                "output_units" : "1",
                "global_step" : "10",
                "early_type" : "2",
                "min_step" : "10",
                "early_key" : "accuracy",
                "early_value" : "0.98",
                "method_type" : "Basic",
                "global_sn" : "0",
                "alg_sn" : "0",
                "algorithm_type" : "classifier",
                "job_type" : "learn"
            },
            "pbounds":{
                "dropout_prob": [0, 0.5],
                "optimizer_fn": ["Adam", "rmsprop", "Adadelta"],
                "learning_rate": [0, 0.8],
                "act_fn": ["Tanh", "ReLU", "Sigmoid"],
                "hidden_units" : [3,1024]
            }
        }
    }
    ga = GeneticAlgorithm(hprs_info=hprs_info)
    best_params = ga._generate([], [])
    print(best_params)
