# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.com
# Powered by Seculayer © 2020 Solution Development 2 Team, R&D Center. 

from hps.common.Constants import Constants
from hps.algorithms.ga.GeneticAlgorithm import GeneticAlgorithm
from hps.algorithms.ga.RandomSearch import RandomSearch
from hps.algorithms.ga.ParticleSwarmOptimization import ParticleSwarmOptimization
from hps.algorithms.ga.ProposedAlgorithm import ProposedAlgorithm
#from hps.algorithms.ga.SimulatedAnealing import SimulatedAnnealing
from hps.algorithms.ga.SimulatedAnnealing_3 import SimulatedAnnealing
from hps.algorithms.ga.PSO_boundary import ParticleSwarmOptimization_boundary
from hps.algorithms.ga.PSO_GA import PSO_GA
from hps.algorithms.ga.SA_PSO import SA_PSO
# class : HPOptimizerFactory
class HPOptimizerFactory(object):
    @staticmethod
    def create(hpo_dict, job_queue, result_queue):
        hpo_alg = hpo_dict["hpo_alg"]
        if hpo_alg == "GA":
            return {
                Constants.HPO_GA: GeneticAlgorithm(hps_info=hpo_dict,
                                                   job_queue=job_queue,
                                                   result_queue=result_queue)
            }.get(hpo_alg, None)
        elif hpo_alg == "Random":
            return {
                Constants.HPO_RS: RandomSearch(hps_info=hpo_dict,
                                               job_queue=job_queue,
                                               result_queue=result_queue)
            }.get(hpo_alg, None)
        elif hpo_alg == "PSO":
            return {
                Constants.HPO_PSO : ParticleSwarmOptimization(hps_info= hpo_dict,
                                                              job_queue = job_queue,
                                                              result_queue = result_queue)
            }.get(hpo_alg, None)
        elif hpo_alg == "proposedPSO":
            return{
                Constants.HPO_proposedPSO : ProposedAlgorithm(hps_info = hpo_dict,
                                                              job_queue = job_queue,
                                                              result_queue = result_queue)
            }.get(hpo_alg, None)
        elif hpo_alg == "SA":
            return {
                Constants.HPO_SA : SimulatedAnnealing(hps_info = hpo_dict,
                                                      job_queue = job_queue,
                                                      result_queue = result_queue)
            }.get(hpo_alg, None)
        elif hpo_alg == "PSO_boundary":
            return {
                Constants.HPO_boundaryPSO : ParticleSwarmOptimization_boundary(hps_info = hpo_dict,
                                                                               job_queue = job_queue,
                                                                               result_queue = result_queue)
            }.get(hpo_alg, None)
        elif hpo_alg == "PSO_GA" :
            return {
                Constants.HPO_PSOGA : PSO_GA(hps_info = hpo_dict,
                                             job_queue = job_queue,
                                             result_queue = result_queue)
            }.get(hpo_alg, None)
        elif hpo_alg == "PSO_SA":
            return {
                Constants.HPO_PSOSA : SA_PSO(hps_info = hpo_dict,
                                             job_queue = job_queue,
                                             result_queue = result_queue)
            }.get(hpo_alg, None)

        else :
            raise NotImplementedError