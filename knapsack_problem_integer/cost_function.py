import numpy as np
from particle_swarm_optimization.knapsack_problem_integer.model_data import Data


class KnapsackProblemInt:

    def __init__(self):

        self._model_data = Data.load()


    def _parse_solution(self, x):

        number_of_items = self._model_data["number_of_items"]
        number_of_items = np.reshape(number_of_items, (1, number_of_items.shape[0]))
        x_parsed = np.floor(np.multiply(x, number_of_items))

        return x_parsed

    def cost_function(self, x):

        x_parsed = self._parse_solution(x)

        values = self._model_data["values"]
        values = np.reshape(values, (1, values.shape[0]))

        weights = self._model_data["weights"]
        weights = np.reshape(weights, (1, weights.shape[0]))

        number_of_items = self._model_data["number_of_items"]
        number_of_items = np.reshape(number_of_items, (1, number_of_items.shape[0]))

        knapsack_capacity = self._model_data["knapsack_capacity"]

        values_gained = np.dot(x_parsed, values.T)[0, 0]
        values_not_gained = np.dot(number_of_items - x_parsed, values.T)[0, 0]

        weights_gained = np.dot(x_parsed, weights.T)[0, 0]
        weights_not_gained = np.dot(number_of_items - x_parsed, weights.T)[0, 0]

        capacity_violation = np.maximum((weights_gained / knapsack_capacity) - 1, 0)

        cost = values_not_gained * (1 + 10 * capacity_violation)

        out = {
            "x_parsed": x_parsed,
            "values_gained": values_gained,
            "values_not_gained": values_not_gained,
            "weights_gained": weights_gained,
            "weights_not_gained": weights_not_gained,
            "capacity_violation": capacity_violation
        }

        return out, cost
