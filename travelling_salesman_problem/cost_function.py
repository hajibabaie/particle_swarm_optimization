from particle_swarm_optimization.travelling_salesman_problem.data import Data
import numpy as np


class TravellingSalesmanProblem:

    def __init__(self):

        self._model_data = Data.load()

    @staticmethod
    def _parse_solution(x):

        x_parsed = np.argsort(x, axis=1)

        return x_parsed

    def cost_function(self, x):

        x_parsed = self._parse_solution(x)

        distances = self._model_data["distances"]

        cost = 0

        for i in range(int(distances.shape[0]) - 1):

            cost += distances[x_parsed[0, i], x_parsed[0, i + 1]]

        cost += distances[x_parsed[0, -1], x_parsed[0, 0]]

        return x_parsed, cost

