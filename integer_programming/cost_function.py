import numpy as np


class IntegerProgramming:

    @staticmethod
    def _parse_solution(x):

        summation = 8

        x_parsed = summation * x + 1

        x_parsed = np.floor(x_parsed)

        return x_parsed

    def cost_function(self, x):

        product = 1000

        x_parsed = self._parse_solution(x)

        violation = np.abs((np.prod(x_parsed) - product) / product)

        cost = np.sum(x_parsed) * (1 + 10 * violation)


        return x_parsed, cost
