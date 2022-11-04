from particle_swarm_optimization.transportation_problem.data import Data
import numpy as np


class TransportationProblem:

    def __init__(self):

        self._model_data = Data.load()

    def _parse_solution(self, x):

        customer_demand = self._model_data["customer_demand"]
        customer_demand = np.reshape(customer_demand, (customer_demand.shape[0], 1))

        x_sum = np.sum(x, axis=1, keepdims=True)
        x_parsed = np.divide(x, x_sum)

        out = np.multiply(x_parsed, customer_demand)

        return out

    def cost_function(self, x):

        distances = self._model_data["distances"]
        customer_demand = self._model_data["customer_demand"]
        supplier_capacity = self._model_data["supplier_capacity"]

        customer_demand = np.reshape(customer_demand, (1, customer_demand.shape[0]))
        supplier_capacity = np.reshape(supplier_capacity, (1, supplier_capacity.shape[0]))

        x_parsed = self._parse_solution(x)

        supplier_sent = np.sum(x_parsed, axis=0, keepdims=True)

        capacity_violation = np.maximum(np.divide(supplier_sent, supplier_capacity) - 1, 0)

        capacity_violation_mean = np.mean(capacity_violation)

        cost = np.sum(np.multiply(x_parsed, distances))

        total_cost = cost * (1 + 10 * capacity_violation_mean)

        out = {
            "x_parsed": x_parsed,
            "supplier_sent": supplier_sent,
            "capacity_violation": capacity_violation,
            "capacity_violation_mean": capacity_violation_mean,
            "cost": cost
        }

        return out, total_cost
