import os
import numpy as np


class Data:

    def __init__(self,
                 number_of_items,
                 min_range_of_values,
                 max_range_of_values,
                 min_range_of_weights,
                 max_range_of_weights,
                 min_number_of_each_item,
                 max_number_of_each_item):

        self._number_of_items = number_of_items
        self._min_range_of_values = min_range_of_values
        self._max_range_of_values = max_range_of_values
        self._min_range_of_weights = min_range_of_weights
        self._max_range_of_weights = max_range_of_weights
        self._min_number_of_each_item = min_number_of_each_item
        self._max_number_of_each_item = max_number_of_each_item

    def create_and_save(self):

        values = np.random.uniform(self._min_range_of_values,
                                   self._max_range_of_values,
                                   self._number_of_items)

        weights = np.random.uniform(self._min_range_of_weights,
                                    self._max_range_of_weights,
                                    self._number_of_items)

        number_of_each_items = np.random.randint(self._min_number_of_each_item,
                                                 self._max_number_of_each_item,
                                                 self._number_of_items)

        total_weights = np.dot(weights, number_of_each_items).ravel()

        knapsack_capacity = total_weights / 5
        knapsack_capacity = np.reshape(knapsack_capacity, (1, 1))

        os.makedirs("./data", exist_ok=True)
        np.savetxt("./data/values.csv", values)
        np.savetxt("./data/weights.csv", weights)
        np.savetxt("./data/number_of_each_items.csv", number_of_each_items)
        np.savetxt("./data/knapsack_capacity.csv", knapsack_capacity)

    @staticmethod
    def load():

        out = {
            "values": np.loadtxt("./data/values.csv"),
            "weights": np.loadtxt("./data/weights.csv"),
            "number_of_items": np.loadtxt("./data/number_of_each_items.csv"),
            "knapsack_capacity": float(np.loadtxt("./data/knapsack_capacity.csv"))
        }

        return out
