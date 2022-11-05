import os
import numpy as np


class Data:


    def __init__(self, number_of_nodes):

        self._number_of_nodes = number_of_nodes

    def create_and_save(self):

        nodes_x = np.random.uniform(0, 100, self._number_of_nodes)
        nodes_y = np.random.uniform(0, 100, self._number_of_nodes)

        distances = np.zeros((self._number_of_nodes, self._number_of_nodes))

        for i in range(self._number_of_nodes):
            for j in range(i + 1, self._number_of_nodes):
                distances[i, j] = np.sqrt(np.square(nodes_x[i] - nodes_x[j]) +
                                          np.square(nodes_y[i] - nodes_y[j]))
                distances[j, i] = distances[i, j]

        os.makedirs("./data", exist_ok=True)
        np.savetxt("./data/nodes_x.csv", nodes_x)
        np.savetxt("./data/nodes_y.csv", nodes_y)
        np.savetxt("./data/distances.csv", distances, delimiter=",")

    @staticmethod
    def load():

        out = {
            "nodes_x": np.loadtxt("./data/nodes_x.csv"),
            "nodes_y": np.loadtxt("./data/nodes_y.csv"),
            "distances": np.loadtxt("./data/distances.csv", delimiter=",")
        }

        return out
