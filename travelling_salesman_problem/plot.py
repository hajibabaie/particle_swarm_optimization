import matplotlib.pyplot as plt



def plot_tour(solution, model_data):


    nodes_x = model_data["nodes_x"]
    nodes_y = model_data["nodes_y"]
    number_of_cities = len(nodes_x)
    x_parsed = solution.solution

    plt.figure(dpi=600, figsize=(10, 6))
    plt.scatter(nodes_x, nodes_y, marker="o", c="green", s=20, label="nodes")
    for i in range(number_of_cities):
        plt.text(nodes_x[i], nodes_y[i], i)

    plt.plot([nodes_x[x_parsed[0, 0]], nodes_x[x_parsed[0, 1]]],
             [nodes_y[x_parsed[0, 0]], nodes_y[x_parsed[0, 1]]], color="green")

    for i in range(1, number_of_cities - 1):

        plt.plot([nodes_x[x_parsed[0, i]], nodes_x[x_parsed[0, i + 1]]],
                 [nodes_y[x_parsed[0, i]], nodes_y[x_parsed[0, i + 1]]], color="black")

    plt.plot([nodes_x[x_parsed[0, -1]], nodes_x[x_parsed[0, 0]]],
             [nodes_y[x_parsed[0, -1]], nodes_y[x_parsed[0, 0]]], color="red")

    plt.savefig("./salesman_tour.png")