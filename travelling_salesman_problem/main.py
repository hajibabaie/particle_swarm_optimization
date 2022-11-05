from particle_swarm_optimization.travelling_salesman_problem.data import Data
from particle_swarm_optimization.travelling_salesman_problem.solution_method import PSO
from particle_swarm_optimization.travelling_salesman_problem.cost_function import TravellingSalesmanProblem
from particle_swarm_optimization.travelling_salesman_problem.plot import plot_tour


def main():

    # model_data = Data(number_of_nodes=50)
    # model_data.create_and_save()

    data = Data.load()

    problem = TravellingSalesmanProblem()
    cost_function = problem.cost_function

    solution_method = PSO(cost_function=cost_function,
                          max_iteration=1000,
                          number_of_particles=200,
                          number_of_variables=int(data["distances"].shape[0]),
                          min_range_of_variables=0,
                          max_range_of_variables=1,
                          inertia_rate=1,
                          inertia_damping_rate=0.98,
                          personal_learning_rate=1.7,
                          global_learning_rate=2)

    solution_best, run_time = solution_method.run()

    plot_tour(solution_best, data)

    return solution_best, run_time, data


if __name__ == "__main__":

    solution, runtime, model_data = main()
