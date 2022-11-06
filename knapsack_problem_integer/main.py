from particle_swarm_optimization.knapsack_problem_integer.model_data import Data
from particle_swarm_optimization.knapsack_problem_integer.cost_function import KnapsackProblemInt
from particle_swarm_optimization.knapsack_problem_integer.solution_method import PSO


def main():


    # data = Data(number_of_items=12,
    #             min_range_of_values=10,
    #             max_range_of_values=20,
    #             min_range_of_weights=10,
    #             max_range_of_weights=50,
    #             min_number_of_each_item=1,
    #             max_number_of_each_item=8)
    #
    # data.create_and_save()

    model_data = Data.load()

    problem = KnapsackProblemInt()
    cost_function = problem.cost_function

    solution_method = PSO(cost_function=cost_function,
                          max_iteration=100,
                          number_of_particles=20,
                          number_of_variables=int(model_data["values"].shape[0]),
                          min_range_of_variables=0,
                          max_range_of_variables=1,
                          inertia_rate=1,
                          inertia_damping_rate=0.99,
                          personal_learning_rate=1.5,
                          global_learning_rate=1.5)

    solution_best, run_time = solution_method.run()


    return solution_best, run_time, model_data


if __name__ == "__main__":

    solution, runtime, data = main()
