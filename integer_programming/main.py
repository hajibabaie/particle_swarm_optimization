from particle_swarm_optimization.integer_programming.cost_function import IntegerProgramming
from particle_swarm_optimization.integer_programming.solution_method import PSO


def main():

    problem = IntegerProgramming()

    cost_function = problem.cost_function

    solution_method = PSO(cost_function=cost_function,
                          max_iteration=200,
                          number_of_particles=40,
                          number_of_variables=4,
                          min_range_of_variables=0,
                          max_range_of_variables=1,
                          inertia_rate=1,
                          inertia_damping_rate=0.99,
                          personal_learning_rate=1.5,
                          global_learning_rate=1.5)

    solution_best, run_time = solution_method.run()

    return solution_best, run_time


if __name__ == "__main__":

    solution, runtime = main()
