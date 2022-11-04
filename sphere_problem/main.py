from particle_swarm_optimization.sphere_problem.cost_function import sphere
from particle_swarm_optimization.sphere_problem.solution_method import PSO


def main():

    cost_function = sphere

    solution_method = PSO(cost_function=cost_function,
                          max_iteration=800,
                          number_of_particles=100,
                          number_of_variables=10,
                          min_range_of_variables=-10,
                          max_range_of_variables=10,
                          personal_learning_rate=1.5,
                          global_learning_rate=1.5,
                          inertia_rate=1,
                          inertia_damping_rate=0.98)

    particles, particle_best, run_time = solution_method.run()

    return particles, particle_best, run_time


if __name__ == "__main__":

    particles_main, particle, runtime = main()

