from particle_swarm_optimization.transportation_problem.data import Data
from particle_swarm_optimization.transportation_problem.cost_function import TransportationProblem
from particle_swarm_optimization.transportation_problem.solution_method import PSO


def main():

    # model_data = Data(number_of_customers=40,
    #                   number_of_suppliers=6,
    #                   customer_demand_min=10,
    #                   customer_demand_max=90)
    #
    # model_data.create_and_save()

    data = Data.load()

    problem = TransportationProblem()
    cost_function = problem.cost_function

    solution_method = PSO(cost_function=cost_function,
                          max_iteration=400,
                          variables_shape=data["distances"].shape,
                          min_range_of_variables=0,
                          max_range_of_variables=1,
                          number_of_particles=40,
                          inertia_rate=1,
                          inertia_damping_rate=0.98,
                          personal_learning_rate=1.5,
                          global_learning_rate=1.5)

    particles, particle, runtime = solution_method.run()

    return particles, particle, data


if __name__ == "__main__":

    particles_main, particle_best, model_data = main()
