##################################### import libraries #########################################
from network import EVOLUNET
import matplotlib.pyplot as plt
import numpy as np
import random
########################################## genetic algorithm #########################
class Evolution:
    def __init__(self, sizes,nb_individuals=50, nb_generations=100):
        self.nb_individuals = nb_individuals
        self.nb_generations = nb_generations
        self.neuro_size = sizes
        self.fitness = []
        self.population = []
        self.fitness_history = []
        self.best_fitness_history = []
        self.tournament_size = 0

    def init_population(self,method = "full"):
        self.population = []
        self.fitness = [] #init fitness eachtime
        if method == "full":
            for i in range(self.nb_individuals):
                self.population.append(EVOLUNET(self.neuro_size))
        elif method == "one":
            self.population.append(EVOLUNET(self.neuro_size))

    def calculate_fitness(self,training_data):
        self.fitness =[] #init fitness
        i_max = 0
        max_fitness = 0
        for i,individual in enumerate(self.population):
            indi_fitness = individual.evaluate(training_data)
            #search for the max fitness individual
            if indi_fitness > max_fitness:
                i_max= i
                max_fitness = indi_fitness
            self.fitness.append(indi_fitness)
        return i_max,max_fitness

    def selection_tournament(self,tournament_size = 3):
        selected = []
        self.tournament_size = tournament_size
        for _ in range(len(self.population)):
            tournament = random.sample(list(zip(self.population,self.fitness)), self.tournament_size)
            winner = max(tournament, key=lambda x: x[1])
            selected.append(winner)
        return selected

    def plot_fitness(self):
        fig,ax = plt.subplots()
        ax.plot(range(len(self.fitness_history)), self.fitness_history, label='Average fitness',linestyle = "-.")
        ax.plot(range(len(self.best_fitness_history)), self.best_fitness_history, label = "Best fitness",linestyle = ":")
        ax.set_title(f"fitness along the {self.nb_generations} generations with \n tournament selection upto {self.tournament_size} in a population of {self.nb_individuals} individuals ")
        ax.set(xlabel="Generation", ylabel=f'negative cost')
        ax.legend()
        plt.show()

    def get_average_fitness(self):
        return sum(self.fitness) / len(self.fitness)
    def reproduce(self,selected,training_data):
        (parent,parent_fitness)= selected
        parent_weights = parent.weights
        parent_bias = parent.bias
        # create new child with mutated weights and bias
        child = EVOLUNET(self.neuro_size, parent_bias, parent_weights)
        child.mutate_param()
        child_fitness = child.evaluate(training_data)
        return child,child_fitness,parent, parent_fitness

    def generation_evaluation(self,training_data):
        self.init_population(method = "full")

        for generation in range(self.nb_generations):
            if generation % 10 ==0:
                print("Generation {}".format(generation))

            i_best_individual,best_fitness = self.calculate_fitness(training_data)
            average_fitness = self.get_average_fitness()
            # print(best_fitness,average_fitness)
            #keep information for plot
            self.fitness_history.append(average_fitness)
            self.best_fitness_history.append(best_fitness)

            sub_population = self.selection_tournament(tournament_size = 10)
            next_population = []
            for i in range(0,len(sub_population)):
                child, child_fitness, parent, parent_fitness= self.reproduce(sub_population[i],training_data)
                if child_fitness > parent_fitness: #if new child perform better, keep child, else keep the parent
                    next_population.append(child)
                else:
                    next_population.append(parent)
            next_population[0] = self.population[i_best_individual]
            self.population = next_population

        print("learning process finished")

        i_optimized_individual,optimized_fitness = self.calculate_fitness(training_data)
        average_fitness = sum(self.fitness)/len(self.fitness)
        self.fitness_history.append(average_fitness)
        self.best_fitness_history.append(optimized_fitness)
        self.plot_fitness()
        return self.population[i_optimized_individual] #use the best individual

    # def explose(self,training_data):

if __name__ == "__main__":
    test = Evolution([2,3,2])
    test.init_population()
