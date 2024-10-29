##################################### import libraries #########################################
from network import EVOLUNET
import matplotlib.pyplot as plt
import numpy as np
from PrettyTable import PrettyTable
import random
########################################## genetic algorithm #########################
class Evolution:
    def __init__(self, nb_individuals, nb_generations):
        self.nb_individuals = nb_individuals
        self.nb_generations = nb_generations
        self.population = []
        self.fitness = []

    def new_population(self,nb_layers):
        for i in range(self.nb_individuals):
            individual = EVOLUNET(nb_layers)
            self.population.append(individual)

    def calculate_fitness(self):
        for individual in self.population:
            self.fitness.append(individual.evaluate())

    def selection_tournament(self,tournament_size = 3):
        selected = []
        for i in range(len(self.population)):
            tournament = random.sample(list(zip(self.population,self.fitness)), tournament_size)
            winner = max(tournament, key=lambda x: x[1])[0]
            selected.append(winner)
        return selected


    def crossover(self,parent1, parent2):
        alpha = random.random()
        for w1,w2,b1,b2 in zip(parent1.weights, parent2.weights, parent1.bias, parent2.bias):
            child1_weights = [alpha*w1[i] + (1-alpha)*w2[i] for i in range(len(w1))]
            child2_weights = [alpha*w2[i] + (1-alpha)*w1[i] for i in range(len(w1))]
            child1_bias = [alpha*b1[i] + (1-alpha)*b2[i] for i in range(len(b1))]
            child2_bias = [alpha*b2[i] + (1-alpha)*b1[i] for i in range(len(b1))]
        return child1_weights,child2_weights,child1_bias,child2_bias
    def genetic_algorithm(self,mutation_rate):
        best_performace = []

        for generation in range(self.nb_generations):
            self.calculate_fitness()
            best_individual = max(self.fitness)

            sub_population = self.selection_tournament(tournament_size = 3)
            next_population = []
            for i in range(0,len(sub_population),2):
                parent1 = self.population[i]
                parent2 = self.population[i+1]

                child1_weights, child2_weights, child1_bias, child2_bias = self.crossover(parent1,parent2)

                #create new population with children
                child1 = EVOLUNET(child1_weights,child1_bias)
                child2 = EVOLUNET(child2_weights,child2_bias)

                child1.mutate_param(mutation_rate)
                child2.mutate_param(mutation_rate)
       
                next_population.append(child1)
                next_population.append(child2)
            next_population[0] = best_individual
            self.population = next_population

