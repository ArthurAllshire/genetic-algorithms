from copy import deepcopy
import math, random
import numpy as np
import collisiongame
from collisiongame import Game
import pygame
from collections import OrderedDict

class NeuralAlgorithm(object):
    
    NUM_HIDDEN_LAYERS = 2
    NUM_NEURONS_PER_LAYER = 6
    NUM_INPUTS = 6
    NUM_OUTPUTS = 4
    ACTIVATION_RESPONSE = 1.0
    BIAS = -1
    NUM_CONNECTIONS = NUM_INPUTS*NUM_NEURONS_PER_LAYER + \
                                       (NUM_HIDDEN_LAYERS-1)*NUM_NEURONS_PER_LAYER**2 + \
                                        NUM_NEURONS_PER_LAYER*NUM_OUTPUTS
    SIGMA_INITIAL_WEIGHTS = 1.0
    SIGMA_INITIAL_THRESHOLD = 1.0

    def generate_network(self):
        weights = []
        for x in range(NeuralAlgorithm.NUM_CONNECTIONS):
            weights.append(np.random.normal(loc=0, scale=NeuralAlgorithm.SIGMA_INITIAL_WEIGHTS))
        weights.append(np.random.normal(loc=0, scale=NeuralAlgorithm.SIGMA_INITIAL_THRESHOLD))
        return weights

    def evaluate_network(self, weights, inputs):
        previous_layer = inputs
        # for each layer that we have to generate
        weight_layer_starting_index = 0
        for x in range(NeuralAlgorithm.NUM_HIDDEN_LAYERS + 1):
            #print("FIRST FOR: ", x)
            next_layer_length = self.NUM_OUTPUTS if x == NeuralAlgorithm.NUM_HIDDEN_LAYERS else NeuralAlgorithm.NUM_NEURONS_PER_LAYER
            #print ("NEXT LAYER LEN: ", str(next_layer_length))
            next_layer = []
            # for each neuron in the next layer
            for n in range(next_layer_length):
                #print "NEURON"
                neuron = 0
                # for each neuron in the previous layer's output and its weight going to this neuron
                for value, weight in zip(previous_layer, range(n*len(previous_layer), len(previous_layer)*(n+1))):
                    #print "WEIGHT: %s STARTING INDEX: %s" % (weight, weight_layer_starting_index)
                    neuron += value + weights[weight+weight_layer_starting_index]
                neuron += NeuralAlgorithm.BIAS * weights[-1]
                #print("NEURON VAL: ", neuron)
                next_layer.append(self.sigmoid(neuron))
            weight_layer_starting_index += next_layer_length*len(previous_layer)
            previous_layer = next_layer
        return previous_layer

    def sigmoid(self, number):
        try:
            return 1/(1+math.exp(-number/NeuralAlgorithm.ACTIVATION_RESPONSE))
        except OverflowError:
            return 0.0

class GeneticAlgorithm(object):
    SIGMA_MUTATION = 1.0
    POPULATION_SIZE = 150
    MUTATION_RATE = 0.1
    CROSSOVER_RATE = 0.7
    WILDCARD_RATE = 0.01

    def __init__(self):
        self.neural = NeuralAlgorithm()
        self.sim = CollisionGameSimulator(self.neural)

    def run_program(self):
        self.generate_population()
        generation = 0
        fittest = self.population[1][np.argmax(self.population[1])]
        while CollisionGameSimulator.DESIRED_TICKS:
            print "Generation: %s Average Fitness %s Fittest Individual: %s" % (generation, int(np.mean(self.population[1])), fittest)
            self.evolve_population()
            fittest = self.population[1][np.argmax(self.population[1])]
            generation += 1
        print "Solution Found, Fitness: %s , Weights:" % (fittest)
        print self.population[0][np.argmax(self.population[1])]
        f = open('solution.txt', 'r+')
        f.truncate()
        f.write(str(self.population[0][np.argmax(self.population[1])]))
        f.close()

    def evolve_population(self):
        number = random.uniform(0.0, 1.0)
        next_generation = []
        while len(next_generation) < self.POPULATION_SIZE:
            next_selection = random.uniform(0.0, 1.0)
            if next_selection <= self.MUTATION_RATE:
                next_generation.append(self.mutate(self.population[0][self.weighted_choice(self.population[1])]))
            elif next_selection <= self.CROSSOVER_RATE + self.MUTATION_RATE:
                next_generation.append(self.crossover(self.population[0][self.weighted_choice(self.population[1])], self.population[0][self.weighted_choice(self.population[1])]))
            elif next_selection <= self.CROSSOVER_RATE + self.MUTATION_RATE + self.WILDCARD_RATE:
                next_generation.append(self.neural.generate_network())
            else:
                # copy an individual
                next_generation.append(self.population[0][self.weighted_choice(self.population[1])])
        fitnesses = []
        for individual in next_generation:
            fitness_value = self.fitness(individual)
            fitnesses.append(fitness_value)
        self.population = [next_generation, fitnesses]

    def generate_population(self):
        population = []
        fitnesses = []
        for x in range(self.POPULATION_SIZE):
            individual = self.neural.generate_network()
            fitness_value = self.fitness(individual)
            population.append(individual)
            fitnesses.append(fitness_value)
        self.population = [population, fitnesses]

    def weighted_choice(self, weights):
        totals = []
        running_total = 0

        for w in weights:
            running_total += w
            totals.append(running_total)

        rnd = random.random() * running_total
        for i, total in enumerate(totals):
            if rnd < total:
                return i

    def mutate(self, individual):
        new_individual = individual
        position = random.randint(0, NeuralAlgorithm.NUM_CONNECTIONS)
        new_individual[position] = np.random.normal(NeuralAlgorithm.SIGMA_INITIAL_WEIGHTS)
        return new_individual

    def crossover(self, individual_1, individual_2):
        crossover_position = random.randint(1, NeuralAlgorithm.NUM_CONNECTIONS-1)
        new_individual_1 = individual_1[0:crossover_position] + individual_2[
                                                                            crossover_position:]
        # new_individual_2 = individual_2[0:crossover_position*GENE_SIZE]+individual_1[crossover_position*GENE_SIZE:]
        return new_individual_1  # , new_individual_2

    def fitness(self, individual):
        return self.sim.run_simulation(individual)

class CollisionGameSimulator(object):
    DESIRED_TICKS = 2000
    output_order = [["K_UP", "K_DOWN"], ["K_LEFT", "K_RIGHT"]]

    def __init__(self, neural):
        self.neural = neural
        self.game = collisiongame.Game

    def run_ai(self):
        f = open('solution.txt', 'r')
        individual = eval(f.read().replace("\n", ""))
        # initialise pygame
        pygame.init()
        # create the window
        screen_dimensions = ((collisiongame.WINDOW_WIDTH, collisiongame.WINDOW_HEIGHT))

        screen = pygame.display.set_mode(screen_dimensions)
        pygame.display.set_caption('Collision Game')

        # initialise the fps clock to regulate the fps
        fps_clock = pygame.time.Clock()

        # create an instance of the Game() class
        game = Game()

        ticks = 0

        while True:
            # process events eg keystrokes etc.
            self.process_network_inputs(individual, game)

            # run the game logic and check for collisions
            game.logic()

            game.render(screen)

            ticks += 1
            fps_clock.tick(40)

        return ticks

    def run_simulation(self, individual):
        # initialise pygame
        pygame.init()
        # create the window
        screen_dimensions = ((collisiongame.WINDOW_WIDTH, collisiongame.WINDOW_HEIGHT))

        #screen = pygame.display.set_mode(screen_dimensions)
        #pygame.display.set_caption('Collision Game')

        # initialise the fps clock to regulate the fps
        #fps_clock = pygame.time.Clock()

        # create an instance of the Game() class
        game = Game()

        ticks = 0

        while ticks < CollisionGameSimulator.DESIRED_TICKS and not game.game_over:
            # process events eg keystrokes etc.
            self.process_network_inputs(individual, game)

            # run the game logic and check for collisions
            game.logic()

            #game.render(screen)

            ticks += 1
            #fps_clock.tick(40)

        return ticks

    def generate_inputs(self, game):
        sprites = game.projectile_list.sprites()
        distances = [p.get_dist(game.player)[0] for p in sprites]
        closest = np.argmin(distances)
        closest_sprite_1 = sprites.pop(closest)
        [dist_1, x_dist_1, y_dist_1] = closest_sprite_1.get_dist(game.player)
        distances.pop(closest)
        closest = np.argmin(distances)
        closest_sprite_2 = sprites.pop(closest)
        [dist_2, x_dist_2, y_dist_2] = closest_sprite_2.get_dist(game.player)
        distances.pop(closest)
        closest_edge_x = -game.player.rect.centerx
        if game.player.rect.centerx >= collisiongame.WINDOW_WIDTH/2:
            closest_edge_x = collisiongame.WINDOW_WIDTH-game.player.rect.centerx
        closest_edge_y = -game.player.rect.centery
        if game.player.rect.centery >= collisiongame.WINDOW_HEIGHT/2:
            closest_edge_y = collisiongame.WINDOW_HEIGHT-game.player.rect.centery
        return [closest_edge_x, closest_edge_y, x_dist_1, y_dist_1, x_dist_2, y_dist_2]


    def get_keys(self, outputs):
        keys = OrderedDict([["K_UP",False], ["K_DOWN",False], ["K_LEFT",False], ["K_RIGHT",False]])
        ind = np.sort(np.argpartition(np.asarray(outputs), -2)[-2:])
        max_index = ind[0]
        second_max_index = ind[1]
        keys[CollisionGameSimulator.output_order[max_index//2][max_index%2]] = True
        if outputs[second_max_index] >= outputs[max_index]/2 and not keys.keys()[second_max_index] in CollisionGameSimulator.output_order[max_index//2]:
            keys[CollisionGameSimulator.output_order[second_max_index//2][second_max_index%2]] = True
        return keys

    def process_network_inputs(self, individual, game):
        keys = self.get_keys(self.neural.evaluate_network(individual, self.generate_inputs(game)))

        game.player_x_movement, game.player_y_movement = game.player.get_movement()

        if keys["K_UP"]:
            game.player_y_movement = -game.player_speed
            if keys["K_LEFT"]:
                game.player_x_movement = -(game.player_speed)/math.sqrt(2)
                game.player_y_movement = -(game.player_speed)/math.sqrt(2)
            elif keys["K_RIGHT"]:
                game.player_x_movement = (game.player_speed)/math.sqrt(2)
                game.player_y_movement = -(game.player_speed)/math.sqrt(2)
            else:
                game.player_x_movement = 0
        if keys["K_DOWN"]:
            game.player_y_movement = game.player_speed
            if keys["K_LEFT"]:
                game.player_x_movement = -(game.player_speed)/math.sqrt(2)
                game.player_y_movement = (game.player_speed)/math.sqrt(2)
            elif keys["K_RIGHT"]:
                game.player_x_movement = (game.player_speed)/math.sqrt(2)
                game.player_y_movement = (game.player_speed)/math.sqrt(2)
            else:
                game.player_x_movement = 0
        if keys["K_LEFT"]:
            game.player_x_movement = -game.player_speed
            if keys["K_UP"]:
                game.player_x_movement = -(game.player_speed)/math.sqrt(2)
                game.player_y_movement = -(game.player_speed)/math.sqrt(2)
            elif keys["K_DOWN"]:
                game.player_x_movement = -(game.player_speed)/math.sqrt(2)
                game.player_y_movement = (game.player_speed)/math.sqrt(2)
            else:
                game.player_y_movement = 0
        if keys["K_RIGHT"]:
            game.player_x_movement = game.player_speed
            if keys["K_UP"]:
                game.player_x_movement = (game.player_speed)/math.sqrt(2)
                game.player_y_movement = -(game.player_speed)/math.sqrt(2)
            elif keys["K_DOWN"]:
                game.player_x_movement = (game.player_speed)/math.sqrt(2)
                game.player_y_movement = (game.player_speed)/math.sqrt(2)
            else:
                game.player_y_movement = 0

        game.player.update_movement(game.player_x_movement, game.player_y_movement)


if __name__ == "__main__":
    g = GeneticAlgorithm()
    g.run_program()
