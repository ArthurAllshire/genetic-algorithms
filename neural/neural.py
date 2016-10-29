import time
import math
import pygame
import random
import argparse
from collections import OrderedDict
import pickle

import numpy as np

import collisiongame
from collisiongame import Game


class NeuralAlgorithm(object):
    """Class that holds parameters for the neural network and has the functions required to use them"""
    sizes = [10, 10, 10, 4]

    def generate_network(self):
        w = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        b = [np.random.randn(y) for y in self.sizes[1:]]
        return [b, w]

    def evaluate_network(self, indiv,inputs):
        biases = indiv[0]
        weights = indiv[1]
        for b, w in zip(biases, weights):
            inputs = self.activation_fuction(np.dot(w, inputs)+b)
        return inputs

    def activation_fuction(self, z):
        #return 1.0 / (1.0 + np.exp(-z))
        return np.tanh(z)


class GeneticAlgorithm(object):
    SIGMA_MUTATION = 1.5
    # maybe lower to 50
    POPULATION_SIZE = 50
    MUTATION_RATE = 0.4
    CROSSOVER_RATE = 0.4
    WILDCARD_RATE = 0.00
    #maybe 2 or 3
    TRIAL_NUMBER = 2

    def __init__(self, threshold=None):
        self.threshold=[False, None]
        if threshold:
            self.threshold = threshold
            try:
                self.threshold[1] = int(self.threshold[1])
            except:
                self.threshold = [False, None]
        self.neural = NeuralAlgorithm()
        self.sim = CollisionGameSimulator(self.neural)
        self.game = Game()
        # mock a couple of the time dependant functions in the game class
        self.game.score_clock_tick = lambda: 1000/collisiongame.FPS
        self.game.projectile_direction_tick = lambda: 1000/collisiongame.FPS

    def run_program(self):
        self.start_time = time.time()
        # initialise pygame
        pygame.init()
        # generate the starting population
        self.generate_population()
        generation = 0
        # variable to hold the individual that is fittest in the current generation
        fittest = self.population[1][np.argmax(self.population[1])]
        while True:
            print "Generation: %s Average Fitness %s Fittest Individual: %s" % (generation, int(np.mean(self.population[1])), fittest)
            # generate the next generation of individuals
            self.evolve_population()
            fittest = self.population[1][np.argmax(self.population[1])]
            # if above a certain threshold, print out the chromosome
            if self.threshold[0]:
                if fittest > self.threshold[1]:
                    metadata= {"sizes":self.neural.sizes, "mutation":self.MUTATION_RATE,
                               "crossover":self.CROSSOVER_RATE, "seconds_elapsed":int(time.time()-self.start_time),
                               "wildcard":self.WILDCARD_RATE, "population_size":self.POPULATION_SIZE, "generations":generation,
                    "time":time.time(), "fitness":fittest, "desired_fitness":self.threshold[1]}
                    individual =  self.population[0][np.argmax(self.population[1])]
                    with open("networks/"+str(int(time.time())), "w") as f:
                        pickle.dump([metadata, individual], f)
                    print "Solution found with fitness %s" % (fittest)
                    break
            generation += 1

    def evolve_population(self):
        """Evolve the population based on simple genetic operators"""
        next_generation = []
        while len(next_generation) < self.POPULATION_SIZE:
            # generate a random number to determine what genetic operator to use, and then apply it
            # in order to generate a new individual for the next generation
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
        # generate a list of fitnesses, with a 1:1 corrospondance to the list of individuals
        # in the next generation
        # NOTE: This is the most time intensive part of the program, as it requires running
        # actual game simulations hundreds or thousands of times per generation
        fitnesses = []
        for individual in next_generation:
            fitness_value = self.fitness(individual)
            fitnesses.append(fitness_value)
        self.population = [next_generation, fitnesses]

    def generate_population(self):
        """Generate an initial population of entirely random networks"""
        population = []
        fitnesses = []
        for x in range(self.POPULATION_SIZE):
            individual = self.neural.generate_network()
            fitness_value = self.fitness(individual)
            population.append(individual)
            fitnesses.append(fitness_value)
        self.population = [population, fitnesses]

    def weighted_choice(self, weights):
        """Choose a random index in given a set of weights"""
        if sum(weights) == 0.0:
            return random.randint(0, len(weights)-1)
        totals = []
        running_total = 0

        for w in weights:
            w = w ** 1.0
            running_total += w
            totals.append(running_total)

        rnd = random.random() * running_total
        for i, total in enumerate(totals):
            if rnd < total:
                return i

    def mutate(self, individual):
        """Mutate a random neuron in a given network by choosing a new values for weight and bias"""
        layer = self.weighted_choice(self.neural.sizes[1:])+1
        neuron = random.randint(0, self.neural.sizes[layer]-1)
        connection = random.randint(0, self.neural.sizes[layer-1]-1)
        # mutate the weight
        individual[1][layer-1][neuron][connection] += np.random.normal()
        #mutate the bias
        individual[0][layer-1][neuron] += np.random.normal()
        return individual

    def crossover(self, individual_1, individual_2):
        layer = self.weighted_choice(self.neural.sizes[1:])+1
        layer_index = layer - 1
        neuron = random.randint(0, self.neural.sizes[layer]-1)
        connection = random.randint(0, self.neural.sizes[layer-1]-1)
        old_w_1 = individual_1[1]
        old_w_2 = individual_2[1]
        new_w_modded_layer = old_w_1[layer_index].tolist()[:neuron]\
                             + [old_w_1[layer_index][neuron][:connection].tolist() + old_w_2[layer_index][neuron][connection:].tolist()]\
                             +  old_w_2[layer_index].tolist()[neuron+1:]
        new_w_modded_layer = np.array(new_w_modded_layer)
        new_w = old_w_1[:layer_index]+ [new_w_modded_layer] + old_w_2[layer_index+1:]
        old_b_1 = individual_1[0]
        old_b_2 = individual_2[0]
        new_b_modded_layer = [old_b_1[layer_index][:neuron].tolist() + old_b_2[layer_index][neuron:].tolist()]
        new_b = old_b_1[:layer_index] + new_b_modded_layer + old_b_2[layer_index+1:]
        new_b[layer_index] = np.array(new_b[layer_index])

        return [new_b, new_w]

    def fitness(self, individual):
        sims = []
        # run multiple sims to get an average, as different simulations for the same network
        # can vary greatly as a result of the random nature of the collision game
        for x in range(self.TRIAL_NUMBER):
            sims.append(self.sim.run_simulation(individual, self.game))
        return np.mean(sims)

class CollisionGameSimulator(object):
    DESIRED_TICKS = 300
    output_order = [["K_UP", "K_DOWN"], ["K_LEFT", "K_RIGHT"]]
    NUMBER_CLOSEST_SQUARES = 4

    def __init__(self, neural):
        self.neural = neural
        self.game = collisiongame.Game

    def run_ai(self, network_time):
        """Run the network with the weights in the file solution.txt"""
        # initialise pygame
        p = None
        with open("networks/"+network_time, "rb") as f:
            p = pickle.load(f)

        print p[0]
        individual = p[1]

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

    def run_simulation(self, individual, game=None):
        """Run the simulation with no rendering for a given network"""

        screen_dimensions = ((collisiongame.WINDOW_WIDTH, collisiongame.WINDOW_HEIGHT))

        # create an instance of the Game() class
        if not game:
            game = Game()
        else:
            game.reset()

        ticks = 0
        total_movement = 0
        total_possible = math.sqrt(2*(game.player_speed**2))
        last_keys = {}
        direction_switches = 0

        while ticks < CollisionGameSimulator.DESIRED_TICKS and not game.game_over:
            # process events, as determined by the neural network
            keys = self.process_network_inputs(individual, game)
            if keys != last_keys:
                direction_switches += 1
            last_keys = keys

            # run the game logic and check for collisions
            game.logic()

            ticks += 1
        #determine if the player died on the edge, and add a fitness penalty if it did
        if game.player.rect.top <= 0 or \
            game.player.rect.bottom >= collisiongame.WINDOW_HEIGHT or \
            game.player.rect.left <= 0 or \
            game.player.rect.right >= collisiongame.WINDOW_WIDTH:
            ticks /= 2
        #boost the fitness if the program actively switches directions, else add a penalty
        if direction_switches > 1:
            return ticks * 2
        else:
            return ticks / 2


    def generate_inputs(self, game):
        """Generate the inputs for the neural network (xy of nearest edges/projectiles)"""
        sprites = game.projectile_list.sprites()
        closest_edge_x = -game.player.rect.centerx
        if game.player.rect.centerx >= collisiongame.WINDOW_WIDTH/2:
            closest_edge_x = collisiongame.WINDOW_WIDTH-game.player.rect.centerx
        closest_edge_y = -game.player.rect.centery
        if game.player.rect.centery >= collisiongame.WINDOW_HEIGHT/2:
            closest_edge_y = collisiongame.WINDOW_HEIGHT-game.player.rect.centery
        outputs = [closest_edge_x/collisiongame.WINDOW_WIDTH, closest_edge_y/collisiongame.WINDOW_HEIGHT]
        distances = [p.get_dist(game.player)[0] for p in sprites]
        for x in range(self.NUMBER_CLOSEST_SQUARES):
            closest = np.argmin(distances)
            closest_sprite = sprites.pop(closest)
            [dist, x_dist, y_dist] = closest_sprite.get_dist(game.player)
            distances.pop(closest)
            outputs.append(x_dist/collisiongame.WINDOW_WIDTH)
            outputs.append(y_dist/collisiongame.WINDOW_HEIGHT)
        return outputs


    def get_keys(self, outputs):
        """Turn the numerical outputs of the neural network into a dictionary of pressed keys"""
        keys = OrderedDict([["K_UP",False], ["K_DOWN",False], ["K_LEFT",False], ["K_RIGHT",False]])
        for x in range(len(outputs)):
            outputs[x] = abs(outputs[x])
        ind = np.sort(np.argpartition(np.asarray(outputs), -2)[-2:])
        max_index = ind[0]
        second_max_index = ind[1]
        keys[CollisionGameSimulator.output_order[max_index//2][max_index%2]] = True
        if outputs[second_max_index] >= (outputs[max_index]/10)*9 and not keys.keys()[second_max_index] in CollisionGameSimulator.output_order[max_index//2]:
            keys[CollisionGameSimulator.output_order[second_max_index//2][second_max_index%2]] = True
        return keys

    def process_network_inputs(self, individual, game):
        """Update the player's movement, using a given neural network"""
        keys = self.get_keys(self.neural.evaluate_network(individual, self.generate_inputs(game)))

        game.player_x_movement, game.player_y_movement = game.player.get_movement()

        # translate the key inputs from the network into actual movement in the game
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
    parser = argparse.ArgumentParser(description="Run genetic algorithm to learn to play collision game")
    parser.add_argument('--save', help="Save the resulting network to a file", action="store_true")
    parser.add_argument('--threshold', help="Set the threshold to save the file at", default=300)
    args = parser.parse_args()
    g = GeneticAlgorithm(threshold=[args.save, args.threshold])
    g.run_program()
