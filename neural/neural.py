from copy import deepcopy
import math, random
import numpy as np
import collisiongame
from collisiongame import Game
import pygame
from collections import OrderedDict

class NeuralAlgorithm(object):
    """Class that holds parameters for the neural network and has the functions required to use them"""
    NUM_HIDDEN_LAYERS = 3
    NUM_NEURONS_PER_LAYER = 8
    NUM_INPUTS = 8
    NUM_OUTPUTS = 4
    ACTIVATION_RESPONSE = 1.0
    BIAS = -1.0
    # the number of individual connections in the network
    NUM_CONNECTIONS = NUM_INPUTS*NUM_NEURONS_PER_LAYER + \
                                       (NUM_HIDDEN_LAYERS-1)*NUM_NEURONS_PER_LAYER**2 + \
                                        NUM_NEURONS_PER_LAYER*NUM_OUTPUTS
    INITIAL_WEIGHTS_RANGE = 1.5
    INITIAL_THRESHOLD_RANGE = 0.5

    def generate_network(self):
        """Generate a random network, with the initial values having a uniform distribution with given min and max values"""
        weights = []
        for x in range(NeuralAlgorithm.NUM_CONNECTIONS):
            weights.append(random.uniform(-self.INITIAL_WEIGHTS_RANGE, self.INITIAL_THRESHOLD_RANGE))
        weights.append(random.uniform(-self.INITIAL_THRESHOLD_RANGE, self.INITIAL_THRESHOLD_RANGE))
        return weights

    def evaluate_network(self, weights, inputs):
        previous_layer = inputs
        # variable to hold where this layer starts in the long list of weights
        weight_layer_starting_index = 0
        # for each layer that we have to generate
        for x in range(NeuralAlgorithm.NUM_HIDDEN_LAYERS + 1):
            # set the next layer length (just the hidden layer length if we are not generating the output layer)
            next_layer_length = self.NUM_OUTPUTS if x == NeuralAlgorithm.NUM_HIDDEN_LAYERS else NeuralAlgorithm.NUM_NEURONS_PER_LAYER
            next_layer = []
            # for each neuron in the next layer
            for n in range(next_layer_length):
                neuron = 0
                # for each neuron in the previous layer's output and its weight going to this neuron
                for value, weight in zip(previous_layer, range(n*len(previous_layer), len(previous_layer)*(n+1))):
                    # add the weighted input to the total for this neuron
                    neuron += value * weights[weight+weight_layer_starting_index]
                #add in the bias
                neuron += NeuralAlgorithm.BIAS * weights[-1]
                #add the neuron to the next layer, after passing the sum through the sigmoid function
                next_layer.append(self.sigmoid(neuron))
            # update the starting index of the next layer
            weight_layer_starting_index += next_layer_length*len(previous_layer)
            previous_layer = next_layer
        return previous_layer

    def sigmoid(self, number):
        """Sigmoid function, used to scale the output of each neuron so it starts to activate at certain values (https://en.wikipedia.org/wiki/Sigmoid_function)"""
        return 1/(1+math.exp(-number/NeuralAlgorithm.ACTIVATION_RESPONSE))

class GeneticAlgorithm(object):
    SIGMA_MUTATION = 1.5
    POPULATION_SIZE = 100
    MUTATION_RATE = 0.1
    CROSSOVER_RATE = 0.7
    WILDCARD_RATE = 0.00
    ELITE = 4

    def __init__(self):
        self.neural = NeuralAlgorithm()
        self.sim = CollisionGameSimulator(self.neural)
        self.game = Game()
        # mock a couple of the time dependant functions in the game class
        self.game.score_clock_tick = lambda: 1000/collisiongame.FPS
        self.game.projectile_direction_tick = lambda: 1000/collisiongame.FPS

    def run_program(self):
        # initialise pygame
        pygame.init()
        # generate the starting population
        self.generate_population()
        generation = 0
        # variable to hold the individual that is fittest in the current generation
        fittest = self.population[1][np.argmax(self.population[1])]
        while CollisionGameSimulator.DESIRED_TICKS:
            print "Generation: %s Average Fitness %s Fittest Individual: %s" % (generation, int(np.mean(self.population[1])), fittest)
            # generate the next generation of individuals
            self.evolve_population()
            fittest = self.population[1][np.argmax(self.population[1])]
            # if above a certain threshold, print out the chromosome
            if fittest > 280:
                print self.population[0][np.argmax(self.population[1])]
            generation += 1
        print "Solution Found, Fitness: %s , Weights:" % (fittest)
        print self.population[0][np.argmax(self.population[1])]
        # save the solution in a file
        f = open('solution.txt', 'r+')
        f.truncate()
        f.write(str(self.population[0][np.argmax(self.population[1])]))
        f.close()

    def evolve_population(self):
        """Evolve the population based on simple genetic operators"""
        # start the next generation with a given number of "elite" individuals - the
        # fittest few of the population from the last generation, to ensure that we
        # do not lose an unusually good crossover or combination
        next_generation = [i[1] for i in sorted(zip(self.population[1], self.population[0]))[:self.ELITE]]
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
        """Mutate a random weight in a given network by choosing a new value in a gaussian distribution around the initial value"""
        new_individual = individual
        position = random.randint(0, NeuralAlgorithm.NUM_CONNECTIONS)
        new_individual[position] = np.random.normal(scale=GeneticAlgorithm.SIGMA_MUTATION, loc=0)
        return new_individual

    def crossover(self, individual_1, individual_2):
        """Crossover two individuals at a random point, and return the offspring"""
        crossover_position = random.randint(1, NeuralAlgorithm.NUM_CONNECTIONS-1)
        new_individual_1 = individual_1[0:crossover_position] + individual_2[crossover_position:]
        return new_individual_1

    def fitness(self, individual):
        sims = []
        # run multiple sims to get an average, as different simulations for the same network
        # can vary greatly as a result of the random nature of the collision game
        for x in range(10):
            sims.append(self.sim.run_simulation(individual, self.game))
        return np.mean(sims)

class CollisionGameSimulator(object):
    DESIRED_TICKS = 300
    output_order = [["K_UP", "K_DOWN"], ["K_LEFT", "K_RIGHT"]]
    NUMBER_CLOSEST_SQUARES = 3

    def __init__(self, neural):
        self.neural = neural
        self.game = collisiongame.Game

    def run_ai(self):
        """Run the network with the weights in the file solution.txt"""
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
    g = GeneticAlgorithm()
    g.run_program()
