import random
from copy import deepcopy

"""
The problem:

Find a way to use numbers 0 through 9 and operators +, -, /, and * in order to make a specified number

Return types:
float
Node types:
constant
variable
function
"""
NODE_TYPES = ["function", "constant", "variable"]
RETURN_TYPES = ["float"]

def add(children):
    num = children[0]
    children.pop(0)
    for child in children:
        num += child
    return num

def subtract(children):
    num = children[0]
    children.pop(0)
    for child in children:
        num -= child
    return num

def multiply(children):
    num = children[0]
    children.pop(0)
    for child in children:
        num *= child
    return num

def divide(children):
    num = children[0]
    children.pop(0)
    for child in children:
        try:
            num /= child
        except ZeroDivisionError:
            pass
    return num

nodes = {
        "function": {"add": [add, "float", ["float", "float"]],
            "subtract": [subtract, "float", ["float", "float"]],
            "divide": [divide, "float", ["float", "float"]],
            "multiply": [multiply, "float", ["float", "float"]]},
        "variable": {},
        "constant": {str(x): [float(x), "float"] for x in range(10)}
        } 

class Node():
    def __init__(self, nodetype, return_type, children=None, function=None, constant=None, variable = None, variable_value=None):
        self.nodetype = nodetype
        if self.nodetype not in NODE_TYPES:
            print("WARNING: node has invalid nodetype")
        self.children = children # also nodes
        self.function = function
        self.constant = constant
        # variable is the id of the variable, variable_value is its value, set during the gp run
        self.variable = variable
        self.variable_value = variable_value
    
    def set_variable_value(self, value):
        if self.nodetype == variable:
            self.variable_value = value

    def get_value(self):
        if self.nodetype == "function":
            if children:
                childreturns = []
                for child in children:
                    childreturns.append(get_value())
                self.function(children)
            else:
                self.function([])
        elif self.nodetype == "constant":
            return constant
        else:
            # variable
            return self.variable_value

class Run():
    MAX_DEPTH = 5
    CONSTANT_CHANCE = 1.0
    FUNCTION_MUTATION_PROB = 0.9
    FUNCTION_CROSSOVER_PROB = 0.9
    POPULATION_SIZE = 100
    DESIRED_NUMBER = 59
    CROSSOVER_RATE = 0.7
    MUTATION_RATE = 0.1

    def __init__(self):
        self.variable_injections = {}
        self.population = self.make_population()

    def run_program(self):
        generations = 0
        solution = None
        while not solution:
            print("G," + str(generations) + ",F," + str(sum(self.population[1])))
            if len(self.population[0]) == 1:
                solution = self.population 
            else:
                self.population = self.next_generation(self.population)
            generations += 1
        print(self.evaluate_tree(solution[0][0]))
        print("Solution Found!: " + solution[0][0])

    def next_generation(self, generation):
        number = random.uniform(0.0, 1.0)
        next_generation = []
        while len(next_generation) < Run.POPULATION_SIZE:
            next_selection = random.uniform(0.0, 1.0)
            if next_selection <= Run.MUTATION_RATE:
                next_generation.append(self.mutate(generation[0][self.weighted_choice(generation[1])]))
            elif next_selection <= Run.CROSSOVER_RATE + Run.MUTATION_RATE:
                next_generation.append(self.crossover(generation[0][self.weighted_choice(generation[1])], generation[0][self.weighted_choice(generation[1])]))
            else:
                # copy an individual
                next_generation.append(generation[0][self.weighted_choice(generation[1])])
        fitnesses = []
        for individual in next_generation:
            fitness_value = self.fitness(individual)
            fitnesses.append(fitness_value)
            if fitness_value == -1:
                return [[individual], [fitness_value]]
        return [next_generation, fitnesses]
    
    def make_population(self):
        pop = [[], []]
        for x in range(Run.POPULATION_SIZE):
            tree = self.make_tree(0)
            fitness = self.fitness(tree)
            pop[0].append(tree)
            pop[1].append(fitness)
            if fitness == -1:
                return [[tree], [fitness]]
        return pop
    
    def fitness(self, tree):
        value = None
        output = self.evaluate_tree(tree)
        try:
            if abs(Run.DESIRED_NUMBER-output) < 1.0:
                value = abs(Run.DESIRED_NUMBER-output)
                if value == 0.0:
                    value = -1
            else:
                value = 1/abs(Run.DESIRED_NUMBER-output)
        except ZeroDivisionError:
            return -1
        return value

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

    def make_tree(self, depth):
        node = None
        if depth >= 5:
            choose = random.uniform(0, 1)
            if choose > Run.CONSTANT_CHANCE:
                #pick a variable
                var_to_use = nodes["variable"][random.choice(list(nodes["variable"].keys()))]
                node = Node("variable", var_to_use[1], variable=var_to_use)
            else:
                const_to_use = nodes["constant"][random.choice(list(nodes["constant"].keys()))]
                node = Node("constant", const_to_use[1], constant=const_to_use[0])
        else:
            func_to_use = nodes["function"][random.choice(list(nodes["function"].keys()))]
            children = []
            for child in func_to_use[2]:
                children.append(self.make_tree(depth+1))
            node = Node("function", func_to_use[1], children=children, function=func_to_use[0])
        return node

    def evaluate_tree(self, node):
        if node.nodetype == "constant":
            return node.constant
        elif node.nodetype == "variable":
            return self.variable_injections[node.variable] 
        else:
            evals = [self.evaluate_tree(child) for child in node.children]
            return node.function(evals)

    def crossover(self, node1, node2):
        depth = random.randint(0, Run.MAX_DEPTH-1)
        node1 = deepcopy(node1)
        node2 = deepcopy(node2)
        temp_node_1 = node1
        temp_node_2 = node2
        for x in range(depth):
            if temp_node_1.nodetype == "function":
                temp_node_1 = random.choice(temp_node_1.children)
            if temp_node_2.nodetype == "function":
                temp_node_2 = random.choice(temp_node_2.children)
        if temp_node_2.children:
            temp_node_1.children[random.randint(0, len(temp_node_1.children)-1)] = random.choice(temp_node_2.children)
        else:
            if temp_node_1.children:
                temp_node_1.children[random.randint(0, len(temp_node_1.children)-1)] = temp_node_2
            else:
                temp_node_1
        return node1

    def mutate(self, node, start_depth=0):
        mutate_prob=1/(Run.MAX_DEPTH**2)
        node = deepcopy(node)
        if random.uniform(0, 1) < mutate_prob:
            return self.make_tree(start_depth)
        else:
            if node == "function":
                for node in node.children:
                    node = self.mutate(node, start_depth+1)
        return node

        """
        node = deepcopy(node)
        mutation_type = random.uniform(0, 1)
        if mutation_type < self.FUNCTION_MUTATION_PROB:
            # take away one from range
            depth = random.randint(0, Run.MAX_DEPTH-1)
            if depth is not 0:
                range_to_iterate = depth - 1
                temp_nd = node
                for x in range(range_to_iterate):
                    print(x)
                    temp_nd = random.choice(temp_nd.children)
                if temp_nd.nodetype != "function":
                    mutate(node)
                new_tree_index = random.randint(0, len(temp_nd.children)-1) 
                temp_nd.children[new_tree_index] = self.make_tree(depth)
            else:
                # depth is 0, replace whole tree
                node = self.make_tree(0)
        else:
            temp_nd = node
            temp_nd_child_index = None
            depth = 0
            while True:
                temp_nd_child_index = random.randint(0, len(temp_nd.children)-1)#random.choice(temp_nd.children)
                if temp_nd.children[temp_nd_child_index].nodetype != "function":
                    break
                else:
                    temp_nd = temp_nd.children[temp_nd_child_index]
                    depth += 1
            node = self.make_tree(depth)
        return node"""

if __name__ == "__main__":
    r = Run()
    r.run_program()
