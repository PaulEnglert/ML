# -*- coding: utf-8 -*-

import numpy as np


def protected_division(v1, v2):
    zeros = v2 == 0
    v1[zeros] = 1
    v2[zeros] = 1
    return np.divide(v1, v2)


class Node:
    """Class to represent a node in the solution tree"""
    # FUNCTION SET
    f_set = [{
        'f': lambda X, v1, v2: np.add(v1, v2),
        'str': '+',
        'arity': 2}, {
        'f': lambda X, v1, v2: np.subtract(v1, v2),
        'str': '-',
        'arity': 2}, {
        'f': lambda X, v1, v2: np.multiply(v1, v2),
        'str': '*',
        'arity': 2}, {
        'f': lambda X, v1, v2: protected_division(v1, v2),
        'str': '/',
        'arity': 2}, {
        'f': lambda X, v1: np.sin(v1),
        'str': 'sin',
        'arity': 1}, {
        'f': lambda X, v1: np.cos(v1),
        'str': 'cos',
        'arity': 1}, {
        'f': lambda X, v1: np.tan(v1),
        'str': 'tan',
        'arity': 1}, {
        'f': lambda X, v1: np.exp(v1),
        'str': 'exp',
        'arity': 1}, {
        'f': lambda X, v1, v2: np.power(v1, v2),
        'str': 'pow',
        'arity': 2},  # {
        # 'f': lambda X, v1: np.log(v1),
        # 'str': 'log',
        # 'arity': 1}
    ]
    constants = [-10, -5, 5, 10]
    num_features = 0

    # RANDOMIZER FOR FUNCTION / TERMINAL SET
    @staticmethod
    def get_random_F():
        r = int(np.random.rand() * len(Node.f_set))
        return Node.f_set[r]

    @staticmethod
    def get_random_T():
        rcidx = int(np.random.rand() * len(Node.constants))
        rfidx = int(np.random.rand() * Node.num_features)
        T = [{
            'f': lambda X: np.ones(X.shape[0]) * float(Node.constants[rcidx]),
            'str': str(Node.constants[rcidx]),
            'arity': 0}, {
            'f': lambda X: X[:, rfidx],  # return random X element
            'str': str('x' + str(rfidx)),
            'arity': 0}]
        return T[1 if np.random.rand() > 0.5 else 0]

    @staticmethod
    def get_random_TF():
        if np.random.rand() > 0.5:
            return Node.get_random_F()
        return Node.get_random_T()

    #  Node Initialization
    def __init__(self, parent, function):
        # reference to parent Node
        self.parent = parent
        # references to children depending on functions arity
        self.children = []
        # this is a function combining inputs
        # or it's a function returning a constant/input value
        self.function = function

    def compute(self, X):
        if self.function['arity'] == 0:
            return self.function['f'](X)
        else:
            child_result = [c.compute(X) for c in self.children]
            return self.function['f'](X, *child_result)

    def calculate_depth(self):
        if len(self.children) > 0:
            return np.max([[c.calculate_depth() for c in self.children]]) + 1
        return 1

    def calculate_size(self):
        return np.sum([c.calculate_size() for c in self.children]) + 1

    def traverse_and_select(self, n, selected):
        n[0] = n[0] + 1  # workaround to pass int by reference
        if int(np.random.rand() * n[0]) == 0:
            selected = self
        if len(self.children) > 0:
            for child in self.children:
                selected = child.traverse_and_select(n, selected)
        return selected

    def __str__(self):
        if self.function['arity'] == 0:
            return self.function['str']
        elif self.function['arity'] == 1:
            return self.function['str'] + '(' + str(self.children[0]) + ')'
        elif self.function['arity'] == 2:
            return '(' + str(self.children[0]) + self.function['str'] + \
                str(self.children[1]) + ')'
        elif self.function['arity'] > 2:
            r = self.function['str'] + '('
            for i in range(self.function['arity']):
                r += str(self.children[i]) + ','
            return r + ')'

    def copy(self, parent=None):
        newself = Node(parent, self.function)
        for c in range(newself.function['arity']):
            newself.children.append(self.children[c].copy(newself))
        return newself


class Individual:
    """Individual in a Genetic Programming Population"""
    count = 0

    def __init__(self, min_depth, max_depth):
        self.id = Individual.count
        Individual.count += 1
        self.last_semantics = {}
        self.last_error = {}
        # initialize tree (root is always a function)
        self.root = Node(None, function=Node.get_random_F())
        self.grow(self.root, 0, min_depth, max_depth)
        self.calculate_dimensions()

    def compute(self, X, data_type='train'):
        np.seterr(all='ignore')
        self.last_semantics[data_type] = self.root.compute(X)
        np.seterr(all='warn')
        return self.last_semantics[data_type]

    def evaluate(self, X, Y, data_type='train'):
        self.compute(X)
        self.last_error[data_type] = np.sum(
            (self.last_semantics[data_type] - Y)**2)
        return self.last_error[data_type]

    def get_fitness(self, data_type):
        return self.last_error[data_type]

    def get_semantics(self, data_type):
        return self.last_semantics[data_type]

    # GENETIC OPERATORS
    def mutate(self):
        copy = self.copy()
        mutation_point = copy.random_node_choice()
        random_branch = copy.create_random()
        # copy self and update references at parent and new child
        if mutation_point.parent is not None:
            idx = mutation_point.parent.children.index(mutation_point)
            mutation_point.parent.children[idx] = random_branch
            random_branch.parent = mutation_point.parent
        else:
            copy.root = random_branch  # root has been replaced
        return copy

    def crossover(self, partner):
        # get cx points and copy to not get tangled up in refs
        copy_1 = self.copy()
        cx_point_1 = copy_1.random_node_choice()  # node to replace
        cx_point_2 = partner.random_node_choice().copy()  # replacement
        # update references of offspring
        if cx_point_1.parent is not None:
            idx = cx_point_1.parent.children.index(cx_point_1)
            cx_point_1.parent.children[idx] = cx_point_2
            cx_point_2.parent = cx_point_1.parent
        else:
            copy_1.root = cx_point_2
        return copy_1

    # UTILITIES for tree management
    def grow(self, parent, cur_depth, min_depth=0, max_depth=None):
        for i in range(parent.function['arity']):
            if cur_depth < min_depth:
                parent.children.append(Node(parent, Node.get_random_F()))
            elif max_depth is None or cur_depth < max_depth:
                parent.children.append(
                    Node(parent, Node.get_random_TF()))
            else:  # force terminal element
                parent.children.append(
                    Node(parent, Node.get_random_T()))
            # grow new branch
            self.grow(parent.children[i], cur_depth + 1, min_depth, max_depth)

    def create_random(self):
        parent = Node(None, Node.get_random_TF())
        self.grow(parent, 0)
        return parent

    def calculate_dimensions(self, only_depth=False):
        self.depth = self.root.calculate_depth()
        if not only_depth:
            self.size = self.root.calculate_size()
        return (self.depth, self.size,)

    def random_node_choice(self):
        n = [0]  # pass integer by reference
        return self.root.traverse_and_select(n, None)

    def __str__(self):
        return str(self.root)

    def copy(self):
        newself = Individual(1, 1)
        newself.root = self.root.copy()
        return newself

    def better(self, other, data_type='train'):  # TODO make this dynamic
        return self.get_fitness(data_type) <= other.get_fitness(data_type)


class Population:
    tournament_size = 4

    def __init__(self, size, selection_type=None):
        if selection_type is None:
            selection_type = Population.tournament
        self.size = size
        self.individuals = []

    def create_individuals(
            self, init_min_depth=1, max_depth=6, init_type=None):
        if init_type is None:
            init_type = Population.ramped
        self.individuals = init_type(
            self.size, init_min_depth, max_depth)

    def select(self, count=1):
        return [Population.tournament(self.individuals) for i in range(count)]

    def get_best(self):
        return Population.filter_best(self.individuals)

    def evaluate(self, X, Y):
        for i in self.individuals:
            i.evaluate(X, Y, 'train')

    @staticmethod
    def filter_best(array):
        best = None
        for i in array:
            if best is None or i.better(best):
                best = i
        return best

    @staticmethod
    def ramped(size, min_depth, max_depth):
        individuals = []
        bucket_size = int(size / (1 + max_depth - min_depth))
        for bucket in range(min_depth, max_depth + 1):
            for i in range(bucket_size):
                if i % 2 == 0:  # force full growth
                    individuals.append(Individual(bucket, bucket))
                else:  # allow normal grow
                    individuals.append(Individual(min_depth, bucket))
        # fill up missing, e.g. due to unclean bucketing:
        while len(individuals) < size:
            individuals.append(Individual(min_depth, max_depth))
        return individuals

    @staticmethod
    def grow(size, min_depth, max_depth, num_features, constants):
        individuals = []
        while len(individuals) < size:
            individuals.append(Individual(min_depth, max_depth))
        return individuals

    @staticmethod
    def tournament(individuals):
        participants = [individuals[int(np.random.rand() * len(individuals))]
                        for i in range(Population.tournament_size)]
        return Population.filter_best(participants)


class GP:
    """Standard Genetic Programming using Tree-based Solutions"""
    reproduction_probability = .1
    mutation_probability = .3
    crossover_probability = .8
    apply_depth_limit = True
    depth_limit = 17

    def __init__(self, num_features, constants, size):
        Node.constants = constants
        Node.num_features = num_features

        self.population = Population(size)
        self.population.create_individuals(
            init_min_depth=1, max_depth=6, init_type=None)

    def evolve(self, X, Y, generations):
        for g in range(generations):
            log_str = '[{0:4}] '.format(g)
            # evaluate
            self.population.evaluate(X, Y)

            new_population = Population(self.population.size)
            # elitism
            new_population.individuals.append(self.population.get_best())
            # evolve
            while len(new_population.individuals) < new_population.size:
                p1, p2 = self.population.select(2)
                r = np.random.rand()
                offspring = p1
                if r < GP.crossover_probability:
                    offspring = p1.crossover(p2)
                elif r < (1 - GP.reproduction_probability):
                    offspring = p1.mutate()
                if GP.apply_depth_limit:
                    d, s = offspring.calculate_dimensions(only_depth=True)
                    if d > GP.depth_limit:
                        offspring = p1  # overwrite an offspring with parent
                new_population.individuals.append(offspring)  # add
            # update new population
            self.population = new_population
            self.population.evaluate(X, Y)

            print log_str + ' best training error={0}'.format(
                self.population.get_best().get_fitness('train'))
