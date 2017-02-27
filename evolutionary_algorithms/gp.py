# -*- coding: utf-8 -*-

import os
from datetime import datetime
import numpy as np
from utilities.util import StopRecursion, setup_logger, reset_logger

import logging as _l

_lmain = _l.getLogger('gp.main')
_lftest = _l.getLogger('gp.ftest')
_lftrain = _l.getLogger('gp.ftrain')


def protected_division(v1, v2):
    zeros = np.abs(v2) < 0.00000000001
    v2[zeros] = 1
    return np.divide(v1, v2)


class Node(object):
    """Class to represent a node in the solution tree"""
    count = 0

    # FUNCTION SET
    f_add = {
        'f': lambda X, v1, v2: np.add(v1, v2),
        'str': '+',
        'arity': 2}
    f_subtract = {
        'f': lambda X, v1, v2: np.subtract(v1, v2),
        'str': '-',
        'arity': 2}
    f_multiply = {
        'f': lambda X, v1, v2: np.multiply(v1, v2),
        'str': '*',
        'arity': 2}
    f_divide = {
        'f': lambda X, v1, v2: protected_division(v1, v2),
        'str': '/',
        'arity': 2}
    f_sin = {
        'f': lambda X, v1: np.sin(v1),
        'str': 'sin',
        'arity': 1}
    f_cos = {
        'f': lambda X, v1: np.cos(v1),
        'str': 'cos',
        'arity': 1}
    f_tan = {
        'f': lambda X, v1: np.tan(v1),
        'str': 'tan',
        'arity': 1}
    f_exp = {
        'f': lambda X, v1: np.exp(v1),
        'str': 'exp',
        'arity': 1}
    f_pow = {
        'f': lambda X, v1, v2: np.power(v1, v2),
        'str': 'pow',
        'arity': 2}
    f_log = {
        'f': lambda X, v1: np.log(v1),
        'str': 'log',
        'arity': 1}

    f_set = [f_add, f_multiply, f_subtract, f_divide]
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
        threshold = float(len(Node.constants)) /\
            (len(Node.constants) + Node.num_features)
        return T[0 if np.random.rand() < threshold else 1]

    @staticmethod
    def get_random_TF():
        if np.random.rand() < 0.5:
            return Node.get_random_F()
        return Node.get_random_T()

    #  Node Initialization
    def __init__(self, parent, function):
        self.id = Node.count
        Node.count += 1
        # reference to parent Node
        self.parent = parent
        # references to children depending on functions arity
        self.children = []
        # this is a function combining inputs
        # or it's a function returning a constant/input value
        self.function = function

    def traverse_and_collect(
            self, node_dict, X, n_steps, early_exit_depth=None):
        # collect all information in one pass,
        # exit early if max depth is reached
        node_dict[self.id] = self
        inputs = []
        size = 1
        depth = -1  # otherwise root is not 0 depth
        # if the traversal already went to deep, exit by Exception
        if early_exit_depth is not None and n_steps > early_exit_depth:
            raise StopRecursion()
        # calculate data for all children
        for c in self.children:
            y, s, d = c.traverse_and_collect(
                node_dict, X, n_steps + 1, early_exit_depth)
            inputs.append(y)
            if d > depth:
                depth = d
            size += s
        # pass on the own data
        return (self.function['f'](X, *inputs), size, depth + 1)

    def compute(self, X):
        if self.function['arity'] == 0:
            return self.function['f'](X)
        else:
            child_result = [c.compute(X) for c in self.children]
            return self.function['f'](X, *child_result)

    def calculate_depth(self):
        if len(self.children) > 0:
            return np.max([[c.calculate_depth() for c in self.children]]) + 1
        return 0

    def calculate_size(self):
        return np.sum([c.calculate_size() for c in self.children]) + 1

    # select a random node with uniform distribution from the tree
    def tree_select_uniform_rand(self, n, selected):
        n[0] = n[0] + 1  # workaround to pass int by reference
        if int(np.random.rand() * n[0]) == 0:
            selected = self
        if len(self.children) > 0:
            for child in self.children:
                selected = child.tree_select_uniform_rand(n, selected)
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

    def copy(self, nodes_dict=None, parent=None):
        newself = Node(parent, self.function)
        if nodes_dict is not None:
            nodes_dict[newself.id] = newself
        for c in range(newself.function['arity']):
            newself.children.append(
                self.children[c].copy(nodes_dict=nodes_dict, parent=newself))
        return newself


class Individual(object):
    """Individual in a Genetic Programming Population"""
    count = 0
    apply_depth_limit = True
    depth_limit = 17

    def __init__(self, min_depth, max_depth):
        self.id = Individual.count
        Individual.count += 1
        self.last_semantics = {'train': None, 'test': None}
        self.last_error = {'train': None, 'test': None}
        # initialize tree
        if max_depth == 0:  # force end of tree at root
            self.root = Node(None, function=Node.get_random_T())
        elif min_depth > 0:  # prevent ending of tree at root
            self.root = Node(None, function=Node.get_random_F())
        else:  # probabilistic ending of tree at root
            self.root = Node(None, function=Node.get_random_TF())
        self.nodes = {self.root.id: self.root}
        self.grow(self.root, 0, min_depth, max_depth, nodes_dict=self.nodes)

    def compute(self, X, data_type):
        # compute output of input matrix
        np.seterr(all='ignore')
        self.last_semantics[data_type] = self.root.compute(X)
        np.seterr(all='warn')
        return self.last_semantics[data_type]

    # evaluate individual data set and collect as much information
    # as possible, including: list of nodes, depth, size and output
    # catch the early exit Exception of the traversal if max depth is reached
    def __evaluate(self, X, Y, data_type='train'):
        self.nodes = {}
        np.seterr(all='ignore')
        dl = Individual.depth_limit if Individual.apply_depth_limit else None
        try:
            result, self.size, self.depth = self.root.traverse_and_collect(
                self.nodes, X, 0, dl)
        except StopRecursion:
            result = [np.Inf for i in X]
            self.depth = np.Inf
            self.size = np.Inf
        np.seterr(all='warn')
        self.last_semantics[data_type] = result
        self.last_error[data_type] = np.sqrt(np.sum(
            (self.last_semantics[data_type] - Y)**2) / X.shape[0])
        return self.last_error[data_type]

    # wrapper to evaluate all three data partitions
    def evaluate(self, X, Y, testX=None, testY=None):
        # collect as much information as possible here
        self.__evaluate(X, Y, 'train')
        if self.depth == np.Inf:
            return
        # for these only run the computation
        if testX is not None and testY is not None:
            self.compute(testX, 'test')
            self.last_error['test'] = np.sqrt(np.sum(
                (self.last_semantics['test'] - testY)**2) / testX.shape[0])

    def get_fitness(self, data_type):
        return self.last_error[data_type]

    def get_semantics(self, data_type):
        return self.last_semantics[data_type]

    # GENETIC OPERATORS
    def mutate(self):
        # before_self = str(self)
        copy = self.copy()
        mutation_point = copy.random_node_choice()
        parent = mutation_point.parent
        random_branch = copy.create_random(
            max_depth=GP.mutation_maximum_depth)
        # copy self and update references at parent and new child
        if parent is not None:
            idx = parent.children.index(mutation_point)
            parent.children[idx] = random_branch
            random_branch.parent = parent
        else:
            copy.root = random_branch  # root has been replaced

        # after_self = str(self)
        # assert after_self == before_self
        return copy

    def crossover(self, partner):
        # get cx points and copy to not get tangled up in refs
        # before_self = str(self)
        newself = self.copy()
        oldbranch = newself.random_node_choice()  # node to replace
        parent = oldbranch.parent
        newbranch = partner.random_node_choice().copy()  # replacement
        # update references of offspring
        if parent is not None:
            idx = parent.children.index(oldbranch)
            parent.children[idx] = newbranch
            newbranch.parent = parent
        else:
            newself.root = newbranch
        # after_self = str(self)
        # assert after_self == before_self
        return newself

    # UTILITIES for tree management
    def grow(
            self, parent, cur_depth, min_depth=0,
            max_depth=None, nodes_dict=None):
        if max_depth is None:
            max_depth = 950  # prevent recursion depth exception
        for i in range(parent.function['arity']):
            if cur_depth < min_depth:
                newnode = Node(parent, Node.get_random_F())
            elif max_depth is None or cur_depth < max_depth:
                newnode = Node(parent, Node.get_random_TF())
            else:  # force terminal element
                newnode = Node(parent, Node.get_random_T())
                if cur_depth == 950:
                    print "WARNING: Reached Maximum Recursion Limit!"
            parent.children.append(newnode)
            if nodes_dict is not None:
                nodes_dict[newnode.id] = newnode
            # grow new branch
            self.grow(
                parent.children[i], cur_depth + 1, min_depth=min_depth,
                max_depth=max_depth, nodes_dict=nodes_dict)

    def create_random(self, max_depth=None):
        parent = Node(None, Node.get_random_TF())
        self.grow(parent, 0, max_depth=max_depth)
        return parent

    def calculate_depth(self):
        self.depth = self.root.calculate_depth()
        return self.depth

    def calculate_size(self):
        self.size = self.root.calculate_size()
        return self.size

    def random_node_choice(self):
        return self.nodes[np.random.choice(self.nodes.keys())]

    def __str__(self):
        return str(self.root)

    def copy(self):
        newself = Individual(0, 0)
        newself.depth = self.depth
        newself.size = self.size
        for k in self.last_semantics.keys():
            if self.last_semantics[k] is not None:
                newself.last_semantics[k] = np.copy(self.last_semantics[k])
        for k in self.last_error.keys():
            if self.last_error[k] is not None:
                newself.last_error[k] = self.last_error[k]
        newself.nodes = {}
        newself.root = self.root.copy(newself.nodes)
        assert len(newself.nodes) == len(self.nodes)
        return newself

    def better(self, other, data_type='train'):
        return self.get_fitness(data_type) < other.get_fitness(data_type)


class Population(object):
    tournament_size = 4

    def __init__(self, size, selection_type=None):
        if selection_type is None:
            self.selection_type = Population.tournament
        else:
            self.selection_type = selection_type
        self.size = size
        self.individuals = []

    def create_individuals(
            self, init_min_depth=0, max_depth=6, init_type=None):
        if init_type is None:
            init_type = Population.ramped
        self.individuals = init_type(
            self.size, init_min_depth, max_depth)

    def select(self, count=1):
        if count == 1:
            return self.selection_type(self.individuals)
        return [self.selection_type(self.individuals)
                for i in range(count)]

    def get_best(self, data_type='train'):
        return Population.filter_best(self.individuals, data_type)

    def evaluate(self, X, Y, testX=None, testY=None):
        for i in self.individuals:
            i.evaluate(X, Y, testX, testY)

    @staticmethod
    def filter_best(array, data_type='train'):
        return array[np.argmin([a.get_fitness(data_type) for a in array])]

    @staticmethod
    def ramped(size, min_depth, max_depth):
        individuals = []
        bucket_size = int(size / (1 + max_depth - min_depth))
        for bucket in range(min_depth, max_depth + 1):
            for i in range(bucket_size):
                if i % 2 == 0:  # allow normal grow
                    individuals.append(Individual(min_depth, bucket))
                else:  # force full growth
                    individuals.append(Individual(bucket, bucket))
        # fill up missing, e.g. due to unclean bucketing:
        full = False
        while len(individuals) < size:
            if not full:
                individuals.append(Individual(min_depth, max_depth))
                full = True
            if full:
                individuals.append(Individual(max_depth, max_depth))
                full = False
        return individuals

    @staticmethod
    def grow(size, min_depth, max_depth, num_features, constants):
        individuals = []
        while len(individuals) < size:
            individuals.append(Individual(min_depth, max_depth))
        return individuals

    @staticmethod
    def full(size, min_depth, max_depth, num_features, constants):
        individuals = []
        while len(individuals) < size:
            individuals.append(Individual(max_depth, max_depth))
        return individuals

    @staticmethod
    def tournament(individuals):
        participants = [individuals[int(np.random.rand() * len(individuals))]
                        for i in range(Population.tournament_size)]
        return Population.filter_best(participants)


class GP(object):
    """Standard Genetic Programming using Tree-based Solutions"""
    reproduction_probability = .0
    mutation_probability = .1
    crossover_probability = .9
    max_initial_depth = 6
    apply_depth_limit = True
    depth_limit = 17
    mutation_maximum_depth = 6
    log_verbose = True
    log_stdout = True
    debug = False
    log_file_path = 'results'

    def __init__(self, num_features, constants, size):
        self.name = "Standard GP"
        Node.constants = constants
        Node.num_features = num_features

        self.population = Population(size)
        self.population.create_individuals(
            init_min_depth=1, max_depth=GP.max_initial_depth)

        self.prepare_logging()
        self.log_config()

    def evolve(
            self, X, Y, testX=None, testY=None, generations=25):
        Individual.depth_limit = GP.depth_limit
        Individual.apply_depth_limit = GP.apply_depth_limit

        # evaluate
        self.population.evaluate(X, Y, testX, testY)
        best = self.population.get_best()

        for g in range(generations):
            log_str = '[{0:4}] '.format(g)
            # new population
            new_population = Population(self.population.size)
            # elitism
            new_population.individuals.append(best)
            # variation
            if GP.log_verbose:
                size_violation_count = 0
                mutation_count = 0
                crossover_count = 0
                size_sum = best.size
                depth_sum = best.depth
            while len(new_population.individuals) < new_population.size:
                # select parents
                p1 = self.population.select()

                # determine operator
                r = np.random.rand()
                offspring = p1
                # do crossover
                if r < GP.crossover_probability:
                    if GP.log_verbose:
                        crossover_count += 1
                    offspring = p1.crossover(self.population.select())
                    offspring.evaluate(X, Y, testX, testY)
                # do mutation
                elif r < GP.crossover_probability + GP.mutation_probability:
                    if GP.log_verbose:
                        mutation_count += 1
                    offspring = p1.mutate()
                    offspring.evaluate(X, Y, testX, testY)

                # check depth
                if GP.apply_depth_limit:
                    if offspring.depth > GP.depth_limit:
                        if GP.log_verbose:
                            size_violation_count += 1
                        offspring = p1  # overwrite an offspring with p1
                if GP.log_verbose:
                    size_sum += offspring.size
                if GP.log_verbose:
                    depth_sum += offspring.depth
                # add offspring to new population
                new_population.individuals.append(offspring)
            # update new population
            self.population = new_population
            best = self.population.get_best()

            # logging
            log_str += ' best training error={0}'.format(
                best.get_fitness('train'))
            log_str += ' with test error={0}'.format(
                best.get_fitness('test'))
            if GP.log_stdout:
                print log_str
            if GP.log_verbose:
                self.log_state(
                    g, best, log_str, size_violation_count=size_violation_count,
                    crossover_count=crossover_count, mutation_count=mutation_count,
                    avg_size=(size_sum / self.population.size), avg_depth=(depth_sum / self.population.size))

    def log_state(self, generation, best, log_str, **kwargs):
        if GP.log_verbose:
            _lmain.info(log_str)
            _lmain.info('best individual: \n{0}'.format(best))
            _lmain.info('number of size violations: {0}'.format(kwargs.get('size_violation_count', '')))
            _lmain.info('number of crossovers: {0}'.format(kwargs.get('crossover_count', '')))
            _lmain.info('number of mutations: {0}'.format(kwargs.get('mutation_count', '')))
            _lmain.info('avg size: {0}'.format(kwargs.get('avg_size'), ''))
            _lmain.info('avg depth: {0}'.format(kwargs.get('avg_depth', '')))
        # log train fitness
        _lftrain.info('{0};{1};{2};{3}'.format(
            generation,
            best.get_fitness('train'),
            best.size,
            best.depth))
        # log test fitness
        _lftest.info('{0};{1}'.format(
            generation,
            best.get_fitness('test')))

    def prepare_logging(self):
        self.rid = str(int(
            (datetime.now() - datetime(1970, 1, 1)).total_seconds()))

        if not GP.log_verbose:
            return

        base = os.path.join(os.getcwd(), GP.log_file_path)
        if not os.path.exists(base):
            os.makedirs(base)

        # update loggers
        level = _l.INFO
        if GP.debug:
            level = _l.DEBUG
        reset_logger(logger_name='gp.main')
        setup_logger(logger_name='gp.main', log_file=os.path.join(base, self.rid + '-gp.log'),
                     level=level)
        reset_logger(logger_name='gp.ftrain')
        setup_logger(logger_name='gp.ftrain', log_file=os.path.join(base, self.rid + '-fitnesstrain.txt'),
                     level=level)
        reset_logger(logger_name='gp.ftest')
        setup_logger(logger_name='gp.ftest', log_file=os.path.join(base, self.rid + '-fitnesstest.txt'),
                     level=level)

        _lmain.info(self.name)
        _lftest.info('Gen;Test Fitness')
        _lftrain.info('Gen;Train Fitness;Size;Depth')

    def log_config(self):
        _lmain.info('------------------------')
        _lmain.info('CONFIG')
        _lmain.info('Number of Features={0}'.format(Node.num_features))
        _lmain.info('Constants={0}'.format(Node.constants))
        _lmain.info('reproduction_probability={0}'.format(GP.reproduction_probability))
        _lmain.info('mutation_probability={0}'.format(GP.mutation_probability))
        _lmain.info('crossover_probability={0}'.format(GP.crossover_probability))
        _lmain.info('max_initial_depth={0}'.format(GP.max_initial_depth))
        _lmain.info('apply_depth_limit={0}'.format(GP.apply_depth_limit))
        _lmain.info('depth_limit={0}'.format(GP.depth_limit))
        _lmain.info('mutation_maximum_depth={0}'.format(GP.mutation_maximum_depth))
        _lmain.info('------------------------')
