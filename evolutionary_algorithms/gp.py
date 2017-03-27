# -*- coding: utf-8 -*-

import os
from datetime import datetime
import numpy as np
from utilities.util import StopRecursion, setup_logger, reset_logger

from uuid import uuid4

import logging as _l

_lmain = _l.getLogger('gp.main')
_lftest = _l.getLogger('gp.ftest')
_lftrain = _l.getLogger('gp.ftrain')


def protected_division(v1, v2):
    zeros = np.abs(v2) < 0.00000000001
    v2[zeros] = 1
    return np.divide(v1, v2)


# FUNCTION SET
class Function:
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

    @staticmethod
    def constant(c):
        return {
            'f': lambda X: np.ones(X.shape[0]) * c,
            'str': str(c),
            'arity': 0}

    @staticmethod
    def variable(idx):
        return {
            'f': lambda X: X[:, idx],  # return random X element
            'str': str('x' + str(idx)),
            'arity': 0}

    @staticmethod
    def __random_F(f_set):
        r = int(np.random.rand() * len(f_set))
        return f_set[r]

    @staticmethod
    def __random_T(num_features, constants):
        rcidx = int(np.random.rand() * len(constants))
        rfidx = int(np.random.rand() * num_features)
        threshold = float(len(constants)) / (len(constants) + num_features)
        if np.random.rand() < threshold:
            return Function.constant(constants[rcidx])
        else:
            return Function.variable(rfidx)

    @staticmethod
    def __random_TF(f_set=None, num_features=None, constants=None):
        if np.random.rand() < 0.5:
            return Function.__random_F(f_set)
        return Function.__random_T(num_features, constants)

    @staticmethod
    def get_random(type='TF', f_set=None, num_features=None, constants=None):
        if type == 'TF':
            return Function.__random_TF(f_set, num_features, constants)
        elif type == 'T':
            return Function.__random_T(num_features, constants)
        elif type == 'F':
            return Function.__random_F(f_set)


class Node(object):
    #  Node Initialization
    def __init__(self, parent, function):
        self.id = uuid4()
        # reference to parent Node
        self.parent = parent
        # references to children depending on functions arity
        self.children = []
        # this is a function combining inputs
        # or it's a function returning a constant/input value
        self.function = function

    def compute_and_collect(
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
            y, s, d = c.compute_and_collect(
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

    def copy(self, nodes_dict=None, parent=None):
        newself = self.__class__(parent, self.function)
        if nodes_dict is not None:
            nodes_dict[newself.id] = newself
        for c in range(newself.function['arity']):
            newself.children.append(
                self.children[c].copy(nodes_dict=nodes_dict, parent=newself))
        return newself

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


class Tree(object):

    def __init__(self, min_depth, max_depth, f_set, num_features, constants, **kwargs):
        self.f_set = f_set
        self.num_features = num_features
        self.constants = constants
        self.scope = kwargs.get('scope', {})
        if self.scope.get('node_class', None) is None:
            self.scope['node_class'] = Node
        # initialize tree
        etype = 'TF'
        if max_depth == 0:  # force end of tree at root
            etype = 'T'
        elif min_depth > 0:  # prevent ending of tree at root
            etype = 'F'
        else:  # probabilistic ending of tree at root
            etype = 'TF'
        f = Function.get_random(etype, self.f_set, self.num_features, self.constants)
        self.root = self.scope['node_class'](None, function=f)
        self.nodes = {self.root.id: self.root}
        self.grow(self.root, 0, min_depth, max_depth)

    def compute(self, X, depth_limit=None, collect_additional=False):
        np.seterr(all='ignore')
        try:
            if collect_additional:
                self.nodes = {}
                result, self.size, self.depth = self.root.compute_and_collect(
                    self.nodes, X, 0, depth_limit)
            else:
                result = self.root.compute(X)
        except StopRecursion:
            result = [np.Inf for i in X]
            self.depth = np.Inf
            self.size = np.Inf
        np.seterr(all='warn')
        return result

    def grow(self, parent, cur_depth, min_depth=0, max_depth=None):
        if max_depth is None:
            max_depth = 950  # prevent recursion depth exception
        for i in range(parent.function['arity']):
            etype = 'TF'
            if cur_depth < min_depth:
                etype = 'F'
            elif max_depth is None or cur_depth < max_depth:
                etype = 'TF'
            else:  # force terminal element
                etype = 'T'
            if cur_depth == 950:
                print "WARNING: Reached Maximum Recursion Limit!"
            f = Function.get_random(etype, self.f_set, self.num_features, self.constants)
            newnode = self.scope['node_class'](parent, f)
            parent.children.append(newnode)
            if self.nodes is not None:
                self.nodes[newnode.id] = newnode
            # grow new branch
            self.grow(
                parent.children[i], cur_depth + 1, min_depth=min_depth,
                max_depth=max_depth)

    def calculate_depth(self):
        self.depth = self.root.calculate_depth()
        return self.depth

    def calculate_size(self):
        self.size = self.root.calculate_size()
        return self.size

    def create_random(self, max_depth=None):
        f = Function.get_random('TF', self.f_set, self.num_features, self.constants)
        parent = self.scope['node_class'](None, f)
        self.grow(parent, 0, max_depth=max_depth)
        return parent

    def random_node_choice(self):
        return self.nodes[np.random.choice(self.nodes.keys())]

    def mutate(self):
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
        return copy

    def crossover(self, partner):
        # get cx points and copy to not get tangled up in refs
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
        return newself

    def copy(self):
        newself = self.__class__(0, 0, self.f_set, self.num_features, self.constants)
        newself.depth = self.depth
        newself.size = self.size
        newself.nodes = {}
        newself.root = self.root.copy(newself.nodes)
        assert len(newself.nodes) == len(self.nodes)
        return newself

    def __str__(self):
        return str(self.root)


class Individual(object):
    """Individual in a Genetic Programming Population"""

    def __init__(self, min_depth, max_depth, gp_instance, **kwargs):
        self.id = uuid4()
        self.last_semantics = {'train': None, 'test': None}
        self.last_error = {'train': None, 'test': None}
        self.apply_depth_limit = kwargs.get('apply_depth_limit', False)
        self.depth_limit = kwargs.get('depth_limit', 17)
        self.context = {}
        self.context['gp_instance'] = gp_instance
        self.scope = kwargs.get('scope', {})
        if self.scope.get('tree_class', None) is None:
            self.scope['tree_class'] = Tree

        # initialize tree
        self.tree = self.scope['tree_class'](min_depth, max_depth, gp_instance.f_set,
                                             gp_instance.num_features, gp_instance.constants)

    def __evaluate_X(self, X, data_type):
        # compute output of input matrix
        np.seterr(all='ignore')
        self.last_semantics[data_type] = self.tree.compute(X)
        np.seterr(all='warn')
        return self.last_semantics[data_type]

    # evaluate individual data set and collect as much information
    # as possible, including: nodes list, depth, size and output
    # catch the early exit Exception of the traversal if max depth is reached
    def __evaluate_all(self, X, Y, data_type):
        dl = self.depth_limit if self.apply_depth_limit else None
        self.last_semantics[data_type] = self.tree.compute(X, dl, collect_additional=True)
        self.last_error[data_type] = np.sqrt(np.sum(
            (self.last_semantics[data_type] - Y)**2) / X.shape[0])
        return self.last_error[data_type]

    # wrapper to evaluate all data partitions
    def evaluate(self, X, Y, testX=None, testY=None):
        # collect as much information as possible here
        self.__evaluate_all(X, Y, 'train')
        if self.tree.depth == np.Inf:
            return
        # for these only run the computation
        if testX is not None and testY is not None:
            self.__evaluate_X(testX, 'test')
            self.last_error['test'] = np.sqrt(np.sum(
                (self.last_semantics['test'] - testY)**2) / testX.shape[0])

    def get_fitness(self, data_type):
        return self.last_error[data_type]

    def get_semantics(self, data_type):
        return self.last_semantics[data_type]

    def mutate(self):
        mutated = self.__class__(0, 0, self.context['gp_instance'], depth_limit=self.depth_limit,
                                 apply_depth_limit=self.apply_depth_limit)
        mutated.tree = self.tree.mutate()
        return mutated

    def crossover(self, partner):
        offspring = self.__class__(0, 0, self.context['gp_instance'], depth_limit=self.depth_limit,
                                   apply_depth_limit=self.apply_depth_limit)
        offspring.tree = self.tree.crossover(partner.tree)
        return offspring

    def better(self, other, data_type='train'):
        return self.get_fitness(data_type) < other.get_fitness(data_type)

    def copy(self):
        newself = self.__class__(0, 0, self.context['gp_instance'], depth_limit=self.depth_limit,
                                 apply_depth_limit=self.apply_depth_limit)
        newself.tree = self.tree.copy()
        for k in self.last_semantics.keys():
            if self.last_semantics[k] is not None:
                newself.last_semantics[k] = np.copy(self.last_semantics[k])
        for k in self.last_error.keys():
            if self.last_error[k] is not None:
                newself.last_error[k] = self.last_error[k]
        return newself

    def __str__(self):
        return str(self.tree)


class Population(object):

    def __init__(self, size, gp_instance, selection_type=None, **kwargs):
        if selection_type is None:
            self.selection_type = self.__class__.tournament
            self.tournament_size = 4
        else:
            self.selection_type = selection_type
        self.size = size
        self.individuals = []
        self.context = {}
        self.context['gp_instance'] = gp_instance
        self.tournament_size = kwargs.get('tournament_size', 0)
        self.scope = kwargs.get('scope', {})
        if self.scope.get('individual_class', None) is None:
            self.scope['individual_class'] = Individual

    def create_individuals(
            self, init_min_depth=0, max_depth=6, init_type=None, **kwargs):
        if init_type is None:
            init_type = self.__class__.ramped
        self.individuals = init_type(
            self.size, init_min_depth, max_depth, self.context['gp_instance'],
            scope=self.scope, **kwargs)

    def select(self, count=1):
        if count == 1:
            return self.selection_type(self.individuals, tournament_size=self.tournament_size)
        return [self.selection_type(self.individuals, tournament_size=self.tournament_size)
                for i in range(count)]

    def get_best(self, data_type='train'):
        return self.__class__.filter_best(self.individuals, data_type)

    def evaluate(self, X, Y, testX=None, testY=None):
        for i in self.individuals:
            i.evaluate(X, Y, testX, testY)

    @staticmethod
    def filter_best(array, data_type='train'):
        return array[np.argmin([a.get_fitness(data_type) for a in array])]

    @staticmethod
    def ramped(size, min_depth, max_depth, gp_instance, **kwargs):
        scope = kwargs.get('scope', {})
        if scope.get('individual_class', None) is None:
            scope['individual_class'] = Individual
        individuals = []
        bucket_size = int(size / (1 + max_depth - min_depth))
        for bucket in range(min_depth, max_depth + 1):
            for i in range(bucket_size):
                if i % 2 == 0:  # allow normal grow
                    individuals.append(scope['individual_class'](
                        min_depth, bucket, gp_instance, **kwargs))
                else:  # force full growth
                    individuals.append(scope['individual_class'](
                        bucket, bucket, gp_instance, **kwargs))
        # fill up missing, e.g. due to unclean bucketing:
        full = False
        while len(individuals) < size:
            if not full:
                individuals.append(scope['individual_class'](
                    min_depth, max_depth, gp_instance, **kwargs))
                full = True
            if full:
                individuals.append(scope['individual_class'](
                    max_depth, max_depth, gp_instance, **kwargs))
                full = False
        return individuals

    @staticmethod
    def grow(size, min_depth, max_depth, gp_instance, **kwargs):
        scope = kwargs.get('scope', {})
        if scope.get('individual_class', None) is None:
            scope['individual_class'] = Individual
        individuals = []
        while len(individuals) < size:
            individuals.append(scope['individual_class'](
                min_depth, max_depth, gp_instance, **kwargs))
        return individuals

    @staticmethod
    def full(size, min_depth, max_depth, gp_instance, **kwargs):
        scope = kwargs.get('scope', {})
        if scope.get('individual_class', None) is None:
            scope['individual_class'] = Individual
        individuals = []
        while len(individuals) < size:
            individuals.append(scope['individual_class'](
                max_depth, max_depth, gp_instance, **kwargs))
        return individuals

    @staticmethod
    def tournament(individuals, **kwargs):
        participants = [individuals[int(np.random.rand() * len(individuals))]
                        for i in range(kwargs.get('tournament_size', 4))]
        return Population.filter_best(participants)


class GP(object):
    """Standard Genetic Programming using Tree-based Solutions"""
    reproduction_probability = .0
    mutation_probability = .1
    crossover_probability = .9
    max_initial_depth = 6
    tournament_size = 4
    apply_depth_limit = True
    depth_limit = 17
    mutation_maximum_depth = 6
    log_verbose = True
    log_stdout = True
    debug = False
    log_file_path = 'results'

    def __init__(self, num_features, constants, size):
        self.name = "Standard GP"
        self.constants = constants
        self.num_features = num_features

        self.set_function_set(
            Function.f_add, Function.f_subtract, Function.f_divide, Function.f_multiply)

        self.population = Population(size, self, tournament_size=GP.tournament_size)
        self.population.create_individuals(
            init_min_depth=1, max_depth=GP.max_initial_depth, depth_limit=GP.depth_limit,
            apply_depth_limit=GP.apply_depth_limit)

        self.prepare_logging()
        self.log_config()

    def set_function_set(self, *functions):
        self.f_set = []
        if len(functions) == 0:
            raise 'Function Set cannot be empty.'
        self.f_set = functions

    def evolve(self, X, Y, testX=None, testY=None, generations=25):
        # evaluate
        self.population.evaluate(X, Y, testX, testY)
        best = self.population.get_best()

        for g in range(generations):
            log_str = '[{0:4}] '.format(g)
            # new population
            new_population = Population(
                self.population.size, self, tournament_size=GP.tournament_size)
            # elitism
            new_population.individuals.append(best)

            # create new population
            for i in range(0, self.population.size):
                p1 = self.population.select()
                r = np.random.rand()
                offspring = p1
                if r < GP.crossover_probability:
                    offspring = p1.crossover(self.population.select())
                    offspring.evaluate(X, Y, testX, testY)
                elif r < GP.crossover_probability + GP.mutation_probability:
                    offspring = p1.mutate()
                    offspring.evaluate(X, Y, testX, testY)
                if GP.apply_depth_limit:
                    if offspring.tree.depth > GP.depth_limit:
                        offspring = p1
                new_population.individuals.append(offspring)

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
                    g, best, log_str)

    def log_state(self, generation, best, log_str, **kwargs):
        if GP.log_verbose:
            _lmain.info(log_str)
            _lmain.info('best individual: \n{0}'.format(best))
        # log train fitness
        _lftrain.info('{0};{1};{2};{3}'.format(
            generation,
            best.get_fitness('train'),
            best.tree.size,
            best.tree.depth))
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
        _lmain.info('Number of Features={0}'.format(self.num_features))
        _lmain.info('Constants={0}'.format(self.constants))
        _lmain.info('Population Size={0}'.format(self.population.size))
        _lmain.info('reproduction_probability={0}'.format(GP.reproduction_probability))
        _lmain.info('mutation_probability={0}'.format(GP.mutation_probability))
        _lmain.info('crossover_probability={0}'.format(GP.crossover_probability))
        _lmain.info('max_initial_depth={0}'.format(GP.max_initial_depth))
        _lmain.info('apply_depth_limit={0}'.format(GP.apply_depth_limit))
        _lmain.info('depth_limit={0}'.format(GP.depth_limit))
        _lmain.info('mutation_maximum_depth={0}'.format(GP.mutation_maximum_depth))
        _lmain.info('------------------------')
