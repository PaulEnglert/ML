# -*- coding: utf-8 -*-

import os
from datetime import datetime
import re
import numpy as np

from utilities.data_utils import make_batches
from utilities.util import Capturing, StopRecursion

from . import gp
from neural_networks.perceptrons.slp import SLP, f_softplus


class Node(gp.Node):
    """Overwrite standard to save a footprint of calculation"""

    def traverse_and_collect(
            self, node_dict, footprint, X, n_steps, early_exit_depth=None):
        # collect all information in one pass,
        # exit early if max depth is reached
        node_dict[self.id] = self
        inputs = []
        size = 1
        depth = 0
        # if the traversal already went to deep, exit by Exception
        if early_exit_depth is not None and n_steps > early_exit_depth:
            raise StopRecursion()
        # calculate data for all children
        for c in self.children:
            y, s, d = c.traverse_and_collect(
                node_dict, footprint, X, n_steps + 1, early_exit_depth)
            inputs.append(y)
            if d > depth:
                depth = d
            size += s
        self.Y = self.function['f'](X, *inputs)
        footprint.append({'Y': self.Y, 'sp': self})
        # pass on the own data
        return (self.Y, size, depth + 1)

    def copy(self, nodes_dict=None, parent=None):
        newself = Node(parent, self.function)
        if nodes_dict is not None:
            nodes_dict[newself.id] = newself
        for c in range(newself.function['arity']):
            newself.children.append(
                self.children[c].copy(nodes_dict=nodes_dict, parent=newself))
        return newself


class Individual(gp.Individual):
    """overwrite of standard to use rnngp classes and logic
        the footprint is a vector of the output of all sub-
        programs. The last column is the root nodes prediciton
    """

    def __init__(self, min_depth, max_depth):
        self.id = Individual.count
        Individual.count += 1
        self.footprint = {'train': [], 'val': [], 'test': []}
        self.last_semantics = {'train': None, 'val': None, 'test': None}
        self.last_error = {'train': None, 'val': None, 'test': None}
        # initialize tree (root is always a function)
        self.root = Node(None, function=Node.get_random_F())
        self.nodes = {self.root.id: self.root}
        self.grow(self.root, 0, min_depth, max_depth, nodes_dict=self.nodes)

    def __evaluate(self, X, Y, data_type='train'):
        self.nodes = {}
        self.footprint[data_type] = []
        np.seterr(all='ignore')
        dl = Individual.depth_limit if Individual.apply_depth_limit else None
        try:
            result, self.size, self.depth = self.root.traverse_and_collect(
                self.nodes, self.footprint[data_type], X, 0, dl)
        except StopRecursion:
            result = [np.Inf for i in X]
            self.depth = np.Inf
            self.size = np.Inf
        np.seterr(all='warn')
        self.last_semantics[data_type] = result
        self.last_error[data_type] = np.sqrt(np.sum(
            (self.last_semantics[data_type] - Y)**2) / X.shape[0])
        return self.last_error[data_type]

    def evaluate(self, X, Y, valX, valY, testX=None, testY=None):
        # collect as much information as possible here
        self.__evaluate(X, Y, 'train')
        if self.depth == np.Inf:
            return
        # for these only run the computation
        if valX is not None and valY is not None:
            self.compute(valX, 'val')
            self.last_error['val'] = np.sqrt(np.sum(
                (self.last_semantics['val'] - valY)**2) / valX.shape[0])
        if testX is not None and testY is not None:
            self.compute(testX, 'test')
            self.last_error['test'] = np.sqrt(np.sum(
                (self.last_semantics['test'] - testY)**2) / testX.shape[0])

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
            parent.children.append(newnode)
            if nodes_dict is not None:
                nodes_dict[newnode.id] = newnode
            # grow new branch
            self.grow(
                parent.children[i], cur_depth + 1, min_depth=min_depth,
                max_depth=max_depth, nodes_dict=nodes_dict)

    def evaluate_footprint(self, num_blames):
        """ Train an MLP on the footprint on a program to identify
            important subprograms
        """
        if num_blames <= 0:
            return []
        raw = []  # convert footprint into learnable data
        assert len(self.footprint['train']) > 0
        for x in self.footprint['train']:
            raw.append(x['Y'])
        raw = np.asarray(raw).T ** 2
        raw = (raw - np.min(raw)) / (np.max(raw) - np.min(raw))
        slp = SLP(
            raw.shape[1] - 1,
            RNNGP.nn_config['num_outputs'],
            activation=RNNGP.nn_config['activation'])
        tX = make_batches(
            raw[:, :-1], RNNGP.nn_config['batch_size'],
            keep_last=True)
        tY = make_batches(
            raw[:, -1], RNNGP.nn_config['batch_size'],
            keep_last=True)
        result = []
        try:
            with Capturing() as out:
                slp.train(
                    tX, tY,
                    epochs=RNNGP.nn_config['epochs'],
                    learning_rate=RNNGP.nn_config['learning_rate'],
                    learning_rate_decay=RNNGP.nn_config['learning_rate_decay'])
        except RuntimeError:
            print 'WARNING: Failed training SLP: {0}'.format(
                out[-1] if len(out) > 0 else 'no-output')
        else:
            weights = slp.weights[:-1] ** 2  # ignore bias
            # determine most important subprogram based on weights
            for i in weights.T.argsort()[:, -num_blames:][0].tolist():
                s = self.footprint['train'][i]['sp'].calculate_size()
                minsize = RNNGP.gp_config['min_program_size']
                maxsize = RNNGP.gp_config['max_program_size']
                if minsize is not None and minsize < 1:
                    minsize = self.size * minsize
                if maxsize is not None and maxsize < 1:
                    maxsize = self.size * maxsize
                # print 'Size of OP:{0}; Min:{1}; Max:{2}'.format(
                #    s, minsize, maxsize)
                if (s >= minsize or
                    minsize is None) and \
                    (s <= maxsize or
                        maxsize is None):
                    result.append(str(self.footprint['train'][i]['sp']))
        return result

    def evaluate_subprogram_goodness(self):
        # count the occurences of offending subprograms in self
        # also calculate the normalized weighted sum of occurrences
        # (weighted by the severity, normalized by number of progs)
        ws = 0
        count = 0
        for sp in Population.offending_sub_programs:
            c = len(re.findall(sp['str'], str(self)))
            count += c
            ws += sp['severity'] * c
        self.offensiveness = ws / len(Population.offending_sub_programs)

    def dominates(self, opponent):
        # less subprogram offenses and better fitness -> domination
        if np.abs(self.offensiveness) < np.abs(opponent.offensiveness) and \
                self.get_fitness('train') < opponent.get_fitness('train'):
            return True
        return False

    def create_random(self, start_depth=0, max_depth=None):
        parent = Node(None, Node.get_random_TF())
        self.grow(parent, start_depth, max_depth=max_depth)
        return parent

    def copy(self):
        newself = Individual(1, 1)
        newself.depth = self.depth
        newself.size = self.size
        for k in self.last_semantics.keys():
            if self.last_semantics[k] is not None:
                newself.last_semantics[k] = np.copy(self.last_semantics[k])
        for k in self.last_error.keys():
            if self.last_error[k] is not None:
                newself.last_error[k] = self.last_error[k]
        for k in self.footprint.keys():
            if self.footprint[k] is not None:
                newself.footprint[k] = self.footprint[k]
        newself.nodes = {}
        newself.root = self.root.copy(newself.nodes)
        assert len(newself.nodes) == len(self.nodes)
        return newself


class Population(gp.Population):
    validation_elite = []
    offending_sub_programs = []

    def __init__(self, size, selection_type):
        self.selection_type = selection_type
        self.size = size
        self.individuals = []

    def create_individuals(
            self, init_min_depth=1, max_depth=6, init_type=None):
        if init_type is None:
            init_type = Population.ramped
        self.individuals = init_type(
            self.size, init_min_depth, max_depth)

    def select(self, count=1):
        return [self.selection_type(self.individuals)
                for i in range(count)]

    def evaluate(self, X, Y, valX, valY, testX=None, testY=None):
        for i in self.individuals:
            i.evaluate(X, Y, valX, valY, testX, testY)

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
    def full(size, min_depth, max_depth, num_features, constants):
        individuals = []
        while len(individuals) < size:
            individuals.append(Individual(max_depth, max_depth))
        return individuals

    def apply_for_validation_elite(self, applicant):
        if len(Population.validation_elite) <\
                RNNGP.gp_config['validation_elite_size']:
            Population.validation_elite.append(applicant)
        else:
            rpl = np.argmax([i.get_fitness('val')
                            for i in Population.validation_elite])
            if Population.validation_elite[rpl].get_fitness('val') >\
                    applicant.get_fitness('val'):
                Population.validation_elite[rpl] = applicant

    def get_best_n(self, count, data_type='train'):
        return Population.get_best_n(self.individuals, count, data_type)

    def get_penalized_best(self):
        if len(self.offending_sub_programs) == 0:
            return self.get_best()
        return Population.filter_penalized_best(self.individuals)

    def get_val_elite_fitnesses(self):
        if RNNGP.gp_config['validation_elite_repr'] == 'median':
            size = len(Population.validation_elite)
            b = np.argsort([i.get_fitness('val')
                            for i in Population.validation_elite])[size // 2]
            return (Population.validation_elite[b].get_fitness('train'),
                    Population.validation_elite[b].get_fitness('val'))
        if RNNGP.gp_config['validation_elite_repr'] == 'avg':
            ft = np.average([i.get_fitness('train')
                            for i in Population.validation_elite])
            fv = np.average([i.get_fitness('val')
                            for i in Population.validation_elite])
            return (ft, fv)

    @staticmethod
    def filter_best_n(array, count, data_type='train'):
        f = [i.get_fitness(data_type) for i in array]
        idcs = np.argsort(f)[-count:].tolist()
        return [array[i] for i in idcs]

    @staticmethod
    def nsga_II_sort(P):
        S, n, F = {}, {}, {}
        count = 0
        F[1] = []
        for p in P:
            S[p.id] = []
            n[p.id] = 0
            for q in P:
                if p.dominates(q):
                    S[p.id].append(q)
                elif q.dominates(p):
                    n[p.id] = n[p.id] + 1
            if n[p.id] == 0:
                p.rank = 1
                count = count + 1
                F[1].append(p)
        i = 1
        while len(F[i]) > 0:
            Q = []
            for p in F[i]:
                for q in S[p.id]:
                    n[q.id] = n[q.id] - 1
                    if n[q.id] == 0:
                        q.rank = i + 1
                        count = count + 1
                        Q.append(q)
            i = i + 1
            F[i] = Q
        print 'Number of Fronts: {0}'.format(i - 1)
        assert count == len(P)  # ensure correctness of nsga-II
        return F

    @staticmethod
    def mo_tournament(individuals):
        """Select based on rank or random"""
        if len(Population.offending_sub_programs) == 0:
            return Population.tournament(individuals)

        participants = [individuals[int(np.random.rand() * len(individuals))]
                        for i in range(Population.tournament_size)]

        return Population.pareto_best(participants)

    @staticmethod
    def so_tournament_penalized(individuals):
        """Select based on penalized fitness"""
        if len(Population.offending_sub_programs) == 0:
            return Population.tournament(individuals)

        participants = [individuals[int(np.random.rand() * len(individuals))]
                        for i in range(Population.tournament_size)]

        return Population.filter_penalized_best(participants)

    @staticmethod
    def filter_penalized_best(P):
        trf = RNNGP.gp_config['so_params']['offensiveness_cost']
        return P[np.argmin([
            np.abs(trf(i.offensiveness)) + i.get_fitness('train')
            for i in P])]

    @staticmethod
    def pareto_best(P):
        rank_sorted = np.argsort([p.rank for p in P]).tolist()
        best = []
        for i in reversed(rank_sorted):
            if P[i].rank == P[rank_sorted[-1]].rank:
                best.append(P[i])
        if len(best) == 1:
            return best[0]
        cfg = RNNGP.gp_config['mo_params']
        if cfg['equality_escape'] == 'random':
            # return a random member of the best array
            return best[int(np.random.rand() * len(best))]
        elif cfg['equality_escape'] == 'offensiveness':
            # return the one with the lowest offensiveness
            b = np.argsort(
                [i.offensiveness for i in best])[0]
            return best[b]
        else:
            # return the member with best fitness
            b = np.argsort(
                [i.get_fitness('train') for i in best])[0]  # Minimization
            return best[b]

    @staticmethod
    def handle_offending_program_size(max_size):
        size = len(Population.offending_sub_programs)
        if size > max_size:
            keep = np.argsort(
                [b['severity']
                    for b in Population.offending_sub_programs]
            )[-max_size:].tolist()
            Population.offending_sub_programs = [
                Population.offending_sub_programs[k] for k in keep]

    @staticmethod
    def handle_offending_program_duplicates():
        new_list = np.unique([
            {'str': p['str'], 'severity': 0, 'count': 0}
            for p in Population.offending_sub_programs])
        for p in new_list:
            # get all from current and average the severity
            for i in Population.offending_sub_programs:
                if i['str'] == p['str']:
                    p['count'] = p['count'] + 1
                    p['severity'] = p['severity'] + i['severity']
            p['severity'] = p['severity'] / p['count']
            p.pop('count', None)


class RNNGP(gp.GP):
    """Hybrid Genetic Programming using NNs to repulse overfitting Solutions"""

    # Configuration regarding the ANN (MLP) that is used for extracting
    # important subprograms
    nn_config = {
        # DON'T CHANGE
        #  - one output neuron, since it's a regression
        'num_outputs': 1,
        'activation': f_softplus,

        # adapt the following according to needs
        'batch_size': 25,
        'epochs': 25,
        'learning_rate': 0.1,
        'learning_rate_decay': 1.01,
    }

    gp_config = {
        # GENERAL
        # number of generations to skip before engaging in the repusler logic
        'skip_generations': 50,
        # prevent the use of elitsm, to avoid letting a newly found repulser
        # survive
        'prevent_elitism': False,
        # Search Operators: use 'mo' for applying multiobjective search to
        # optimize fitness and offensiveness (see 'mo_params') for
        # additional configuration (this will fallback to standard fitness
        # search, if no offending programs have been collected yet)
        # Use 'so' to apply single objective search with the 'so_params'
        # configuration
        'search_operator': 'so',
        'mo_params': {
            # use a pareto optimal solution as surviving elite
            # (if 'prevent_elitism' is True then this will have no effect!)
            'pareto_elitism': True,
            # if two individuals are in the same pareto front, how should
            # they be compared to get a single winner (this affects tournament
            # and elitism)
            'equality_escape': 'fitness',  # 'random','offensiveness','fitness'
        },
        'so_params': {
            # The property to which the search should be directed to
            'search_for': 'fitness',  # 'fitness', 'offensiveness'
            # Penalize solutions based on their offensiveness, this is
            # only used if 'search_for' is 'fitness'
            'penalize_offensiveness': True,
            # Transformation applied to the offensiveness before adding
            # to the fitness
            'offensiveness_cost': lambda o: o * 0.01,
        },
        # SUBPROGRAMS
        # how many subprograms are allowed to be identified as
        # the key programs of a full program
        'num_blames_per_individual': 5,
        # size constraints on the blames: after blaming, this means
        # that even though n subprograms have been identified, any of them
        # can be rejected based on the size
        # (set to None if no size constraint should be applied)
        # (set to float if a fraction of the program size should be used)
        'min_program_size': 0.2,
        'max_program_size': 0.3,
        # number of blamed structures to keep over time
        # if number is exceeded blames with less severity will be purged
        'max_offending_progs': 50,

        # OVERFITTING
        # number of individuals to keep in the best-of list on the
        # validation data
        'validation_elite_size': 50,
        # quantify overfitting of an individual
        # f1, f2: training & test fitness of individual i
        # bf1, bf2: of the best found (median/avg of validation elite)
        # 'overfit_severity': lambda f1, f2, bf1, bf2: (
        #     (f2 - bf2) ** 2),  # squared difference of val fitness
        # this is from Vanneschi et al. (Measuring Bloat, Overfitting
        # and Functional Complexity in Genetic Programming):
        'overfit_severity': lambda f1, f2, bf1, bf2: (
            np.abs(f1 - f2) - np.abs(bf1 - bf2)),
        # determine the representative of the validation elite by avg/median
        'validation_elite_repr': 'avg'  # 'median' or 'avg'
    }

    def __init__(self, num_features, constants, size):
        self.name = "RNNGP"
        Node.constants = constants
        gp.Node.constants = constants
        Node.num_features = num_features
        gp.Node.num_features = num_features

        Population.validation_elite = []
        Population.offending_sub_programs = []

        if RNNGP.gp_config['search_operator'] == 'mo':
            self.sel_type = Population.mo_tournament
        else:
            self.sel_type = Population.so_tournament_penalized
        self.population = Population(
            size, selection_type=self.sel_type)
        self.population.create_individuals(
            init_min_depth=1, max_depth=RNNGP.max_initial_depth,
            init_type=None)

        self.prepare_logging()
        self.log_config()

    def evolve(self, X, Y, valX, valY, testX=None, testY=None, generations=25):
        Individual.depth_limit = RNNGP.depth_limit
        Individual.apply_depth_limit = RNNGP.apply_depth_limit

        # evaluate on training and validation
        self.population.evaluate(X, Y, valX, valY, testX, testY)

        for g in range(generations):
            log_str = '[{0:4}] '.format(g)

            if RNNGP.log_verbose:
                size_violation_count = 0
                mutation_count = 0
                crossover_count = 0
                size_sum = 0
                depth_sum = 0

            # modify validation elite
            # only the current best on training is a candidate
            # TODO only save the necessary data,
            # e.g train/val semantics & fitness
            self.population.apply_for_validation_elite(
                self.population.get_best('train'))

            # create new population
            new_population = Population(
                self.population.size, selection_type=self.sel_type)

            # copy the elite individual over to the next population
            # unless it's forbidden by configuration
            if not RNNGP.gp_config['prevent_elitism']:
                elitist = self.select_elitist()
                if RNNGP.log_verbose:
                    size_sum = elitist.size
                    depth_sum = elitist.depth
                new_population.individuals.append(elitist)

            # vary the current population to fill the new population
            while len(new_population.individuals) < new_population.size:
                # select parents
                # this will if no subprograms have been collected yet,
                # fall back to standard tournament
                p1, p2 = self.population.select(2)

                # determine operators
                r = np.random.rand()
                # first try to copying parent
                offspring = p1.copy()
                # do crossover
                if r < RNNGP.crossover_probability:
                    if RNNGP.log_verbose:
                        crossover_count += 1
                    offspring = p1.crossover(p2)
                    offspring.evaluate(X, Y, valX, valY, testX, testY)
                # do mutation
                elif r < RNNGP.crossover_probability +\
                        RNNGP.mutation_probability:
                    if RNNGP.log_verbose:
                        mutation_count += 1
                    offspring = p1.mutate()
                    offspring.evaluate(X, Y, valX, valY, testX, testY)
                # apply depth limit
                if RNNGP.apply_depth_limit:
                    if offspring.depth > RNNGP.depth_limit:
                        size_violation_count += 1
                        offspring = p1.copy()  # overwrite an offspring with p1

                # add individual to new population
                new_population.individuals.append(offspring)
                if RNNGP.log_verbose:
                    size_sum += offspring.size
                    depth_sum += offspring.depth

            # update to new population (already evaluated)
            self.population = new_population

            # determine fitness best individual
            best = self.population.get_best('train')

            # start testing for overfitting after n generations
            if g > RNNGP.gp_config['skip_generations']:
                # determine if best is overfitting
                is_overfitting = False
                f1, f2 = best.get_fitness('train'), best.get_fitness('val')
                bf1, bf2 = self.population.get_val_elite_fitnesses()
                if f2 > bf2:  # if best is worse on validation then val elite
                    best.overfits_by = (RNNGP.gp_config
                                        ['overfit_severity'])(f1, f2, bf1, bf2)
                    is_overfitting = True

                # process new repulser with NN
                if is_overfitting:
                    blames = best.evaluate_footprint(
                        RNNGP.gp_config['num_blames_per_individual'])
                    # add new blames to memory
                    Population.offending_sub_programs += [
                        {'str': b, 'severity': best.overfits_by}
                        for b in blames]
                    # control duplicates (average the severity)
                    Population.handle_offending_program_duplicates()
                    # control size
                    Population.handle_offending_program_size(
                        RNNGP.gp_config['max_offending_progs'])

            # prevent unecessary computation if there are no
            print 'num offending programs {0}'.format(
                len(self.population.offending_sub_programs))
            # subprograms collected yet
            if len(Population.offending_sub_programs) > 0:
                # update individuals offensiveness
                for i in self.population.individuals:
                    i.evaluate_subprogram_goodness()

                if RNNGP.gp_config['search_operator'] == 'mo':
                    # NSGA II Sort
                    Population.nsga_II_sort(self.population.individuals)
                else:
                    # no post processing needed for 'so' search
                    pass

            # logging
            self.log_state(
                g, best, log_str, size_violation_count=size_violation_count,
                crossover_count=crossover_count, mutation_count=mutation_count,
                size_sum=size_sum, depth_sum=depth_sum,
                last=(g == generations - 1))

    def select_elitist(self, fronts=None):
        cfg = RNNGP.gp_config
        # single objective search
        if cfg['search_operator'] == 'so':
            # search for fitness
            if cfg['so_params']['search_for'] == 'fitness':
                # don't penalize offensiveness
                if not cfg['so_params']['penalize_offensiveness']:
                    return self.population.get_best()
                # penalize offensiveness
                else:
                    return self.population.get_penalized_best()
            # search for offensiveness
            else:
                raise Exception('search_for {0} not implemented').format(
                    cfg['so_params']['search_for'])
        # multi objective
        if len(Population.offending_sub_programs) == 0 or not\
                cfg['mo_params']['pareto_elitism']:
            # fall back to standard if nothing has been collected yet or
            # it's configured
            return self.population.get_best()
        else:
            # elitist is the pareto optimal solution as configured
            return Population.pareto_best(
                self.population.individuals)

    def log_state(self, generation, best, logstr='', **kwargs):
        logstr += ' best training error={0}'.format(
            best.get_fitness('train'))
        logstr += ' with test error={0}'.format(
            best.get_fitness('test'))
        if RNNGP.log_stdout:
            print logstr
        if not RNNGP.log_verbose:
            return

        logstr += ' best individual: \n{0}\n'.format(best)
        logstr += ' number of size violations: {0}\n'.format(
            kwargs['size_violation_count'])
        logstr += ' number of crossovers: {0}\n'.format(
            kwargs['crossover_count'])
        logstr += ' number of mutations: {0}\n'.format(
            kwargs['mutation_count'])
        logstr += ' avg size: {0}\n'.format(
            kwargs['size_sum'] / self.population.size)
        logstr += ' avg depth: {0}\n'.format(
            kwargs['depth_sum'] / self.population.size)
        logstr += 'num offending subprograms: {0}\n'.format(
            len(Population.offending_sub_programs))
        logstr += 'validation elite representative ' + \
            '(train, val): {0}\n'.format(
                self.population.get_val_elite_fitnesses())
        logstr += '-------------------------------------------'
        if kwargs['last']:
            logstr += '\n offending programm list:{0}'.format(
                Population.offending_sub_programs)

        base = os.path.join(os.getcwd(), RNNGP.log_file_path)
        # log logstr
        if logstr != '':
            with open(os.path.join(
                    base,
                    self.rid + '-gp.log'), 'ab') as log:
                log.write(logstr + '\n')
        # log train fitness
        with open(os.path.join(
                base,
                self.rid + '-fitnesstrain.txt'), 'ab') as log:
            log.write('{0};{1};{2};{3};{4};{5}\n'.format(
                generation,
                best.get_fitness('train'),
                best.size,
                best.depth,
                best.rank if hasattr(best, 'rank') else None,
                best.offensiveness if hasattr(best, 'rank') else None))
        # log val fitness
        with open(os.path.join(
                base,
                self.rid + '-fitnessvalidation.txt'), 'ab') as log:
            log.write('{0};{1}\n'.format(
                generation,
                best.get_fitness('val')))
        # log test fitness
        with open(os.path.join(
                base,
                self.rid + '-fitnesstest.txt'), 'ab') as log:
            log.write('{0};{1}\n'.format(
                generation,
                best.get_fitness('test')))

    def log_config(self):
        if not RNNGP.log_verbose:
            return
        base = os.path.join(os.getcwd(), RNNGP.log_file_path)
        with open(os.path.join(
                base,
                self.rid + '-gp.log'), 'ab') as log:
            log.write('config: \n' + str(RNNGP.gp_config) + '\n')
            log.write('nn-config: \n' + str(RNNGP.nn_config) + '\n')

    def prepare_logging(self):
        if not RNNGP.log_verbose:
            return
        base = os.path.join(os.getcwd(), RNNGP.log_file_path)
        if not os.path.exists(base):
            os.makedirs(base)

        self.rid = str(int(
            (datetime.now() - datetime(1970, 1, 1)).total_seconds()))
        with open(os.path.join(
                base,
                self.rid + '-gp.log'), 'ab') as log:
            log.write(self.name + '\n')
        with open(os.path.join(
                base,
                self.rid + '-fitnesstrain.txt'), 'ab') as log:
            log.write('Gen;Train Fitness;Size;Depth;Rank;Offensiveness\n')
        with open(os.path.join(
                base,
                self.rid + '-fitnessvalidation.txt'), 'ab') as log:
            log.write('Gen;Test Fitness\n')
        with open(os.path.join(
                base,
                self.rid + '-fitnesstest.txt'), 'ab') as log:
            log.write('Gen;Test Fitness\n')
