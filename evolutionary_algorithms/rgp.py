# -*- coding: utf-8 -*-

import os
from datetime import datetime
import re
import dill
import numpy as np
from timeit import default_timer as _t

from utilities.data_utils import make_batches
from utilities.util import Capturing, StopRecursion, setup_logger, reset_logger
from utilities.stats import rv_coefficient, rv2_coefficient, distance_correlation

from . import gp_v2 as gp
from neural_networks.perceptrons.slp import SLP, f_softplus
from sklearn.linear_model import LinearRegression

import logging as _l


_lmain = _l.getLogger('rgp.main')
_lftest = _l.getLogger('rgp.ftest')
_lftrain = _l.getLogger('rgp.ftrain')
_lfval = _l.getLogger('rgp.fval')
_ltime = _l.getLogger('rgp.time')


class Node(gp.Node):
    """Overwrite standard to save a footprint of calculation"""
    def __init__(self, parent, function):
        super(Node, self).__init__(parent, function)

    def calculate_depth(self):
        if len(self.children) > 0:
            return np.max([[c.calculate_depth() for c in self.children]]) + 1
        return 0

    def calculate_size(self):
        return np.sum([c.calculate_size() for c in self.children]) + 1

    # overwrite to collect footprints in addition
    def compute_and_collect(
            self, node_dict, footprint, X, n_steps, early_exit_depth=None):
        # collect all information in one pass,
        # exit early if max depth is reached
        node_dict[self.id] = self
        inputs = []
        size = 1
        depth = -1
        # if the traversal already went to deep, exit by Exception
        if early_exit_depth is not None and n_steps > early_exit_depth:
            raise StopRecursion()
        # calculate data for all children
        for c in self.children:
            y, s, d = c.compute_and_collect(
                node_dict, footprint, X, n_steps + 1, early_exit_depth)
            inputs.append(y)
            if d > depth:
                depth = d
            size += s
        self.Y = self.function['f'](X, *inputs)
        footprint.append({'Y': self.Y, 'sp': self})
        # pass on the own data
        return (self.Y, size, depth + 1)


class Tree(gp.Tree):

    # overwrite init to pass in scope (changes the used Node class)
    def __init__(self, min_depth, max_depth, f_set, num_features, constants):
        super(Tree, self).__init__(min_depth, max_depth, f_set, num_features,
                                   constants, scope={'node_class': Node})

        self.footprint = {'train': [], 'val': [], 'test': []}

    def compute(self, X, depth_limit=None, collect_additional=False):
        np.seterr(all='ignore')
        try:
            if collect_additional:
                fp = []
                self.nodes = {}
                result, self.size, self.depth = self.root.compute_and_collect(
                    self.nodes, fp, X, 0, depth_limit)
            else:
                result = self.root.compute(X)
        except StopRecursion:
            result = [np.Inf for i in X]
            self.depth = np.Inf
            self.size = np.Inf
        np.seterr(all='warn')
        if collect_additional:
            return result, fp
        return result


class Individual(gp.Individual):
    """overwrite of standard to use RGP classes and logic
        the footprint is a vector of the output of all sub-
        programs. The last column is the root nodes prediciton
    """

    def __init__(self, min_depth, max_depth, gp_instance, **kwargs):
        super(Individual, self).__init__(min_depth, max_depth, gp_instance,
                                         scope={'tree_class': Tree})
        self.footprint = {'train': [], 'val': [], 'test': []}
        self.last_semantics = {'train': None, 'val': None, 'test': None}
        self.last_error = {'train': None, 'val': None, 'test': None}

    # overwrite to recieve footprint as well
    def __evaluate_all(self, X, Y, data_type):
        dl = self.depth_limit if self.apply_depth_limit else None
        self.last_semantics[data_type], self.footprint[data_type] = self.tree.compute(
            X, dl, collect_additional=True)
        self.last_error[data_type] = np.sqrt(np.sum(
            (self.last_semantics[data_type] - Y)**2) / X.shape[0])
        return self.last_error[data_type]

    # overwrite to evaluate validation data as well
    def evaluate(self, X, Y, valX, valY, testX=None, testY=None):
        # collect as much information as possible here
        super(Individual, self).evaluate(X, Y, testX, testY)
        # also run validation data
        if valX is not None and valY is not None:
            self.__evaluate_X(valX, 'val')
            self.last_error['val'] = np.sqrt(np.sum(
                (self.last_semantics['val'] - valY)**2) / valX.shape[0])

    def evaluate_footprint(self):
        """ Train an MLA on the footprint on a program to identify
            important subprograms
        """
        start_t = _t()
        if RGP.r_general['blame_full_program']:
            return []
        raw = []  # convert footprint into learnable data
        assert len(self.footprint['train']) > 0
        if len(self.footprint['train']) == 1:
            return []
        for x in self.footprint['train']:
            raw.append(x['Y'])
        raw = np.asarray(raw).T ** 2
        raw = (raw - np.min(raw)) / (np.max(raw) - np.min(raw))
        # get weighting of features
        if RGP.r_general['sub_prog_classifier'] == 'nn':
            weights = self.__class__.process_by_nn(raw)
        elif RGP.r_general['sub_prog_classifier'] == 'lr':
            weights = self.__class__.process_by_lr(raw)
        else:
            raise Exception('Unknown Sub Program Classifier')
        # determine most important subprogram based on weights
        _ltime.debug('4;evaluate-footprint;{0}'.format(_t() - start_t))
        if weights.shape[0] == 0:
            return []
        return weights

    # ############## SYNTACTIC REPULSING DEPENDENCIES ##############
    def blame_subprogram_syntax(self, weights, num_blames):
        start_t = _t()
        if RGP.r_general['blame_full_program']:
            return [str(self)]
        if len(weights) == 0:
            return []
        extracted_positions = weights.argsort()[-num_blames:].tolist()
        result = []
        for i in extracted_positions:
            s = self.footprint['train'][i]['sp'].calculate_size()
            minsize = RGP.r_general['min_program_size']
            maxsize = RGP.r_general['max_program_size']
            if minsize is not None and minsize < 1:
                minsize = self.tree.size * minsize
            if maxsize is not None and maxsize < 1:
                maxsize = self.tree.size * maxsize
            # print 'Size of OP:{0}; Min:{1}; Max:{2}'.format(
            #    s, minsize, maxsize)
            if (s >= minsize or
                minsize is None) and \
                (s <= maxsize or
                    maxsize is None):
                result.append(str(self.footprint['train'][i]['sp']))
        _ltime.debug('4;blame-subprogram-syntax;{0}'.format(_t() - start_t))
        return result

    @staticmethod
    def process_by_lr(X):
        reg = LinearRegression(fit_intercept=RGP.lr_config['intercept'])
        reg.fit(X[:, :-1], X[:, -1])
        return reg.coef_ ** 2

    @staticmethod
    def process_by_nn(X):
        slp = SLP(
            X.shape[1] - 1,
            RGP.nn_config['num_outputs'],
            activation=RGP.nn_config['activation'])
        tX = make_batches(
            X[:, :-1], RGP.nn_config['batch_size'],
            keep_last=True)
        tY = make_batches(
            X[:, -1], RGP.nn_config['batch_size'],
            keep_last=True)
        try:
            with Capturing() as out:
                slp.train(
                    tX, tY,
                    epochs=RGP.nn_config['epochs'],
                    learning_rate=RGP.nn_config['learning_rate'],
                    learning_rate_decay=RGP.nn_config['learning_rate_decay'])
        except RuntimeError:
            print 'WARNING: Failed training SLP: {0}'.format(
                out[-1] if len(out) > 0 else 'no-output')
            return []
        else:
            weights = slp.weights[:-1] ** 2  # ignore bias
            return weights.T[0]

    def evaluate_syntactic_goodness(self):
        # count the occurences of offending subprograms in self
        # also calculate the normalized weighted sum of occurrences
        # (weighted by the severity, normalized by number of progs)
        start_t = _t()
        ws = 0
        count = 0
        for sp in Population.repulsers:
            try:
                c = len(re.findall(sp['str'], str(self)))
            except:
                c = 100
            count += c
            ws += sp['severity'] * c
        self.offensiveness = ws / len(Population.repulsers)
        _ltime.debug('4;evaluate-syntactic-goodness;{0}'.format(_t() - start_t))

    # ############## SEMANTIC REPULSING DEPENDENCIES ############
    def blame_subprogram_semantics(self, weights, num_blames):
        start_t = _t()
        if RGP.r_general['blame_full_program']:
            r_data = {'train': [], 'val': [], 'test': []}
            for k in r_data:
                for x in self.footprint[k]:
                    r_data[k].append(x['Y'])
                r_data[k] = np.asarray(r_data[k]).T
            return [r_data]
        if len(weights) == 0:
            return []
        extracted_positions = weights.argsort()[-num_blames:].tolist()
        result = []
        for i in extracted_positions:
            # ensure size contraints
            sp = self.footprint['train'][i]['sp']
            s = sp.calculate_size()
            minsize = RGP.r_general['min_program_size']
            maxsize = RGP.r_general['max_program_size']
            if minsize is not None and minsize <= 1:
                minsize = self.tree.size * minsize
            if maxsize is not None and maxsize <= 1:
                maxsize = self.tree.size * maxsize
            if (s >= minsize or
                minsize is None) and \
                (s <= maxsize or
                    maxsize is None):
                # extract subprogram footprint
                r_data = {'train': [], 'val': [], 'test': []}
                d_types = []
                for k in r_data:
                    if len(self.footprint[k]) > 0:
                        d_types.append(k)
                start_i = int(i - s + 1)
                for p in range(start_i, i + 1):
                    for k in d_types:
                        r_data[k].append(self.footprint[k][p]['Y'])
                for k in r_data:
                    r_data[k] = np.asarray(r_data[k]).T
                result.append(r_data)
        _ltime.debug('4;blame-subprogram-semantics;{0}'.format(_t() - start_t))
        return result

    def evaluate_semantic_goodness(self, data_type='train'):
        start_t = _t()
        fp = []
        for x in self.footprint[data_type]:
            fp.append(x['Y'])
        fp = np.asarray(fp).T
        self.offensiveness_l = []  # list of similarities to all repulsers
        for r in Population.repulsers:
            if RGP.r_sema['similarity_measure'] == 'RV':
                # needs to be inversed, since it denotes
                # similarity not distance
                self.offensiveness_l.append(1 - np.abs(rv_coefficient(
                    fp, r['str'][data_type])))
            elif RGP.r_sema['similarity_measure'] == 'RV2':
                self.offensiveness_l.append(1 - np.abs(rv2_coefficient(
                    fp, r['str'][data_type])))
            elif RGP.r_sema['similarity_measure'] == 'DC':
                self.offensiveness_l.append(1 - np.abs(distance_correlation(
                    fp, r['str'][data_type])))
        # aggregate offensiveness
        self.offensiveness = np.average(np.abs(self.offensiveness_l))
        _ltime.debug('4;evaluate-semantic-goodness;{0}'.format(_t() - start_t))

    # ############## GENERAL DEPENDENCIES ##############
    def dominates(self, opponent):
        fit = self.get_fitness('train') < opponent.get_fitness('train')
        if RGP.repulse_by == 'semantics' and\
                not RGP.r_sema['mo_search']['aggregated_goodness']:
            off = np.abs(self.offensiveness_l) < np.abs(opponent.offensiveness)
            return fit and (np.sum(off) == len(self.offensiveness_l))

        # less offenses and better fitness -> domination
        if np.abs(self.offensiveness) < np.abs(opponent.offensiveness) and fit:
            return True
        return False

    def copy(self):
        newself = super(Individual, self).copy()
        for k in self.footprint.keys():
            if self.footprint[k] is not None:
                newself.footprint[k] = self.footprint[k]
        return newself


class Population(gp.Population):
    validation_elite = []
    repulsers = []

    def __init__(self, size, gp_instance, selection_type=None, **kwargs):
        super(Population, self).__init__(size, gp_instance, selection_type=None,
                                             scope={'individual_class': Individual}, **kwargs)

    # overwrite to time execution and evaluate validation data
    def evaluate(self, X, Y, valX, valY, testX=None, testY=None):
        start_t = _t()
        for i in self.individuals:
            i.evaluate(X, Y, valX, valY, testX, testY)
        _ltime.debug('3;population-evaluation;{0}'.format(_t() - start_t))

    def apply_for_validation_elite(self, applicant):
        start_t = _t()
        if len(self.__class__.validation_elite) < RGP.gp_config['validation_elite_size']:
            self.__class__.validation_elite.append(applicant)
        else:
            rpl = np.argmax([i.get_fitness('val')
                            for i in self.__class__.validation_elite])
            if self.__class__.validation_elite[rpl].get_fitness('val') > applicant.get_fitness('val'):
                self.__class__.validation_elite[rpl] = applicant
        _ltime.debug('3;apply-for-validation-elite;{0}'.format(_t() - start_t))

    def get_best_n(self, count, data_type='train'):
        return self.__class__.filter_best_n(self.individuals, count, data_type)

    def get_penalized_best(self):
        if len(self.repulsers) == 0:
            return self.get_best()
        return self.__class__.filter_penalized_best(self.individuals)

    def get_val_elite_fitnesses(self):
        if RGP.gp_config['validation_elite_repr'] == 'median':
            size = len(self.__class__.validation_elite)
            b = np.argsort([i.get_fitness('val')
                            for i in self.__class__.validation_elite])[size // 2]
            return (self.__class__.validation_elite[b].get_fitness('train'),
                    self.__class__.validation_elite[b].get_fitness('val'))
        if RGP.gp_config['validation_elite_repr'] == 'avg':
            ft = np.average([i.get_fitness('train')
                            for i in self.__class__.validation_elite])
            fv = np.average([i.get_fitness('val')
                            for i in self.__class__.validation_elite])
            return (ft, fv)

    @staticmethod
    def filter_best_n(array, count, data_type='train'):
        f = [i.get_fitness(data_type) for i in array]
        idcs = np.argsort(f)[:count].tolist()
        return [array[i] for i in idcs]

    @staticmethod
    def nsga_II_sort(P):
        start_t = _t()
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
        if RGP.log_stdout:
            print 'Number of Fronts: {0}'.format(i - 1)
        assert count == len(P)  # ensure correctness of nsga-II
        _ltime.debug('3;nsga-II-sort;{0}'.format(_t() - start_t))
        return F

    @staticmethod
    def mo_tournament(individuals, **kwargs):
        """Select based on rank or random"""
        if len(Population.repulsers) == 0:
            return Population.tournament(individuals)

        participants = [individuals[int(np.random.rand() * len(individuals))]
                        for i in range(kwargs.get('tournament_size', 4))]

        return Population.pareto_best(participants)

    @staticmethod
    def so_tournament_penalized(individuals, **kwargs):
        """Select based on penalized fitness"""
        if len(Population.repulsers) == 0:
            return Population.tournament(individuals)

        participants = [individuals[int(np.random.rand() * len(individuals))]
                        for i in range(kwargs.get('tournament_size', 4))]

        return Population.filter_penalized_best(participants)

    @staticmethod
    def filter_penalized_best(P):
        trf = RGP.gp_config['so_params']['offensiveness_cost']
        return P[np.argmin([
            np.abs(trf(i.offensiveness)) + i.get_fitness('train')
            for i in P])]

    @staticmethod
    def pareto_best(P):
        rank_sorted = np.argsort([p.rank for p in P]).tolist()
        best = []
        for i in rank_sorted:
            if P[i].rank == P[rank_sorted[0]].rank:
                best.append(P[i])
        if len(best) == 1:
            return best[0]
        cfg = RGP.gp_config['mo_params']
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
    def handle_repulser_list_size(max_size):
        size = len(Population.repulsers)
        if size > max_size:
            keep = np.argsort(
                [b['severity']
                    for b in Population.repulsers]
            )[-max_size:].tolist()
            Population.repulsers = [
                Population.repulsers[k] for k in keep]

    @staticmethod
    def handle_repulser_duplicates():
        new_list = np.unique([
            {'cp': str(p['str']), 'str': p['str'], 'severity': 0, 'count': 0}
            for p in Population.repulsers])
        for p in new_list:
            # get all from current and average the severity
            for i in Population.repulsers:
                if str(i['str']) == p['cp']:
                    p['count'] = p['count'] + 1
                    p['severity'] = p['severity'] + i['severity']
            p['severity'] = p['severity'] / p['count']
            p.pop('count', None)
            p.pop('cp', None)


class RGP(gp.GP):
    """Hybrid GP using any ML to repulse overfitting Solutions"""
    debug = False
    # collect footprints (activate, from generation, number if individuals per generation)
    collect_footprints = (False, 250, 4)
    footprints_path = 'footprints.txt'
    timeit = False
    # Technique to use for repulsing individuals
    repulse_by = 'semantics'  # semantics | syntax

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
        'search_operator': 'mo',
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
            # to the fitness (for semantic repulers o will be  in [0,1])
            'offensiveness_cost': lambda o: o * 0.005,
        },
        # number of blamed structures to keep over time
        # if number is exceeded blames with less severity will be purged
        'max_num_repulser': 50,
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

    r_general = {
        # blame full program for overfitting, or evaluate footprint and
        # determine important subprograms
        'blame_full_program': False,
        # how many subprograms are allowed to be identified as
        # the key programs of a full program
        'num_blames_per_individual': 5,
        # size constraints on the blames: after blaming, this means
        # that even though n subprograms have been identified, any of them
        # can be rejected based on the size
        # (set to None if no size constraint should be applied)
        # (set to float if a fraction of the program size should be used)
        # (this is used by syntactic and semantic repulsing)
        'min_program_size': 0.2,
        'max_program_size': 0.7,
        # classifier to use when determining importance of sub programs
        # see their own configuration for further adaptions
        'sub_prog_classifier': 'lr',  # lr: linear regression, nn: neural ntwrk
    }

    # configurations for semantic repulsing
    r_sema = {
        'mo_search': {
            # if false uses a dynamic number of objectives in the domination
            # process, each objective representing a similarity to a repulser
            # which has to be maximized
            # all other processes use a aggregated indicator! (E.g. equality
            # escape in elitism, or penalized 'so' search)
            'aggregated_goodness': True,
        },
        # similarity measure to use when evaluating individuals semantic
        # goodness
        'similarity_measure': 'RV'  # RV, RV2 or DC
    }

    # configuration for syntactic repulsing
    r_synt = {
    }

    # Configuration regarding Algorithm that is used for extracting
    # important subprograms
    nn_config = {
        # DON'T CHANGE
        #  - one output neuron, since it's a regression
        'num_outputs': 1,
        'activation': f_softplus,

        # adapt the following according to needs
        'batch_size': 25,
        'epochs': 50,
        'learning_rate': 0.1,
        'learning_rate_decay': 1.01,
    }

    lr_config = {
        'intercept': True,
    }

    def __init__(self, num_features, constants, size):
        self.name = "RGP"
        self.constants = constants
        self.num_features = num_features

        Population.validation_elite = []
        Population.repulsers = []

        self.set_function_set(
            gp.Function.f_add, gp.Function.f_subtract, gp.Function.f_divide, gp.Function.f_multiply)

        if RGP.gp_config['search_operator'] == 'mo':
            self.sel_type = Population.mo_tournament
        else:
            self.sel_type = Population.so_tournament_penalized
        self.population = Population(
            size, self, selection_type=self.sel_type, tournament_size=RGP.tournament_size)
        self.population.create_individuals(
            init_min_depth=1, max_depth=RGP.max_initial_depth, init_type=None,
            depth_limit=RGP.depth_limit, apply_depth_limit=RGP.apply_depth_limit)

        self.prepare_logging()
        self.log_config()

    def evolve(self, X, Y, valX, valY, testX=None, testY=None, generations=25):
        # evaluate on training and validation
        self.population.evaluate(X, Y, valX, valY, testX, testY)

        for g in range(generations):
            start_g = _t()
            start_pp = _t()
            log_str = '[{0:4}] '.format(g)

            if RGP.log_verbose:
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
                self.population.size, self, selection_type=self.sel_type,
                tournament_size=RGP.tournament_size)

            # copy the elite individual over to the next population
            # unless it's forbidden by configuration
            if not RGP.gp_config['prevent_elitism']:
                elitist = self.select_elitist()
                if RGP.log_verbose:
                    size_sum = elitist.tree.size
                    depth_sum = elitist.tree.depth
                new_population.individuals.append(elitist)
            _ltime.debug('1;prepocessing-phase;{0}'.format(_t() - start_pp))

            # vary the current population to fill the new population
            start_vp = _t()
            while len(new_population.individuals) < new_population.size:
                # select parents
                # this will if no subprograms have been collected yet,
                # fall back to standard tournament
                p1 = self.population.select()

                # determine operators
                r = np.random.rand()
                # first try to copying parent
                offspring = p1.copy()
                # do crossover
                if r < RGP.crossover_probability:
                    if RGP.log_verbose:
                        crossover_count += 1
                    offspring = p1.crossover(self.population.select())
                    offspring.evaluate(X, Y, valX, valY, testX, testY)
                # do mutation
                elif r < RGP.crossover_probability +\
                        RGP.mutation_probability:
                    if RGP.log_verbose:
                        mutation_count += 1
                    offspring = p1.mutate()
                    offspring.evaluate(X, Y, valX, valY, testX, testY)
                # apply depth limit
                if RGP.apply_depth_limit:
                    if offspring.tree.depth > RGP.depth_limit:
                        size_violation_count += 1
                        offspring = p1.copy()  # overwrite an offspring with p1

                # add individual to new population
                new_population.individuals.append(offspring)
                if RGP.log_verbose:
                    size_sum += offspring.tree.size
                    depth_sum += offspring.tree.depth

            _ltime.debug('1;variation-phase;{0}'.format(_t() - start_vp))
            start_pp = _t()

            # update to new population (already evaluated)
            self.population = new_population

            # determine fitness best individual
            best = self.population.get_best('train')

            # if whished log footprint information 
            if RGP.collect_footprints[0] and g > RGP.collect_footprints[1]:
                bf1, bf2 = self.population.get_val_elite_fitnesses()
                for i in self.population.get_best_n(RGP.collect_footprints[2]):
                    f1, f2 = i.get_fitness('train'), i.get_fitness('val')
                    fp = []
                    for x in i.footprint['train']:
                        fp.append(x['Y'])
                    fp = np.asarray(fp).T
                    fp = np.append(fp, [[f1] for i in range(len(fp))], axis=1)
                    fp = np.append(fp, [[f2] for i in range(len(fp))], axis=1)
                    fp = np.append(fp, [[bf1] for i in range(len(fp))], axis=1)
                    fp = np.append(fp, [[bf2] for i in range(len(fp))], axis=1)
                    with open(RGP.footprints_path, 'ab') as log_file:
                        np.savetxt(log_file, fp, delimiter=";")

            # start testing for overfitting after n generations
            if g > RGP.gp_config['skip_generations']:
                # determine if best is overfitting
                f1, f2 = best.get_fitness('train'), best.get_fitness('val')
                bf1, bf2 = self.population.get_val_elite_fitnesses()
                if f2 > bf2:  # if best is worse on validation then val elite
                    start_oe = _t()
                    best.overfits_by = (RGP.gp_config
                                        ['overfit_severity'])(f1, f2, bf1, bf2)

                    _lmain.debug('Evaluating best individuals footprint')
                    evaluation = best.evaluate_footprint()
                    _lmain.debug('Result: {0}'.format(evaluation))
                    if RGP.repulse_by == 'syntax':
                        new_reps = best.blame_subprogram_syntax(
                            evaluation,
                            RGP.r_general['num_blames_per_individual'])
                    elif RGP.repulse_by == 'semantics':
                        new_reps = best.blame_subprogram_semantics(
                            evaluation,
                            RGP.r_general['num_blames_per_individual'])
                    # add new repulsers to memory
                    Population.repulsers += [
                        {'str': b, 'severity': best.overfits_by}
                        for b in new_reps]
                    _ltime.debug('2;evaluate-new-repulser;{0}'.format(_t() - start_oe))
                    _lmain.debug(
                        'Added new repulsers (blame full prog: {0})'.format(
                            RGP.r_general['blame_full_program']))

                    start_lm = _t()
                    # control duplicates (average the severity)
                    Population.handle_repulser_duplicates()
                    # control size
                    Population.handle_repulser_list_size(
                        RGP.gp_config['max_num_repulser'])
                    _ltime.debug('2;repulser-list-management;{0}'.format(_t() - start_lm))

            # prevent unecessary computation if there are no
            # subprograms collected yet
            if RGP.log_stdout:
                print 'num repulsers {0}'.format(
                    len(self.population.repulsers))
            avg_offensiveness = 0
            if len(Population.repulsers) > 0:
                start_ou = _t()
                # update individuals offensiveness
                for i in self.population.individuals:
                    if RGP.repulse_by == 'syntax':
                        i.evaluate_syntactic_goodness()
                    elif RGP.repulse_by == 'semantics':
                        i.evaluate_semantic_goodness()
                    avg_offensiveness += i.offensiveness
                avg_offensiveness /= self.population.size
                _ltime.debug('2;p-calculate-offensiveness;{0}'.format(_t() - start_ou))

                if RGP.gp_config['search_operator'] == 'mo':
                    # NSGA II Sort
                    Population.nsga_II_sort(self.population.individuals)
                else:
                    # no post processing needed for 'so' search
                    pass

            _ltime.debug('1;postprocessing-phase;{0}'.format(_t() - start_pp))

            # logging
            self.log_state(
                g, best, log_str, size_violation_count=size_violation_count,
                crossover_count=crossover_count, mutation_count=mutation_count,
                size_sum=size_sum, depth_sum=depth_sum,
                last=(g == generations - 1), avg_off=avg_offensiveness)

            _ltime.debug('0;generation;{0}'.format(_t() - start_g))

    def select_elitist(self):
        cfg = RGP.gp_config
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
        if len(Population.repulsers) == 0 or not\
                cfg['mo_params']['pareto_elitism']:
            # fall back to standard if nothing has been collected yet or
            # it's configured
            return self.population.get_best()
        else:
            # elitist is the pareto optimal solution as configured
            return Population.pareto_best(
                self.population.individuals)

    def log_state(self, generation, best, logstr='', **kwargs):
        # generation wrap up:
        logstr += ' best training error={0}'.format(
            best.get_fitness('train'))
        logstr += ' with test error={0}'.format(
            best.get_fitness('test'))
        if RGP.log_stdout:
            print logstr

        # log state to file
        _lmain.info(logstr)
        _lmain.info('best individual:\n{0}'.format(best))
        _lmain.info('number of size violations: {0}'.format(
            kwargs['size_violation_count']))
        _lmain.info('number of crossovers: {0}'.format(
            kwargs['crossover_count']))
        _lmain.info('number of mutations: {0}'.format(
            kwargs['mutation_count']))
        _lmain.info('avg size: {0}'.format(
            kwargs['size_sum'] / self.population.size))
        _lmain.info('avg depth: {0}'.format(
            kwargs['depth_sum'] / self.population.size))
        _lmain.info('num repulsers: {0}'.format(
            len(Population.repulsers)))
        _lmain.info('validation elite representative (train, val): {0}'.format(
            self.population.get_val_elite_fitnesses()))
        _lmain.info('average offensiveness: {0}'.format(
            kwargs['avg_off']))
        _lmain.info('-------------------------------------------')
        if kwargs['last']:
            _lmain.info('repulser list:\n{0}'.format(
                Population.repulsers))

        # log train fitness
        _lftrain.info('{0};{1};{2};{3};{4};{5};{6}'.format(
            generation,
            best.get_fitness('train'),
            best.tree.size,
            best.tree.depth,
            best.rank if hasattr(best, 'rank') else None,
            best.offensiveness if hasattr(best, 'offensiveness') else None,
            len(Population.repulsers)))
        # log val fitness
        _lfval.info('{0};{1}'.format(
            generation,
            best.get_fitness('val')))
        # log test fitness
        _lftest.info('{0};{1}'.format(
            generation,
            best.get_fitness('test')))

    def log_config(self):
        _lmain.info('-----------------------------')
        _lmain.info('CONFIGURATION')
        _lmain.info('Number of Features={0}'.format(self.num_features))
        _lmain.info('Constants={0}'.format(self.constants))
        _lmain.info('Population Size={0}'.format(self.population.size))
        _lmain.info('reproduction_probability={0}'.format(RGP.reproduction_probability))
        _lmain.info('mutation_prob={0}'.format(RGP.mutation_probability))
        _lmain.info('crossover_prob={0}'.format(RGP.crossover_probability))
        _lmain.info('max_initial_depth={0}'.format(RGP.max_initial_depth))
        _lmain.info('apply_depth_limit={0}'.format(RGP.apply_depth_limit))
        _lmain.info('depth_limit={0}'.format(RGP.depth_limit))
        _lmain.info('mutation_maximum_depth={0}'.format(RGP.mutation_maximum_depth))

        _lmain.info('repulse_by={0}'.format(RGP.repulse_by))
        _lmain.info('r_general: {0}'.format(RGP.r_general))
        _lmain.info('r_synt: {0}'.format(RGP.r_synt))
        _lmain.info('r_sema: {0}'.format(RGP.r_sema))
        _lmain.info('gp_config: {0}'.format(RGP.gp_config))
        _lmain.info('nn-config: {0}'.format(RGP.nn_config))
        _lmain.info('lr-config: {0}'.format(RGP.lr_config))
        _lmain.info("".join(dill.source.getsource(RGP.gp_config['so_params']['offensiveness_cost']).split()))
        _lmain.info("".join(dill.source.getsource(RGP.gp_config['overfit_severity']).split()))
        _lmain.info('-----------------------------')

    def prepare_logging(self):
        self.rid = str(int(
            (datetime.now() - datetime(1970, 1, 1)).total_seconds()))

        base = os.path.join(os.getcwd(), RGP.log_file_path)
        if not os.path.exists(base):
            os.makedirs(base)

        # update RGP.footprints_path
        RGP.footprints_path = os.path.join(base, self.rid + '-footprints.txt')
        if RGP.collect_footprints[0]:
            with open(RGP.footprints_path, 'ab') as log_file:
                log_file.write('footprint; last 4 columns: fitnesstrain, fitnessval, validation best fitnesstrain, validation best fitnessval\n')

        # update loggers
        level = _l.INFO
        if RGP.debug:
            level = _l.DEBUG
        reset_logger(logger_name='rgp.main')
        setup_logger(logger_name='rgp.main', log_file=os.path.join(base, self.rid + '-gp.log'),
                     level=level)
        reset_logger(logger_name='rgp.time')
        setup_logger(logger_name='rgp.time', log_file=os.path.join(base, self.rid + '-time.log'),
                     level=_l.DEBUG if RGP.timeit else _l.INFO)
        reset_logger(logger_name='rgp.ftrain')
        setup_logger(logger_name='rgp.ftrain', log_file=os.path.join(base, self.rid + '-fitnesstrain.txt'),
                     level=level)
        reset_logger(logger_name='rgp.fval')
        setup_logger(logger_name='rgp.fval', log_file=os.path.join(base, self.rid + '-fitnessvalidation.txt'),
                     level=level)
        reset_logger(logger_name='rgp.ftest')
        setup_logger(logger_name='rgp.ftest', log_file=os.path.join(base, self.rid + '-fitnesstest.txt'),
                     level=level)

        _lmain.info(self.name)
        _ltime.info(datetime.now())
        _lftest.info('Gen;Test Fitness')
        _lfval.info('Gen;Validation Fitness')
        _lftrain.info('Gen;Train Fitness;Size;Depth;Rank;Offensiveness;Number Repulsers')
