#!/usr/bin/python3

from TSPClasses import Scenario
from TSPSolver import TSPSolver
import random

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
    from PyQt5.QtWidgets import *
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
elif PYQT_VER == 'PYQT4':
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *
elif PYQT_VER == 'PYQT6':
    from PyQt6.QtWidgets import *
    from PyQt6.QtGui import *
    from PyQt6.QtCore import *
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))


# Get results of branchAndBound method of TSPSolver object
def branchAndBoundDriver(
        random_seed,
        number_points,
        time_limit,
        *,
        difficulty="Hard (Deterministic)",
        multiprocessing_results=None):

    random.seed(random_seed)
    points = []
    solver = TSPSolver(None)

    # Generate points (cities)
    while len(points) < number_points:
        x = -1.5 + 3*random.uniform(0.0,1.0)
        y = -1 + 2*random.uniform(0.0,1.0)
        points.append(QPointF(x,y))

    scenario = Scenario(city_locations=points, difficulty=difficulty, rand_seed=random_seed)

    solver.setupWithScenario(scenario)

    if multiprocessing_results is not None:
        results = solver.branchAndBound(time_allowance=time_limit)
        for key, value in results.items():
            multiprocessing_results[key] = value
    else:
        return solver.branchAndBound(time_allowance=time_limit)

# Get results of fancy method of TSPSolver object
def fancyDriver(
        random_seed,
        number_points,
        time_limit,
        *,
        difficulty="Hard (Deterministic)",
        multiprocessing_results=None):

    random.seed(random_seed)
    points = []
    solver = TSPSolver(None)

    # Generate points (cities)
    while len(points) < number_points:
        x = -1.5 + 3*random.uniform(0.0,1.0)
        y = -1 + 2*random.uniform(0.0,1.0)
        points.append(QPointF(x,y))

    scenario = Scenario(city_locations=points, difficulty=difficulty, rand_seed=random_seed)

    solver.setupWithScenario(scenario)

    if multiprocessing_results is not None:
        results = solver.fancy(time_allowance=time_limit)
        for key, value in results.items():
            multiprocessing_results[key] = value
    else:
        return solver.fancy(time_allowance=time_limit)


# Get results of greedy method of TSPSolver object
def greedyDriver(
        random_seed,
        number_points,
        time_limit,
        *,
        difficulty="Hard (Deterministic)",
        multiprocessing_results=None):

    random.seed(random_seed)
    points = []
    solver = TSPSolver(None)

    # Generate points (cities)
    while len(points) < number_points:
        x = -1.5 + 3*random.uniform(0.0,1.0)
        y = -1 + 2*random.uniform(0.0,1.0)
        points.append(QPointF(x,y))

    scenario = Scenario(city_locations=points, difficulty=difficulty, rand_seed=random_seed)

    solver.setupWithScenario(scenario)

    if multiprocessing_results is not None:
        results = solver.greedy(time_allowance=time_limit)
        for key, value in results.items():
            multiprocessing_results[key] = value
    else:
        return solver.greedy(time_allowance=time_limit)


# Get results of defaultRandomTour method of TSPSolver object
def randomDriver(
        random_seed,
        number_points,
        time_limit,
        *,
        difficulty="Hard (Deterministic)",
        multiprocessing_results=None):

    random.seed(random_seed)
    points = []
    solver = TSPSolver(None)

    # Generate points (cities)
    while len(points) < number_points:
        x = -1.5 + 3*random.uniform(0.0,1.0)
        y = -1 + 2*random.uniform(0.0,1.0)
        points.append(QPointF(x,y))

    scenario = Scenario(city_locations=points, difficulty=difficulty, rand_seed=random_seed)

    solver.setupWithScenario(scenario)

    if multiprocessing_results is not None:
        results = solver.defaultRandomTour(time_allowance=time_limit)
        for key, value in results.items():
            multiprocessing_results[key] = value
    else:
        return solver.defaultRandomTour(time_allowance=time_limit)
