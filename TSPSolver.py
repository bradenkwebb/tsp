#!/usr/bin/python3

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT6':
	from PyQt6.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))


import time
import numpy as np
from TSPClasses import *
from heapq import *
import copy
import itertools


class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None
		self._bssf = None

	def setupWithScenario( self, scenario ):
		self._scenario = scenario


	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution,
		time spent to find solution, number of permutations tried during search, the
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''

	def defaultRandomTour( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		print(f"DefaultRandomTour: {results['cost']}")
		if self._bssf is None or results['cost'] < self._bssf.cost:
			self._bssf = bssf
		return results


	''' <summary>
		This algorithm finds a (generally sub-optimal) solution to the TSP. It returns the first solution it finds,
		and while it is somewhat greedy, it is not actually as greedy as it could be - it instead uses the same general
		heuristic that I use in the branch-and-bound algorithm, always picking the state closest to a solution, and only
		then optimizing for cost. It is a branching algorithm without bound, that prioritizes deep, low-cost states. 
		This is a good algorithm to use as a baseline for comparison, as it is very fast and relatively simple to implement.

		The time and space complexity of this algorithm depend largely on the sparsity of the graph. If the graph is
		very sparse and asymmetric, then the algorithm will have to search a large number of states, and will have a high time complexity.
		If the graph is very dense, then almost any state of sufficient depth will be a solution, and the algorithm will
		have a lower time complexity.

		If we were searching for the optimal solution, I calculate that in the worst case, there are O(n!) states, where n is the
		number of cities, which could need to be searched. However, since we terminate as soon as we find a solution, the time complexity
		is much closer to O(n)*O(self._expand()) = O(n^4), and we see such polynomial-time behavior empirically as well. The space
		complexity is O(n^3), since we store a matrix of size n^2 for each state, and we store a queue of these states that shrinks
		and grows as we search the tree.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''
	def greedy( self, time_allowance=60.0):
		results = {}
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		results['count'] = 0
		bssf = None
		cities = self._scenario.getCities()
		S = []
		bssf = None
		heappush(S, TSPCostMatrix(cities))

		start_time = time.time()
		while S and time.time() - start_time < time_allowance: 
			P = heappop(S)
			states = self._expand(P) # O(n^3)
			for state in states: # In the worst case, this is O(n) in the number of states
				if len(state.path) == len(cities):
					potSolution = TSPSolution(state.getRoute(cities))
					if potSolution.cost < math.inf:
						bssf = potSolution
						print(f"new bssf found: {bssf.cost}")
						S.clear() # this allows us to break out of both loops
						break
				else:
					heappush(S, state)
		end_time = time.time()

		results['cost'] = bssf.cost if bssf else math.inf
		results['time'] = end_time - start_time
		results['soln'] = bssf
		print(f"Greedy: {results}")
		if self._bssf is None or results['cost'] < self._bssf.cost:
			self._bssf = bssf
		return results

	''' <summary>
		This method actually runs the Branch-and-Bound algorithm, to find an optimal solution to the TSP if given
		enough time, or a high-quality sub-optimal approximation if given less time. It uses a priority queue to store
		states of partial solutions, and it expands the state closest to a solution, and with the lowest cost, first.
		The algorithm first calls the greedy algorithm to find a solution, and then uses that solution as the initial
		best-solution-so-far (BSSF). It uses the cost of that BSSF as a bound on the cost of any other solution, allowing
		it to prune states that are not worth exploring. 
		
		In the worst-case scenario, as in the greedy algorithm, the algorithm will have to search O(n!) states, where n is the
		number of cities. However, in practice, the algorithm will terminate much sooner, as it will prune many states that
		are not worth exploring. The exact time complexity is difficult to calculate, but it empirically seems to be
		exponential and of the form O(c^n) for some constant c. The space complexity is O(n^3), since we store a matrix of
		size n^2 for each state, and the size of the priority queue is shown empirically to grow linearly with n.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints:
		max queue size, total number of states created, and number of pruned states.</returns>
	'''
	def branchAndBound( self, time_allowance=60.0 ):
		cities = self._scenario.getCities()
		S = []
		start_time = time.time()
		results = self.greedy(time_allowance=time_allowance)
		results['count'] = 0
		results['max'] = 0
		results['total'] = 1
		results['pruned'] = 0
		bssf = results['soln']
		if bssf is None:
			print("Greedy failed to find a solution")
		else:
			heappush(S, TSPCostMatrix(cities))
		while S and time.time() - start_time < time_allowance: # Exponential, O(c^n) for some constant c
			results['max'] = max(results['max'], len(S))
			P = heappop(S)
			if P.cost < bssf.cost:
				states = self._expand(P) # O(n^3)
				results['total'] += len(states)
				for state in states: # O(n)
					if len(state.path) == len(cities):
						potSolution = TSPSolution(state.getRoute(cities))
						if potSolution.cost < bssf.cost:
							bssf = potSolution
							print(f"new bssf found: {bssf.cost}")
							results['count'] += 1
						else:
							results['pruned'] += 1
					elif state.cost < bssf.cost:
						heappush(S, state)
					else:
						results['pruned'] += 1
			else:
				results['pruned'] += 1
		end_time = time.time()
		elapsed_time = end_time - start_time
		if elapsed_time > time_allowance and bssf:
			results['pruned'] += len([state for state in S if state.cost > bssf.cost])
		results['time'] = elapsed_time
		results['soln'] = bssf
		results['cost'] = bssf.cost if bssf else math.inf
		print(f"Branch and Bound: {results}")
		self._bssf = bssf
		if bssf is not None and len(S) == 0 and elapsed_time < time_allowance:
			print("Optimal result!!!")
		return results
	
	''' 
	This takes a cost matrix and returns a list of all possible states that can be reached from it
	that have a finite cost. This is a helper function for the branch and bound algorithm. Since it
	loops through all possible edges in the graph from the most recent city in the path, it has a time
	complexity of O(n)*O(TSPCostMatrix.addCity) = O(n)*O(n^2) = O(n^3). 
	The space complexity is, in the worst case, also O(n)*O(TSPCostMatrix) = O(n)(O(n^2)) = O(n^3), as it 
	must store a copy of the cost matrix for each possible state.
	'''
	def _expand(self, costMatrix):
		states = []
		mostRecentCityIndex = costMatrix.path[-1]
		for cityIndex in range(len(costMatrix.matrix[mostRecentCityIndex])): # O(n)
			if costMatrix.matrix[mostRecentCityIndex][cityIndex] < math.inf:
				newMatrix = copy.deepcopy(costMatrix) # O(n^2)
				newMatrix.addCity(cityIndex) # O(n^2)
				states.append(newMatrix)
		return states



	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found during search, the
		best solution found.  You may use the other three field however you like.
		algorithm</returns>
	'''

	def fancy( self,time_allowance=60.0 ):
		pass
