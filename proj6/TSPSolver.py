#!/usr/bin/python3

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
# elif PYQT_VER == 'PYQT4':
# 	from PyQt4.QtCore import QLineF, QPointF
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

	def greedy( self,time_allowance=60.0 ):
		print('inside greedy-------------------')
		cities = self._scenario.getCities()
		start_time = time.time()
		bssf = None

		tourFound = False
		for startCity in cities: # O(n) max
			if tourFound or time.time() - start_time > time_allowance:
				break
			route = [startCity]
			visited = set(route)
			unvisited = set(cities)
			unvisited.remove(startCity)
			while not tourFound and time.time() - start_time < time_allowance:
				nextCity = min(unvisited, key=lambda city: route[-1].costTo(city))
				if route[-1].costTo(nextCity) == math.inf:
					break
				route.append(nextCity)
				visited.add(nextCity)
				unvisited.remove(nextCity)
				if not unvisited:
					bssf = TSPSolution(route)
					tourFound = bssf.cost < math.inf
					break

		#end results O(1) in time
		results = {}
		end_time = time.time()
		results['cost'] = bssf.cost
		results['time'] = end_time - start_time
		results['count'] = 0
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		print('ending greedy----------------')
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


	"""
	This is my implementation of an MMAS Ant Colony Optimization algorithm for the TSP problem.
	
	Originally, I began working off of this paper: https://arxiv.org/abs/2203.02228
	but I ended up beginning to work on the Max-Min Ant System algorithm instead as they described
	as their baseline algorithm. So ultimately, I used both their description of MMAS as well as 
	the original paper here: http://www.cs.ubc.ca/~hoos/Publ/fgcs00.pdf as references.
	"""
	def fancy(self, time_allowance=60.0):
		cities = self._scenario.getCities()
		num_ants = 10 # the smaller this is, the more likely the algorithm will break bc all ants can't find a solution
		rho = .97 # evaporation rate (this probably isn't where I should put this)
		p_best = 0.99 # probability that constructed solution will contain solely the highest pheromone edges
		alpha = 1 # pheromone importance
		beta = 2 # heuristic importance
		np.set_printoptions(precision=5)

		iterationTolerance = 50000 #the number of iterations the algorithm can do before terminating

		# Initialize the best-solution-so-far with the greedy algorithm
		results = self.greedy(time_allowance=time_allowance)
		results['count'] = 0
		results['max'] = 0
		results['total'] = 1
		results['pruned'] = 0
		bssf = results['soln']
		
		pdec = np.power(p_best, 1 / len(cities))
		tau_min_coeff = (1 - pdec) / (pdec * (len(cities) / 2 - 1))

		dist_matrix = np.array([[cities[i].costTo(cities[j]) for j in range(len(cities))] for i in range(len(cities))]) # Initialize the distance matrix
		tau_max, tau_min = self._calcTauLimits(bssf.cost, rho, tau_min_coeff) # Calculate the tau limits

		# Initialize the pheromone matrix
		pheromone_matrix = np.array([[tau_max for _ in range(len(cities))] for _ in range(len(cities))], dtype=float) # initialize to 1 (maybe we should go way higher)

		# Initialize the heuristic matrix
		heuristic_matrix = np.ones_like(dist_matrix)
		for i in range(len(cities)):
			for j in range(len(cities)):
				if dist_matrix[i][j] != 0:
					heuristic_matrix[i][j] = 1 / dist_matrix[i][j]
				else:
					heuristic_matrix[i][j] = 1

		print('starting iteration ------------------------------------')
		# Iterate until time runs out or the algorithm converges
		converged = False
		start_time = time.time()
		num_iterations = 0
		bssfUpdateIteration = num_iterations
		while not converged and time.time() - start_time < time_allowance:
			num_iterations += 1
			#print(num_iterations)
			converged = self._check_convergence(pheromone_matrix, tau_min, tau_max)
			#print(pheromone_matrix)
			ants = set(Ant(cities, alpha, beta) for _ in range(num_ants))	# Initialize the ant population
			invalid_ants = set()
			for ant in ants:
				for _ in range(len(cities)):
					# chooseNextCity isn't done quite right, but it's close
					if not ant.chooseNextCity(pheromone_matrix, heuristic_matrix):
						invalid_ants.add(ant)
						break
			best_ant = min(ants.difference(invalid_ants), key=lambda ant: ant.getCost(dist_matrix))
			if best_ant.getCost(dist_matrix) < bssf.cost:
				bssf = TSPSolution(best_ant.getCityRoute())
				print(f"New best solution!!!: {bssf.cost}")
				results['count'] += 1
				tau_max, tau_min = self._calcTauLimits(bssf.cost, rho, tau_min_coeff)
				bssfUpdateIteration = num_iterations
			pheromone_matrix = self._updatePheromones(pheromone_matrix, rho, best_ant, dist_matrix, tau_max, tau_min)
			if (bssfUpdateIteration + iterationTolerance < num_iterations):
				converged = True

		
		# print(f"final pheromone matrix:")
		# print(pheromone_matrix)
		print(f"MAX-MIN ACO COMPLETE")
		print(f"Number of iterations: {num_iterations}")
		print(f"final tau_max: {tau_max}")
		print(f"final tau_min: {tau_min}")
		print(f"final bssf cost: {bssf.cost}")
		results['time'] = time.time() - start_time
		results['soln'] = bssf
		results['cost'] = bssf.cost
		return results

	def _calcTauLimits(self, bssf_cost, rho, tau_min_coeff):
		tau_max = 1 / ((1 - rho) * bssf_cost)
		tau_min = min(tau_max, tau_max * tau_min_coeff) # this should be 0 when p_best = 1
		print(f"\tbssf_cost: {bssf_cost}, rho: {rho}, tau_max:{tau_max}, tau_min: {tau_min}")
		return tau_max, tau_min

	def _updatePheromones(self, pheromone_matrix, rho, best_ant, dist_matrix, tau_max, tau_min):
		pheromone_matrix *= rho
		best_ant_cost = best_ant.getCost(dist_matrix)
		delta_tau = 1 / best_ant_cost
		if delta_tau < tau_max:
			for route_index, city_index in enumerate(best_ant.route):
				if route_index < len(best_ant.route) - 1:
					pheromone_matrix[city_index][best_ant.route[route_index + 1]] += delta_tau
				else:
					pheromone_matrix[city_index][best_ant.route[0]] += delta_tau
		return np.clip(pheromone_matrix, tau_min, tau_max, out=pheromone_matrix)
	
	""" 
	Convergenece is defined on page 15 of Stützle and Hoos (1999) as when "for each choice point,
	one of the solution components has tau_max as associated pheromone trail, while all 
	alternative solution components have a pheromone trail value tau_min".
	"""
	def _check_convergence(self, pheromone_matrix, tau_min, tau_max):
		convergenceParamater = .001
		for row in pheromone_matrix:
			if not np.where(tau_max - row < convergenceParamater, 1, 0).sum() == 1 \
				or not np.where(row - tau_min < convergenceParamater, 1, 0).sum() == len(row) - 1: #why -1?
				return False
		return True
		
