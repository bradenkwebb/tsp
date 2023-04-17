#!/usr/bin/python3


import math
import numpy as np
import random
import time

class Ant():
	def __init__(self, cities, alpha=1, beta=1):
		self.cities = cities
		self.route = []
		self.route_set = set()
		self.alpha = alpha
		self.beta = beta
		self.complete = False
	
	"""
	The method by which an ant chooses the next city to visit. 

	This hasn't been constructed quite correctly, because here we have that at each step, the ants
	consider going to ANY of the cities that they haven't visited yet. However, in the paper, there
	is a fixed-length for the number of nearest neighbors that ants should consider: see 
	section 4.4 of Stützle and Hoos (1999).
	"""
	def chooseNextCity(self, pheromone_matrix, heuristic_matrix):
		if len(self.route) == len(self.cities):
			print("I'm not sure if this should run...")
			print(self.route)
			return False
		# if we haven't visited any cities yet, choose a random one
		if len(self.route) == 0:
			rand_city_index = random.randrange(len(self.cities)) # start at a random city
			self.route.append(rand_city_index)
			self.route_set.add(rand_city_index)
		else:
			# if we have visited some cities, choose the next one based on 
			# the pheromone matrix and the heuristic matrix
			# calculate the probabilities of choosing each city
			probs = np.zeros(len(self.cities))
			i = self.route[-1]
			
			for j in range(len(self.cities)):
				if j not in self.route_set:
					probs[j] = pheromone_matrix[i][j]**self.alpha * heuristic_matrix[i][j]**self.beta
					assert(not np.isnan(probs[j]))
			total = sum(probs)
			if total == 0:
				return False
			# normalize the probabilities
			probs /= total
			# choose the next city
			next_city_index = np.random.choice(len(self.cities), p=probs)
			self.route.append(next_city_index)
			self.route_set.add(next_city_index)
		return True
	
	def getCost(self, dist_matrix):
		if len(self.route) < len(dist_matrix) - 1:
			return np.inf
		
		cost = 0
		for i in range(len(self.route)-1):
			cost += dist_matrix[self.route[i]][self.route[i+1]]
		cost += dist_matrix[self.route[-1]][self.route[0]]
		return cost
	
	def getCityRoute(self):
		cityRoute = []
		for cityIndex in self.route:
			cityRoute.append(self.cities[cityIndex])
		return cityRoute


class TSPCostMatrix:
	""" 
	Initializes a cost matrix for the TSP problem with the given cities.
	The process of initialization has time complexity O(n^2) in the number of cities,
	and space complexity O(n^2).
	"""
	def __init__(self, cities):
		self.path = [0]
		self.cost = 0
		self.matrix = np.zeros((len(cities), len(cities)))
		for row in range(len(cities)):
			for col in range(len(cities)):
				if row != col:
					self.matrix[row][col] = cities[row].costTo(cities[col])
				else:
					self.matrix[row][col] = math.inf
		self.reduceMatrix()

	"""
	Reduces the cost matrix by subtracting the minimum value in each row and column.
	In general, this has time complexity of 2*O(n)*O(np-subtraction-broadcasting). Most ways
	that I can think of implementing numpy broadcasting, in order to calculate each
	'row -= minVal' expression, likely require O(n) time. As such, if (in the worst) case
	we have to do this n times because each row and each column has a finite value, then
	the time complexity is O(n^2). The space complexity is O(1).
	"""
	def reduceMatrix(self):
		for row in self.matrix:
			minVal = min(row)
			if minVal < math.inf:
				row -= minVal # probably O(n)
				self.cost += minVal

		for col in self.matrix.T:
			minVal = min(col)
			if minVal < math.inf:
				col -= minVal # probably O(n)
				self.cost += minVal
	
	""" 
	The heuristic used to determine the order in which states are removed from the priority queue.
	We want to remove states that are deepest in the search tree first, and when there is a tie,
	prioritize those with the lowest cost. This has time complexity O(1) and space complexity O(1).
	"""
	def compVal(self):
		return (-len(self.path), self.cost)
	
	""" 
	This method calculates the new state obtained by adding the given city to the current state.
	This has time complexity O(n) + O(reduceMatrix) = O(n^2) in the number of cities, 
	and space complexity O(1).
	"""
	def addCity(self, city):
		rowIndex = self.path[-1]
		colIndex = city
		self.path.append(city)
		self.cost += self.matrix[rowIndex][colIndex]
		for row in self.matrix:
			row[colIndex] = math.inf
		for col in self.matrix.T:
			col[rowIndex] = math.inf
		self.matrix[colIndex][rowIndex] = math.inf # we can't return to the city we just came from
		self.reduceMatrix()

	""" 
	This method returns the route represented by the current state as a list of cities.
	This has time complexity O(n) in the number of cities, and space complexity O(n).
	"""
	def getRoute(self, originalCities):
		route = []
		for cityIndex in self.path:
			route.append(originalCities[cityIndex])
		return route
		
	def __lt__(self, otherMatrix):
		assert(isinstance(otherMatrix, TSPCostMatrix))
		return self.compVal() < otherMatrix.compVal()

	def __gt__(self, otherMatrix):
		assert(isinstance(otherMatrix, TSPCostMatrix))
		return self.compVal() > otherMatrix.compVal()
	
	def __eq__(self, otherMatrix):
		assert(isinstance(otherMatrix, TSPCostMatrix))
		return self.compVal() == otherMatrix.compVal()
	
	def __le__(self, otherMatrix):
		assert(isinstance(otherMatrix, TSPCostMatrix))
		return self.compVal() <= otherMatrix.compVal()
	
	def __ge__(self, otherMatrix):
		assert(isinstance(otherMatrix, TSPCostMatrix))
		return self.compVal() >= otherMatrix.compVal()
	
	def __str__(self):
		return str(self.matrix)


class GreedyMatrix(TSPCostMatrix):
	def __init__(self, cities):
		super().__init__(cities)
	
	def compVal(self):
		return self.cost
		

class TSPSolution:
	def __init__( self, listOfCities):
		self.route = listOfCities
		self.cost = self._costOfRoute()

	def _costOfRoute( self ):
		cost = 0
		last = self.route[0]
		for city in self.route[1:]:
			cost += last.costTo(city)
			last = city
		cost += self.route[-1].costTo( self.route[0] )
		return cost

	def enumerateEdges( self ):
		elist = []
		c1 = self.route[0]
		for c2 in self.route[1:]:
			dist = c1.costTo( c2 )
			if dist == np.inf:
				return None
			elist.append( (c1, c2, int(math.ceil(dist))) )
			c1 = c2
		dist = self.route[-1].costTo( self.route[0] )
		if dist == np.inf:
			return None
		elist.append( (self.route[-1], self.route[0], int(math.ceil(dist))) )
		return elist


def nameForInt( num ):
	if num == 0:
		return ''
	elif num <= 26:
		return chr( ord('A')+num-1 )
	else:
		return nameForInt((num-1) // 26 ) + nameForInt((num-1)%26+1)


class Scenario:

	HARD_MODE_FRACTION_TO_REMOVE = 0.20 # Remove 20% of the edges

	def __init__( self, city_locations, difficulty, rand_seed ):
		self._difficulty = difficulty

		if difficulty == "Normal" or difficulty == "Hard":
			self._cities = [City( pt.x(), pt.y(), \
								  random.uniform(0.0,1.0) \
								) for pt in city_locations]
		elif difficulty == "Hard (Deterministic)":
			random.seed( rand_seed )
			self._cities = [City( pt.x(), pt.y(), \
								  random.uniform(0.0,1.0) \
								) for pt in city_locations]
		else:
			self._cities = [City( pt.x(), pt.y() ) for pt in city_locations]


		num = 0
		for city in self._cities:
			city.setScenario(self)
			city.setIndexAndName( num, nameForInt( num+1 ) )
			num += 1

		# Assume all edges exists except self-edges
		ncities = len(self._cities)
		self._edge_exists = ( np.ones((ncities,ncities)) - np.diag( np.ones((ncities)) ) ) > 0

		if difficulty == "Hard":
			self.thinEdges()
		elif difficulty == "Hard (Deterministic)":
			self.thinEdges(deterministic=True)

	def getCities( self ):
		return self._cities


	def randperm( self, n ):		
		perm = np.arange(n)
		for i in range(n):
			randind = random.randint(i,n-1)
			save = perm[i]
			perm[i] = perm[randind]
			perm[randind] = save
		return perm

	def thinEdges( self, deterministic=False ):
		ncities = len(self._cities)
		edge_count = ncities*(ncities-1) # can't have self-edge
		num_to_remove = np.floor(self.HARD_MODE_FRACTION_TO_REMOVE*edge_count)

		can_delete	= self._edge_exists.copy()

		# Set aside a route to ensure at least one tour exists
		route_keep = np.random.permutation( ncities )
		if deterministic:
			route_keep = self.randperm( ncities )
		for i in range(ncities):
			can_delete[route_keep[i],route_keep[(i+1)%ncities]] = False

		# Now remove edges until 
		while num_to_remove > 0:
			if deterministic:
				src = random.randint(0,ncities-1)
				dst = random.randint(0,ncities-1)
			else:
				src = np.random.randint(ncities)
				dst = np.random.randint(ncities)
			if self._edge_exists[src,dst] and can_delete[src,dst]:
				self._edge_exists[src,dst] = False
				num_to_remove -= 1


class City:
	def __init__( self, x, y, elevation=0.0 ):
		self._x = x
		self._y = y
		self._elevation = elevation
		self._scenario	= None
		self._index = -1
		self._name	= None

	def setIndexAndName( self, index, name ):
		self._index = index
		self._name = name

	def setScenario( self, scenario ):
		self._scenario = scenario

	''' <summary>
		How much does it cost to get from this city to the destination?
		Note that this is an asymmetric cost function.
		 
		In advanced mode, it returns infinity when there is no connection.
		</summary> '''
	MAP_SCALE = 1000.0
	def costTo( self, other_city ):

		assert( type(other_city) == City )

		# In hard mode, remove edges; this slows down the calculation...
		# Use this in all difficulties, it ensures INF for self-edge
		if not self._scenario._edge_exists[self._index, other_city._index]:
			return np.inf

		# Euclidean Distance
		cost = math.sqrt( (other_city._x - self._x)**2 +
						  (other_city._y - self._y)**2 )

		# For Medium and Hard modes, add in an asymmetric cost (in easy mode it is zero).
		if not self._scenario._difficulty == 'Easy':
			cost += (other_city._elevation - self._elevation)
			if cost < 0.0:
				cost = 0.0	


		return int(math.ceil(cost * self.MAP_SCALE))

