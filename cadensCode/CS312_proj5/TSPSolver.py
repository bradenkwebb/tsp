#!/usr/bin/python3
import math

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
import heapq
import itertools

from queue import PriorityQueue

#a branch and bound node which stores each state for the path taken
class BBNode:
	state = [[]]
	lowerBound = math.inf
	citiesFrom = []
	citiesTo = []
	route = []
	currCity = -1
	depthFactor = 100
	boundFactor = 0.5

	def __init__(self, state, citiesFrom, citiesTo, currRoute, prevLowerBound = 0):
		self.lowerBound = prevLowerBound
		self.citiesFrom = citiesFrom
		self.citiesTo = citiesTo
		self.currCity = currRoute[len(currRoute) - 1]
		self.route = currRoute
		self.state = self.reduceStateAndGetBound(state)

	def reduceStateAndGetBound(self, state):
		newState = state
		for j in self.citiesFrom:
			minimum = math.inf
			for i in self.citiesTo:
				if(newState[j][i] < minimum):
					minimum = newState[j][i]
			for i in self.citiesTo:
				newState[j][i] -= minimum
			self.lowerBound += minimum

		for i in self.citiesTo:
			minimum = math.inf
			for j in self.citiesFrom:
				if (newState[j][i] < minimum):
					minimum = newState[j][i]
			for j in self.citiesFrom:
				newState[j][i] -= minimum
			self.lowerBound += minimum

		return newState

	def getCurrCity(self):
		return self.currCity

	def getState(self):
		return self.state

	def getCitiesTo(self):
		return self.citiesTo

	def getCitiesFrom(self):
		return self.citiesFrom

	def isSuccessful(self):
		return (len(self.citiesFrom) == 0 and len(self.citiesTo) == 0)

	def getLowerBound(self):
		return self.lowerBound

	def getRoute(self):
		return self.route

	def getPrioritization(self):
		return (self.boundFactor * self.lowerBound) - (self.depthFactor * len(self.route))

	def __lt__(self, other):
		if not isinstance(other, BBNode):
			# don't attempt to compare against unrelated types
			return NotImplemented

		return self.lowerBound == other.getLowerBound()


class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None

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
		return results


	''' <summary>
		This is the entry point for the greedy solver, which you must implement for
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''

	def greedy( self,time_allowance=60.0 ):
		print('inside greedy-------------------')
		cities = self._scenario.getCities()
		start_time = time.time()
		ncities = len(cities)

		#going to start at the first city
		route = [cities[0]]
		currCity = route[len(route) - 1]
		while (len(route) < ncities): #O(n)
			#find next city with lowest cost
			minCityInd = -1
			minCost = math.inf
			for cityInd in range(0, len(cities)): #O(n)
				if(not route.__contains__(cities[cityInd])):
					thisCost = currCity.costTo(cities[cityInd])
					if (thisCost <= minCost):
						minCityInd = cityInd
						minCost = thisCost
			route.append(cities[minCityInd])

		#loop overall takes O(n^2) in worst case

		bssf = TSPSolution(route)

		#check if we found the solution
		foundTour = (len(route) == ncities + 1)

		#end results O(1) in time
		results = {}
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf #will be math.inf if there is no path (hopefully?)
		results['time'] = end_time - start_time
		results['count'] = 1
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		print('ending greedy----------------')
		return results
		pass



	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints:
		max queue size, total number of states created, and number of pruned states.</returns>
	'''

	def branchAndBound( self, time_allowance=60.0 ):
		print('inside Branch and Bound +++++++++++++++++++++')
		#get initial values
		cities = self._scenario.getCities()
		ncities = len(cities)
		solutionCount = 0
		start_time = time.time()
		startBssf = self.greedy(time_allowance)
		#if (startBssf["cost"] == math.inf):
		#	startBssf = self.defaultRandomTour(time_allowance)

		maxQueueSize = 0
		totalStatesCreated = 0
		totalStatesPruned = 0

		#initial population of the state
		print('initial population of state')
		currCosts = []
		infinityCosts = []
		for j in range(0, ncities):
			newArray = []
			infinityArray = []
			for i in range(0, ncities):
				newArray.append(cities[j].costTo(cities[i]))
				infinityArray.append(math.inf)
			currCosts.append(newArray)
			infinityCosts.append(infinityArray)

		# converting initial bssf route to indexes
		bssfSoln = startBssf["soln"]
		bssfSolnRouteCities = bssfSoln.route
		bssfSolnRouteIndexes = []
		for i in range(0, len(bssfSolnRouteCities)):
			for j in range(0, len(cities)):
				if (cities[j] == bssfSolnRouteCities[i]):
					bssfSolnRouteIndexes.append(j)
		bssfSolnRouteIndexes.append(0)

		#create node for initial bssf
		bssf = BBNode(infinityCosts, [], [], bssfSolnRouteIndexes, startBssf["cost"])

		#experiment: getting the average cost of edges for the greedy approach to serve as a depth factor
		#averageEdge = 0
		#validEdgeCount = 0
		#for i in range(0, len(bssfSolnRouteIndexes) - 1):
		#	edge = cities[bssfSolnRouteIndexes[i]].costTo(cities[bssfSolnRouteIndexes[i + 1]])
		#	if (edge != math.inf):
		#		averageEdge += edge
		#		validEdgeCount += 1
		#averageEdge = averageEdge / validEdgeCount
		#print('averageEdge:')
		#print(averageEdge)


		#creating root node
		print('creating root Node')
		citiesFrom = []
		citiesTo = []
		for i in range(0,ncities):
			citiesFrom.append(i)
			citiesTo.append(i)
		rootNode = BBNode(currCosts, citiesFrom, citiesTo, [0], 0)
		nodeQueue = PriorityQueue()
		if (rootNode.getLowerBound() < bssf.getLowerBound()):
			totalStatesCreated += 1
			nodeQueue.put((rootNode.getPrioritization(), rootNode))

		#start of branch and bound iteration
		print('starting while loop')
		while (not nodeQueue.empty() and time.time()-start_time < time_allowance):
			currNode = nodeQueue.get()[1] #nodes in the queue are coupled with lower bound in tuples
			if (currNode.isSuccessful()):
				#add to the solution count and create new bssf if applicable
				solutionCount += 1
				if (currNode.getLowerBound() < bssf.getLowerBound()):
					bssf = currNode
					prunedTuple = self.prune(nodeQueue, bssf, totalStatesPruned)
					nodeQueue = prunedTuple[0]
					totalStatesPruned = prunedTuple[1]
			else:
				#expand current node
				statesTuple = self.expand(currNode, nodeQueue, bssf, ncities, totalStatesCreated)
				totalStatesCreated = statesTuple[0]
				totalStatesPruned += statesTuple[1]
				if (maxQueueSize < len(nodeQueue.queue)): #update max queue size
					maxQueueSize = len(nodeQueue.queue)

		# still count pruned states if the timer runs out
		if (time.time()-start_time < time_allowance):
			for nodeTuple in nodeQueue.queue:
				if (nodeTuple[1].getLowerBound() > bssf.getLowerBound()):
					totalStatesPruned += 1


		print('exited while loop')
		foundTour = len(bssf.getCitiesTo()) == 0

		#convert indexed cities into a route holding the actual cities
		foundCityRoute = []
		foundIndexRoute = bssf.getRoute()
		foundIndexRoute = foundIndexRoute[:len(foundIndexRoute) - 1] #exclude the repeated start node
		for i in range(0, len(foundIndexRoute)):
			foundCityRoute.append(cities[foundIndexRoute[i]])

		#return the data collected
		solution = TSPSolution(foundCityRoute)
		results = {}
		end_time = time.time()
		if (foundTour):
			results['cost'] = solution.cost
		else:
			results['cost'] = math.inf
		results['time'] = end_time - start_time
		results['count'] = solutionCount
		results['soln'] = solution
		results['max'] = maxQueueSize
		results['total'] = totalStatesCreated
		results['pruned'] = totalStatesPruned
		print('exiting Branch and Bound ++++++++++++++++++++')
		return results
		pass

	def prune(self, nodeQueue, bssf, totalStatesPruned):
		prunedArray = []
		for i in range(0, len(nodeQueue.queue)):
			currNode = nodeQueue.get()
			if (currNode[1].getLowerBound() <= bssf.getLowerBound()):
				prunedArray.append(currNode)
			else:
				totalStatesPruned += 1
		for i in range(0, len(prunedArray)):
			nodeQueue.put(prunedArray[i])
		return (nodeQueue, totalStatesPruned)

	def expand(self, currNode, nodeQueue, bssf, ncities, totalStatesCreated):
		currRow = currNode.getCurrCity()
		#make sure only cities that do not complete the loop are considered
		citiesTo = currNode.getCitiesTo()
		if (len(citiesTo) > 1): #we have other cities than the start node
			citiesTo = citiesTo[1:]
		#else we only have the root city to consider

		prunedStatesCount = 0

		for i in citiesTo: #for each city considered
			#get new state
			currState = currNode.getState()
			newState = []
			for j in range(0, ncities):
				newArray = []
				for k in range(0, ncities):
					newArray.append(currState[j][k])
				newState.append(newArray)
			#before modifying retreive the cost
			startLowerBound = currNode.getLowerBound() + currState[currRow][i]
			#modifying state
			for j in range(0, ncities):
				newState[j][i] = math.inf
			for k in range(0, ncities):
				newState[currRow][k] = math.inf
			#edit cities to go to and come from
			newFromCities = currNode.getCitiesFrom().copy() #note:includes the start node
			newToCities = currNode.getCitiesTo().copy()
			newFromCities.remove(currRow)
			newToCities.remove(i)
			#add onto the route
			route = currNode.getRoute().copy()
			route.append(i)
			#create new node
			newNode = BBNode(newState, newFromCities, newToCities, route, startLowerBound)
			totalStatesCreated += 1

			if (newNode.getLowerBound() <= bssf.getLowerBound()):
				nodeQueue.put((newNode.getPrioritization(), newNode))
			else:
				prunedStatesCount += 1

		return (totalStatesCreated, prunedStatesCount)




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
