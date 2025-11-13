# Import libraries
import random
import numpy as np
import os
import math 
import matplotlib.pyplot as plt
import copy
from typing import List, Dict
from math import sqrt
import time

# Clear console (optional, for readability if running in a terminal)
os.system('cls')

class Depot:
    def __init__(self, id, x, y, maxVehicles, maxDuration, maxVehicleLoad):
        """
        Constructor for the Depot class.

        :param id: ID of the depot
        :param x: x-coordinate of the depot
        :param y: y-coordinate of the depot
        :param maxVehicles: Maximum number of vehicles that the depot can handle
        :param maxDuration: Maximum duration that vehicles can be assigned
        :param maxVehicleLoad: Maximum load capacity of the vehicles
        """
        self.id = id
        self.x = x
        self.y = y
        self.maxVehicles = maxVehicles
        self.maxDuration = maxDuration
        self.maxVehicleLoad = maxVehicleLoad

    # Getters
    def getId(self):
        return self.id

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def getMaxDuration(self):
        return self.maxDuration

    def getMaxVehicles(self):
        return self.maxVehicles

    def getMaxVehicleLoad(self):
        return self.maxVehicleLoad

    # Setters (Protected access equivalent)
    def setX(self, x):
        self.x = x

    def setY(self, y):
        self.y = y

    def getDepotCopy(self):
        """
        Return a copy of the depot.
        """
        return Depot(self.id, self.x, self.y, self.maxVehicles, self.maxDuration, self.maxVehicleLoad)

    def __str__(self):
        """
        Return a string representation of the depot (its id).
        """
        return str(self.id)
     
class Customer:
    def __init__(self, id, x, y, demand, service_time, ready_time, due_time):
        """
        Constructor for the Customer class.

        :param id: ID of the customer
        :param x: x-coordinate of the customer
        :param y: y-coordinate of the customer
        :param duration: Service duration
        :param ready time: earliest service time in time window
        :param due time: latest service time in time window
        :param demand: Service demand
        :param onBorderline: boolean value to determine if customer is borderline or not
        :param possibleDepotsIDs: list of possible depots a customer can be assigned to based on distance
        """
        self.id = id # customer id
        self.x = x  # x coordinate
        self.y = y  # y coordinate
        self.service_time = service_time  # service duration
        self.ready_time = ready_time # earliest time to serve the customer
        self.due_time = due_time # latest time to serve the customer
        self.demand = demand  # service demand
        self.onBorderline = False  # default value
        self.possibleDepotsIDs = [] # possible depots list for customer

    def addPossibleDepot(self, possibleDepotsID):
        """
        Add a possible depot ID to the customer's list of possible depots.
        """
        self.possibleDepotsIDs.append(possibleDepotsID)

    def setOnBorderline(self):
        """
        Mark the customer as being on the borderline if it's not already marked.
        """
        if not self.onBorderline:
            self.onBorderline = True

    # Getters
    def getId(self):
        return self.id

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def getDuration(self):
        return self.duration

    def getDemand(self):
        return self.demand
    
    def getServiceTime(self):
        return self.service_time
    
    def getReadyTime(self):
        return self.ready_time
    
    def getDueTime(self):
        return self.due_time

    def getOnBorder(self):
        return self.onBorderline

    def getPossibleDepots(self):
        return self.possibleDepotsIDs

    # Setters
    def setX(self, x):
        self.x = x  

    def setY(self, y):
        self.y = y

    def __str__(self):
        """
        Return a string representation of the customer (its id).
        """
        return str(self.id)    
    
class CrowdedDepot(Depot):
    def __init__(self, id, x, y, maxVehicles, maxDuration, maxVehicleLoad):
        """
        Constructor for the CrowdedDepot class, which inherits from Depot.

        :param id: ID of the depot
        :param x: x-coordinate of the depot
        :param y: y-coordinate of the depot
        :param maxVehicles: Maximum number of vehicles for this depot
        :param maxDuration: Maximum duration allowed for this depot
        :param maxVehicleLoad: Maximum load capacity of the vehicles at this depot
        """
        super().__init__(id, x, y, maxVehicles, maxDuration, maxVehicleLoad)
        self.customers = []

    def __init__(self, depot):
        """
        Constructor that creates a CrowdedDepot object from another Depot object.

        :param depot: Another Depot object
        """
        super().__init__(depot.getId(), depot.getX(), depot.getY(), depot.getMaxVehicles(),
                         depot.getMaxDuration(), depot.getMaxVehicleLoad())
        self.customers = []

    # Getters
    def getCustomers(self):
        return self.customers

    def getCustomer(self, id):
        return next((customer for customer in self.customers if customer.getId() == id), None)

    def getCustomerIds(self):
        customerIds = [str(customer.getId()) for customer in self.customers]
        return f"Depot {self.getId()} has customers: {', '.join(customerIds)}"

    # Setters
    def setX(self, x):
        super().setX(x)  # Used to shift coordinates before plotting

    def setY(self, y):
        super().setY(y)

    def addCustomer(self, customer):
        self.customers.append(customer)

    def removeCustomer(self, customer):
        if customer in self.customers:
            self.customers.remove(customer)          

class Manager:
    def __init__(self, problemFilepath, borderlineThreshold):
        """
        Constructor manager that reads data from instance and manages customers by assigning them.

        :param borderlineThreshold: threshold that determines whether a given customer is borderline
        :param customers: list of all the customers objects of the problem instance
        :param swappable customers: list of the borderline customers objects that can be swapped between depots later
        :param depots: list of all the depots objects of the problem instance
        :param readData: method used to read and extract data from the problem instance
        """
        self.borderlineThreshold = borderlineThreshold
        self.customers = []
        self.swappableCustomers = []
        self.depots = []
        self.readData(problemFilepath)

    # Getters
    def getCustomers(self):
        return self.customers

    def getSwappableCustomers(self):
        return self.swappableCustomers

    def getDepots(self):
        return self.depots

    def getCustomer(self, id):
        return next((customer for customer in self.customers if customer.getId() == id), None)

    def getDepot(self, id):
        return next((depot for depot in self.depots if depot.getId() == id), None)

    def copyDepots(self):
        # Create a copy of depots
        return [depot.getDepotCopy() for depot in self.depots]

    def readData(self, problemFilepath):
       
        try:
            with open(problemFilepath, 'r') as fileReader:
                # Read the first line to get number of vehicles, number of customers, and number of depots
                firstLine = fileReader.readline().strip()
                _ , nVehicles, nCustomers, nDepots = map(int, firstLine.split()[0:4])
   
                # Read each depot's maximum specifications
                maxRouteDurationsPerDepot = []
                maxLoadEachVehiclePerDepot = []
                for _ in range(nDepots):
                    line = fileReader.readline().strip()
                    maxDuration, maxVehicleLoad = map(int, line.split()[0:2])
                    maxRouteDurationsPerDepot.append(maxDuration)
                    maxLoadEachVehiclePerDepot.append(maxVehicleLoad)

                # Read customer data and create customers
                for _ in range(nCustomers):
                    line = fileReader.readline().strip()
                    customerId, x, y, service_time, ready_time, due_time, demand = int(line.split()[0]), float(line.split()[1]), float(line.split()[2]), int(line.split()[4]), int(line.split()[11])/36, int(line.split()[12])/36,int(line.split()[4])
    
                    customer = Customer(customerId, x, y, demand, service_time, ready_time, due_time)
                    self.customers.append(customer)

                # Read depot data and create depots
                for i in range(nDepots):
                    line = fileReader.readline().strip()
                    _, x, y = map(float, line.split()[0:3])
                    maxDuration = maxRouteDurationsPerDepot[i]
                    maxVehicleLoad = maxLoadEachVehiclePerDepot[i]
                    depot = Depot(i + 1, x, y, nVehicles, maxDuration, maxVehicleLoad)
                    self.depots.append(depot)
        
        except FileNotFoundError:
            print("Cannot find file...")
            raise

    def assignCustomersToDepots(self):
        depots = [CrowdedDepot(depot) for depot in self.copyDepots()]
        for customer in self.customers:
            isBorderlineCustomer = False
            depotDistances = {}
            customerCoordinates = [customer.getX(), customer.getY()]

            # Initial values
            firstDepot = depots[0]
            shortestDistance = float('inf')
            currentShortestDepot = firstDepot

            # Loop through depots to find the nearest depot and the distance to that depot
            for depot in depots:
                depotCoordinates = [depot.getX(), depot.getY()]
                distance = Euclidian.distance(customerCoordinates, depotCoordinates)
                depotDistances[depot.getId()] = distance
                if distance < shortestDistance:
                    shortestDistance = distance
                    currentShortestDepot = depot

            # Add customer to the depot closest to the customer
            currentShortestDepot.addCustomer(customer)
            customer.addPossibleDepot(currentShortestDepot.getId())

            # Check if the customer is on the borderline with another depot
            for depotId, distance in depotDistances.items():
                if (distance - shortestDistance) / shortestDistance < self.borderlineThreshold and depotId != currentShortestDepot.getId():
                    customer.addPossibleDepot(depotId)
                    customer.setOnBorderline()
                    isBorderlineCustomer = True

            if isBorderlineCustomer:
                self.swappableCustomers.append(customer)

        return depots

    def assignCustomerToDepotsFromIndividual(self, individual):
        depots = [CrowdedDepot(depot) for depot in self.copyDepots()]
        for depot in depots:
            routes = individual.getChromosome().get(depot.getId())
            for route in routes:
                for customerId in route.getRoute():
                    customer = self.getCustomer(customerId)
                    depot.addCustomer(customer)
        return depots            
    
class Euclidian:
    @staticmethod
    def distance(coordinatesA, coordinatesB):
        """
        Returns Euclidean distance between a pair of (x, y) coordinates
        """
        xDistance = abs(coordinatesA[0] - coordinatesB[0])
        yDistance = abs(coordinatesA[1] - coordinatesB[1])
        return math.sqrt((xDistance * xDistance) + (yDistance * yDistance))
       
class Route:
    def __init__(self, route, demand, distance, RouteDuration, time_window_feasible, routeEarlyViolation, routeLateViolation, routeViolation):
        """
        Represents a route from and to a particular depot.
        :param route: List of customer IDs
        :param demand: Total demand of customers
        :param distance: Total route distance
        :param routeDuration: Total route duration (including travel time, service time, waiting time)
        :param time_window_feasible: boolean used to determine whether all customers of the route are served within time window
        :param routeViolation: Total time violation (early + late)
        :param routeEarlyViolation: Time violation due to early arrivals (total waiting time)
        :param routeLateViolation: Time violation due to late arrivals (total lateness time)
        """
        self.route = route    # List of customer IDs
        self.demand = demand  # Total demand of customers
        self.distance = distance  # Total route distance
        self.RouteDuration = RouteDuration # Total route duration
        self.time_window_feasible = time_window_feasible
        self.routeViolation = routeViolation
        self.routeEarlyViolation = routeEarlyViolation
        self.routeLateViolation = routeLateViolation

    # Getters
    def getRoute(self):
        return self.route

    def getDemand(self):
        return self.demand

    def getDistance(self):
        return self.distance
    
    def getRouteDuration(self):
        return self.RouteDuration
    
    def getTimeFeasibility(self):
        return self.time_window_feasible
    
    def getRouteViolation(self):
        return self.routeViolation
    
    def getRouteEarlyViolation(self):
        return self.routeEarlyViolation
    
    def getRouteLateViolation(self):
        return self.routeLateViolation

    # Setters
    def setDemand(self, demand):
        self.demand = demand

    def setDistance(self, distance):
        self.distance = distance
        
    def setRoute(self, new_route):
        self.route = new_route
        
    def setRouteDuration(self, RouteDuration):
        self.RouteDuration = RouteDuration    
        
    def setTimeFeasibility(self, time_window_feasible):
        self.time_window_feasible = time_window_feasible  
        
    def setRouteViolation(self, routeViolation):
        self.routeViolation = routeViolation         
        
    def setRouteEarlyViolation(self, routeEarlyViolation):
        self.routeEarlyViolation = routeEarlyViolation    
        
    def setRouteLateViolation(self, routeLateViolation):
        self.routeLateViolation = routeLateViolation   
        
            
    def removeCustomer(self, ID):
        """
        Removes a customer from the route if present.
        :param ID: ID of the customer to remove
        :return: True if customer was removed, False otherwise
        """
        if ID in self.route:
            self.route.remove(ID)
            return True
        return False

    def isEmpty(self):
        """
        Checks if the route is empty.
        :return: True if route is empty, False otherwise
        """
        return len(self.route) == 0

    def getClone(self):
        """
        Returns a clone of the current route.
        :return: A new Route object with the same route, demand, and distance
        """
        return Route(self.route.copy(), self.demand, self.distance, self.RouteDuration, self.time_window_feasible, self.routeEarlyViolation, self.routeLateViolation, self.routeViolation)      
    
class Individual:
    def __init__(self, initial_chromosome: Dict[int, List['Route']]):
        """
        Represents a solution to the problem.
        :param initial_chromosome: Map of DepotID -> DepotRoutes
        """
        self.distance = 0  
        self.chromosome = initial_chromosome
        self.duration = 0
        self.time_violation = 0 
        self.early_time_violation = 0
        self.late_time_violation = 0
        self.environmental_cost = 0
        self.economic_cost = 0
        self.social_cost = 0
        self.isFeasible = False
        

    # Getters
    def getChromosome(self):
        return self.chromosome 
    
    def getFeasibility(self):
        return self.isFeasible

    def getTotalDistance(self):
        return self.distance
       
    def getEnvironmentalCost(self):
        return self.environmental_cost
    
    def getSocialCost(self):
        return self.social_cost
    
    def getEconomicCost(self):
        return self.economic_cost
    
    def getTotalDuration(self):
        return self.duration
    
    def getTotalTimeViolation(self):
        return self.time_violation
    
    def getEarlyTimeViolation(self):
        return self.early_time_violation
    
    def getLateTimeViolation(self):
        return self.late_time_violation
    
    # Setters
    def setIsFeasible(self, feasible: bool):
        self.isFeasible = feasible

    def setTotalDistance(self, distance: float):
        if distance < 0:
            print("Error: Distance cannot be negative (negative fitness in: Individual.setFitness)")
            return
        self.distance = distance
          
    def setEnvironmentalCost(self, environmental_cost):
        self.environmental_cost = environmental_cost
        
    def setSocialCost(self, social_cost):
        self.social_cost = social_cost
        
    def setEconomicCost(self, economic_cost):
        self.economic_cost = economic_cost
        
    def setTotalDuration(self, duration):
        self.duration = duration  
        
    def setTotalTimeViolation(self, time_violation):
        self.time_violation = time_violation  

    def setEarlyTimeViolation(self, early_time_violation):
        self.early_time_violation = early_time_violation
    
    def setLateTimeViolation(self, late_time_violation):
        self.late_time_violation = late_time_violation

    def __str__(self):
        """
        String representation of the individual.
        """
        result = ""
        for key, chromosome_depot in self.chromosome.items():
            result += f"Depot{key}: "
            for route in chromosome_depot:
                result += str(route.getRoute()) + " " + str(route.getDemand()) + " - " + str(route.getRouteDuration())
             # Remove the last ' - '
            result += "| "
        
        return result[:-4]  # Remove the last ' |  '

    def getClone(self):
        """
        Clone individual without fitness and isFeasible
        """
        chromosome_copy = {}

        for key, chromosome_depot in self.chromosome.items():
            chromosome_depot_copy = [route.getClone() for route in chromosome_depot]
            chromosome_copy[key] = chromosome_depot_copy

        cloned_individual = Individual(chromosome_copy)
        cloned_individual.distance = self.distance  # Copy the distance value
        cloned_individual.isFeasible = self.isFeasible  # Copy the feasibility status
        cloned_individual.environmental_cost = self.environmental_cost
        cloned_individual.economic_cost = self.economic_cost
        cloned_individual.social_cost = self.social_cost
        cloned_individual.early_time_violation = self.early_time_violation
        cloned_individual.late_time_violation = self.late_time_violation
        cloned_individual.duration = self.duration
        return cloned_individual

    def __lt__(self, other):
        """
        Comparison based on feasibility and fitness.
        """
        if self.isFeasible and not other.isFeasible:
            return True
        if not self.isFeasible and other.isFeasible:
            return False
        return self.environmental_cost < other.environmental_cost     

class Initializer:
    @staticmethod
    def init(populationSize, depots, metrics, manager):
        """
        Returns a list of new individuals. Each individual chromosome is initialized with depots
        containing the appropriate customers according to depots (initial assignments), but the routes
        within each depot are generated randomly with a bias towards feasibility.
        """
        population = []
        
        for idx in range(populationSize):
            
            chromosome = {}
            if idx <= 20:
                for depot in depots:
                    
                    # Assuming RouteScheduler.get_initial_routes() is defined in Python
                    chromosomeDepot = RouteScheduler.getInitialRoutes(depot, metrics,"savings", manager)                   
                        
                    chromosome[depot.getId()] = chromosomeDepot

                individual = Individual(chromosome)
                population.append(individual)
                  
            elif idx <= 40:
                for depot in depots:
                    # Assuming RouteScheduler.get_initial_routes() is defined in Python
                    chromosomeDepot = RouteScheduler.getInitialRoutes(depot, metrics,"sweep", manager)
                    chromosome[depot.getId()] = chromosomeDepot

                individual = Individual(chromosome)
                population.append(individual)
             
            else:
                for depot in depots:
                    # Assuming RouteScheduler.get_initial_routes() is defined in Python
                    chromosomeDepot = RouteScheduler.getInitialRoutes(depot, metrics,"random", manager)
                    chromosome[depot.getId()] = chromosomeDepot
                    
                individual = Individual(chromosome)
                population.append(individual)
                
        return population

class RouteScheduler:
    @staticmethod
    
    def heuristic_amelioration(tab):
        new_tab = copy.deepcopy(tab)

        if len(tab) != 2:
            return new_tab

        route_x = new_tab[0].getRoute()
        route_y = new_tab[1].getRoute()

        # Skip if either route is too short for 2-opt
        if len(route_x) < 2 or len(route_y) < 2:
            return new_tab

        if random.random() > 0.5:
            # 2-opt swap only if both routes have at least 2 customers
            route_x[-2], route_y[-2] = route_y[-2], route_x[-2]

        else:
            # OR-opt: pick random positions, ensuring they exist
            if len(route_x) > 1 and len(route_y) > 1:
                a = random.randint(0, len(route_x) - 1)
                c = random.randint(0, len(route_y) - 1)
                route_x[a], route_y[c] = route_y[c], route_x[a]

        return new_tab
    
    
    def get_initial_routes(depot, metrics):
        """
        Creates routes from a crowded depot, with a bias towards feasibility.
        """
        customers = list(depot.getCustomers())  # make copy of customers
        random.shuffle(customers)  # shuffle the copied list

        max_vehicle_load = depot.getMaxVehicleLoad()

        routes = []

        current_aggregated_vehicle_load = 0

        route = []

        for customer in customers:
            demand_constraint_holds = current_aggregated_vehicle_load + customer.getDemand() <= max_vehicle_load
            if demand_constraint_holds:
                route.append(customer.getId())
                current_aggregated_vehicle_load += customer.getDemand()
            else:
                # Add current route to routes
                route_distance = metrics.getRouteDistance(depot.getId(), route)
                routes.append(Route(route, current_aggregated_vehicle_load, route_distance))
                # Reset current route values
                route = [customer.getId()]
                current_aggregated_vehicle_load = customer.getDemand()

        if route:  # Add last route if there are any remaining customers
            route_distance = metrics.getRouteDistance(depot.getId(), route)
            routes.append(Route(route, current_aggregated_vehicle_load, route_distance))

        return routes
    
    
    
    def getInitialRoutes(depot, metrics, method, manager):
        """
        Creates routes from a depot using different heuristics.
        method can be:
        - "savings"    : Clarke & Wright savings (default, current behavior)
        - "sweep"      : Sweep heuristic
        - "salhi_sari" : Salhi & Sari heuristic (simplified for single depot)
        """
        customers = list(depot.getCustomers())  # copy customers

        # ------------------------------------------------------------------
        # Method 1: Savings heuristic (Clarke & Wright)
        # ------------------------------------------------------------------
        if method == "savings":
          
            random.shuffle(customers)

           # Step 1: compute savings
            savings_list = []
            for ci in customers:
                for cj in customers:
                    if ci.getId() >= cj.getId():
                        continue
                    dist_di = Euclidian.distance([depot.getX(), depot.getY()],
                                             [ci.getX(), ci.getY()])
                    dist_dj = Euclidian.distance([depot.getX(), depot.getY()],
                                             [cj.getX(), cj.getY()])
                    dist_ij = Euclidian.distance([ci.getX(), ci.getY()],
                                             [cj.getX(), cj.getY()])
                    s = dist_di + dist_dj - dist_ij
                    savings_list.append((s, ci.getId(), cj.getId()))
            random.shuffle(savings_list) 
            epsilon = 1e-6       
            savings_list.sort(key=lambda x: x[0] + random.uniform(-epsilon, epsilon), reverse=True)

            # Step 2: initialize routes
            routes = {c.getId(): [c.getId()] for c in customers}
            route_loads = {c.getId(): c.getDemand() for c in customers}

            # Step 3: merge routes with stochastic behavior
            top_k = 10  # choose randomly among top-k savings
            merged_keys = set()

            while savings_list:
                # pick randomly from top-k savings
                candidates = savings_list[:top_k] if len(savings_list) >= top_k else savings_list
                s, i, j = random.choice(candidates)
                savings_list.remove((s, i, j))

                if i in merged_keys or j in merged_keys:
                    continue

                route_i = route_j = None
                for key, r in routes.items():
                    if key in merged_keys:
                        continue
                    if r[0] == i or r[-1] == i:
                        route_i = key
                    if r[0] == j or r[-1] == j:
                        route_j = key

                if route_i is None or route_j is None or route_i == route_j:
                    continue

                new_load = route_loads[route_i] + route_loads[route_j]
                if new_load > depot.getMaxVehicleLoad():
                    continue

                # random flip routes to allow more diversity
                if random.random() > 0.5:
                    routes[route_i] = routes[route_i][::-1]
                if random.random() > 0.5:
                    routes[route_j] = routes[route_j][::-1]

                # merge
                if routes[route_i][-1] == i and routes[route_j][0] == j:
                    routes[route_i] += routes[route_j]
                    route_loads[route_i] = new_load
                    merged_keys.add(route_j)
                    del routes[route_j]
                elif routes[route_j][-1] == j and routes[route_i][0] == i:
                    routes[route_j] += routes[route_i]
                    route_loads[route_j] = new_load
                    merged_keys.add(route_i)
                    del routes[route_i]

            # Step 4: build Route objects
            final_routes = []
            for route in routes.values():
                route_distance = metrics.getRouteDistance(depot.getId(), route)
                route_load = sum(manager.getCustomer(cid).getDemand() for cid in route)
                route_duration = metrics.CalculateRouteDuration(depot.getId(), route)
                time_feasible = metrics.checkTimeFeasibility(depot.getId(), route)
                time_early_violation, time_violation, time_late_violation = metrics.calculateTimeWindowViolation(depot.getId(), route)

                final_routes.append(Route(route, route_load, route_distance, route_duration,
                                  time_feasible, time_early_violation, time_late_violation, time_violation))

            return final_routes

        # ------------------------------------------------------------------
        # Method 2: Sweep heuristic
        # ------------------------------------------------------------------
        elif method == "sweep":
            angle_offset = random.uniform(0, 2 * math.pi)
            depot_x, depot_y = depot.getX(), depot.getY()

            # sort customers by polar angle
            
            customers.sort(key=lambda c: math.atan2(c.getY() - depot_y,
                                                c.getX() - depot_x) + angle_offset)
            
            # --- NEW: start from a random customer ---
            start_index = random.randint(0, len(customers) - 1)
            customers = customers[start_index:] + customers[:start_index]
            final_routes, current_route = [], []
            current_load = 0
            for c in customers:
                demand = c.getDemand()
                if current_load + demand <= depot.getMaxVehicleLoad():
                   current_route.append(c.getId())
                   current_load += demand
                else:
                   dist = metrics.getRouteDistance(depot.getId(), current_route)
                   time_early_violation, time_violation, time_late_violation = metrics.calculateTimeWindowViolation(depot.getId(), current_route) 
                   route_duration = metrics.CalculateRouteDuration(depot.getId(), current_route)
                   time_feasible = metrics.checkTimeFeasibility(depot.getId(), current_route)
                   final_routes.append(Route(current_route, current_load, dist, route_duration, time_feasible, time_early_violation, time_violation, time_late_violation))
                   current_route = [c.getId()]
                   current_load = demand
            if current_route:
                dist = metrics.getRouteDistance(depot.getId(), current_route)
                route_duration = metrics.CalculateRouteDuration(depot.getId(), current_route)
                time_feasible = metrics.checkTimeFeasibility(depot.getId(), current_route)
                time_early_violation, time_violation, time_late_violation = metrics.calculateTimeWindowViolation(depot.getId(), current_route) 
                final_routes.append(Route(current_route, current_load, dist, route_duration, time_feasible, time_early_violation, time_violation, time_late_violation))
            return final_routes

        # ------------------------------------------------------------------
        # Method 3: Random
        # ------------------------------------------------------------------
        elif method == "clark_wright":
            # Step 0: shuffle customers to introduce diversity in initial individuals
            random.shuffle(customers)

            # Step 1: compute savings list (s, cust1_id, cust2_id)
            savings_list = []
            depot_x, depot_y = depot.getX(), depot.getY()
            for ci in customers:
                for cj in customers:
                    if ci.getId() >= cj.getId():
                        continue
                    dist_di = Euclidian.distance([depot_x, depot_y], [ci.getX(), ci.getY()])
                    dist_dj = Euclidian.distance([depot_x, depot_y], [cj.getX(), cj.getY()])
                    dist_ij = Euclidian.distance([ci.getX(), ci.getY()], [cj.getX(), cj.getY()])
                    s = dist_di + dist_dj - dist_ij
                    savings_list.append((s, ci.getId(), cj.getId()))

            # sort by descending savings
            random.shuffle(savings_list)
            savings_list.sort(key=lambda x: x[0], reverse=True)

            # Step 2: initialize each customer as a separate route
            routes = {c.getId(): [c.getId()] for c in customers}
            route_loads = {c.getId(): c.getDemand() for c in customers}

            # Step 3: merge routes according to Clarke & Wright
            for s, i, j in savings_list:
                route_i = route_j = None
                for key, r in routes.items():
                    if r[0] == i or r[-1] == i:
                        route_i = key
                    if r[0] == j or r[-1] == j:
                        route_j = key
                if route_i is None or route_j is None or route_i == route_j:
                    continue

                # Check vehicle capacity
                new_load = route_loads[route_i] + route_loads[route_j]
                if new_load > depot.getMaxVehicleLoad():
                    continue

                # Merge at the ends if possible
                if routes[route_i][-1] == i and routes[route_j][0] == j:
                    routes[route_i] += routes[route_j]
                    route_loads[route_i] = new_load
                    del routes[route_j]
                elif routes[route_j][-1] == j and routes[route_i][0] == i:
                    routes[route_j] += routes[route_i]
                    route_loads[route_j] = new_load
                    del routes[route_i]

            # Step 4: convert merged routes to Route objects
            final_routes = []
            for route in routes.values():
                route_distance = metrics.getRouteDistance(depot.getId(), route)
                route_duration = metrics.CalculateRouteDuration(depot.getId(), route)
                time_feasible = metrics.checkTimeFeasibility(depot.getId(), route)
                time_early_violation, time_violation, time_late_violation = metrics.calculateTimeWindowViolation(depot.getId(), route) 
                final_routes.append(Route(route, new_load, route_distance, route_duration, time_feasible, time_early_violation, time_violation, time_late_violation))
    
                final_routes = RouteScheduler.heuristic_amelioration(final_routes)
            
            for r in final_routes:
                updated_route = r.getRoute()
              
                new_load = sum(manager.getCustomer(cid).getDemand() for cid in updated_route)
                new_distance = metrics.getRouteDistance(depot.getId(), updated_route)
                r.setDemand(new_load)
                r.setDistance(new_distance)
            
            return final_routes
        
        
        
        
        elif method == "random":
            # Shuffle customers globally
            random.shuffle(customers)

            final_routes = []
            current_route = []
            current_load = 0

            for c in customers:
                demand = c.getDemand()
                # If adding this customer does not exceed vehicle load, add to current route
                if current_load + demand <= depot.getMaxVehicleLoad():
                    current_route.append(c.getId())
                    current_load += demand
                else:
                    # Finish current route
                    dist = metrics.getRouteDistance(depot.getId(), current_route)
                    route_duration = metrics.CalculateRouteDuration(depot.getId(), current_route)
                    time_feasible = metrics.checkTimeFeasibility(depot.getId(), current_route)
                    time_early_violation, time_violation, time_late_violation = metrics.calculateTimeWindowViolation(depot.getId(), current_route) 
                    final_routes.append(Route(current_route, current_load, dist, route_duration, time_feasible, time_early_violation, time_violation, time_late_violation))
                    # Start a new route
                    current_route = [c.getId()]
                    current_load = demand

            # Add last route if not empty
            if current_route:
                dist = metrics.getRouteDistance(depot.getId(), current_route)
                route_duration = metrics.CalculateRouteDuration(depot.getId(), current_route)
                time_feasible = metrics.checkTimeFeasibility(depot.getId(), current_route)
                time_early_violation, time_violation, time_late_violation = metrics.calculateTimeWindowViolation(depot.getId(), current_route) 
                final_routes.append(Route(current_route, current_load, dist, route_duration, time_feasible, time_early_violation, time_violation, time_late_violation))
            

            return final_routes 

        else:
            raise ValueError(f"Unknown initialization method: {method}")
        
        
class Insertion:
    def __init__(self, manager, metrics, depot, augmentedRoutes, customerID, routeLoc, index):
        self.manager = manager
        self.metrics = metrics
        self.depot = depot
        self.cost = self.insertionCost(augmentedRoutes, customerID, routeLoc, index)
        self.isFeasible = metrics.checkRoutes(depot, augmentedRoutes[routeLoc])
        self.result = augmentedRoutes

    def insertionCost(self, routes, customerID, routeLoc, index):
        """"
        Computes the cost of the insertion, based on which route and location the insertion is applied.
        """
        augmentedRoute = routes[routeLoc]
        customer = self.manager.getCustomer(customerID)
        
        # Fuel-to-air mass ratio 
        ksi = 1

        # Heating value for diesel fuel (J/kg) 
        k = 44e6

        # Conversion factor (gram/second to liter/second)
        psi = 0.832

        # Engine friction factor (kJ/rev/liter) yes
        F = 15

        # Engine speed (rev/s)
        N = 34

        # Engine displacement (liters)
        V = 10

        # Vehicle drive train efficiency yes
        epsilon = 0.45

        # Efficiency parameter for diesel engines yes
        omega = 0.4

        # Gravity (m/s2) yes
        g = 9.81

        # Rolling resistance coefficient yes
        C_r = 0.01

        # Road angle yes
        phi = math.radians(0)

        # Aerodynamic drag coefficient yes
        C_d = 0.7

        # Frontal surface area of a vehicle (m2)
        A = 8.5

        # Air density (kg/m3)
        ro = 1.2041


        # Vehicle speed (m/s2)
        speed = 10

        # Curb weight of vehicle (kg) yes
        curb_weight = 5000
        
        payload = 100
        
        # First term
        # Lambda: scaling factor
        lamda = ksi / k

        
        def EnvironmentalCostFromDistance(distance_km):
            """Compute CO₂ cost for a given distance (m)."""
            distance = distance_km * 1000
            # First term
            
            # Term 1: Engine friction
            term1 = (lamda * F * N * V * distance) / speed
        
            M = curb_weight + payload
            # Term 2: Rolling resistance + slope (curb weight + payload)
            gamma = 1 / (epsilon * omega)
            term2 = lamda * gamma * M * g * C_r * distance 

            # Term 3: Aerodynamic drag
            term3 = lamda * gamma * 0.5 * C_d * A * ro * distance * speed ** 2
       

            # Fuel consumption formula
            fuel_consumption_liters = (term1 + term2 + term3) / psi
        
            # CO2 factor
            CO2_factor = 2.64
        
            gas_emissions = fuel_consumption_liters * CO2_factor
            
            return gas_emissions
        
        # If this insertion created a new route distance, the cost is simply twice the distance between depot and customer
        if len(augmentedRoute) == 1:
            
            return 2 * EnvironmentalCostFromDistance(Euclidian.distance([self.depot.getX(), self.depot.getY()], [customer.getX(), customer.getY()]))

        # If this insertion was at start or end of route,
        # cost = distance from depot to customer and customer to old first/last customer minus distance from depot to old first/last customer
        if index == 0 or index == len(augmentedRoute) - 1:
            offset = 1 if index == 0 else -1
            otherCustomer = self.manager.getCustomer(augmentedRoute[index + offset])
            EnvironmentalcostToDepot = EnvironmentalCostFromDistance(Euclidian.distance([self.depot.getX(), self.depot.getY()], [customer.getX(), customer.getY()]))
            EnvironmentalcostToOther = EnvironmentalCostFromDistance(Euclidian.distance([customer.getX(), customer.getY()], [otherCustomer.getX(), otherCustomer.getY()]))
            EnvironmentalcostDepotToOther = EnvironmentalCostFromDistance(Euclidian.distance([self.depot.getX(), self.depot.getY()], [otherCustomer.getX(), otherCustomer.getY()]))
            return EnvironmentalcostToDepot + EnvironmentalcostToOther - EnvironmentalcostDepotToOther

        # Else insertion is at location between two customers, and
        # cost = distance from customer to both other customers minus distance between both other customers
        customerBefore = self.manager.getCustomer(augmentedRoute[index - 1])
        customerAfter = self.manager.getCustomer(augmentedRoute[index + 1])
        EnvironmentalcostToCustomerBefore = EnvironmentalCostFromDistance(Euclidian.distance([customer.getX(), customer.getY()], [customerBefore.getX(), customerBefore.getY()]))
        EnvironmentalcostToCustomerAfter = EnvironmentalCostFromDistance(Euclidian.distance([customer.getX(), customer.getY()], [customerAfter.getX(), customerAfter.getY()]))
        EnvironmentalcostBetweenBeforeAfter = EnvironmentalCostFromDistance(Euclidian.distance([customerBefore.getX(), customerBefore.getY()], [customerAfter.getX(), customerAfter.getY()]))
        return (EnvironmentalcostToCustomerBefore + EnvironmentalcostToCustomerAfter) - EnvironmentalcostBetweenBeforeAfter
        
    """ def insertionCost(self, routes, customerID, routeLoc, index):
        augmentedRoute = routes[routeLoc]
        
        routeEarlyViolation, routeLateViolation, routeViolation = self.metrics.calculateTimeWindowViolation(self.depot.getId(), augmentedRoute)
        augmentedRouteObject = Route(augmentedRoute,0,0,0,False, routeEarlyViolation, routeLateViolation, routeViolation)
        self.metrics.evaluateRoute(self.depot.getId(), augmentedRouteObject)
        
        if len(augmentedRoute)==1:
            
            return self.metrics.evaluateEnvironmentalCostRoute(augmentedRouteObject, self.depot)
        
        
        oldRoute = augmentedRoute[:index]+augmentedRoute[index+1:]
        
        routeEarlyViolation, routeLateViolation, routeViolation = self.metrics.calculateTimeWindowViolation(self.depot.getId(), augmentedRoute)
        augmentedRouteObject = Route(augmentedRoute,0,0,0,False, routeEarlyViolation, routeLateViolation, routeViolation)
        self.metrics.evaluateRoute(self.depot.getId(), augmentedRouteObject)
        
        routeEarlyViolation_old, routeLateViolation_old, routeViolation_old = self.metrics.calculateTimeWindowViolation(self.depot.getId(), oldRoute)
        oldRouteObject = Route(oldRoute,0,0,0,False, routeEarlyViolation_old, routeLateViolation_old, routeViolation_old)
        self.metrics.evaluateRoute(self.depot.getId(), oldRouteObject)
        
        #print(self.metrics.evaluateEnvironmentalCostRoute(augmentedRouteObject, self.depot) 
            #- self.metrics.evaluateEnvironmentalCostRoute(oldRouteObject, self.depot))
        return (self.metrics.evaluateEnvironmentalCostRoute(augmentedRouteObject, self.depot) 
            - self.metrics.evaluateEnvironmentalCostRoute(oldRouteObject, self.depot))
        """
       
         
        
    def getDepot(self):
        return self.depot

    def getCost(self):
        return self.cost

    def getFeasibility(self):
        return self.isFeasible

    def getResult(self):
        return self.result

    def __str__(self):
        return str(self.cost)

    def __lt__(self, other):
        return self.cost < other.getCost()
    
    def __eq__(self, other):
        # equality (==) comparison
        return self.cost == other.get_cost()
    
    def __gt__(self, other):
        # greater than (>) comparison
        return self.cost > other.get_cost()
    
    def __le__(self, other):
        # less than or equal to (<=) comparison
        return self.cost <= other.get_cost()

    def __ge__(self, other):
        # greater than or equal to (>=) comparison
        return self.cost >= other.get_cost()
    
    @staticmethod
    def findBest(insertions):
        leader = insertions[0]
        for insertion in insertions:
            if insertion.getCost() < leader.getCost():
                leader = insertion
        return leader

class Inserter:
    def __init__(self, manager, metrics):
        self.manager = manager
        self.metrics = metrics

    def insertCustomerID(self, depot, individual, customerID, balanceParameter):
        feasibleInsertions = []  # Store all possible feasible solutions
        unFeasibleInsertions = []  # Store all possible non-feasible solutions
        routes = individual.getChromosome()[depot.getId()]
        numberOfRoutes = 0

        # Try all possible insertions in existing routes in depot
        for routeLoc in range(len(routes)):
            for index in range(len(routes[routeLoc].getRoute()) + 1):
                routesCopy = Inserter.copyDepotRoutes(routes)
                routesCopy[routeLoc].insert(index, customerID)
                insertion = Insertion(self.manager, self.metrics, depot, routesCopy, customerID, routeLoc, index)

                if insertion.getFeasibility():
                    feasibleInsertions.append(insertion)
                else:
                    unFeasibleInsertions.append(insertion)
            numberOfRoutes += 1

        # If depot can add a route and still satisfy constraints, suggest this as well
        if numberOfRoutes < depot.getMaxVehicles():
            routesCopy = Inserter.copyDepotRoutes(routes)
            newRoute = [customerID]
            routesCopy.append(newRoute)

            insertion = Insertion(self.manager, self.metrics, depot, routesCopy, customerID, len(routesCopy) - 1, 0)
            if insertion.getFeasibility():
                feasibleInsertions.append(insertion)
            else:
                unFeasibleInsertions.append(insertion)

        # Find best feasible insertion
        if random.random() < balanceParameter and len(feasibleInsertions) > 0:
            chosenInsertion = Insertion.findBest(feasibleInsertions)
        else:
            allInsertions = feasibleInsertions + unFeasibleInsertions
            chosenInsertion = Insertion.findBest(allInsertions)  # Else, take best infeasible

        result = self.getResult(chosenInsertion)

        individual.getChromosome()[chosenInsertion.getDepot().getId()] = result  # Apply insertion

    def getResult(self, insertion):
        result = []
        for route in insertion.getResult():
            
            resultingRoute = Route(route,0,0,0,False,0,0,0)
            self.metrics.evaluateRoute(insertion.getDepot().getId(), resultingRoute)
            EarlyTimeViolation, LateTimeViolation, TimeViolation = self.metrics.calculateTimeWindowViolation(insertion.getDepot().getId(), route)
            routeDuration = self.metrics.CalculateRouteDuration(insertion.getDepot().getId(), route)
            resultingRoute.setRouteEarlyViolation(EarlyTimeViolation)
            resultingRoute.setRouteLateViolation(LateTimeViolation)
            resultingRoute.setRouteViolation(TimeViolation)
            resultingRoute.setRouteDuration(routeDuration)
            result.append(resultingRoute)
        return result

    @staticmethod
    def copyDepotRoutes(routes):
        copy = []
        for route in routes:
            routeCopy = list(route.getRoute())
            copy.append(routeCopy)
        return copy
        
class Metrics:
    def __init__(self, manager):
        self.manager = manager

    def getTotalDistance(self, individual):
        totalDistance = 0
        for key, chromosomeDepot in individual.getChromosome().items():
            for route in chromosomeDepot:
                totalDistance += route.getDistance()  # get distance
        return totalDistance
    
    def getTotalDuration(self, individual):
        totalDuration = 0
        for key, chromosomeDepot in individual.getChromosome().items():
            for route in chromosomeDepot:
                totalDuration += route.getRouteDuration()  # get distance
        return totalDuration
    
    def getEarlyTimeViolation(self, individual):
        EarlyTimeViolation = 0
        for key, chromosomeDepot in individual.getChromosome().items():
            for route in chromosomeDepot:
                EarlyTimeViolation += route.getRouteEarlyViolation()  # get distance
        return EarlyTimeViolation
        
    def getLateTimeViolation(self, individual):
        LateTimeViolation = 0
        for key, chromosomeDepot in individual.getChromosome().items():
            for route in chromosomeDepot:
                LateTimeViolation += route.getRouteLateViolation()  # get distance
        return LateTimeViolation
        
    def getTotalTimeViolation(self, individual):
        TotalTimeViolation = 0
        for key, chromosomeDepot in individual.getChromosome().items():
            for route in chromosomeDepot:
                TotalTimeViolation += route.getRouteViolation()  # get distance
        return TotalTimeViolation    
    
    def getWorkload(self, individual):
        workloads = []  # Store workloads of each route
        num_tech = 0
        for key, chromosomeDepot in individual.getChromosome().items():
            for route in chromosomeDepot:
                workload = route.getDemand()
                workloads.append(workload)
                num_tech += 1
    
            if not workloads:
                return 0  # Avoid errors if there are no routes
        mean_workload = sum(workloads)/num_tech
        sum_squared_diff = sum((workload - mean_workload) ** 2 for workload in workloads)
        return sqrt(sum_squared_diff / num_tech)
    
    
    def getNumTech(self, individual):
        num_tech = 0
        for key, chromosomeDepot in individual.getChromosome().items():
            for route in chromosomeDepot:
                num_tech += 1  # get distance
        return num_tech

    def evaluateRoute(self, depotID, route):
        """
        Iterates through a route and calculates total demand and distance.
        Assigns these values to the route object.
        """
        if len(route.getRoute()) == 0:
            return

        totalDistance = 0
        totalDemand = 0

        depot = self.manager.getDepot(depotID)
        depotCoordinates = [depot.getX(), depot.getY()]

        # Add distance from depot to first customer
        toCustomer = self.manager.getCustomer(route.getRoute()[0])
        toCustomerCoordinates = [toCustomer.getX(), toCustomer.getY()]
        totalDistance += Euclidian.distance(depotCoordinates, toCustomerCoordinates)

        # Add distances between customers
        fromCustomer = toCustomer
        for i in range(1, len(route.getRoute())):
            totalDemand += fromCustomer.getDemand()

            fromCustomerCoordinates = [fromCustomer.getX(), fromCustomer.getY()]

            toCustomer = self.manager.getCustomer(route.getRoute()[i])
            toCustomerCoordinates = [toCustomer.getX(), toCustomer.getY()]

            totalDistance += Euclidian.distance(fromCustomerCoordinates, toCustomerCoordinates)
            fromCustomer = toCustomer

        totalDemand += fromCustomer.getDemand()

        # Add distance from last customer and back to depot
        totalDistance += Euclidian.distance(toCustomerCoordinates, depotCoordinates)

        # Assign values to route
        route.setDistance(totalDistance)
        route.setDemand(totalDemand)
        

    def getRouteDistance(self, depotID, route):
        if len(route) == 0:
            return 0

        totalDistance = 0

        depot = self.manager.getDepot(depotID)
        depotCoordinates = [depot.getX(), depot.getY()]

        # Add distance from depot to first customer
        toCustomer = self.manager.getCustomer(route[0])
        toCustomerCoordinates = [toCustomer.getX(), toCustomer.getY()]
        totalDistance += Euclidian.distance(depotCoordinates, toCustomerCoordinates)

        # Add distances between customers
        fromCustomer = toCustomer
        for i in range(1, len(route)):
            fromCustomerCoordinates = [fromCustomer.getX(), fromCustomer.getY()]

            toCustomer = self.manager.getCustomer(route[i])
            toCustomerCoordinates = [toCustomer.getX(), toCustomer.getY()]

            totalDistance += Euclidian.distance(fromCustomerCoordinates, toCustomerCoordinates)

            fromCustomer = toCustomer

        # Add distance from last customer and back to depot
        totalDistance += Euclidian.distance(toCustomerCoordinates, depotCoordinates)

        return totalDistance

    def isIndividualFeasible(self, individual):
        depots = self.manager.getDepots()
        for key, routes in individual.getChromosome().items():
            depot = next((d for d in depots if d.getId() == key), None)
            if depot is None or not self.areRoutesFeasible(depot, routes):
                return False
        return True

    def areRoutesFeasible(self, depot, routes):
        """
        Checks if the routes belonging to a particular depot are feasible.
        Assumes all routes have updated demands and distances.
        """
        numberOfVehiclesInUse = 0
        for route in routes:
            demand = route.getDemand()
            maxDuration = depot.getMaxDuration()

            duration = maxDuration if maxDuration == 0 else route.getDistance()
            if demand > depot.getMaxVehicleLoad():
                return False
            numberOfVehiclesInUse += 1

        return numberOfVehiclesInUse <= depot.getMaxVehicles()

    def checkRoutes(self, depot, route):
        """
        Checks if a given route within a depot is feasible.
        Iterates through customers to calculate values.
        """
        demand = 0
        for customerID in route:
            customer = self.manager.getCustomer(customerID)
            demand += customer.getDemand()

        maxDuration = depot.getMaxDuration()
        duration = maxDuration if maxDuration == 0 else self.getRouteDistance(depot.getId(), route)
        return duration <= maxDuration and demand <= depot.getMaxVehicleLoad()

    def getRouteDemand(self, route):
        totalDemand = 0
        for customerId in route:
            totalDemand += self.manager.getCustomer(customerId).getDemand()
        return totalDemand
    
    
    def getCosts(self, individual): 
        
        totalDistance = 0
        totalDuration = 0
        tech_hourly_wage = 20
        early_arrival_duration = 0
        late_arrival_duration = 0
        for key, chromosomeDepot in individual.getChromosome().items():
            for route in chromosomeDepot:
                totalDuration += route.getRouteDuration()  # get duration
                totalDistance += route.getDistance()   # get distance
                early_arrival_duration += route.getRouteEarlyViolation()
                late_arrival_duration += route.getRouteLateViolation()
        
        labor_cost = totalDuration * tech_hourly_wage
        
        # Fuel-to-air mass ratio 
        ksi = 1

        # Heating value for diesel fuel (kJ/g) 
        k = 45e6

        # Conversion factor from grams to liters
        psi = 0.737

        # Engine friction factor (kJ/rev/liter)
        F = 0.2

        # Engine speed (rev/s)
        N = 34

        # Engine displacement (liters) 
        V = 7

        # Vehicle drive train efficiency 
        epsilon = 0.4

        # Efficiency parameter for diesel engines
        omega = 0.9

        # Gravity (m/s2)
        g = 9.81

        # Rolling resistance coefficient
        C_r = 0.009

        # Road angle
        phi = math.radians(0)

        # Aerodynamic drag coefficient
        C_d = 0.55

        # Frontal surface area of a vehicle (m2)
        A = 7.6

        # Air density (kg/m3)
        ro = 1.2041

        # Total travel distance 
        total_distance = totalDistance*1000

        # Vehicle speed 
        speed = 10

        # Curb weight of vehicle 
        curb_weight = 4000

        alpha = (ksi * F * N * V * total_distance) / (k * psi)

        beta = (ksi * g * math.sin(phi) + g * C_r * math.cos(phi)* total_distance) /  (epsilon * omega * k * psi)

        gamma = (0.5 * C_d * A * ro * total_distance) / (epsilon * omega * k * psi)

        # Fuel consumption formula
        fuel_consumption = alpha * (1 / speed) + beta * curb_weight + gamma * speed ** 2
      
              
        # Fuel consumption formula
        fuel_consumption = alpha * (1 / speed) + beta * curb_weight + gamma * speed ** 2
        
        
        
        # CO2 factor
        CO2_factor = 2.64
        gas_emissions = fuel_consumption * CO2_factor
        
        fuel_cost_liter = 1.6
        fuel_cost = fuel_consumption * fuel_cost_liter
        
        carbon_tax = 0.05 
        emission_cost = gas_emissions * carbon_tax
        
        vehicle_usage_unit = 20
        vehicle_usage_cost = vehicle_usage_unit * totalDuration
        
        early_arrival_cost_unit = 7
        early_arrival_cost = early_arrival_cost_unit * early_arrival_duration
        
        late_arrival_cost_unit = 25
        late_arrival_cost = late_arrival_cost_unit * late_arrival_duration
        
        
        economic_cost = labor_cost + fuel_cost + emission_cost + vehicle_usage_cost + late_arrival_cost + early_arrival_cost 
        
        # CO2 factor
        CO2_factor = 2.64
        
        gas_emissions = fuel_consumption * CO2_factor
        
        
        environmental_cost = gas_emissions 
        
        """
        Returns the Gini coefficient (0..1) of workloads across routes.
        0 => perfectly equal workloads, 1 => totally unequal.
        """
        workloads = []
        for _, chromosomeDepot in individual.getChromosome().items():
            for route in chromosomeDepot:
                workloads.append(float(route.getDemand()))

        if not workloads:
            return 0.0

        # If all workloads are zero, inequality is zero
        total = sum(workloads)
        if total == 0:
            return 0.0

        # Sort workloads (non-decreasing)
        workloads.sort()
        n = len(workloads)

        # Compute Gini using the "rank" formula:
        # G = (2 * sum(i * x_i) / (n * sum(x))) - (n + 1) / n
        # where i is 1-based index after sorting
        cumulative = 0.0
        for i, x in enumerate(workloads, start=1):
            cumulative += i * x

        gini = (2.0 * cumulative) / (n * total) - (n + 1.0) / n
        # Numerical safety: clamp to [0,1]
        gini = max(0.0, min(1.0, gini))
        
        social_cost = gini
        return economic_cost, environmental_cost, social_cost
    
    
    
    def evaluateEnvironmentalCost(self, individual):
        
        totalDistance = 0
       
        for chromosomeDepot in individual.getChromosome().values():
            for route in chromosomeDepot:
          
                totalDistance += route.getDistance()   # get distance
                
                   
        # Fuel-to-air mass ratio 
        ksi = 1

        # Heating value for diesel fuel (J/kg) 
        k = 44e6

        # Conversion factor (gram/second to liter/second)
        psi = 0.832

        # Engine friction factor (kJ/rev/liter) yes
        F = 15

        # Engine speed (rev/s)
        N = 34

        # Engine displacement (liters)
        V = 10

        # Vehicle drive train efficiency yes
        epsilon = 0.45

        # Efficiency parameter for diesel engines yes
        omega = 0.4

        # Gravity (m/s2) yes
        g = 9.81

        # Rolling resistance coefficient yes
        C_r = 0.01

        # Road angle yes
        phi = math.radians(0)

        # Aerodynamic drag coefficient yes
        C_d = 0.7

        # Frontal surface area of a vehicle (m2)
        A = 8.5

        # Air density (kg/m3)
        ro = 1.2041

        # Total travel distance (m)
        total_distance = totalDistance * 1000

        # Vehicle speed (m/s2)
        speed = 10

        # Curb weight of vehicle (kg) yes
        curb_weight = 5000
        
        payload = 100
        
        # First term
        # Lambda: scaling factor
        lamda = ksi / k

        # Term 1: Engine friction
        term1 = (lamda * F * N * V * total_distance) / speed
        
        M = curb_weight + payload
        # Term 2: Rolling resistance + slope (curb weight + payload)
        gamma = 1 / (epsilon * omega)
        term2 = lamda * gamma * M * g * C_r * total_distance 

        # Term 3: Aerodynamic drag
        term3 = lamda * gamma * 0.5 * C_d * A * ro * total_distance * speed ** 2
       

        # Fuel consumption formula
        fuel_consumption_liters = (term1 + term2 + term3) / psi
        
        
        # CO2 factor
        CO2_factor = 2.64
        
        gas_emissions = fuel_consumption_liters * CO2_factor
        
        environmental_cost = gas_emissions
        
       
        
        return environmental_cost
    
    def evaluateEconomicCost(self, individual):
        
        totalDistance = 0
        totalDuration = 0
        tech_hourly_wage = 20
        early_arrival_duration = 0
        late_arrival_duration = 0
        for key, chromosomeDepot in individual.getChromosome().items():
            for route in chromosomeDepot:
                totalDuration += route.getRouteDuration()  # get duration
                totalDistance += route.getDistance()   # get distance
                early_arrival_duration += route.getRouteEarlyViolation()
                late_arrival_duration += route.getRouteLateViolation()
        
        labor_cost = totalDuration * tech_hourly_wage
        
        # Fuel-to-air mass ratio 
        ksi = 1

        # Heating value for diesel fuel (J/kg) 
        k = 44e6

        # Conversion factor (gram/second to liter/second)
        psi = 0.832

        # Engine friction factor (kJ/rev/liter) yes
        F = 15

        # Engine speed (rev/s)
        N = 34

        # Engine displacement (liters)
        V = 10

        # Vehicle drive train efficiency yes
        epsilon = 0.45

        # Efficiency parameter for diesel engines yes
        omega = 0.4

        # Gravity (m/s2) yes
        g = 9.81

        # Rolling resistance coefficient yes
        C_r = 0.01

        # Road angle yes
        phi = math.radians(0)

        # Aerodynamic drag coefficient yes
        C_d = 0.7

        # Frontal surface area of a vehicle (m2)
        A = 8.5

        # Air density (kg/m3)
        ro = 1.2041

        # Total travel distance (m)
        total_distance = totalDistance * 1000

        # Vehicle speed (m/s2)
        speed = 10

        # Curb weight of vehicle (kg) yes
        curb_weight = 5000
        
        payload = 100
        
        # First term
        # Lambda: scaling factor
        lamda = ksi / k

        # Term 1: Engine friction
        term1 = (lamda * F * N * V * total_distance) / speed
        
        M = curb_weight + payload
        # Term 2: Rolling resistance + slope (curb weight + payload)
        gamma = 1 / (epsilon * omega)
        term2 = lamda * gamma * M * g * C_r * total_distance 

        # Term 3: Aerodynamic drag
        term3 = lamda * gamma * 0.5 * C_d * A * ro * total_distance * speed ** 2
       

        # Fuel consumption formula
        fuel_consumption_liters = (term1 + term2 + term3) / psi
        
        fuel_cost_liter = 1.6
        fuel_cost = fuel_consumption_liters * fuel_cost_liter
        
        # CO2 factor
        CO2_factor = 2.64
        gas_emissions = fuel_consumption_liters * CO2_factor
        
        carbon_tax = 0.05 
        emission_cost = gas_emissions * carbon_tax
        
        vehicle_usage_unit = 20
        vehicle_usage_cost = vehicle_usage_unit * totalDuration
        
        early_arrival_cost_unit = 7
        early_arrival_cost = early_arrival_cost_unit * early_arrival_duration
        
        late_arrival_cost_unit = 25
        late_arrival_cost = late_arrival_cost_unit * late_arrival_duration
        
        
        economic_cost = labor_cost + fuel_cost + emission_cost + vehicle_usage_cost + late_arrival_cost + early_arrival_cost 
        
        return economic_cost
    
    def getNumSatisfiedCustomers(self, individual):
        num_satisfied_customers = 0
        speed = 1
        for key, chromosomeDepot in individual.getChromosome().items():
            for route in chromosomeDepot:
                depot = self.manager.getDepot(key)
                depotCoordinates = [depot.getX(), depot.getY()]
                currentTime = 0  # Assume departure from depot at time 0
                route = route.getRoute()
                # First customer
                toCustomer = self.manager.getCustomer(route[0])
                toCustomerCoordinates = [toCustomer.getX(), toCustomer.getY()]
                travelTime = (Euclidian.distance(depotCoordinates, toCustomerCoordinates)) / speed
                arrivalTime = currentTime + travelTime
        
                if arrivalTime > toCustomer.getDueTime():
                    num_satisfied_customers += 1  
        
                # Adjust service start time: must wait if early
                startServiceTime = max(arrivalTime, toCustomer.getReadyTime())
        
                departureTime = startServiceTime + toCustomer.getServiceTime()
        
                currentTime = departureTime
        
                # Remaining customers
                for i in range(1, len(route)):
                    fromCustomer = self.manager.getCustomer(route[i - 1])
                    toCustomer = self.manager.getCustomer(route[i])
                    travelTime = (Euclidian.distance(
                    [fromCustomer.getX(), fromCustomer.getY()],
                    [toCustomer.getX(), toCustomer.getY()]
                    )) / speed
                    arrivalTime = currentTime + travelTime
            
                    if arrivalTime > toCustomer.getDueTime():
                        num_satisfied_customers += 1  
            
                    # Wait if early
                    startServiceTime = max(arrivalTime, toCustomer.getReadyTime())
                    departureTime = startServiceTime + toCustomer.getServiceTime()
          
                    currentTime = departureTime
            
        return num_satisfied_customers
    
    
    
    def evaluateSocialCost(self, individual):
        """
        Returns the Gini coefficient (0..1) of workloads across routes.
        0 => perfectly equal workloads, 1 => totally unequal.
        """
        num_satisfied_customers = 0
        workloads = []
        for _ , chromosomeDepot in individual.getChromosome().items():
            for route in chromosomeDepot:
                workloads.append(float(route.getDemand()))
            

        if not workloads:
            return 0.0

        # If all workloads are zero, inequality is zero
        total = sum(workloads)
        if total == 0:
            return 0.0

        # Sort workloads (non-decreasing)
        workloads.sort()
        n = len(workloads)

        # Compute Gini using the "rank" formula:
        # G = (2 * sum(i * x_i) / (n * sum(x))) - (n + 1) / n
        # where i is 1-based index after sorting
        cumulative = 0.0
        for i, x in enumerate(workloads, start=1):
            cumulative += i * x

        gini = (2.0 * cumulative) / (n * total) - (n + 1.0) / n
        
        # Numerical safety: clamp to [0,1]
        gini = max(0.0, min(1.0, gini))
        
        num_satisfied_customers = self.getNumSatisfiedCustomers(individual)
        customers = self.manager.getCustomers()
        customer_satisfaction = num_satisfied_customers / len(customers)
        
        social_cost = (gini + (1 - customer_satisfaction))/2 
        
 
        return social_cost
    
    
    
    
    def evaluateEnvironmentalCostRoute(self, route, depot):
        
        
        totalDistance = route.getDistance()   # get route distance
      
        
        # Fuel-to-air mass ratio 
        ksi = 1

        # Heating value for diesel fuel (J/kg) 
        k = 44e6

        # Conversion factor (gram/second to liter/second)
        psi = 0.832

        # Engine friction factor (kJ/rev/liter) yes
        F = 15

        # Engine speed (rev/s)
        N = 34

        # Engine displacement (liters)
        V = 10

        # Vehicle drive train efficiency yes
        epsilon = 0.45

        # Efficiency parameter for diesel engines yes
        omega = 0.4

        # Gravity (m/s2) yes
        g = 9.81

        # Rolling resistance coefficient yes
        C_r = 0.01

        # Road angle yes
        phi = math.radians(0)

        # Aerodynamic drag coefficient yes
        C_d = 0.7

        # Frontal surface area of a vehicle (m2)
        A = 8.5

        # Air density (kg/m3)
        ro = 1.2041

        # Total travel distance (m)
        total_distance = totalDistance * 1000

        # Vehicle speed (m/s2)
        speed = 10

        # Curb weight of vehicle (kg) yes
        curb_weight = 5000
        
        payload = 100
        
        # First term
        # Lambda: scaling factor
        lamda = ksi / k

        # Term 1: Engine friction
        term1 = (lamda * F * N * V * total_distance) / speed
        
        M = curb_weight + payload
        # Term 2: Rolling resistance + slope (curb weight + payload)
        gamma = 1 / (epsilon * omega)
        term2 = lamda * gamma * M * g * C_r * total_distance 

        # Term 3: Aerodynamic drag
        term3 = lamda * gamma * 0.5 * C_d * A * ro * total_distance * speed ** 2
       

        # Fuel consumption formula
        fuel_consumption_liters = (term1 + term2 + term3) / psi
        
        # Conversion from m3 to liters
        
        
        # CO2 factor
        CO2_factor = 2.64
        
        gas_emissions = fuel_consumption_liters * CO2_factor
        
        environmental_cost = gas_emissions
        
        return environmental_cost
    
    """   def EnvironmentalCostBetween(self, fromCustomer, toCustomer, currentTime):
        tech_hourly_wage = 20
        # Travel
        distance = Euclidian.distance([fromCustomer.getX(), fromCustomer.getY()],
                                  [toCustomer.getX(), toCustomer.getY()])
        speed = 50/3.6
        travel_time = distance / speed

        arrival_time = currentTime + travel_time
        wait_time = max(0, toCustomer.getReadyTime() - arrival_time)
        early_arrival_duration = wait_time
        late_arrival_duration = max(0, arrival_time - toCustomer.getDueTime())
        start_service_time = arrival_time + wait_time
        departure_time = start_service_time + toCustomer.getServiceTime()

        duration = departure_time - currentTime  # time spent from leaving fromCustomer to finishing service

        # Labor cost
        labor_cost = duration * tech_hourly_wage  # tech_hourly_wage

           
        # Fuel-to-air mass ratio 
        ksi = 1

        # Heating value for diesel fuel (J/kg) 
        k = 44

        # Conversion factor (gram/second to liter/second)
        psi = 737

        # Engine friction factor (kJ/rev/liter) yes
        F = 0.2

        # Engine speed (rev/s)
        N = 33

        # Engine displacement (liters)
        V = 5

        # Vehicle drive train efficiency yes
        epsilon = 0.4

        # Efficiency parameter for diesel engines yes
        omega = 0.9

        # Gravity (m/s2) yes
        g = 9.81

        # Rolling resistance coefficient yes
        C_r = 0.01

        # Road angle yes
        phi = math.radians(0)

        # Aerodynamic drag coefficient yes
        C_d = 0.7

        # Frontal surface area of a vehicle (m2)
        A = 3.912

        # Air density (kg/m3)
        ro = 1.2041

        # Total travel distance (m)
        total_distance = totalDistance * 1000

        # Vehicle speed (m/s2)
        speed = 15

        # Curb weight of vehicle (kg) yes
        curb_weight = 4000
        f = 100
        # First term
        lamda = ksi / (k * psi)
        term1 = (lamda * F * N * V) / speed 
        
        gamma =  1 / (1000 * omega * epsilon)
        # Second term
        term2 = lamda * curb_weight * gamma * g * C_r
        
        # Third term 
        term3 = lamda * gamma * g * f * C_r
        
        # Fourth term
        beta = 0.5 * C_d * ro * A
        term4 = lamda * beta * gamma * speed ** 2
        
        alpha = (ksi * F * N * V ) / (k * psi)

        beta = (ksi * g * math.sin(phi) + g * C_r * math.cos(phi)) /  (epsilon * omega * k * psi)

        gamma = (0.5 * C_d * A * ro ) / (epsilon * omega * k * psi)

        # Fuel consumption formula
        fuel_consumption = (term1 + term2 + term3 + term4) * total_distance
        # CO2 factor
        CO2 = 2640
        gas_emissions = (CO2 * fuel_consumption) / 1000
        
        fuel_cost_liter = 1.6
        fuel_cost = fuel_consumption * fuel_cost_liter
        
        carbon_tax = 0.05 
        emission_cost = gas_emissions * carbon_tax
        
        vehicle_usage_unit = 20
        vehicle_usage_cost = vehicle_usage_unit * duration
        
        early_arrival_cost_unit = 7
        early_arrival_cost = early_arrival_cost_unit * early_arrival_duration
        
        late_arrival_cost_unit = 25
        late_arrival_cost = late_arrival_cost_unit * late_arrival_duration
        
        
        economic_cost = labor_cost + fuel_cost + emission_cost + vehicle_usage_cost + late_arrival_cost + early_arrival_cost 
        
        return economic_cost, departure_time
     """
    
    def CalculateRouteDuration(self, depotID, route):   
        
        speed = 10
        if len(route) == 0:
           return 0
        totalTime = 0
        depot = self.manager.getDepot(depotID)
        depotCoordinates = [depot.getX(), depot.getY()]
        currentTime = 0 # or 0 if depot has no time window
        
        # Add duration from depot to first customer
        toCustomer = self.manager.getCustomer(route[0])
        toCustomerCoordinates = [toCustomer.getX(), toCustomer.getY()]
        travelTime = (Euclidian.distance(depotCoordinates,toCustomerCoordinates)) / speed
        arrivalTime = currentTime + travelTime
        waitTime = max(0, toCustomer.getReadyTime() - arrivalTime)
        startServiceTime = arrivalTime + waitTime
        departureTime = startServiceTime + toCustomer.getServiceTime()
        totalTime = departureTime
        prevCustomer = toCustomer
        
        # Visit rest of the route
        for i in range(1, len(route)):
            customer = self.manager.getCustomer(route[i])
            travelTime = (Euclidian.distance([prevCustomer.getX(), prevCustomer.getY()],[customer.getX(), customer.getY()])) / speed
            arrivalTime = totalTime + travelTime
            waitTime = max(0, customer.getReadyTime() - arrivalTime)
            startServiceTime = arrivalTime + waitTime
            departureTime = startServiceTime + customer.getServiceTime()
            totalTime = departureTime
            
            prevCustomer = customer
        
        # Return to depot
        travelTimeBack = (Euclidian.distance(
        [prevCustomer.getX(), prevCustomer.getY()],
        depotCoordinates
        )) / speed
        
        totalTime += travelTimeBack
       
        return totalTime
    
    def checkTimeFeasibility(self, depotID, route):
        """
        Check if the route satisfies all customer time windows.
        """
        if len(route) == 0:
           return True  # Empty route is trivially feasible

        depot = self.manager.getDepot(depotID)
        depotCoordinates = [depot.getX(), depot.getY()]
        currentTime = 0  # Assume departure from depot at time 0

        # First customer
        toCustomer = self.manager.getCustomer(route[0])
        toCustomerCoordinates = [toCustomer.getX(), toCustomer.getY()]
        travelTime = Euclidian.distance(depotCoordinates, toCustomerCoordinates)
        arrivalTime = currentTime + travelTime

        if arrivalTime > toCustomer.getDueTime():
           print("time window violated")
           return False  # Violation of time window

        waitTime = max(0, toCustomer.getReadyTime() - arrivalTime)
        serviceStart = arrivalTime + waitTime
        departureTime = serviceStart + toCustomer.getServiceTime()
        currentTime = departureTime

        # Remaining customers
        for i in range(1, len(route)):
            fromCustomer = self.manager.getCustomer(route[i - 1])
            toCustomer = self.manager.getCustomer(route[i])
            travelTime = Euclidian.distance(
            [fromCustomer.getX(), fromCustomer.getY()],
            [toCustomer.getX(), toCustomer.getY()]
            )
            arrivalTime = currentTime + travelTime

            if arrivalTime > toCustomer.getDueTime():
                
                return False

            waitTime = max(0, toCustomer.getReadyTime() - arrivalTime)
            serviceStart = arrivalTime + waitTime
            departureTime = serviceStart + toCustomer.getServiceTime()
            currentTime = departureTime
            
        # Return to depot (optional: if depot has time constraints)
        return True
    
    def checkTimeWindow(self, depotID, route):
        depot = self.manager.getDepot(depotID)
        depotCoordinates = [depot.getX(), depot.getY()]
        currentTime = 0  # Assume departure from depot at time 0

        # First customer
        toCustomer = self.manager.getCustomer(route[0])
        toCustomerCoordinates = [toCustomer.getX(), toCustomer.getY()]
        travelTime = Euclidian.distance(depotCoordinates, toCustomerCoordinates)
        arrivalTime = currentTime + travelTime

        if arrivalTime > toCustomer.getDueTime():
           return False  # Violation of time window
       
        waitTime = max(0, toCustomer.getReadyTime() - arrivalTime)
        departureTime = arrivalTime + waitTime + toCustomer.getServiceTime()
        currentTime = departureTime
        
        # Remaining customers
        for i in range(1, len(route)):
            fromCustomer = self.manager.getCustomer(route[i - 1])
            toCustomer = self.manager.getCustomer(route[i])
            travelTime = Euclidian.distance(
            [fromCustomer.getX(), fromCustomer.getY()],
            [toCustomer.getX(), toCustomer.getY()]
            )
            arrivalTime = currentTime + travelTime

            if arrivalTime > toCustomer.getDueTime():
                
                return False
            
            departureTime = arrivalTime + toCustomer.getServiceTime()
            currentTime = departureTime
            
            
        # Return to depot (optional: if depot has time constraints)
        return True
    
    def calculateTimeWindowViolation(self, depotID, route):
        speed = 10
        depot = self.manager.getDepot(depotID)
        depotCoordinates = [depot.getX(), depot.getY()]
        currentTime = 0  # Assume departure from depot at time 0
        TimeViolation = 0
        LateTimeViolation = 0
        EarlyTimeViolation = 0
        # First customer
        toCustomer = self.manager.getCustomer(route[0])
        toCustomerCoordinates = [toCustomer.getX(), toCustomer.getY()]
        travelTime = (Euclidian.distance(depotCoordinates, toCustomerCoordinates)) / speed
        arrivalTime = currentTime + travelTime

       
           
        LateTimeViolation += max(0,arrivalTime - toCustomer.getDueTime())
        
      
        EarlyTimeViolation += max(0,toCustomer.getReadyTime() - arrivalTime)  
        
        # Adjust service start time: must wait if early
        startServiceTime = max(arrivalTime, toCustomer.getReadyTime())
        
        departureTime = startServiceTime + toCustomer.getServiceTime()
        currentTime = departureTime
        
        # Remaining customers
        for i in range(1, len(route)):
            fromCustomer = self.manager.getCustomer(route[i - 1])
            toCustomer = self.manager.getCustomer(route[i])
            travelTime = (Euclidian.distance(
            [fromCustomer.getX(), fromCustomer.getY()],
            [toCustomer.getX(), toCustomer.getY()]
            )) / speed
            arrivalTime = currentTime + travelTime

            
           
            LateTimeViolation += max(0,arrivalTime - toCustomer.getDueTime()) 
            
          
            EarlyTimeViolation += max(0,toCustomer.getReadyTime() - arrivalTime)      
            
            # Wait if early
            startServiceTime = max(arrivalTime, toCustomer.getReadyTime())
            departureTime = startServiceTime + toCustomer.getServiceTime()
          
            currentTime = departureTime
            
        TimeViolation = EarlyTimeViolation + LateTimeViolation  # Violation of time window    
        
        # Return to depot (optional: if depot has time constraints)
        return EarlyTimeViolation, LateTimeViolation, TimeViolation
    
    
class Selection:
    @staticmethod
    def selectCompetitorPair(population):
        """
        Selects two random individuals from the population.
        """
        p1 = random.choice(population)
        p2 = random.choice(population)
        return [p1, p2]

    @staticmethod
    def runTournamentSelection(player1, player2, fitness_bias):
        """
        Runs tournament selection between two candidates.
        Fitness bias controls how often the tournament is played out. If it is, the best individual wins.
        If not, a random individual is chosen.
        """
        chosen_individual = player1
        if random.random() < fitness_bias:
            if player2.getEnvironmentalCost() > player1.getEnvironmentalCost():
                chosen_individual = player2
        else:
            if random.random() < 0.5:
                chosen_individual = player2
        
        return chosen_individual

class Crossover:

    def __init__(self, manager, inserter, crossover_rate, balance_parameter):
        self.crossover_rate = crossover_rate
        self.balance_parameter = balance_parameter
        self.manager = manager
        self.inserter = inserter

    def set_crossover_rate(self, crossover_rate):
        print(f"Lowering crossover rate to {crossover_rate}")
        self.crossover_rate = crossover_rate

    def apply(self, p1, p2):
        """
        Recombines two individuals (parents) to produce two new individuals (offspring),
        by applying Best Cost Route Crossover:
        1. Randomly select depot to undergo reproduction
        2. Randomly select a route from each parent
        3. For each selected route, remove all customers in that route from the other parent
        4. For each customer c that was removed from parent p, insert customer c back in parent p at best location.
        """
        if random.random() > self.crossover_rate:
            # By some probability according to the crossover rate parameter, do not apply crossover
            return [p1, p2]

        # Step 1: Randomly select depot to undergo reproduction
        depots = self.manager.getDepots()
        depot_ids = list(p1.getChromosome().keys())
        chosen_depot_id = random.choice(depot_ids)
        chosen_depot = next(d for d in depots if d.getId() == chosen_depot_id)
        
        parent1 = p1.getClone()  # Clone parents to avoid cross reference bugs
        parent2 = p2.getClone()

        # Step 2: Randomly select a route from each parent
        parent1_random_route = parent1.getChromosome()[chosen_depot_id][random.randint(0, len(parent1.getChromosome()[chosen_depot_id]) - 1)].getRoute()
        parent2_random_route = parent2.getChromosome()[chosen_depot_id][random.randint(0, len(parent2.getChromosome()[chosen_depot_id]) - 1)].getRoute()

        # Step 3: For each selected route, remove all customers in that route from the other parent
        self.remove_customer_ids_from_routes(list(parent2.getChromosome().values()), parent1_random_route)
        self.remove_customer_ids_from_routes(list(parent1.getChromosome().values()), parent2_random_route)

        # Step 4: For each customer c that was removed from parent p, insert customer c in parent p at best location.
        for customer_id in parent1_random_route:
            # add all ids somewhere in parent2
            self.inserter.insertCustomerID(chosen_depot, parent2, customer_id, self.balance_parameter)

        for customer_id in parent2_random_route:
            # add all ids somewhere in parent1
            self.inserter.insertCustomerID(chosen_depot, parent1, customer_id, self.balance_parameter)

        return [parent1, parent2]

    def remove_customer_ids_from_routes(self, routes_across_all_depots, ids):
        for routes in routes_across_all_depots:
            for route in routes:
                for customer_id in ids:
                    route.removeCustomer(customer_id)
            routes[:] = [route for route in routes if not route.isEmpty()]  # Remove empty routes

    @staticmethod
    def print_routes(routes):
        s = ""
        for route in routes:
            s += str(route) + " - "
        print(s)
        
class Mutation:
    def __init__(self, manager, metrics, inserter, mutation_rate, inter_depot_freq):
        self.manager = manager
        self.metrics = metrics
        self.inserter = inserter
        self.mutation_rate = mutation_rate
        self.inter_depot_freq = inter_depot_freq
        self.refinement_phase = False

    def set_refinement_phase(self):
        print("Setting mutation operator in refinement state")
        self.refinement_phase = True

    def apply(self, individual, generation):
        if random.random() < self.mutation_rate:  # Apply mutation by probability
            if generation > 100 and generation % self.inter_depot_freq == 0:  # Inter-depot mutation
                self.apply_inter_depot_swapping(individual)
            else:  # Intra-depot mutation
                if self.refinement_phase:
                    self.apply_relocation(individual)
                else:
                    self.apply_reversal(individual)

    def apply_reversal(self, individual):
        """
        Reversal mutation:
        1. Choose random depot
        2. Flatten routes in depot to get one list of customers
        3. Select two random indexes and reverse all customers between them
        """
        depot_ids = list(individual.chromosome.keys())
        chosen_depot_id = random.choice(depot_ids)

        routes = individual.chromosome[chosen_depot_id]
        routes_flattened = [customer for route in routes for customer in route.getRoute()]

        if len(routes_flattened) < 2:
            return

        cut_point1, cut_point2 = random.sample(range(len(routes_flattened)), 2)
        cut_from, cut_to = min(cut_point1, cut_point2), max(cut_point1, cut_point2)

        reversed_segment = routes_flattened[cut_from:cut_to + 1][::-1]
        augmented_routes_flattened = (
            routes_flattened[:cut_from] + reversed_segment + routes_flattened[cut_to + 1:]
        )

        augmented_routes = []
        flat_index = 0
        for route in routes:
            augmented_route = []
            for _ in route.getRoute():
                augmented_route.append(augmented_routes_flattened[flat_index])
                flat_index += 1

            resulting_route = Route(augmented_route,0,0,0,False,0,0,0)
            self.metrics.evaluateRoute(chosen_depot_id, resulting_route)
            EarlyTimeViolation, LateTimeViolation, TimeViolation = self.metrics.calculateTimeWindowViolation(chosen_depot_id, augmented_route)
            routeDuration = self.metrics.CalculateRouteDuration(chosen_depot_id, augmented_route)
            resulting_route.setRouteEarlyViolation(EarlyTimeViolation)
            resulting_route.setRouteLateViolation(LateTimeViolation)
            resulting_route.setRouteViolation(TimeViolation)
            resulting_route.setRouteDuration(routeDuration)
            augmented_routes.append(resulting_route)

        # Apply mutation to chromosome
        individual.chromosome[chosen_depot_id] = augmented_routes

    def apply_relocation(self, individual):
        """
        Relocation mutation:
        1. Select a random customer
        2. Insert customer at the best location in the same depot
        """
        depot_ids = list(individual.chromosome.keys())
        chosen_depot_id = random.choice(depot_ids)
        routes = individual.chromosome[chosen_depot_id]

        candidates = [customer for route in routes for customer in route.get_route()]

        if not candidates:
            return

        chosen_customer_id = random.choice(candidates)

        # Remove customer
        for route in routes:
            if route.remove_customer(chosen_customer_id):
                break

        # Insert customer
        chosen_depot = self.manager.get_depot(chosen_depot_id)
        self.inserter.insert_customer_id(chosen_depot, individual, chosen_customer_id, 1)

    def apply_inter_depot_swapping(self, individual):
        """
        Inter-depot swapping mutation:
        1. Select a random borderline customer
        2. Swap customer to a randomly selected depot (from customer's possible depots)
        3. Insert customer at the best location in its new depot
        """
        candidates = self.manager.getSwappableCustomers()
        if not candidates:
            return

        chosen_customer = random.choice(candidates)
        chosen_customer_id = chosen_customer.id

        # Remove customer from current depot
        was_at_depot_id = None
        for depot_id, routes in individual.chromosome.items():
            for route in routes:
                if route.removeCustomer(chosen_customer_id):
                    was_at_depot_id = depot_id
                    self.metrics.evaluateRoute(depot_id, route)
                    break
            routes[:] = [route for route in routes if not route.isEmpty()]
            if was_at_depot_id is not None:
                break

        possible_depot_ids = list(chosen_customer.getPossibleDepots())
        if was_at_depot_id in possible_depot_ids:
            possible_depot_ids.remove(was_at_depot_id)

        if not possible_depot_ids:
            return

        chosen_next_depot_id = random.choice(possible_depot_ids)
        chosen_depot = self.manager.getDepot(chosen_next_depot_id)
        self.inserter.insertCustomerID(chosen_depot, individual, chosen_customer_id, 1)
        
class Algorithm:
    def __init__(self, manager):
        # ------------------- PARAMS -------------------- #
        self.populationSize = 90  # 80 - 100
        self.numberOfGenerations = 200  # 1200
        self.fitnessGoal = 0
        self.refinementAfter = 700  # 600 - 800
        self.fitnessBias = 0.8
        self.crossoverRateRefinementMode = 0.4
        self.eliteReplacement = 20
        self.manager = manager
        self.metrics = Metrics(manager)
        inserter = Inserter(manager, self.metrics)

        self.crossover = Crossover(manager, inserter, 0.8, 1)
        self.mutation = Mutation(manager, self.metrics, inserter, 0.05, 10)

    def run_no_args(self):
        
        return self.run([])
    
    def run(self, historyShortestDistance):
        

        customers = self.manager.getCustomers()
        crowdedDepots = self.manager.assignCustomersToDepots()

        numCustomers = len(customers)
        numDepots = len(crowdedDepots)

        print(f"Number of customer: {numCustomers}")
        print(f"Number of depots:   {numDepots}")
        print("------------------------")

        for depot in crowdedDepots:
            print(f"Depot {depot.getId()} has {len(depot.getCustomers())} customers")

        # Here, you would implement the algorithm's main functionality,
        # such as running the generations, crossover, mutation, and tracking the fitness.
        # This part is omitted as the core logic was not fully provided.
        # Initialize population
        population = Initializer.init(self.populationSize, crowdedDepots, self.metrics, self.manager)
        print(f"\nInitial population size: {len(population)}")
        
        # Evaluate fitness
        average_distance = self.evaluatePopulation(population)
        
        print_individual(population[0])
        self.evaluateFeasibility(population)
        print(f"Generation: 0  |  Population size: {len(population)}  | Average total cost: {average_distance} | Best individual: {population[0].getEnvironmentalCost()}")

        # Sort population based on fitness (ascending order)
        population.sort(key=lambda individual: individual.getEnvironmentalCost()) 
        

        # Add the fitness of the best individual to the history
        historyShortestDistance.append(population[0].getEnvironmentalCost())

        # Create a list to hold the best parents
        bestParents = []

        # Add the top eliteReplacement individuals to bestParents by cloning them
        for i in range(self.eliteReplacement):
            bestParents.append(population[i].getClone())

        # Evaluate the population of best parents
        self.evaluatePopulation(bestParents)

        # Evaluate the feasibility of the best parents
        self.evaluateFeasibility(bestParents)
        
        
        # For each generation
        for generation in range(1, self.numberOfGenerations + 1):
            offspring = []

            # While offspring size < parents size
            while len(offspring) < self.populationSize:
            # Select 2 competitor pairs
                competitorPair1 = Selection.selectCompetitorPair(population)
                competitorPair2 = Selection.selectCompetitorPair(population)

                # Select parents from tournament selection
                parent1 = Selection.runTournamentSelection(competitorPair1[0], competitorPair1[1], self.fitnessBias)
                parent2 = Selection.runTournamentSelection(competitorPair2[0], competitorPair2[1], self.fitnessBias)

                # Apply crossover
                offspringPair = self.crossover.apply(parent1, parent2)

                # Apply mutation
                for offspringIndividual in offspringPair:
                    self.mutation.apply(offspringIndividual, generation)

                # Add offspring pair to offspring
                offspring.extend(offspringPair)

            averageDistance = self.evaluatePopulation(offspring)
            population = offspring
            self.evaluateFeasibility(population)
            population.sort(key=lambda individual: individual.getEnvironmentalCost()) 

            # Elitism
            population = population[:self.populationSize - self.eliteReplacement]
            population.extend(bestParents)
            population.sort(key=lambda individual: individual.getEnvironmentalCost()) 


            if generation % 10 == 0:
               historyShortestDistance.append(population[0].getEnvironmentalCost())

            if generation % 100 == 0:
               print(f"Generation: {generation}  |  Population size: {len(population)}"
              f"  | Average total cost: {averageDistance}"
              f"  | Best individual cost {population[0].getEnvironmentalCost()}"
              f"  | Best individual is feasible: {population[0].getFeasibility()}"
              f"  | Worst individual cost {population[-1].getEnvironmentalCost()}")

            bestParents = []
            for i in range(self.eliteReplacement):
                bestParents.append(population[i].getClone())
    
            self.evaluatePopulation(bestParents)
            self.evaluateFeasibility(bestParents)
            
            
        # Sort last population
        population.sort(key=lambda individual: individual.getEnvironmentalCost()) 

        fCounter = 0
        for individual in population:
            if individual.getFeasibility():  # Assuming isFeasible() is a method that returns a boolean
                fCounter += 1

        print(f"Number of feasible solutions: {fCounter}")

        # Get result
        #bestIndividual = population[0]
        bestIndividual = population[0]
        for individual in population:
            if individual.isFeasible:  
                bestIndividual = individual
                break
        
        print(f"Result is feasible {bestIndividual.getFeasibility()}")
        
        solutionDepots = self.manager.assignCustomerToDepotsFromIndividual(bestIndividual)
        return solutionDepots,bestIndividual
    
    def evaluatePopulation(self, population):
        totalDistanceForPop = 0
        for individual in population:
            #if individual.getFitness() == 0:
                distance = self.metrics.getTotalDistance(individual)
            
                environmental_cost = self.metrics.evaluateEnvironmentalCost(individual)
                economic_cost = self.metrics.evaluateEconomicCost(individual)
                social_cost = self.metrics.evaluateSocialCost(individual)
                total_duration = self.metrics.getTotalDuration(individual)
                EarlyTimeViolation  = self.metrics.getEarlyTimeViolation(individual)
                LateTimeViolation = self.metrics.getLateTimeViolation(individual)
                TotalTimeViolation = self.metrics.getTotalTimeViolation(individual)
                
                individual.setTotalDistance(distance)
        
             
                
                # Update Environmental Cost
                individual.setEnvironmentalCost(environmental_cost)
                
                # Update Economic Cost
                individual.setEconomicCost(economic_cost)
                
                # Update Social Cost
                individual.setSocialCost(social_cost)
                
                # Update Total Duration
                individual.setTotalDuration(total_duration)
                
                # Update Early Time Violation
                individual.setEarlyTimeViolation(EarlyTimeViolation)
                
                # Update Late Time Violation
                individual.setLateTimeViolation(LateTimeViolation)
                
                # Update Total Time Violation
                individual.setTotalTimeViolation(TotalTimeViolation)
                
                
                totalDistanceForPop += distance
            #else:
                #totalDistanceForPop += individual.getFitness()
        
        return totalDistanceForPop / len(population)

    
    def evaluateFeasibility(self, population):
        for individual in population:
            isFeasible = self.metrics.isIndividualFeasible(individual)
            individual.setIsFeasible(isFeasible)


manager = Manager(r"C:\Users\Haifa Zaidi\OneDrive\Pictures\Research\data\test_data2.txt", 0.5)

def print_individual(individual):
    """
    Prints the string representation of an individual along with fitness and workload difference.
    """
    result = individual.__str__()  # Get the string representation of the individual
    result += (
        #f"\nTotal travel distance: {individual.getTotalDistance():.2f} Km | \n"
        #f"Total travel duration: {individual.getTotalDuration():.2f} | \n"
        #f"Fuel Consumption: {individual.getEnvironmentalCost()/2.64:.2f} L | \n"    
        f"\n \nEconomic cost: {individual.getEconomicCost()} $ | \n"
        f"Environmental Cost: {individual.getEnvironmentalCost()} Kg | \n"
        f"Social cost: {individual.getSocialCost()} | \n"
        

    
    )
    print(result)

ga = Algorithm(manager)   
 
historyShortestDistance=[]
start_time = time.time()
solutionDepots,bestIndividual = ga.run_no_args()
end_time = time.time()
    
    
def plot_solution(solutionDepots,individual):
    
    plt.figure(figsize=(10, 8))
    
    
    for depot in solutionDepots:
        # Get the location of the depot
        depot_x, depot_y = depot.getX(), depot.getY()

        # Plot the depot location
        plt.plot(depot_x, depot_y, 'ks', markersize=10, label=f'Depot {depot}')
        # Loop over each depot and its routes in the individual
        routes = individual.getChromosome()[depot.getId()]
       
        

        # Plot each route starting from this depot
        for route in routes:
            if not route:
                continue
            
            # Get the sequence of customer locations
            route_coords = [(depot_x, depot_y)]  # Start from the depot
            for customerID in route.getRoute():
                customer = depot.getCustomer(customerID)
                route_coords.append((customer.getX(), customer.getY()))
            route_coords.append((depot_x, depot_y))  # Return to the depot

            # Convert route coordinates to lists of x and y for plotting
            route_x, route_y = zip(*route_coords)

            # Plot the route with a random color
            plt.plot(route_x, route_y, marker='o', label=f'Route from Depot {depot}')

            # Plot each customer in the route
            for customerID in route.getRoute():
                customer = depot.getCustomer(customerID)
                customer_x, customer_y = customer.getX(), customer.getY()
                plt.plot(customer_x, customer_y, 'bo', markersize=5)  # Blue circle for customers

    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Solution Plot: Depots and Routes")
    plt.legend()
    plt.grid()
    plt.show()
 
print_individual(bestIndividual)

plot_solution(solutionDepots,bestIndividual)

# Calculate the execution time
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")