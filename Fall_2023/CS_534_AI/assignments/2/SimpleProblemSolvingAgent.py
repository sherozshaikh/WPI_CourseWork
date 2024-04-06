import numpy as np

class Graph:
    """A graph connects nodes (vertices) by edges (links). Each edge can also
    have a length associated with it. The constructor call is something like:
        g = Graph({'A': {'B': 1, 'C': 2})
    this makes a graph with 3 nodes, A, B, and C, with an edge of length 1 from
    A to B,  and an edge of length 2 from A to C. You can also do:
        g = Graph({'A': {'B': 1, 'C': 2}, directed=False)
    This makes an undirected graph, so inverse links are also added. The graph
    stays undirected; if you add more links with g.connect('B', 'C', 3), then
    inverse link is also added. You can use g.nodes() to get a list of nodes,
    g.get('A') to get a dict of links out of A, and g.get('A', 'B') to get the
    length of the link from A to B. 'Lengths' can actually be any object at
    all, and nodes can be any hashable object."""

    def __init__(self, graph_dict=None, directed=True):
        self.graph_dict = graph_dict or {}
        self.directed = directed
        if not directed:
            self.make_undirected()

    def make_undirected(self):
        """Make a digraph into an undirected graph by adding symmetric edges."""
        for a in list(self.graph_dict.keys()):
            for (b, dist) in self.graph_dict[a].items():
                self.connect1(b, a, dist)

    def connect(self, A, B, distance=1):
        """Add a link from A and B of given distance, and also add the inverse
        link if the graph is undirected."""
        self.connect1(A, B, distance)
        if not self.directed:
            self.connect1(B, A, distance)

    def connect1(self, A, B, distance):
        """Add a link from A to B of given distance, in one direction only."""
        self.graph_dict.setdefault(A, {})[B] = distance

    def get(self, a, b=None):
        """Return a link distance or a dict of {node: distance} entries.
        .get(a,b) returns the distance or None;
        .get(a) returns a dict of {node: distance} entries, possibly {}."""
        links = self.graph_dict.setdefault(a, {})
        if b is None:
            return links
        else:
            return links.get(b)

    def nodes(self):
        """Return a list of nodes in the graph."""
        s1 = set([k for k in self.graph_dict.keys()])
        s2 = set([k2 for v in self.graph_dict.values() for k2, v2 in v.items()])
        nodes = s1.union(s2)
        return list(nodes)

def UndirectedGraph(graph_dict=None):
    """Build a Graph where every edge (including future ones) goes both ways."""
    return Graph(graph_dict=graph_dict, directed=False)

""" [Figure 3.2]
Simplified road map of Romania
"""

romania_map = UndirectedGraph(dict(
    Oradea=dict(Zerind=71, Sibiu=151),
    Zerind=dict(Arad=75, Oradea=71),
    Arad=dict(Zerind=75, Sibiu=140, Timisoara=118),
    Timisoara=dict(Arad=118, Lugoj=111),
    Lugoj=dict(Timisoara=111, Mehadia=70),
    Mehadia=dict(Drobeta=75, Lugoj=70),
    Drobeta=dict(Mehadia=75, Craiova=120),
    Craiova=dict(Pitesti=138, Drobeta=120, Rimnicu=146),
    Rimnicu=dict(Sibiu=80, Pitesti=97, Craiova=146),
    Sibiu=dict(Oradea=151, Arad=140, Fagaras=99, Rimnicu=80),
    Fagaras=dict(Sibiu=99, Bucharest=211),
    Pitesti=dict(Craiova=138, Rimnicu=97, Bucharest=101),
    Bucharest=dict(Urziceni=85, Pitesti=101, Giurgiu=90, Fagaras=211),
    Giurgiu=dict(Bucharest=90),
    Urziceni=dict(Vaslui=142, Bucharest=85, Hirsova=98),
    Hirsova=dict(Urziceni=98, Eforie=86),
    Eforie=dict(Hirsova=86),
    Vaslui=dict(Urziceni=142, Iasi=92),
    Iasi=dict(Vaslui=92, Neamt=87),
    Neamt=dict(Iasi=87),
))

romania_map.locations = dict(
    Arad=(91, 492),
    Bucharest=(400, 327),
    Craiova=(253, 288),
    Drobeta=(165, 299),
    Eforie=(562, 293),
    Fagaras=(305, 449),
    Giurgiu=(375, 270),
    Hirsova=(534, 350),
    Iasi=(473, 506),
    Lugoj=(165, 379),
    Mehadia=(168, 339),
    Neamt=(406, 537),
    Oradea=(131, 571),
    Pitesti=(320, 368),
    Rimnicu=(233, 410),
    Sibiu=(207, 457),
    Timisoara=(94, 410),
    Urziceni=(456, 350),
    Vaslui=(509, 444),
    Zerind=(108, 531),
    )

def exp_schedule(k=30, lam=0.05, limit=100):
    """One possible schedule function for simulated annealing"""
    return lambda t: (k * np.exp(-lam * t) if t < limit else 0)

class SimpleProblemSolvingAgent_SPSAP():
    """ A class that implements a Simple Problem Solving Agent (SPSA) for solving pathfinding problems on a Romania map. """
    def __init__(self):
        pass

    def calculate_euclidean_distance(self, x1 : float, y1 : float, x2 : float, y2 : float) -> int:
        """ Calculate the Euclidean distance between two points given their coordinates. """
        return int(round((((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5),0))

    # Heuristic Table
    def _get_heuristic_table(self, end_state)->dict:
        """ Generate a heuristic table based on the Euclidean distances to the goal state. """
        end_state_x, end_state_y = romania_map.locations.get(end_state)
        distances : dict = {city : self.calculate_euclidean_distance(x1 = romania_map.locations[city][0], y1 = romania_map.locations[city][1], x2 = end_state_x, y2 = end_state_y) for city in romania_map.nodes()}
        return distances

    def _generate_initial_path(self,initial,goal) -> list:
        """ Generate an initial path from 'initial' to 'goal' """
        open_list : list = list()
        close_list : list = list()
        break_cond : bool = True
        while break_cond:
            neighbors = list(romania_map.get(initial).keys())
            if goal in neighbors:
                close_list.extend(neighbors)
                break_cond : bool = False
            else:
                for i in neighbors:
                    if i not in close_list:
                        open_list.append(i)
            initial = open_list[0]
            del open_list[0]
            close_list.append(initial)
        initial_path : list = list()
        for i in close_list + open_list:
            if i not in initial_path and i!=initial and i!=goal:
                initial_path.append(i)
        initial_path.append(goal)
        return initial_path

    def _permute_initial_path(self,initial_path):
        """ Permute a given initial path randomly. """
        left = np.random.randint(0,len(initial_path) - 1)
        right = np.random.randint(0,len(initial_path) - 1)
        if left > right:
            left,right = right,left
        initial_path[left:right+1] = reversed(initial_path[left:right+1])
        return initial_path

    def _get_cost_path(self, current_state,goal_state,distance_table,schedule=exp_schedule(),t_range=999) -> tuple:
        """ Calculate the cost and path of a simulated annealing search. """
        path_taken : list = [current_state]
        total_cost : int = 0
        for t in range(t_range):
            if current_state == goal_state:
                break
            T = schedule(t)
            if T == 0:
                continue
            neighbors = list(romania_map.get(current_state).keys())
            if not neighbors:
                continue
            next_state = np.random.choice(neighbors)
            delta_e = distance_table[current_state] - distance_table[next_state]
            if delta_e > 0 or self.probability(np.exp(-delta_e / T)):
                total_cost += romania_map.get(current_state)[next_state]
                current_state = next_state
                path_taken.append(current_state)
        return path_taken,total_cost

    def _check_city(self, total_nodes : list, stxt : str) -> str:
        """ Prompt the user to input a city and validate if it exists in the list of possible cities. """
        city : str = input(f'\nPlease enter the {stxt} city: ')
        city_cond : bool = True
        while city_cond:
            if city in total_nodes:
                city_cond : bool = False
            else:
                print(f'\nCould not find {city}, please try again')
                city : str = input(f'\nPlease enter the {stxt} city: ')
                if city in total_nodes:
                    city_cond : bool = False
                else:
                    continue
        return city

    def _get_a_b_cities(self, all_nodes : list) -> tuple:
        """ Prompt the user to input origin and destination cities and ensure they are different else ask again. """
        origin_city = self._check_city(total_nodes=all_nodes, stxt='origin')
        destination_city = self._check_city(total_nodes=all_nodes, stxt='destination')
        same_city_condition : bool = True
        while same_city_condition:
            if origin_city != destination_city:
                same_city_condition : bool = False
            else:
                print("\nThe same city can't be both origin and destination. Please try again.")
                destination_city = self._check_city(total_nodes=all_nodes, stxt='destination')
        return origin_city,destination_city

    def _sort_heuristic_values(self, n_visit_nodes : list) -> list:
        """ Sort a list of nodes based on their heuristic cost. """
        n_visit_nodes.sort(key = lambda x: x[1]) 
        return n_visit_nodes

    def probability(self, p) -> bool:
        """Return true with probability p."""
        return p > np.random.uniform(0.0, 1.0)

    def greedy_best_first_search(self, current_state,goal_state) -> tuple:
        """ Implement the Greedy Best-First Search algorithm to find the optimal path and its cost. """
        limit_counter : int = 0
        distance_table : dict = self._get_heuristic_table(end_state = goal_state)
        open_list : list = list()
        closed_list : list = [current_state]
        path_taken : list = [current_state]
        total_cost : int = 0
        bool_cond : bool = True
        while bool_cond or limit_counter > 1000:
            next_available_states : list = list(romania_map.get(current_state).keys())
            if len(next_available_states) == 0:
                bool_cond : bool = False
            elif goal_state in next_available_states:
                path_taken.append(goal_state)
                total_cost += distance_table[current_state]
                bool_cond : bool = False
            else:
                for i in next_available_states:
                    if i not in closed_list:
                        open_list.append([i,distance_table[i]])
                open_list = self._sort_heuristic_values(n_visit_nodes = open_list)
                current_state = open_list[0][0]
                total_cost += open_list[0][1]
                del open_list[0]
                path_taken.append(current_state)
                closed_list.append(current_state)
            limit_counter += 1
        return r' → '.join(path_taken), total_cost

    def astar_search(self, current_state,goal_state) -> tuple:
        """ Implement the A* Search algorithm to find the optimal path and its cost. """
        limit_counter : int = 0
        distance_table : dict = self._get_heuristic_table(end_state = goal_state)
        open_list : list = list()
        closed_list : list = [current_state]
        path_taken : list = [current_state]
        total_cost : int = 0
        bool_cond : bool = True
        while bool_cond or limit_counter > 1000:
            next_available_states : list = list(romania_map.get(current_state).keys())
            if len(next_available_states) == 0:
                bool_cond : bool = False
            elif goal_state in next_available_states:
                path_taken.append(goal_state)
                total_cost += romania_map.get(current_state)[goal_state]
                bool_cond : bool = False
            else:
                for i in next_available_states:
                    if i not in closed_list:
                        open_list.append([i,distance_table[i] + romania_map.get(current_state)[i]])
                open_list = self._sort_heuristic_values(n_visit_nodes = open_list)
                total_cost += romania_map.get(current_state)[open_list[0][0]]
                current_state = open_list[0][0]
                del open_list[0]
                path_taken.append(current_state)
                closed_list.append(current_state)
            limit_counter += 1
        return r' → '.join(path_taken), total_cost

    def hill_climbing(self,initial,goal):
        """ Implement the Hill Climbing algorithm to find the optimal path and its cost. """
        distance_table : dict = self._get_heuristic_table(end_state = goal)
        number_of_neighbors=10
        hc_min : dict = {'cost' : 99999999, 'path' : list()}
        initial_path : list = self._generate_initial_path(initial,goal)
        iterations = 100
        while iterations:
            for _ in range(number_of_neighbors):
                initial_path : list = self._permute_initial_path(initial_path)
                current_cost : int = sum([distance_table[i] for i in initial_path])
                if current_cost > 0 and current_cost < hc_min['cost']:
                    hc_min['cost'] = current_cost
                    hc_min['path'] = initial_path
                    break
                else:
                    pass
            iterations -= 1
        if hc_min['cost'] != 99999999:
            followed_path : list = [initial]
            cond1 : bool = True
            while cond1:
                min_val = 9999
                min_city = ''
                neighbors = list(romania_map.get(initial).keys())
                if goal in neighbors:
                    followed_path.append(goal)
                    cond1 : bool = False
                else:
                    for j in neighbors:
                        if distance_table[j] < min_val:
                            min_val = distance_table[j]
                            min_city = j
                    initial = min_city
                if min_city != '':
                    followed_path.append(min_city)
            hc_min['path'] = followed_path
            hc_min['cost'] = 0
            for i in range(len(followed_path)):
                current_s,next_s = followed_path[i],followed_path[i+1]
                hc_min['cost'] += romania_map.get(current_s)[next_s]
                if next_s == goal:
                    break
        else:
            hc_min : dict = {'cost' : None, 'path' : list()}
        return r' → '.join(hc_min['path']), hc_min['cost']

    def simulated_annealing_full(self, current_state,goal_state,schedule=exp_schedule(),t_range = 999) -> tuple:
        """ Implement Simulated Annealing to find an optimal path and its cost over multiple epochs. """
        distance_table : dict = self._get_heuristic_table(end_state = goal_state)
        simulated_annealing_min : dict = {'cost' : 99999999, 'path' : list()}
        for epoch in range(t_range):
            m_path,m_cost = self._get_cost_path(current_state,goal_state,distance_table=distance_table,schedule=exp_schedule(),t_range=999)
            if m_cost > 1 and simulated_annealing_min['cost'] > m_cost:
                simulated_annealing_min['cost'] = m_cost
                simulated_annealing_min['path'] = m_path
            else:
                pass
        return r' → '.join(simulated_annealing_min['path']), simulated_annealing_min['cost']

    def simulated_annealing(self, current_state,goal_state,schedule=exp_schedule(),t_range=999) -> tuple:
        """ Implement Simulated Annealing to find an optimal path and its cost. """
        distance_table : dict = self._get_heuristic_table(end_state = goal_state)
        path_taken : list = [current_state]
        total_cost : int = 0
        for t in range(t_range):
            if current_state == goal_state:
                break
            T = schedule(t)
            if T == 0:
                continue
            neighbors = list(romania_map.get(current_state).keys())
            if not neighbors:
                continue
            next_state = np.random.choice(neighbors)
            delta_e = distance_table[current_state] - distance_table[next_state]
            if delta_e > 0 or self.probability(np.exp(-delta_e / T)):
                total_cost += romania_map.get(current_state)[next_state]
                current_state = next_state
                path_taken.append(current_state)
        return r' → '.join(path_taken), total_cost

    def run_models(self, current_state,goal_state) -> None:
        """ Run various pathfinding algorithms and display their results. """
        print('\nGreedy Best-First Search')
        path1,cost1=self.greedy_best_first_search(current_state,goal_state)
        print(path1)
        print(f'Total Cost: {cost1}')

        print('\nA* Search')
        path1,cost1=self.astar_search(current_state,goal_state)
        print(path1)
        print(f'Total Cost: {cost1}')

        print('\nHill Climbing Search')
        path1,cost1=self.hill_climbing(current_state,goal_state)
        print(path1)
        print(f'Total Cost: {cost1}')

        print('\nSimulated Annealing Search')
        path1,cost1=self.simulated_annealing(current_state,goal_state)
        print(path1)
        print(f'Total Cost: {cost1}')

        # print('\nSimulated Annealing Search - Quick Path')
        # path1,cost1=self.simulated_annealing_full(current_state,goal_state)
        # print(path1)
        # print(f'Total Cost: {cost1}')

        return None

# if __name__ == '__main__':
    # For Testing
