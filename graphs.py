from typing import List, Dict, Tuple
import numpy as np
import pandas as pd


class Vertex:
    def __init__(self, name):
        self.name = str(name)

    def __str__(self):
        return f"Vertex({self.name})"

    def __repr__(self):
        return f"Vertex({self.name})"


class DirectedEdge:
    """
    Represents a directed edge between two vertices.

    E.g., my_edge = Edge(('A', 'B'))
    """

    def __init__(self, start_vertex, end_vertex):
        self.start = start_vertex
        self.end = end_vertex

    def __str__(self):
        return f"Edge({self.start.name} -> {self.end.name})"

    def __repr__(self):
        return f"DirectedEdge({self.start}, {self.end})"


class Graph:
    def __init__(self):
        self.vertices: List[Vertex] = []
        self.edges: List[DirectedEdge] = []
        self.vertex_map: Dict[str, Vertex] = {}
        self.edge_map: Dict[Tuple[str, str], DirectedEdge] = {}
        self.undirected_matrix: np.ndarray = None
        self.components: List[Graph|DirectedGraph] = []

    def add_vertex(self, vertex_name: str):
        vertex_name = str(vertex_name)
        if vertex_name not in self.vertex_map:
            vertex = Vertex(vertex_name)
            self.vertices.append(vertex)
            self.vertex_map[vertex_name] = vertex

        self.undirected_matrix = self.make_undirected_matrix()

    def add_vertices(self, vertex_name_list: list):
        if len(vertex_name_list) < 1:
            print("Empty list of vertex names. Aborting.")
            return

        for vertex_name in vertex_name_list:
            vertex_name = str(vertex_name)
            if vertex_name in self.vertex_map:
                print(f"Vertex '{vertex_name}' already exists in the graph. Ignoring.")
                continue

            self.add_vertex(vertex_name)

    def add_vertices_by_string(self, vertex_name_string: str):
        if len(vertex_name_string) < 1:
            print("Empty string. Aborting.")
            return

        for vertex_name in vertex_name_string:
            if vertex_name in self.vertex_map:
                print(f"Vertex '{vertex_name}' already exists in the graph. Aborting.")
                continue

            self.add_vertex(vertex_name)

    def add_edge(self, edge_tuple: tuple):
        if len(edge_tuple) != 2:
            print(f"{edge_tuple} is an invalid edge. Aborting.")
            return

        start_name, end_name = edge_tuple
        start_name = str(start_name)
        end_name = str(end_name)

        if start_name not in self.vertex_map:
            print(f"Vertex '{start_name}' does not exist in the graph. Aborting.")
            return
        if end_name not in self.vertex_map:
            print(f"Vertex '{end_name}' does not exist in the graph. Aborting.")
            return

        edge_name_1 = (start_name, end_name)
        edge_name_2 = (end_name, start_name)

        vertex1 = self.vertex_map[start_name]
        vertex2 = self.vertex_map[end_name]

        if edge_name_1 not in self.edge_map:
            edge1 = DirectedEdge(vertex1, vertex2)
            self.edges.append(edge1)
            self.edge_map[edge_name_1] = edge1

        if edge_name_2 not in self.edge_map:
            edge2 = DirectedEdge(vertex2, vertex1)
            self.edges.append(edge2)
            self.edge_map[edge_name_2] = edge2

        self.undirected_matrix = self.make_undirected_matrix()

    def add_edges(self, edge_name_list: list):
        if len(edge_name_list) < 1:
            print("Empty edge list. Aborting.")
            return

        for edge_tuple in edge_name_list:
            self.add_edge(edge_tuple)

    def remove_vertex(self, vertex_name):
        vertex_name = str(vertex_name)
        if vertex_name not in self.vertex_map:
            print(f"Vertex {vertex_name} does not exist in the graph. Aborting.")
            return

        vertex_indices = {vertex.name: idx for idx, vertex in enumerate(self.vertices)}
        vertex_index = vertex_indices[vertex_name]
        self.vertices.pop(vertex_index)
        self.vertex_map.pop(vertex_name)
        for vertex_name_tuple, edge in self.edge_map.items():
            if vertex_name not in vertex_name_tuple:
                continue

            # if found the associated edge
            popped_edge = self.edge_map.pop(vertex_name_tuple)
            self.edges.remove(popped_edge)

    def make_undirected_matrix(self):
        size = len(self.vertices)
        adj_matrix = np.zeros((size, size), dtype=int)
        vertex_index = {
            vertex.name: idx for idx, vertex in enumerate(self.vertices)
        }  # ensure each vertex has a unique row & column

        # add edges to the matrix
        for edge in self.edges:
            start_idx = vertex_index[edge.start.name]
            end_idx = vertex_index[edge.end.name]
            adj_matrix[start_idx][end_idx] = 1
            adj_matrix[end_idx][start_idx] = 1

        self.undirected_matrix = adj_matrix

    def show_undirected_matrix(self):
        # extract vertex names for lavels
        vertex_names = [vertex.name for vertex in self.vertices]

        return pd.DataFrame(
            self.undirected_matrix, index=vertex_names, columns=vertex_names
        )

    def get_vertices(self):
        return self.vertices

    def get_edges(self):
        return self.edges

    def path_existence(self, start, target):
        """
        An algoritm for detecting path existence.
        Iterate through all edges. If an edge (u, v ) is found, such that
        u is marked and v is not, then mark v as visited.

        Parameters:
            s (str): the name of the start vertex.
            t (str): the name of the target vertex.

        Returns:
            bool: True if ath exists from s to t, otherwise False.
        """

        start = str(start)
        target = str(target)
        if start not in self.vertex_map:
            print(f"The vertex {start} does not exist, aborting.")
            return False
        if target not in self.vertex_map:
            print(f"The vertex {target} does not exist, aborting.")
            return False

        visited = set()  # track visited vertices
        original_start_vertex = self.vertex_map[start]
        target_vertex = self.vertex_map[target]
        visited.add(original_start_vertex)

        # Step 2: repeat until no more vertices can be marked
        marked = True
        while marked:
            marked = False
            for edge in self.edges:
                # If edge (u, v) exists, and u is visited but v is not
                # print(f'Considering edge ({edge.start}, {edge.end})')
                if edge.start in visited and edge.end not in visited:
                    # print(f'{edge.start} is visited but {edge.end} is not, adding {edge.end} to visited set')
                    visited.add(edge.end)
                    marked = True

        # Step 3: check if t is visited
        return target_vertex in visited

    def degree(self, vertex_name):
        """
        Calculate the degree of a given vertex.

        Parameters:
            vertex_name (str): The name of the vertex whose degree is being calculated.

        Return:
            int: The degree of the vertex.
            None: If the vertex does not exist in the graph.
        """

        vertex_name = str(vertex_name)
        if vertex_name not in self.vertex_map:
            print(f"The vertex {vertex_name} does not exist in the graph, aborting.")
            return None
        else:
            degree = 0
            for edge_name in self.edge_map.keys():
                start_name, end_name = edge_name
                if vertex_name == start_name or vertex_name == end_name:
                    degree += 1

            return degree

    def get_pendant_vertex(self):
        result = []
        for vertex in self.vertices:
            if self.degree(vertex.name) == 1:
                result.append(vertex)

        return result

    def bfs(self, start_name):
        start_name = str(start_name)
        if start_name not in self.vertex_map:
            print(f"Vertex {start_name} does not exist. Aborting.")
            return None

        vertex_indices = {vertex.name: idx for idx, vertex in enumerate(self.vertices)}
        start_vertex = self.vertex_map[start_name]
        visited = [False] * len(self.vertices)
        q = [start_vertex]
        result = []

        while q:
            visited_vertex = q[0]
            vertex_idx = vertex_indices[visited_vertex.name]

            result.append(visited_vertex.name)
            visited[vertex_idx] = True
            q.pop(0)

            # for every adjacent vertex to the current
            for i in range(len(self.vertices)):
                if (self.undirected_matrix[vertex_idx, i] == 1) and (not visited[i]):
                    q.append(self.vertices[i])
                    visited[i] = True

        return result

    def dfs(self, start_name, visited=None, result=None):
        if visited is None:
            visited = [False] * len(self.vertices)
        if result is None:
            result = []

        start_name = str(start_name)
        if start_name not in self.vertex_map:
            print(f"Vertex {start_name} does not exist. Aborting.")
            return []

        vertex_indices = {vertex.name: idx for idx, vertex in enumerate(self.vertices)}
        start_idx = vertex_indices[start_name]

        # mark the current vertex as visited
        result.append(self.vertex_map[start_name].name)
        visited[start_idx] = True

        size = len(self.vertices)
        for i in range(size):
            if (self.undirected_matrix[start_idx][i] == 1) and (not visited[i]):
                self.dfs(self.vertices[i].name, visited, result)

        return result

    def find_components(self):
        components = []

        vertex_indices = {vertex.name: idx for idx, vertex in enumerate(self.vertices)}
        visited = [False] * len(self.vertices)

        for i in range(len(self.vertices)):
            if visited[i]:
                continue
            visited_vertex_name = self.vertices[i].name
            results = self.dfs(visited_vertex_name, visited)
            # print(f"Results: {results}")

            component_vertices = []
            component_vertex_map = {}
            component_edges = []
            component_edge_map = {}
            for result in results:
                for vertex_name in result:
                    # assign vertex to component
                    vertex_idx = vertex_indices[vertex_name]
                    vertex = self.vertices[vertex_idx]
                    component_vertices.append(vertex)
                    component_vertex_map[vertex_name] = vertex

                    # assign edge to component
                    for (start_name, end_name), edge in self.edge_map.items():
                        if vertex_name not in (start_name, end_name):
                            continue

                        component_edges.append(edge)
                        component_edge_map[(start_name, end_name)] = edge
                
            component = self.generate_component(component_vertices, component_vertex_map, component_edges, component_edge_map)
            
            components.append(component)

        self.components = components
        return components
    
    def generate_component(self, vertices, vertex_map, edges, edge_map):
        component = Graph()
        component.vertices = vertices
        component.vertex_map = vertex_map
        component.edges = edges
        component.edge_map = edge_map
        print(f"Component vertex map: {component.vertex_map}")
        component.make_undirected_matrix()


class DirectedGraph(Graph):
    def __init__(self):
        super().__init__()
        self.directed_matrix: np.ndarray = None
        self.scc: List[DirectedGraph] = []

    def add_directed_edge(self, edge_tuple):
        if len(edge_tuple) != 2:
            print(f"{edge_tuple} is an invalid edge. Aborting.")
            return

        start_name, end_name = edge_tuple
        start_name = str(start_name)
        end_name = str(end_name)

        if start_name not in self.vertex_map:
            print(f"Vertex '{start_name}' does not exist in the graph. Aborting.")
            return
        if end_name not in self.vertex_map:
            print(f"Vertex '{end_name}' does not exist in the graph. Aborting.")
            return

        edge_name = (start_name, end_name)

        start_vertex = self.vertex_map[start_name]
        end_vertex = self.vertex_map[end_name]

        if edge_name not in self.edge_map:
            edge = DirectedEdge(start_vertex, end_vertex)
            self.edges.append(edge)
            self.edge_map[edge_name] = edge

        self.make_undirected_matrix()
        # self.make_undirected_graph()
        self.make_directed_matrix()

    def add_directed_edges(self, edge_name_list: list):
        if len(edge_name_list) < 1:
            print("Empty edge list. Aborting.")
            return

        for edge_tuple in edge_name_list:
            self.add_directed_edge(edge_tuple)

    def idegree(self, vertex_name):
        """
        Calculate the in-degree of a given vertex.

        Parameters:
            vertex_name (str): The name of the vertex whose in-degree is being calculated.

        Return:
            int: The in-degree of the vertex.
            None: If the vertex does not exist in the graph.
        """

        vertex_name = str(vertex_name)
        if vertex_name not in self.vertex_map:
            print(f"The vertex {vertex_name} does not exist in the graph, aborting.")
            return None
        else:
            degree = 0
            for edge_name in self.edge_map.keys():
                start_name, end_name = edge_name
                if vertex_name == end_name:
                    degree += 1

            return degree

    def odegree(self, vertex_name):
        """
        Calculate the out-degree of a given vertex.

        Parameters:
            vertex_name (str): The name of the vertex whose out-degree is being calculated.

        Return:
            int: The out-degree of the vertex.
            None: If the vertex does not exist in the graph.
        """

        vertex_name = str(vertex_name)
        if vertex_name not in self.vertex_map:
            print(f"The vertex {vertex_name} does not exist in the graph, aborting.")
            return None
        else:
            degree = 0
            for edge_name in self.edge_map.keys():
                start_name, end_name = edge_name
                if vertex_name == start_name:
                    degree += 1

            return degree

    def make_undirected_matrix(self):
        size = len(self.vertices)
        adj_matrix = np.zeros((size, size), dtype=int)
        vertex_index = {
            vertex.name: idx for idx, vertex in enumerate(self.vertices)
        }  # ensure each vertex has a unique row & column

        # add edges to the matrix
        for edge in self.edges:
            start_idx = vertex_index[edge.start.name]
            end_idx = vertex_index[edge.end.name]
            adj_matrix[start_idx][end_idx] = 1
            adj_matrix[end_idx][start_idx] = 1

        self.undirected_matrix = adj_matrix

    def make_directed_matrix(self):
        size = len(self.vertices)
        adj_matrix = np.zeros((size, size), dtype=int)
        vertex_index = {
            vertex.name: idx for idx, vertex in enumerate(self.vertices)
        }  # ensure each vertex has a unique row & column

        # add edges to the matrix
        for edge in self.edges:
            start_idx = vertex_index[edge.start.name]
            end_idx = vertex_index[edge.end.name]
            adj_matrix[start_idx][end_idx] = 1

        self.directed_matrix = adj_matrix

    def show_directed_matrix(self):
        # extract vertex names for lavels
        vertex_names = [vertex.name for vertex in self.vertices]

        return pd.DataFrame(
            self.directed_matrix, index=vertex_names, columns=vertex_names
        )

    def bfs_directed(self, start_name):
        start_name = str(start_name)
        if start_name not in self.vertex_map:
            print(f"Vertex {start_name} does not exist. Aborting.")
            return None

        vertex_indices = {vertex.name: idx for idx, vertex in enumerate(self.vertices)}
        start_vertex = self.vertex_map[start_name]
        visited = [False] * len(self.vertices)
        q = [start_vertex]
        result = []

        while q:
            visited_vertex = q[0]
            vertex_idx = vertex_indices[visited_vertex.name]

            result.append(visited_vertex)
            visited[vertex_idx] = True
            q.pop(0)

            # for every adjacent vertex to the current
            for i in range(len(self.vertices)):
                if (self.directed_matrix[vertex_idx, i] == 1) and (not visited[i]):
                    q.append(self.vertices[i])
                    visited[i] = True

        return result

    def dfs_directed(self, start_name, visited=None, result=None):
        """
        Depth First Search implementation

        Parameters:
            start_name (str): name of the starting node.

            visited (list<bool>): the array to keep track of visited nodes.

            result (list<str>): the array containing DFS results.

        Return:
            An array containing vertex names in DFS order.
        """
        if visited is None:
            visited = [False] * len(self.vertices)
        if result is None:
            result = []

        start_name = str(start_name)
        if start_name not in self.vertex_map:
            print(f"Vertex {start_name} does not exist. Aborting.")
            return None

        vertex_indices = {vertex.name: idx for idx, vertex in enumerate(self.vertices)}
        start_idx = vertex_indices[start_name]

        # mark the current vertex as visited
        if not visited[start_idx]:
            result.append(self.vertex_map[start_name].name)
            visited[start_idx] = True

        size = len(self.vertices)
        for i in range(size):
            if (self.directed_matrix[start_idx][i] == 1) and not visited[i]:
                self.dfs_directed(self.vertices[i].name, visited, result)

        return result
    
    def generate_component(self, vertices, vertex_map, edges, edge_map):
        component = DirectedGraph()
        component.vertices = vertices
        component.vertex_map = vertex_map
        component.edges = edges
        component.edge_map = edge_map
        # print(f"Component vertex map: {component.vertex_map}")
        component.make_undirected_matrix()
        component.make_directed_matrix()
        
        return component
    
    def find_scc_from_matrix(self):
        matrix = self.directed_matrix
        n = len(matrix)
        vertex_indices = {vertex.name: idx for idx, vertex in enumerate(self.vertices)}

        # find complete order
        def dfs_order(graph):
            visited = [False] * n
            order = []
            
            def dfs(vertex_name):
                # print(f"name: {vertex_name}")
                idx = vertex_indices[vertex_name]
                visited[idx] = True
                for w in range(n):
                    if graph[idx][w] == 1 and not visited[w]:
                        dfs(self.vertices[w].name)
                order.append(vertex_name)
            
            for v in range(n):
                if not visited[v]:
                    dfs(self.vertices[v].name)
            return order[::-1]
        
        # create transpose matrix
        def transpose_matrix(m:np.ndarray):
            return np.transpose(m.copy())
        
        # find
        def find_scc(graph, order):
            visited = [False] * n
            sccs = []
            
            def dfs(v, current_scc):
                visited[v] = True
                current_scc.append(self.vertices[v].name)
                for w in range(n):
                    if graph[v][w] == 1 and not visited[w]:
                        dfs(w, current_scc)
            
            for vertex_name in order:
                idx = vertex_indices[vertex_name]
                if not visited[idx]:
                    current_scc = []
                    dfs(idx, current_scc)
                    if current_scc:
                        sccs.append(current_scc)
            
            return sccs
        
        # Kosaraju's algo
        first_order = dfs_order(matrix)
        # print(f"First order: {first_order}")
        
        transposed = transpose_matrix(matrix)
        
        sccs = find_scc(transposed, first_order)

        self.scc = sccs
        return sccs

class Helper:
    nb = 0

    def getNumber():
        Helper.nb += 1
        return Helper.nb - 1
    
    def resetNumber():
        Helper.nb = 0