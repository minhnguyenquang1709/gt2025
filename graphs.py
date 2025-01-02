import numpy as np
import pandas as pd

"""
Breadth First Search

Depth First Search

"""


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
        self.vertices = []
        self.vertex_map = {}
        self.edges = []
        self.edge_map = {}
        self.undirected_matrix = None

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


class DirectedGraph(Graph):
    def __init__(self):
        super().__init__()
        # self.directed_graph = None
        self.directed_matrix = None

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

    def make_undirected_graph(self):
        graph = Graph()
        graph.vertices = self.vertices.copy()
        graph.vertex_map = self.vertex_map.copy()
        graph.edges = self.edges.copy()
        graph.edge_map = self.edge_map.copy()

        for start_name, end_name in self.edge_map.keys():
            if (end_name, start_name) not in graph.edge_map:
                graph.add_edge((end_name, start_name))

        self.directed_graph = graph

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