import numpy as np


def Path_Existence(G, s, t):
    """
    An algoritm for detecting path existence.
    Iterate through all edges. If an edge (u, v ) is found, such that
    u is marked and v is not, then mark v as visited.

    Parameters:
        G (Graph): The graph object.
        s (str): the name of the start vertex.
        t (str): the name of the target vertex.

    Returns:
        bool: True if ath exists from s to t, otherwise False.
    """
    # Step 1: mark s as visited
    visited = set()  # track visited vertices
    visited.add(s)

    # Step 2: repeat until no more vertices can be marked
    marked = True
    while marked:
        marked = False
        for edge in G.edges:
            # If edge (u, v) exists, and u is visited but v is not
            # print(f'Considering edge ({edge.start}, {edge.end})')
            if edge.start in visited and edge.end not in visited:
                # print(f'{edge.start} is visited but {edge.end} is not, adding {edge.end} to visited set')
                visited.add(edge.end)
                marked = True

    # Step 3: check if t is visited
    return t in visited


class Vertex:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"Vertex({self.name})"


class DirectedEdge:
    """
    Represents a directed edge between two vertices.

    E.g., my_edge = Edge(('A', 'B'))
    """

    def __init__(self, tuple):
        self.start, self.end = tuple

    def __str__(self):
        return f"Edge({self.start} -> {self.end})"


class Graph:
    """
    Initialize the graph with a list of Vertex objects and a list of Edge objects.

    Parameters:
        vertices (list): List of Vertex objects.
        edges (list): List of Edge objects.
    """

    def __init__(self, vertex_name_list=None, edge_name_list=None):
        self.vertex_set = set()
        self.edge_set = set()
        self.vertices = []
        self.edges = []
        if vertex_name_list is not None:
            self.add_vertices(vertex_name_list)

        if edge_name_list is not None:
            self.add_edges(edge_name_list)

        self.matrix_rep = self.matrix()

    def get_vertices(self):
        return self.vertices

    def get_edges(self):
        return self.edges

    def add_vertex(self, vertex_name):
        """
        Add a vertex into the graph

        Parameters:
            vertex (str): A string of length 1 representing the name of node
        """

        if vertex_name not in self.vertex_set:
            self.vertices.append(Vertex(vertex_name))

    def add_vertices(self, vertex_name_list):
        """
        Add vertices into the graph.

        Parameters:
            vertex_name_list (list:str): A list of strings for the names of nodes.
        """

        if len(vertex_name_list) < 1:
            print("Empty list of vertex, abort.")
            return
        for vertex_name in vertex_name_list:
            vertex_name = str(vertex_name)
            self.vertex_set.add(vertex_name)
            self.add_vertex(vertex_name)

    def add_edge(self, edge_name):
        """
        Add an edge into the graph

        Parameters:
            edge (str): string of length 2 consisting of the start and the end node. E.g., 'AB'
        """

        if len(edge_name) != 2:
            print(f"{edge_name} is not a correct edge and will be ignored")
            return
        else:
            if edge_name not in self.edge_set:
                start, end = edge_name[0], edge_name[1]
                self.edges.append(DirectedEdge((start, end)))

    def add_edges(self, edge_name_list):
        """
        Add edges into the graph

        Parameters:
            edge_name_list (list:str)
        """
        if len(edge_name_list) < 1:
            print("Empty list of edges, abort.")
            return

        for edge_name in edge_name_list:
            edge_name = str(edge_name)

            self.add_edge(edge_name)

    def matrix(self):
        matrix_temp = []
        # for each row
        for row_vertex in self.vertices:
            row_temp = []
            for column_vertex in self.vertices:
                # set to 1 if not a vertex connecting to itself there exists an edge start from the row vertex and go the column vertex
                connection = False
                for edge in self.edges:
                    if edge.start == row_vertex.name and edge.end == column_vertex.name:
                        connection = True
                        break

                if connection:
                    row_temp.append(1)
                else:
                    row_temp.append(0)

            row = np.array(row_temp)
            matrix_temp.append(row)

        return np.array(matrix_temp)

    def make_matrix(self):
        self.matrix_rep = self.matrix()

    def count_weakly_connected_components(self):
        """
        Count the number of weakly connected components from the matrix representation of the graph

        Def: Weakly connected components are subgraphs having a connection b/w very 2 nodes no matter the direction of the edges.
        In other words: A directed graph is weakly connected if the underlying undirected graph is connected
        """
        pass

    def count_strongly_connected_components(self):
        """
        Count the number of strongly connected components from the matrix representation of the graph

        Def: Strongly connected components are subgraphs having a connection b/w every 2 nodes considering the direction of the edges.
        """
        pass

    def __str__(self):
        return f"Graph(V, E): V = {self.vertices}, E = {self.edges}"
