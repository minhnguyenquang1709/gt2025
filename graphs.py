import numpy as np
import pandas as pd




class Vertex:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"Vertex({self.name})"
    
    def __repr__(self):
        return self.__str__()


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
        return self.__str__()


class Graph:
    """
    Initialize the graph with a list of Vertex objects and a list of Edge objects.

    Parameters:
        vertices (list): List of Vertex objects.
        edges (list): List of Edge objects.
    """

    def __init__(self):
        self.vertex_set = set()
        self.edge_set = set() # set of tuple (start_vertex, end_vertex)
        self.vertices = []
        self.edges = []
        self.vertex_map = {} # map from vertex name to Vertex object
        self.matrix_rep = None

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

        if vertex_name not in self.vertex_map:
            vertex = Vertex(vertex_name)
            self.vertex_set.add(vertex)
            self.vertices.append(vertex)
            self.vertex_map[vertex_name] = vertex
        else:
            print(f'The vertex {vertex_name} is already in the graph, aborting.')

    def add_vertices(self, vertex_name_list:list):
        """
        Add multiple vertices into the graph.

        Parameters:
            vertex_name_list (list:str): A list of strings for the names of nodes.
        """

        if len(vertex_name_list) < 1:
            print("Empty list of vertex, aborting.")
            return
        for vertex_name in vertex_name_list:
            self.add_vertex(str(vertex_name))

    def add_vertices_by_string(self, vertex_string:str):
        """
        Add vertices from a string into the graph.

        Parameters:
            vertex_string (str): A string from which each element will be the name of a vertex.
        """

        for vertex_name in vertex_string:
            self.add_vertex(vertex_name)

    def add_edge(self, edge_tuple):
        """
        Add a directed edge into the graph.

        Parameters:
            edge_tuple (tuple): a tuple (start_vertex_name, end_vertex_name)
        """

        if len(edge_tuple) != 2:
            print(f'{edge_tuple} is not a valid edge, ignoring.')
            return
        
        start_name, end_name = edge_tuple
        start_name = str(start_name)
        end_name = str(end_name)
        if start_name not in self.vertex_map or end_name not in self.vertex_map:
            print(f"Cannot add edge {start_name} -> {end_name}: one or both vertices do not exist.")
            return
        
        start_vertex = self.vertex_map[start_name]
        end_vertex = self.vertex_map[end_name]

        if (start_vertex, end_vertex) not in self.edge_set:
            edge = DirectedEdge(start_vertex, end_vertex)
            self.edges.append(edge)
            self.edge_set.add((start_vertex, end_vertex))
        else:
            print(f'The edge {start_name} -> {end_name} already exists, aborting.')

    def add_edges(self, edge_name_list:list):
        """
        Add edges into the graph.

        Parameters:
            edge_name_list (list:tuple): a list of tuples (start_vertex_name, end_vertex_name).
        """

        if len(edge_name_list) < 1:
            print("Empty list of edges, aborting.")
            return

        for edge_tuple in edge_name_list:
            start_vertex_name, end_vertex_name = edge_tuple
            self.add_edge((start_vertex_name, end_vertex_name))

    def matrix(self):
        """
        Create the adjacency matrix representation of the graph.
        """
        size = len(self.vertices)
        adj_matrix = np.zeros((size, size), dtype=int)
        vertex_index = {vertex.name: idx for idx, vertex in enumerate(self.vertices)} # ensure each vertex has a unique row & column

        # add edges to the matrix
        for edge in self.edges:
            start_idx = vertex_index[edge.start.name]
            end_idx = vertex_index[edge.end.name]
            adj_matrix[start_idx][end_idx] = 1

        return adj_matrix

    def make_matrix(self):
        self.matrix_rep = self.matrix()

    def show_matrix(self):
        # extract vertex names for lavels
        vertex_names = [vertex.name for vertex in self.vertices]

        return pd.DataFrame(self.matrix_rep, index= vertex_names, columns=vertex_names)
    
    def path_existence(self, s, t):
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

      s = str(s)
      t = str(t)
      if s not in self.vertex_map:
          print(f"The vertex {s} does not exist, aborting.")
          return False
      if t not in self.vertex_map:
          print(f"The vertex {t} does not exist, aborting.")
          return False
      
      visited = set()  # track visited vertices
      original_start_vertex = self.vertex_map[s]
      target_vertex = self.vertex_map[t]
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
