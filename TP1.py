def Path_Existence(G, s, t):
    """An algoritm for detecting path existence"""
    pass


class Vertex:
    def __init__(self, name):
        self.name = name


class Edge:
    """
    Please provide a tuple of start & end vertex names.
    
    E.g., my_edge = Edge(('A', 'B'))
    """

    def __init__(self, tuple):
        self.start, self.end = tuple


class Graph:
    def __init__(self, vertices, edges):
        self.vertices = vertices
        self.edges = edges


A = {("A", "B"), ("B", "C")}
