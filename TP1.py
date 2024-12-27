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
    visited = set() # track visited vertices
    visited.add(s)
    u = s # current vertex

    # Step 2: repeat until no more vertices can be marked
    marked = True
    while marked:
        marked = False
        for edge in G.edges:
            # If edge (u, v) exists, and u is visited but v is not
            if edge.start in visited and edge.end not in visited:
                visited.add(edge.end)
                marked = True

    # Step 3: check if t is visited
    return t in visited


class Vertex():
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f'Vertex({self.name})'


class Edge():
    """
    Represents a directed edge between two vertices.
    
    E.g., my_edge = Edge(('A', 'B'))
    """

    def __init__(self, tuple):
        self.start, self.end = tuple

    def __str__(self):
        return f'Edge({self.start} -> {self.end})'


class Graph():
    """
    Initialize the graph with a list of Vertex objects and a list of Edge objects.

    Parameters:
        vertices (list): List of Vertex objects.
        edges (list): List of Edge objects.
    """
    def __init__(self, vertices, edges):
        self.vertices = vertices
        self.edges = edges

    def __str__(self):
        return f'Graph(V, E): V = {self.vertices}, E = {self.edges}'


A = Vertex('A')
B = Vertex('B')
C = Vertex('C')
D = Vertex('D')

AB = Edge(('A', 'B'))
BC = Edge(('B', 'C'))
CD = Edge(('C', 'D'))

graph = Graph([A, B, C, D], [AB, BC, CD])

print(Path_Existence(graph, 'A', 'D'))
print(Path_Existence(graph, 'D', 'A'))