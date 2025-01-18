import numpy as np
import pandas as pd

# from typing import List, Dict, Tuple


class Vertex:
    def __init__(self, name):
        self.name = str(name)

    def __str__(self):
        return f"Vertex({self.name})"

    def __repr__(self):
        return f"Vertex({self.name})"


class Edge:
    """
    Represents a directed edge between two vertices.

    E.g., my_edge = Edge(('A', 'B'))
    """

    def __init__(self, start_vertex, end_vertex, is_weighted=False, weight=1):
        self.start = start_vertex
        self.end = end_vertex
        self.is_weighted = is_weighted
        if self.is_weighted:
            self.weight = weight

    def __str__(self):
        return f"Edge({self.start.name} -> {self.end.name})"

    def __repr__(self):
        return f"Edge({self.start}, {self.end})"


class Graph:
    def __init__(self):
        self.vertices: list[Vertex] = []
        self.edges: list[Edge] = []
        self.vertex_map: dict[str, Vertex] = {}
        self.edge_map: dict[tuple[str, str], Edge] = {}
        self.undirected_matrix: np.ndarray = None

    def make_vertex(vertex_name) -> Vertex:
        pass

    def make_vertices_by_string(string) -> list[Vertex]:
        pass

    def make_edge(start_vertex: Vertex, end_vertex: Vertex) -> Edge:
        pass

    def add_vertex(self, vertex: Vertex):
        pass

    def add_vertices(self, vertex_list: list[Vertex]):
        pass

    def add_edge(self, edge: Edge):
        pass

    def add_edges(self, edge_list: list[Edge]):
        pass

    def remove_vertex(self, vertex_name: str):
        pass

    def remove_edge(self, edge_name_tuple: tuple[str, str]):
        pass

    def make_umatrix(self):
        pass

    def show_umatrix(self):
        pass

    def get_vertex(self, vertex_name: str) -> Vertex:
        pass

    def get_edge(self, edge_name_tuple: tuple[str, str]) -> Edge:
        pass

    def path_existence(self, s_vertex_name: str, t_vertex_name: str) -> bool:
        pass

    def degree(self, vertex_name: str) -> int:
        pass

    def bfs(self, s_vertex_name: str) -> list[list[Vertex]]:
        pass

    def dfs(self, s_vertex_name: str) -> list[list[Vertex]]:
        pass

    def gen_component(vertex_list: list[Vertex], edge_list: list[Edge]):
        pass

    def update(self):
        pass


class DGraph(Graph):
    pass


class DTree(DGraph):
    pass


class DWTree(DTree):
    pass


class Tree(Graph):
    pass


class WGraph(Graph):
    pass


class WTree(Graph):
    pass
