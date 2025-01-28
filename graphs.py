import numpy as np
import pandas as pd

class Vertex:
    def __init__(self, name: str):
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

    def __init__(self, start_vertex: Vertex, end_vertex: Vertex):
        self.start = start_vertex
        self.end = end_vertex

    def __str__(self):
        return f"Edge({self.start.name} -> {self.end.name})"

    def __repr__(self):
        return f"Edge({self.start}, {self.end})"


class Graph:
    def __init__(self):
        self._vertices: list[Vertex] = []
        self._edges: list[Edge] = []
        self._vertex_map: dict[str, Vertex] = {}
        self._edge_map: dict[tuple[str, str], Edge] = {}
        self._umatrix: np.ndarray = None

        self._vertex_indices: dict[str, int] = {}

        self._weight_map: dict[tuple[str, str], int] = {}
        self._wmatrix: np.ndarray = None
        

    def update(self):
        self.make_umatrix()
        self.indexing()
        self.make_wmatrix()

    def indexing(self):
        self._vertex_indices = {vertex.name: idx for idx, vertex in enumerate(self._vertices)}

    def add_vertex(self, vertex_name: str) -> bool:
        vertex_name = str(vertex_name)

        if vertex_name in self._vertex_map:
            print(f"Vertex {vertex_name} already exists, ignore.")
            return False

        v = Vertex(vertex_name)
        self._vertices.append(v)
        self._vertex_map[vertex_name] = v

        self.update()
        return True

    def add_vertices(self, vertex_name_list: list[str]):
        for vertex_name in vertex_name_list:
            self.add_vertex(vertex_name)

    def add_vertices_by_string(self, vertex_name_string: str):
        if len(vertex_name_string) < 1:
            print("Empty string, aborting.")
            return

        for vertex_name in vertex_name_string:
            self.add_vertex(vertex_name)

    def add_directed_edge(self, edge_name_tuple: tuple[str, str]) -> bool:
        weight = 1
        if len(edge_name_tuple) == 2:
            u_name, v_name = edge_name_tuple
        elif len(edge_name_tuple) == 3:
            u_name, v_name, newweight = edge_name_tuple
            weight = newweight
        u_name = str(u_name)
        v_name = str(v_name)

        if (u_name, v_name) in self._edge_map:
            print(f"Edge {u_name} -> {v_name} already exists, ignore.")
            return False

        if u_name not in self._vertex_map or v_name not in self._vertex_map:
            print("One or both vertex do not exist, ignore.")
            return False

        u = self._vertex_map[u_name]
        v = self._vertex_map[v_name]
        e = Edge(u, v)
        self._edges.append(e)
        self._edge_map[(u_name, v_name)] = e
        self._weight_map[(u_name, v_name)] = weight

        self.update()
        return True

    def add_undirected_edge(self, edge_name_tuple: tuple[str, str]) -> bool:
        if len(edge_name_tuple) != 2 and len(edge_name_tuple) != 3:
            print(f"Invalid input, abort.")
            return False
        
        weight = 1
        if len(edge_name_tuple) == 2:
            u_name, v_name = edge_name_tuple
        elif len(edge_name_tuple) == 3:
            u_name, v_name, newweight = edge_name_tuple
            weight = newweight

        # print("Haaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        print(f"Adding edge ({u_name}) -> ({v_name})")
        self.add_directed_edge((u_name, v_name, weight))
        print(f"Adding edge ({v_name}) -> ({u_name})")
        self.add_directed_edge((v_name, u_name, weight))
        return True

    def add_undirected_edges(self, edge_name_list: list[tuple[str, str]]):
        for edge_name_tuple in edge_name_list:
            self.add_undirected_edge(edge_name_tuple)

    def add_directed_edges(self, edge_name_list: list[tuple[str, str]]):
        for edge_name_tuple in edge_name_list:
            self.add_directed_edge(edge_name_tuple)

    def remove_vertex(self, vertex_name: str):
        m = len(self._vertices)
        n = len(self._vertex_map)
        vertex_name = str(vertex_name)

        if vertex_name not in self._vertex_map:
            print(f"Vertex {vertex_name} does not exist, ignore.")
            return

        # find all associated edges
        associated_edges = []
        for edge_name_tuple in self._edge_map.keys():
            for name in edge_name_tuple:
                if name == vertex_name:
                    associated_edges.append(edge_name_tuple)

        for edge in associated_edges:
            self.remove_edge(edge)

        v = self._vertex_map[vertex_name]
        self._vertices.remove(v)
        del self._vertex_map[vertex_name]
        del v
        if len(self._vertices) == m - 1 and len(self._vertex_map) == n - 1 and m == n:
            print(f"Successfully removed Vertex {vertex_name}")
        else:
            print(f"Failed to remove Vertex {vertex_name}")

        self.update()

    def remove_edge(self, edge_name_tuple: tuple[str, str]):
        u_name, v_name = edge_name_tuple
        u_name = str(u_name)
        v_name = str(v_name)

        if (u_name, v_name) not in self._edge_map:
            print(f"Edge {u_name} -> {v_name} does not exist, ignore.")
            return

        e = self._edge_map[(u_name, v_name)]
        self._edges.remove(e)
        del self._edge_map[(u_name, v_name)]
        del e

        self.update()

    def make_umatrix(self):
        size = len(self._vertices)
        adj_matrix = np.zeros((size, size), dtype=int)
        vertex_indices = {
            vertex.name: idx for idx, vertex in enumerate(self._vertices)
        }  # ensure each vertex has a unique row & column

        for edge in self._edges:
            start_idx = vertex_indices[edge.start.name]
            end_idx = vertex_indices[edge.end.name]
            adj_matrix[start_idx][end_idx] = 1
            adj_matrix[end_idx][start_idx] = 1

        self._umatrix = adj_matrix

    def make_wmatrix(self):
        size = len(self._vertices)
        adj_matrix = np.zeros((size, size), dtype=int)
        wmatrix = np.zeros((size, size), dtype=int)
        vertex_indices = {
            vertex.name: idx for idx, vertex in enumerate(self._vertices)
        }  # ensure each vertex has a unique row & column

        for edge in self._edges:
            start_idx = vertex_indices[edge.start.name]
            end_idx = vertex_indices[edge.end.name]
            adj_matrix[start_idx][end_idx] = 1
            adj_matrix[end_idx][start_idx] = 1

            weight = self._weight_map[(edge.start.name, edge.end.name)]
            wmatrix[start_idx][end_idx] = weight
            wmatrix[end_idx][start_idx] = weight

        self._umatrix = adj_matrix
        self._wmatrix = wmatrix

    def show_wmatrix(self):
        vertex_names = [vertex.name for vertex in self._vertices]

        return pd.DataFrame(self._wmatrix, index=vertex_names, columns=vertex_names)

    def show_umatrix(self):
        vertex_names = [vertex.name for vertex in self._vertices]

        return pd.DataFrame(self._umatrix, index=vertex_names, columns=vertex_names)

    def get_vertex(self, vertex_name: str) -> Vertex:
        return self._vertex_map[str(vertex_name)]

    def get_edge(self, edge_name_tuple: tuple[str, str]) -> Edge:
        return self._edge_map[(str(edge_name_tuple[0]), str(edge_name_tuple[1]))]
    
    def w(self, edge_name_tuple: tuple[str, str]) -> int:
        u_name, v_name = edge_name_tuple
        u_name = str(u_name)
        v_name = str(v_name)
        if (u_name, v_name) not in self._edge_map:
            return -1
        else:
            u_idx = self._vertex_indices[u_name]
            v_idx = self._vertex_indices[v_name]
            return self._wmatrix[u_idx][v_idx]

    def path_existence(self, s_vertex_name: str, t_vertex_name: str) -> bool:
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

        s_vertex_name = str(s_vertex_name)
        t_vertex_name = str(t_vertex_name)
        if s_vertex_name not in self._vertex_map:
            print(f"The vertex {s_vertex_name} does not exist, aborting.")
            return False
        if t_vertex_name not in self._vertex_map:
            print(f"The vertex {t_vertex_name} does not exist, aborting.")
            return False

        visited = set()  # track visited vertices
        original_start_vertex = self._vertex_map[s_vertex_name]
        target_vertex = self._vertex_map[t_vertex_name]
        visited.add(original_start_vertex)

        # Step 2: repeat until no more vertices can be marked
        marked = True
        while marked:
            marked = False
            for edge in self._edges:
                # If edge (u, v) exists, and u is visited but v is not
                # print(f'Considering edge ({edge.start}, {edge.end})')
                if edge.start in visited and edge.end not in visited:
                    # print(f'{edge.start} is visited but {edge.end} is not, adding {edge.end} to visited set')
                    visited.add(edge.end)
                    marked = True

        # Step 3: check if t is visited
        return target_vertex in visited

    def degree(self, vertex_name: str) -> int:
        """
        Calculate the degree of a given vertex.

        Parameters:
            vertex_name (str): The name of the vertex whose degree is being calculated.

        Return:
            int: The degree of the vertex.
            -1: If the vertex does not exist in the graph.
        """

        vertex_name = str(vertex_name)
        if vertex_name not in self._vertex_map:
            print(f"The vertex {vertex_name} does not exist in the graph, aborting.")
            return -1
        else:
            in_degrees = np.sum(self._umatrix, axis=0)
            out_degrees = np.sum(self._umatrix, axis=1)
            idx = self._vertex_indices[vertex_name]

            return in_degrees[idx] + out_degrees[idx]
        
    def tdegree(self):
        """
        Calculate the total degree of the graph.

        Return:
            int: The degree of the vertex.
        """
        return np.sum(self._umatrix)

    def bfs(self, s_vertex_name: str) -> list[list[Vertex]]:
        start_name = str(s_vertex_name)
        if start_name not in self._vertex_map:
            print(f"Vertex {start_name} does not exist, abort.")
            return None

        vertex_indices = {vertex.name: idx for idx, vertex in enumerate(self._vertices)}
        start_vertex = self._vertex_map[start_name]
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
            for i in range(len(self._vertices)):
                if (self._umatrix[vertex_idx, i] == 1) and (not visited[i]):
                    q.append(self._vertices[i])
                    visited[i] = True

        return result

    def dfs(self, start_name, visited=None, result=None):
        if visited is None:
            visited = [False] * len(self._vertices)
        if result is None:
            result = []

        start_name = str(start_name)
        if start_name not in self._vertex_map:
            print(f"Vertex {start_name} does not exist. Aborting.")
            return []

        vertex_indices = {vertex.name: idx for idx, vertex in enumerate(self._vertices)}
        start_idx = vertex_indices[start_name]

        # mark the current vertex as visited
        result.append(self._vertex_map[start_name].name)
        visited[start_idx] = True

        size = len(self._vertices)
        for i in range(size):
            if (self._umatrix[start_idx][i] == 1) and (not visited[i]):
                self.dfs(self._vertices[i].name, visited, result)

        return result

    def wcc(self):
        components = []

        visited = [False] * len(self._vertices)

        for i in range(len(self._vertices)):
            if visited[i]:
                continue
            visited_vertex_name = self._vertices[i].name
            results = self.dfs(visited_vertex_name, visited)
            # print(f"Results: {results}")

            components.append(results)
        return components, len(components)
    
    def prim(self, start_vertex: str):
        import heapq
        start_vertex = str(start_vertex)
        v1 = set()  # vertices included in the MST
        e1 = set()  # edges included in the MST
        priority_queue = []  # min-heap for edges
        
        v1.add(start_vertex)
        
        # add all edges from the start vertex to the priority queue
        start_idx = self._vertex_indices[start_vertex]
        for neighbor_idx, weight in enumerate(self._wmatrix[start_idx]):
            if weight > 0:  # Edge exists
                neighbor_name = self._vertices[neighbor_idx].name
                heapq.heappush(priority_queue, (weight, start_vertex, neighbor_name))
        
        # process the priority queue
        while priority_queue:
            weight, u, w = heapq.heappop(priority_queue)
            if w not in v1:
                v1.add(w)
                e = self._edge_map[(u, w)]
                e1.add(e)
                
                # add all edges from the new vertex to the priority queue
                w_idx = self._vertex_indices[w]
                for neighbor_idx, weight in enumerate(self._wmatrix[w_idx]):
                    if weight > 0:  # Edge exists
                        neighbor_name = self._vertices[neighbor_idx].name
                        if neighbor_name not in v1:
                            heapq.heappush(priority_queue, (weight, w, neighbor_name))
        
        return v1, e1
    
    def find(self, parent, vertex_name):
        """Find the root of the set containing the vertex with path compression."""
        if parent[vertex_name] != vertex_name:
            parent[vertex_name] = self.find(parent, parent[vertex_name])
        return parent[vertex_name]

    def union(self, parent, rank, root1, root2):
        """Union by rank."""
        if rank[root1] > rank[root2]:
            parent[root2] = root1
        elif rank[root1] < rank[root2]:
            parent[root1] = root2
        else:
            parent[root2] = root1
            rank[root1] += 1

    def kruskal(self):
        # Step 1: Sort edges by weight
        sorted_edges = sorted(self._edges, key=lambda edge: self._weight_map[(edge.start.name, edge.end.name)])
        
        # Step 2: Initialize union-find structures
        parent = {vertex.name: vertex.name for vertex in self._vertices}
        rank = {vertex.name: 0 for vertex in self._vertices}
        
        mst = []  # List to store edges in the MST
        
        # Step 3: Process sorted edges
        for edge in sorted_edges:
            root_u = self.find(parent, edge.start.name)
            root_v = self.find(parent, edge.end.name)

            # If u and v are in different components, add edge to the MST
            if root_u != root_v:
                mst.append(edge)
                self.union(parent, rank, root_u, root_v)

                # Stop early if MST is complete
                if len(mst) == len(self._vertices) - 1:
                    break

        return mst




class DGraph(Graph):
    def __init__(self):
        super().__init__()
        self._dmatrix = None

        self._in_degrees = []
        self._out_degrees = []

    def update(self):
        super().update()
        self.make_dmatrix()
        self.cal_idegree()
        self.cal_odegree()

    def make_dmatrix(self):
        size = len(self._vertices)
        adj_matrix = np.zeros((size, size), dtype=int)
        vertex_indices = {
            vertex.name: idx for idx, vertex in enumerate(self._vertices)
        }  # ensure each vertex has a unique row & column

        for edge in self._edges:
            start_idx = vertex_indices[edge.start.name]
            end_idx = vertex_indices[edge.end.name]
            adj_matrix[start_idx][end_idx] = 1

        self._dmatrix = adj_matrix

    def degree(self, vertex_name: str) -> int:
        """
        Calculate the degree of a given vertex.

        Parameters:
            vertex_name (str): The name of the vertex whose degree is being calculated.

        Return:
            int: The degree of the vertex.
            -1: If the vertex does not exist in the graph.
        """

        vertex_name = str(vertex_name)
        if vertex_name not in self._vertex_map:
            print(f"The vertex {vertex_name} does not exist in the graph, aborting.")
            return -1
        else:
            in_degrees = np.sum(self._dmatrix, axis=0)
            out_degrees = np.sum(self._dmatrix, axis=1)
            idx = self._vertex_indices[vertex_name]

            return in_degrees[idx] + out_degrees[idx]

    def cal_idegree(self) -> None:
        """
        Calculate the in-degrees of the vertices.
        """
    
        in_degrees = np.sum(self._dmatrix, axis=0)
        self._in_degrees = in_degrees
        # out_degrees = np.sum(self._dmatrix, axis=1)

        # print(f"in-degrees: \n{in_degrees}")
        # print(f"out-degrees: \n{out_degrees}")

    def cal_odegree(self) -> bool:
        """
        Calculate the out-degrees of the vertices.
        """
    
        out_degrees = np.sum(self._dmatrix, axis=1)
        self._out_degrees = out_degrees
        
    def tdegree(self) -> int:
        """
        Calculate the total degree of the graph.

        Return:
            int: The degree of the vertex.
        """
        return np.sum(self._dmatrix)
    
    def idegree(self, vertex_name: str) -> int:
        vertex_name = str(vertex_name)
        if vertex_name not in  self._vertex_map:
            print(f"Vertex {vertex_name} does not exist, abort.")
            return -1
        else:
            idx = self._vertex_indices[vertex_name]
            return self._in_degrees[idx]
        
    def odegree(self, vertex_name: str) -> int:
        vertex_name = str(vertex_name)
        if vertex_name not in  self._vertex_map:
            print(f"Vertex {vertex_name} does not exist, abort.")
            return -1
        else:
            idx = self._vertex_indices[vertex_name]
            return self._out_degrees[idx]

    def show_dmatrix(self):
        vertex_names = [vertex.name for vertex in self._vertices]

        return pd.DataFrame(self._dmatrix, index=vertex_names, columns=vertex_names)
    
    def scc(self):
        matrix = self._dmatrix
        n = len(matrix)
        vertex_indices = {vertex.name: idx for idx, vertex in enumerate(self._vertices)}

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
                        dfs(self._vertices[w].name)
                order.append(vertex_name)
            
            for v in range(n):
                if not visited[v]:
                    dfs(self._vertices[v].name)
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
                current_scc.append(self._vertices[v].name)
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
            
            return sccs, len(sccs)
        
        # Kosaraju's algo
        first_order = dfs_order(matrix)
        # print(f"First order: {first_order}")
        
        transposed = transpose_matrix(matrix)
        
        sccs = find_scc(transposed, first_order)

        self.scc = sccs
        return sccs
    
    def inorder_traversal(self,node, visited=None):
        """
        Traverse the tree in the order: the leftmost child → Node → Visit remaining children recursively.
        """
        if visited is None:
            visited = set()
        node = str(node)
        idx = self._vertex_indices[node]
        visited.add(node)

        # find all children of the current node
        children = [self._vertices[i].name for i in range(len(self._dmatrix[idx])) if self._dmatrix[idx][i] == 1]
        # print(f"Children of node {node}: {children}")

        # traverse the first child (if exists)
        if children:
            self.inorder_traversal(children[0], visited)

        # visit the current node
        print(f"Visited node: {node}")

        # traverse the remaining children (if any)
        for child in children[1:]:
            self.inorder_traversal(child, visited)


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
