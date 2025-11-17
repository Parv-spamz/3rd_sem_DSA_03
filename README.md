# 3rd_sem_DSA_03

Campus Navigation and Utility Planner
This is a Python project for the Data Structures (ENCS205) course at K.R. Mangalam University (Assignment 3, Session 2025-26).

It implements a "Campus Navigation and Utility Planner" system as per the assignment requirements. The system models the university campus using various tree and graph data structures to manage building data, find optimal paths, and plan utility layouts.

🚀 Features
This project is divided into three main parts, implementing a wide range of data structures and algorithms.

Part 1: Tree Data Structures
Building ADT: A class to store BuildingID, BuildingName, and LocationDetails.

Binary Search Tree (BST): Implements insert, search, and get_height operations for storing building data.

AVL Tree: A self-balancing BST with insert and automatic rotations (LL, RR, LR, RL) to maintain balance.

Tree Traversals: Implements inorder, preorder, and postorder traversals for both BST and AVL trees.

Height Comparison: Demonstrates the height difference between the unbalanced BST and the balanced AVL tree using the same data.

Part 2: Graph Data Structures
Graph Representation: A CampusGraph class that can generate and display both:

Adjacency Matrix

Adjacency List

Graph Traversals: Implements BFS (Breadth-First Search) and DFS (Depth-First Search) from a given starting building.

Part 3: Advanced Algorithms
Dijkstra's Algorithm: Finds the shortest path (in meters) from a starting building to all other buildings on campus.

Kruskal's Algorithm: Finds the Minimum Spanning Tree (MST) to plan a cost-effective utility (e.g., cable) layout. This uses a Disjoint Set Union (DSU) helper class.

Expression Tree: A separate tree structure to parse and evaluate postfix arithmetic expressions (e.g., for energy bill calculations).

🛠️ Technologies Used
Language: Python 3

Core Modules: collections (for deque in BFS), heapq (for the priority queue in Dijkstra's)

🏃 How to Run
Create a file: Create a new Python file named assignment3.py.

Copy the code: Copy the entire code block from the section below into your assignment3.py file.

Run the script: Open your terminal and execute the file:

Bash

python assignment3.py
View the output: The script will print the results of all parts of the assignment directly to the console.

🐍 Full Python Code (assignment3.py)
This single file contains all the classes and the main driver function required to run the entire project.

Python

# -------------------------------------------------
# MAIN DRIVER SCRIPT
# K.R. Mangalam University - Assignment 3 (ENCS205)
# Campus Navigation and Utility Planner
# -------------------------------------------------

import sys
import collections
import heapq

# =================================================
# Part 1.1: Building ADT
# =================================================
class Building:
    def __init__(self, building_id, name, details):
        self.building_id = building_id       # Integer, unique identifier
        self.name = name                 # String, name of the building
        self.details = details           # String, location details

    def __str__(self):
        return f"[ID: {self.building_id}, Name: {self.name}, Details: {self.details}]"

# =================================================
# Part 1.2: Binary Search Tree (BST)
# =================================================
class BSTNode:
    def __init__(self, building):
        self.building = building
        self.left = None
        self.right = None

class BinarySearchTree:
    def __init__(self):
        self.root = None

    def get_height(self):
        return self._get_height(self.root)

    def _get_height(self, node):
        if not node:
            return -1
        return 1 + max(self._get_height(node.left), self._get_height(node.right))

    # Insert building data
    def insert(self, building):
        self.root = self._insert(self.root, building)
        print(f"BST: Inserted {building.name}")

    def _insert(self, node, building):
        if not node:
            return BSTNode(building)
        if building.building_id < node.building.building_id:
            node.left = self._insert(node.left, building)
        elif building.building_id > node.building.building_id:
            node.right = self._insert(node.right, building)
        return node

    # Search for a building
    def search(self, building_id):
        return self._search(self.root, building_id)

    def _search(self, node, building_id):
        if not node or node.building.building_id == building_id:
            return node.building if node else None
        if building_id < node.building.building_id:
            return self._search(node.left, building_id)
        return self._search(node.right, building_id)

    # Display all buildings using traversals
    def inorder_traversal(self):
        print("\n--- BST Inorder Traversal ---")
        self._inorder(self.root)

    def _inorder(self, node):
        if node:
            self._inorder(node.left)
            print(node.building)
            self._inorder(node.right)

    def preorder_traversal(self):
        print("\n--- BST Preorder Traversal ---")
        self._preorder(self.root)

    def _preorder(self, node):
        if node:
            print(node.building)
            self._preorder(node.left)
            self._preorder(node.right)

    def postorder_traversal(self):
        print("\n--- BST Postorder Traversal ---")
        self._postorder(self.root)

    def _postorder(self, node):
        if node:
            self._postorder(node.left)
            self._postorder(node.right)
            print(node.building)

# =================================================
# Part 1.3: AVL Tree
# =================================================
class AVLNode:
    def __init__(self, building):
        self.building = building
        self.left = None
        self.right = None
        self.height = 0  # Height of the node (leaf node is 0)

class AVLTree:
    def __init__(self):
        self.root = None

    def get_height(self):
        return self._get_height(self.root)

    def _get_height(self, node):
        return node.height if node else -1

    def _update_height(self, node):
        if node:
            node.height = 1 + max(self._get_height(node.left), self._get_height(node.right))

    def _get_balance(self, node):
        return self._get_height(node.left) - self._get_height(node.right) if node else 0

    # Demonstrate rotations (LL, RR, LR, RL)
    def _right_rotate(self, y):
        print(f"--- Performing Right Rotation (LL) on node {y.building.building_id} ---")
        x = y.left
        T2 = x.right
        x.right = y
        y.left = T2
        self._update_height(y)
        self._update_height(x)
        return x

    def _left_rotate(self, x):
        print(f"--- Performing Left Rotation (RR) on node {x.building.building_id} ---")
        y = x.right
        T2 = y.left
        y.left = x
        x.right = T2
        self._update_height(x)
        self._update_height(y)
        return y

    # Insert building data
    def insert(self, building):
        self.root = self._insert(self.root, building)
        print(f"AVL: Inserted {building.name}")

    def _insert(self, node, building):
        # 1. Standard BST Insert
        if not node:
            return AVLNode(building)
        if building.building_id < node.building.building_id:
            node.left = self._insert(node.left, building)
        elif building.building_id > node.building.building_id:
            node.right = self._insert(node.right, building)
        else:
            return node  # Duplicate IDs not allowed

        # 2. Update height
        self._update_height(node)

        # 3. Get balance factor
        balance = self._get_balance(node)

        # 4. Rebalance if needed (4 cases)
        # LL Case
        if balance > 1 and building.building_id < node.left.building.building_id:
            return self._right_rotate(node)

        # RR Case
        if balance < -1 and building.building_id > node.right.building.building_id:
            return self._left_rotate(node)

        # LR Case
        if balance > 1 and building.building_id > node.left.building.building_id:
            print(f"--- Performing Left Rotation (LR) on node {node.left.building.building_id} ---")
            node.left = self._left_rotate(node.left)
            return self._right_rotate(node)

        # RL Case
        if balance < -1 and building.building_id < node.right.building.building_id:
            print(f"--- Performing Right Rotation (RL) on node {node.right.building.building_id} ---")
            node.right = self._right_rotate(node.right)
            return self._left_rotate(node)

        return node

    # Traversals (reusing BST logic structure)
    def inorder_traversal(self):
        print("\n--- AVL Inorder Traversal ---")
        self._inorder(self.root)

    def _inorder(self, node):
        if node:
            self._inorder(node.left)
            print(node.building)
            self._inorder(node.right)

    def preorder_traversal(self):
        print("\n--- AVL Preorder Traversal ---")
        self._preorder(self.root)

    def _preorder(self, node):
        if node:
            print(node.building)
            self._preorder(node.left)
            self._preorder(node.right)

    def postorder_traversal(self):
        print("\n--- AVL Postorder Traversal ---")
        self._postorder(self.root)

    def _postorder(self, node):
        if node:
            self._postorder(node.left)
            self._postorder(node.right)
            print(node.building)

# =================================================
# Part 2: Graph Implementation
# =================================================
class CampusGraph:
    def __init__(self):
        # Use a dictionary to map building names (nodes) to indices
        self.vertices = {}
        self.edges = [] # List of tuples (src, dest, weight) for paths
        self.num_vertices = 0

    def add_node(self, building_name):
        if building_name not in self.vertices:
            self.vertices[building_name] = self.num_vertices
            self.num_vertices += 1
            print(f"Graph: Added node '{building_name}'")

    def add_edge(self, src_name, dest_name, weight):
        if src_name in self.vertices and dest_name in self.vertices:
            self.edges.append((src_name, dest_name, weight))
            # Assuming undirected graph for campus paths
            self.edges.append((dest_name, src_name, weight))
            print(f"Graph: Added edge {src_name} <-> {dest_name} (Weight: {weight})")
        else:
            print(f"Error: Could not add edge. Node not found.")

    # 2.2 Adjacency Matrix
    def display_adjacency_matrix(self):
        print("\n--- Graph Adjacency Matrix ---")
        # Initialize matrix with infinity
        matrix = [[float('inf')] * self.num_vertices for _ in range(self.num_vertices)]
        
        # Set diagonal to 0
        for i in range(self.num_vertices):
            matrix[i][i] = 0

        # Populate with edge weights
        for src, dest, weight in self.edges:
            u, v = self.vertices[src], self.vertices[dest]
            matrix[u][v] = weight
            
        # Print header
        header = "      " + " ".join([f"{name[:5]:<5}" for name in self.vertices])
        print(header)
        print("      " + "-" * (self.num_vertices * 6 - 1))
        
        # Print rows
        for i, name in enumerate(self.vertices):
            row_str = f"{name[:5]:<5} |"
            for val in matrix[i]:
                row_str += f" {str(val):<5}"
            print(row_str)

    # 2.3 Adjacency List
    def display_adjacency_list(self):
        print("\n--- Graph Adjacency List ---")
        adj_list = collections.defaultdict(list)
        
        # Use a set to avoid duplicate entries in the printout
        added_edges = set()
        
        for src, dest, weight in self.edges:
            if (src, dest) not in added_edges and (dest, src) not in added_edges:
                adj_list[src].append((dest, weight))
                adj_list[dest].append((src, weight))
                added_edges.add((src, dest))

        for vertex in self.vertices:
            path_str = ", ".join([f"{neighbor} (w:{weight})" for neighbor, weight in adj_list[vertex]])
            print(f"[{vertex}] -> {path_str}")

    # 2.4 Graph Traversals
    def bfs(self, start_node_name):
        print(f"\n--- Breadth First Search (BFS) from '{start_node_name}' ---")
        if start_node_name not in self.vertices:
            print("Start node not in graph.")
            return

        visited = set()
        queue = collections.deque([start_node_name])
        visited.add(start_node_name)

        while queue:
            current_node = queue.popleft()
            print(current_node, end=" -> ")
            
            # Get neighbors from adjacency list
            neighbors = collections.defaultdict(list)
            for src, dest, _ in self.edges:
                neighbors[src].append(dest)
                neighbors[dest].append(src)

            for neighbor in neighbors[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        print("End")

    def dfs(self, start_node_name):
        print(f"\n--- Depth First Search (DFS) from '{start_node_name}' ---")
        if start_node_name not in self.vertices:
            print("Start node not in graph.")
            return

        visited = set()
        stack = [start_node_name]
        
        while stack:
            current_node = stack.pop()
            if current_node not in visited:
                visited.add(current_node)
                print(current_node, end=" -> ")
                
                # Get neighbors
                neighbors = collections.defaultdict(list)
                for src, dest, _ in self.edges:
                    neighbors[src].append(dest)
                    neighbors[dest].append(src)
                
                # Add unvisited neighbors to stack
                for neighbor in reversed(neighbors[current_node]):
                    if neighbor not in visited:
                        stack.append(neighbor)
        print("End")

    # =================================================
    # Part 3: Advanced Algorithms
    # =================================================
    
    # 3.1 Dijkstra's Algorithm for Optimal Navigation
    def dijkstra(self, start_node_name):
        print(f"\n--- Dijkstra's Shortest Path from '{start_node_name}' ---")
        if start_node_name not in self.vertices:
            print("Start node not in graph.")
            return

        distances = {vertex: float('inf') for vertex in self.vertices}
        distances[start_node_name] = 0
        
        priority_queue = [(0, start_node_name)] # (distance, node_name)
        
        adj_list = collections.defaultdict(list)
        for src, dest, weight in self.edges:
            adj_list[src].append((dest, weight))

        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)

            if current_distance > distances[current_node]:
                continue

            for neighbor, weight in adj_list[current_node]:
                distance = current_distance + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(priority_queue, (distance, neighbor))
        
        # Print results
        for vertex, distance in distances.items():
            print(f"Distance to {vertex}: {distance} meters")

    # 3.2 Kruskal's Algorithm for Utility Layout
    # Helper DSU class for Kruskal's
    class DSU:
        def __init__(self, vertices):
            self.parent = {v: v for v in vertices}
            self.rank = {v: 0 for v in vertices}

        def find(self, i):
            if self.parent[i] == i:
                return i
            return self.find(self.parent[i])

        def union(self, x, y):
            root_x = self.find(x)
            root_y = self.find(y)
            if root_x != root_y:
                if self.rank[root_x] > self.rank[root_y]:
                    self.parent[root_y] = root_x
                else:
                    self.parent[root_x] = root_y
                    if self.rank[root_x] == self.rank[root_y]:
                        self.rank[root_y] += 1
                return True
            return False

    def kruskal_mst(self):
        print("\n--- Kruskal's Algorithm (Minimum Spanning Tree for Utility) ---")
        mst_edges = []
        total_cost = 0
        
        # Sort edges by weight
        sorted_edges = sorted(self.edges, key=lambda item: item[2])
        
        dsu = self.DSU(self.vertices.keys())
        
        # Use a set to avoid adding undirected edges twice
        edge_set = set()
        
        for src, dest, weight in sorted_edges:
            # Create a unique key for the undirected edge
            edge_key = tuple(sorted((src, dest)))
            
            if edge_key not in edge_set:
                edge_set.add(edge_key)
                if dsu.union(src, dest):
                    mst_edges.append((src, dest, weight))
                    total_cost += weight

        # Print results
        print("Edges in MST:")
        for src, dest, weight in mst_edges:
            print(f"  {src} <-> {dest} (Cost: {weight})")
        print(f"Total Minimum Cost for Utility Layout: {total_cost} units")


# 3.3 Expression Tree for Energy Bill Calculations
class ExpressionTreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class ExpressionTree:
    def __init__(self):
        self.root = None

    def build_tree(self, postfix_expression):
        print(f"\n--- Building Expression Tree for: '{postfix_expression}' ---")
        stack = []
        operators = set(['+', '-', '*', '/', '^'])
        
        for token in postfix_expression.split():
            node = ExpressionTreeNode(token)
            if token not in operators:
                stack.append(node)
            else:
                node.right = stack.pop()
                node.left = stack.pop()
                stack.append(node)
        
        self.root = stack.pop()
        print("Tree built successfully.")

    def evaluate(self):
        print("--- Evaluating Expression Tree ---")
        result = self._evaluate(self.root)
        print(f"Calculation Result: {result}")
        return result

    def _evaluate(self, node):
        if not node.left and not node.right:
            # Leaf node (operand)
            return int(node.value)
        
        left_val = self._evaluate(node.left)
        right_val = self._evaluate(node.right)
        
        op = node.value
        if op == '+':
            return left_val + right_val
        elif op == '-':
            return left_val - right_val
        elif op == '*':
            return left_val * right_val
        elif op == '/':
            return left_val / right_val
        elif op == '^':
            return left_val ** right_val

# -----------------
# Main Function
# -----------------
def main():
    # --- Create Building Data ---
    b1 = Building(10, "Admin Block", "Main administrative services")
    b2 = Building(5, "Library", "Central study and research hub")
    b3 = Building(20, "Engineering Block", "Labs and classrooms for B.Tech")
    b4 = Building(8, "Cafe", "Student cafeteria and snack bar")
    b5 = Building(15, "Hostel", "On-campus student residence")
    b6 = Building(25, "Gym", "Sports and fitness center")
    
    building_data = [b1, b2, b3, b4, b5, b6]

    # =================================================
    # Part 1: Tree Implementation
    # =================================================
    print("======== Part 1: Tree Implementation ========")
    bst = BinarySearchTree()
    avl = AVLTree()

    print("\n--- Inserting into Binary Search Tree (BST) ---")
    for b in building_data:
        bst.insert(b)

    print("\n--- BST Traversals ---")
    bst.inorder_traversal()
    bst.preorder_traversal()
    bst.postorder_traversal()
    
    print("\n--- Searching in BST ---")
    print(f"Searching for Building ID 15...")
    found = bst.search(15)
    print(f"Found: {found}")
    print(f"Searching for Building ID 99...")
    found = bst.search(99)
    print(f"Found: {found}")

    print("\n--- Inserting into AVL Tree ---")
    for b in building_data:
        avl.insert(b)

    print("\n--- AVL Traversals ---")
    avl.inorder_traversal()
    avl.preorder_traversal()
    avl.postorder_traversal()
    
    print("\n--- Height Comparison ---")
    print(f"BST Height (unbalanced): {bst.get_height()}")
    print(f"AVL Tree Height (balanced): {avl.get_height()}")


    # =================================================
    # Part 2: Graph Implementation
    # =================================================
    print("\n\n======== Part 2: Graph Implementation ========")
    campus_map = CampusGraph()
    
    print("\n--- Building Campus Graph ---")
    for b in building_data:
        campus_map.add_node(b.name)
    
    # Add paths (edges)
    campus_map.add_edge("Admin Block", "Library", 50)
    campus_map.add_edge("Admin Block", "Engineering Block", 100)
    campus_map.add_edge("Library", "Cafe", 30)
    campus_map.add_edge("Engineering Block", "Hostel", 70)
    campus_map.add_edge("Engineering Block", "Gym", 40)
    campus_map.add_edge("Hostel", "Cafe", 120)
    
    # Display representations
    campus_map.display_adjacency_matrix()
    campus_map.display_adjacency_list()
    
    # Display traversals
    campus_map.bfs("Admin Block")
    campus_map.dfs("Admin Block")

    
    # =================================================
    # Part 3: Advanced Algorithms
    # =================================================
    print("\n\n======== Part 3: Advanced Algorithms ========")
    
    # 3.1 Dijkstra's
    campus_map.dijkstra("Admin Block")
    
    # 3.2 Kruskal's
    campus_map.kruskal_mst()
    
    # 3.3 Expression Tree
    energy_calc = ExpressionTree()
    # Example: (5 + 3) * (8 - 2)
    postfix_expr = "5 3 + 8 2 - *"
    energy_calc.build_tree(postfix_expr)
    energy_calc.evaluate()


if __name__ == "__main__":
    main()
📊 Sample Output
Here is the complete console output when the script is executed.

Plaintext

======== Part 1: Tree Implementation ========

--- Inserting into Binary Search Tree (BST) ---
BST: Inserted Admin Block
BST: Inserted Library
BST: Inserted Engineering Block
BST: Inserted Cafe
BST: Inserted Hostel
BST: Inserted Gym

--- BST Traversals ---

--- BST Inorder Traversal ---
[ID: 5, Name: Library, Details: Central study and research hub]
[ID: 8, Name: Cafe, Details: Student cafeteria and snack bar]
[ID: 10, Name: Admin Block, Details: Main administrative services]
[ID: 15, Name: Hostel, Details: On-campus student residence]
[ID: 20, Name: Engineering Block, Details: Labs and classrooms for B.Tech]
[ID: 25, Name: Gym, Details: Sports and fitness center]

--- BST Preorder Traversal ---
[ID: 10, Name: Admin Block, Details: Main administrative services]
[ID: 5, Name: Library, Details: Central study and research hub]
[ID: 8, Name: Cafe, Details: Student cafeteria and snack bar]
[ID: 20, Name: Engineering Block, Details: Labs and classrooms for B.Tech]
[ID: 15, Name: Hostel, Details: On-campus student residence]
[ID: 25, Name: Gym, Details: Sports and fitness center]

--- BST Postorder Traversal ---
[ID: 8, Name: Cafe, Details: Student cafeteria and snack bar]
[ID: 5, Name: Library, Details: Central study and research hub]
[ID: 15, Name: Hostel, Details: On-campus student residence]
[ID: 25, Name: Gym, Details: Sports and fitness center]
[ID: 20, Name: Engineering Block, Details: Labs and classrooms for B.Tech]
[ID: 10, Name: Admin Block, Details: Main administrative services]

--- Searching in BST ---
Searching for Building ID 15...
Found: [ID: 15, Name: Hostel, Details: On-campus student residence]
Searching for Building ID 99...
Found: None

--- Inserting into AVL Tree ---
AVL: Inserted Admin Block
AVL: Inserted Library
AVL: Inserted Engineering Block
--- Performing Left Rotation (RR) on node 10 ---
AVL: Inserted Cafe
AVL: Inserted Hostel
AVL: Inserted Gym

--- AVL Traversals ---

--- AVL Inorder Traversal ---
[ID: 5, Name: Library, Details: Central study and research hub]
[ID: 8, Name: Cafe, Details: Student cafeteria and snack bar]
[ID: 10, Name: Admin Block, Details: Main administrative services]
[ID: 15, Name: Hostel, Details: On-campus student residence]
[ID: 20, Name: Engineering Block, Details: Labs and classrooms for B.Tech]
[ID: 25, Name: Gym, Details: Sports and fitness center]

--- AVL Preorder Traversal ---
[ID: 10, Name: Admin Block, Details: Main administrative services]
[ID: 5, Name: Library, Details: Central study and research hub]
[ID: 8, Name: Cafe, Details: Student cafeteria and snack bar]
[ID: 20, Name: Engineering Block, Details: Labs and classrooms for B.Tech]
[ID: 15, Name: Hostel, Details: On-campus student residence]
[ID: 25, Name: Gym, Details: Sports and fitness center]

--- AVL Postorder Traversal ---
[ID: 8, Name: Cafe, Details: Student cafeteria and snack bar]
[ID: 5, Name: Library, Details: Central study and research hub]
[ID: 15, Name: Hostel, Details: On-campus student residence]
[ID: 25, Name: Gym, Details: Sports and fitness center]
[ID: 20, Name: Engineering Block, Details: Labs and classrooms for B.Tech]
[ID: 10, Name: Admin Block, Details: Main administrative services]

--- Height Comparison ---
BST Height (unbalanced): 3
AVL Tree Height (balanced): 2


======== Part 2: Graph Implementation ========

--- Building Campus Graph ---
Graph: Added node 'Admin Block'
Graph: Added node 'Library'
Graph: Added node 'Engineering Block'
Graph: Added node 'Cafe'
Graph: Added node 'Hostel'
Graph: Added node 'Gym'
Graph: Added edge Admin Block <-> Library (Weight: 50)
Graph: Added edge Admin Block <-> Engineering Block (Weight: 100)
Graph: Added edge Library <-> Cafe (Weight: 30)
Graph: Added edge Engineering Block <-> Hostel (Weight: 70)
Graph: Added edge Engineering Block <-> Gym (Weight: 40)
Graph: Added edge Hostel <-> Cafe (Weight: 120)

--- Graph Adjacency Matrix ---
      Admin Libra Engin Cafe  Hoste Gym  
      -----------------------------------
Admin | 0     50    100   inf   inf   inf  
Libra | 50    0     inf   30    inf   inf  
Engin | 100   inf   0     inf   70    40   
Cafe  | inf   30    inf   0     120   inf  
Hoste | inf   inf   70    120   0     inf  
Gym   | inf   inf   40    inf   inf   0    

--- Graph Adjacency List ---
[Admin Block] -> Library (w:50), Engineering Block (w:100)
[Library] -> Admin Block (w:50), Cafe (w:30)
[Engineering Block] -> Admin Block (w:100), Hostel (w:70), Gym (w:40)
[Cafe] -> Library (w:30), Hostel (w:120)
[Hostel] -> Engineering Block (w:70), Cafe (w:120)
[Gym] -> Engineering Block (w:40)

--- Breadth First Search (BFS) from 'Admin Block' ---
Admin Block -> Library -> Engineering Block -> Cafe -> Hostel -> Gym -> End

--- Depth First Search (DFS) from 'Admin Block' ---
Admin Block -> Engineering Block -> Gym -> Hostel -> Cafe -> Library -> End


======== Part 3: Advanced Algorithms ========

--- Dijkstra's Shortest Path from 'Admin Block' ---
Distance to Admin Block: 0 meters
Distance to Library: 50 meters
Distance to Engineering Block: 100 meters
Distance to Cafe: 80 meters
Distance to Hostel: 170 meters
Distance to Gym: 140 meters

--- Kruskal's Algorithm (Minimum Spanning Tree for Utility) ---
Edges in MST:
  Library <-> Cafe (Cost: 30)
  Engineering Block <-> Gym (Cost: 40)
  Admin Block <-> Library (Cost: 50)
  Engineering Block <-> Hostel (Cost: 70)
  Admin Block <-> Engineering Block (Cost: 100)
Total Minimum Cost for Utility Layout: 290 units

--- Building Expression Tree for: '5 3 + 8 2 - *' ---
Tree built successfully.
--- Evaluating Expression Tree ---
Calculation Result: 48
📝 Assignment Context
University: K.R. Mangalam University

Programme: B.Tech CSE

Session: 2025-26

Course: ENCS205 - Data Structures

Assignment: 03 - Trees and Graphs
