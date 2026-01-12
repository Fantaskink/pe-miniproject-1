from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
import random
import matplotlib.pyplot as plt

class Topology(Enum):
    RING = 1,
    TREE = 2,
    COMPLETE = 3,
    STAR = 4


class Node:
    def __init__(self, index: int, value: float) -> None:
        self.index = index
        self.initial_value = value
        self.value = value
        self.neighbours: set['Node'] = set()

    def add_neighbour(self, neighbour: 'Node') -> None:
        self.neighbours.add(neighbour)

    def __str__(self) -> str:
        return f"Index: {self.index}, Value: {self.value:.4f}, Neighbours: {[node.index for node in self.neighbours]}"


class Network(ABC):
    def __init__(self, topology: Topology) -> None:
        self.nodes = self._create_network_nodes(topology)
        self.true_average = self._calculate_true_average()

    def _create_network_nodes(self, topology: Topology) -> list[Node]:
        if topology == Topology.TREE:
            return self._create_tree_topology()
        elif topology == Topology.RING:
            return self._create_ring_topology()
        elif topology == Topology.COMPLETE:
            return self._create_complete_topology()
        elif topology == Topology.STAR:
            return self._create_star_topology()
        
    def _create_tree_topology(self):
        n_nodes = random.randint(10,100)
        network : list[Node] = []

        for i in range(n_nodes):
            if len(network) == 0:
                new_node = Node(i, random.randint(10, 100))
                network.append(new_node)
                continue

            connected_node = random.choice(network)
            new_node = Node(i, random.randint(10, 100))
            connected_node.add_neighbour(new_node)
            new_node.add_neighbour(connected_node)
            network.append(new_node)
        
        return network
    
    def _create_ring_topology(self):
        n_nodes = random.randint(10,100)
        network : list[Node] = []

        for i in range(n_nodes):
            new_node = Node(i, random.randint(10, 100))
            
            # FIX 1: Only connect if network is not empty
            if network:
                last_node = network[-1]
                last_node.add_neighbour(new_node)
                new_node.add_neighbour(last_node)
                
            network.append(new_node) # Append the new node here

        # Connect the last node to the first node to close the ring
        first = network[0]
        last = network[-1]
        first.add_neighbour(last)
        last.add_neighbour(first)
        
        return network
    
    def _create_complete_topology(self):
        n_nodes = random.randint(10,100)
        network : list[Node] = []

        for i in range(n_nodes):
            new_node = Node(i, random.randint(10, 100))
            
            # Connect the new node to ALL existing nodes
            for existing_node in network:
                existing_node.add_neighbour(new_node)
                new_node.add_neighbour(existing_node)
                
            network.append(new_node) # Add the new node to the list of existing nodes
        
        return network
    
    def _create_star_topology(self):
        n_nodes = random.randint(10,100)
        network : list[Node] = []

        for i in range(n_nodes):    
            new_node = Node(i, random.randint(10, 100))
            network.append(new_node)
            
        # Choose the central hub node
        hub_node = random.choice(network) 

        for node in network:
            # FIX: Skip connecting the hub node to itself
            if node is not hub_node:
                node.add_neighbour(hub_node)
                hub_node.add_neighbour(node)
        
        return network

    
    def _calculate_true_average(self) -> float:
        sum = 0
        for node in self.nodes:
            sum += node.initial_value
        
        return sum / len(self.nodes)
    
    def share_random_numbers(self) -> None:
        shared_pairs = set() 
        
        for node in self.nodes:
            for neighbour in node.neighbours:
                if (node.index, neighbour.index) not in shared_pairs and \
                   (neighbour.index, node.index) not in shared_pairs:
                    
                    random_number = random.randint(10, 100)
                    
                    node.value -= random_number
                    neighbour.value += random_number
                    
                    shared_pairs.add((node.index, neighbour.index))

    @abstractmethod
    def exchange(self):
        pass

    def get_max_error(self) -> float:
        """Calculates the maximum error among all nodes."""
        current_values = np.array([node.value for node in self.nodes])
        # Max difference between any node value and the true average
        return np.max(np.abs(current_values - self.true_average))


class SynchronousNetwork(Network):
    def __init__(self, topology: Topology) -> None:
        super().__init__(topology)
        self.weight_matrix = self.get_weight_matrix()

    def exchange(self) -> None:
        """
        Performs one synchronous consensus step: x(t+1) = W * x(t).
        """
        # 1. Get the vector of current values x(t)
        current_values = np.array([node.value for node in self.nodes])
        
        # 2. Calculate the new values x(t+1) = W * x(t)
        new_values = self.weight_matrix @ current_values
        
        # 3. Update the node values (synchronous update)
        for i, node in enumerate(self.nodes):
            node.value = new_values[i]

    def get_weight_matrix(self) -> np.array:
        degree_matrix = self._get_degree_matrix()
        adjacency_matrix = self._get_adjacency_matrix()
        laplacian_matrix = degree_matrix - adjacency_matrix
        
        d_max = np.max(np.diag(degree_matrix)) # Largest value in the degree matrix
        alpha = 1 / (d_max + 1) 

        identity_matrix = np.identity(len(degree_matrix))
        return identity_matrix - (alpha * laplacian_matrix)
        

    def _get_degree_matrix(self) -> np.array:
        n_nodes = len(self.nodes)
        D = np.zeros((n_nodes, n_nodes))
        
        for node in self.nodes:
            index = node.index
            degree = len(node.neighbours)
            D[index, index] = degree
        
        return D
    
    def _get_adjacency_matrix(self) -> np.array:
        n_nodes = len(self.nodes)
        A = np.zeros((n_nodes, n_nodes))
        
        for node in self.nodes:
            for neighbour in node.neighbours:
                A[node.index, neighbour.index] = 1

        return A
    
class AsynchronousNetwork(Network):
    def __init__(self, topology: Topology) -> None:
        super().__init__(topology)

    def exchange(self):
        random_node = random.choice(self.nodes)
        random_neighbour = random.choice(list(random_node.neighbours))

        average = 0.5 * (random_node.value + random_neighbour.value)

        random_node.value = average
        random_neighbour.value = average


if __name__ == "__main__":
    topology = Topology.RING

    USE_ASYNC = False
    
    if USE_ASYNC:
        network = AsynchronousNetwork(topology)
        title = "Asynchronous Average Consensus Convergence"
    else:
        network = SynchronousNetwork(topology)
        title = "Synchronous Average Consensus Convergence"

    true_average = network._calculate_true_average()
    network.share_random_numbers()

    MAX_ITERATIONS = 100000
    CONVERGENCE_TOLERANCE = 1e-4
    
    errors = []
    
    print(f"Topology: {topology}")
    print(f"Total Nodes: {len(network.nodes)}")
    print(f"True Average (based on initial values): {network.true_average:.4f}")
    
    # 4. Iterative Consensus
    for t in range(MAX_ITERATIONS):
        # Perform synchronous exchange
        network.exchange()
        
        # Calculate error (difference of the true average and the computed one)
        max_error = network.get_max_error()
        errors.append(max_error)
        
        # Check for convergence
        if max_error < CONVERGENCE_TOLERANCE:
            print(f"Convergence achieved at iteration {t+1}")
            break

    # 5. Convergence Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(errors)
    plt.yscale('log') # Log scale is typical for plotting convergence error
    plt.title(title)
    plt.xlabel('Iteration (t)')
    plt.ylabel(r'Max Error $|\mathbf{x}(t) - \mu|$ (Log Scale)')
    plt.grid(True, which="both", ls="--")
    plt.show()
    
    # Final verification
    final_average = np.mean([node.value for node in network.nodes])
    print("\n--- Final Results ---")
    print(f"Final Max Error: {errors[-1]:.6e}")
    print(f"Calculated Final Average: {final_average:.4f}")
    print(f"True Average: {network.true_average:.4f}")
    print(f"Total iterations: {len(errors)}")