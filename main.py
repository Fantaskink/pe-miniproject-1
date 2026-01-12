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
    def __init__(self, topology: Topology, n_nodes: int | None = None) -> None:
        self._n_nodes = n_nodes if n_nodes is not None else random.randint(10, 100)
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
        n_nodes = self._n_nodes
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
        n_nodes = self._n_nodes
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
        n_nodes = self._n_nodes
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
        n_nodes = self._n_nodes
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
    def __init__(self, topology: Topology, n_nodes: int | None = None) -> None:
        super().__init__(topology, n_nodes)
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
    def __init__(self, topology: Topology, n_nodes: int | None = None) -> None:
        super().__init__(topology, n_nodes)

    def exchange(self):
        random_node = random.choice(self.nodes)
        random_neighbour = random.choice(list(random_node.neighbours))

        average = 0.5 * (random_node.value + random_neighbour.value)

        random_node.value = average
        random_neighbour.value = average


if __name__ == "__main__":
    topology = Topology.RING
    N_NODES = 10 
    MAX_ITERATIONS = 100000
    CONVERGENCE_TOLERANCE = 1e-6 

    # 1. Create the Master Synchronous Network
    sync_network = SynchronousNetwork(topology, N_NODES)
    
    # 2. Create the Async Network (constructor creates random nodes we will replace)
    async_network = AsynchronousNetwork(topology, N_NODES)

    # 3. CLONE THE TOPOLOGY
    # Create new Node objects for Async so they have independent .value attributes
    async_network.nodes = [Node(n.index, n.initial_value) for n in sync_network.nodes]
    
    # Mirror the neighbor connections exactly from the sync_network
    for i in range(len(sync_network.nodes)):
        master_node = sync_network.nodes[i]
        copy_node = async_network.nodes[i]
        for neighbor in master_node.neighbours:
            # Connect using the index to find the corresponding 'new' node object
            copy_node.add_neighbour(async_network.nodes[neighbor.index])

    # Ensure metadata is synced
    async_network.true_average = sync_network.true_average

    # 4. Perform Secret Sharing on Sync and copy resulting state to Async
    # This ensures both start from the exact same "secret shared" values
    sync_network.share_random_numbers()
    for i in range(len(sync_network.nodes)):
        async_network.nodes[i].value = sync_network.nodes[i].value

    # 5. Independent Execution Loops
    errors_sync = []
    errors_async = []

    # Run Synchronous until it hits tolerance
    while sync_network.get_max_error() > CONVERGENCE_TOLERANCE:
        sync_network.exchange()
        errors_sync.append(sync_network.get_max_error())
        if len(errors_sync) >= MAX_ITERATIONS: break

    # Run Asynchronous until it hits tolerance
    while async_network.get_max_error() > CONVERGENCE_TOLERANCE:
        async_network.exchange()
        errors_async.append(async_network.get_max_error())
        if len(errors_async) >= MAX_ITERATIONS: break

    # 6. Final Plotting
    plt.figure(figsize=(12, 7))
    plt.plot(errors_sync, label='ASS Sync', linewidth=1.2, marker='.', markersize=2)
    plt.plot(errors_async, label='ASS Async', linewidth=1.2, marker='.', markersize=2)
    
    plt.yscale('log')
    plt.title(f'Convergence Comparison - {topology.name} Topology')
    plt.xlabel('Iterations')
    plt.ylabel('Maximum Error (Log Scale)')
    plt.legend()
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.tight_layout()
    plt.show()

    print(f"Sync converged in {len(errors_sync)} iterations.")
    print(f"Async converged in {len(errors_async)} iterations.")