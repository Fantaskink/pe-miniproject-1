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
    def __init__(self, topology: Topology, n_nodes: int | None = None, value_range: tuple[int, int] = (10, 100)) -> None:
        self._n_nodes = n_nodes if n_nodes is not None else random.randint(10, 100)
        self._value_range = value_range
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
                new_node = Node(i, random.randint(self._value_range[0], self._value_range[1]))
                network.append(new_node)
                continue

            connected_node = random.choice(network)
            new_node = Node(i, random.randint(self._value_range[0], self._value_range[1]))
            connected_node.add_neighbour(new_node)
            new_node.add_neighbour(connected_node)
            network.append(new_node)
        
        return network
    
    def _create_ring_topology(self):
        n_nodes = self._n_nodes
        network : list[Node] = []

        for i in range(n_nodes):
            new_node = Node(i, random.randint(self._value_range[0], self._value_range[1]))
            
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
            new_node = Node(i, random.randint(self._value_range[0], self._value_range[1]))
            
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
            new_node = Node(i, random.randint(self._value_range[0], self._value_range[1]))
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
    def __init__(self, topology: Topology, n_nodes: int | None = None, value_range: tuple[int, int] = (10, 100)) -> None:
        super().__init__(topology, n_nodes, value_range)
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
    def __init__(self, topology: Topology, n_nodes: int | None = None, value_range: tuple[int, int] = (10, 100)) -> None:
        super().__init__(topology, n_nodes, value_range)

    def exchange(self):
        random_node = random.choice(self.nodes)
        random_neighbour = random.choice(list(random_node.neighbours))

        average = 0.5 * (random_node.value + random_neighbour.value)

        random_node.value = average
        random_neighbour.value = average


class DifferentialPrivacySynchronousNetwork(SynchronousNetwork):
    def __init__(self, topology: Topology, n_nodes: int | None = None, value_range: tuple[int, int] = (10, 100), mechanism: str = 'laplace', scale: float = 1.0) -> None:
        super().__init__(topology, n_nodes, value_range)
        self.mechanism = mechanism.lower()
        self.scale = scale

    def _sample_noise(self, size: int) -> np.ndarray:
        if self.mechanism in ('laplace', 'laplacian'):
            return np.random.laplace(loc=0.0, scale=self.scale, size=size)
        elif self.mechanism in ('gaussian', 'normal'):
            return np.random.normal(loc=0.0, scale=self.scale, size=size)
        elif self.mechanism in ('uniform',):
            return np.random.uniform(low=-self.scale, high=self.scale, size=size)
        else:
            raise ValueError(f"Unknown mechanism: {self.mechanism}")

    def apply_input_perturbation(self) -> None:
        """Add one-time noise to the current node values (input perturbation)."""
        noise = self._sample_noise(len(self.nodes))
        for i, node in enumerate(self.nodes):
            node.value = node.value + noise[i]

    def exchange(self) -> None:
        # Regular synchronous consensus step (no per-iteration DP noise)
        super().exchange()


if __name__ == "__main__":
    # --- 1. Global Settings ---
    topology_type = Topology.RING
    N_NODES = 1000
    VALUE_RANGE = (10, 100)
    MAX_ITERATIONS = 100000
    CONVERGENCE_TOLERANCE = 1e-6 
    DP_NOISE_SCALE = 2.0  # Increased slightly to make the "floor" visible

    # --- 2. Generate Ground Truth Data ---
    # We create the original values and the topology once so everyone starts 
    # from the exact same "true" state.
    base_network = SynchronousNetwork(topology_type, N_NODES, VALUE_RANGE)
    true_avg = base_network.true_average
    
    # Helper to clone the topology (nodes + connections) from the base_network
    def clone_into(target_net, source_nodes):
        target_net.nodes = [Node(n.index, n.initial_value) for n in source_nodes]
        for i in range(len(source_nodes)):
            for neighbor in source_nodes[i].neighbours:
                target_net.nodes[i].add_neighbour(target_net.nodes[neighbor.index])
        if hasattr(target_net, 'get_weight_matrix'):
            target_net.weight_matrix = target_net.get_weight_matrix()
        target_net.true_average = true_avg

    # --- 3. Setup Additive Secret Sharing (ASS) Networks ---
    # These start with massive errors because of the random number exchange.
    ass_sync = SynchronousNetwork(topology_type, N_NODES, VALUE_RANGE)
    clone_into(ass_sync, base_network.nodes)
    ass_sync.share_random_numbers() # This "scrambles" the values

    ass_async = AsynchronousNetwork(topology_type, N_NODES, VALUE_RANGE)
    clone_into(ass_async, base_network.nodes)
    # Copy the scrambled values from ass_sync so we compare sync vs async on same data
    for i in range(N_NODES):
        ass_async.nodes[i].value = ass_sync.nodes[i].value

    # --- 4. Setup Differential Privacy (DP) Networks ---
    # These start with low error (just the original values + a bit of noise).
    dp_mechanisms = ['laplace', 'gaussian', 'uniform']
    dp_networks: dict[str, DifferentialPrivacySynchronousNetwork] = {}

    for mech in dp_mechanisms:
        dp_net = DifferentialPrivacySynchronousNetwork(topology_type, N_NODES, mechanism=mech, scale=DP_NOISE_SCALE)
        clone_into(dp_net, base_network.nodes)
        # Apply noise to the CLEAN initial values (NOT the scrambled ASS values)
        dp_net.apply_input_perturbation()
        dp_networks[mech] = dp_net

    # --- 5. Execution Loops ---
    results = {
        "ASS Sync": [],
        "ASS Async": []
    }
    for mech in dp_mechanisms:
        results[f"DP: {mech.capitalize()}"] = []

    # Run ASS Sync
    while ass_sync.get_max_error() > CONVERGENCE_TOLERANCE and len(results["ASS Sync"]) < MAX_ITERATIONS:
        ass_sync.exchange()
        results["ASS Sync"].append(ass_sync.get_max_error())

    # Run ASS Async
    while ass_async.get_max_error() > CONVERGENCE_TOLERANCE and len(results["ASS Async"]) < MAX_ITERATIONS:
        ass_async.exchange()
        results["ASS Async"].append(ass_async.get_max_error())

    # Run DP Networks
    for mech, net in dp_networks.items():
        label = f"DP: {mech.capitalize()}"
        while net.get_max_error() > CONVERGENCE_TOLERANCE and len(results[label]) < MAX_ITERATIONS:
            net.exchange()
            results[label].append(net.get_max_error())

    # --- 6. Visualization ---
    plt.figure(figsize=(12, 7))
    
    # Plot ASS (High start, goes to zero)
    plt.plot(results["ASS Sync"], label='ASS (Synchronous)', color='blue', linewidth=2)
    plt.plot(results["ASS Async"], label='ASS (Asynchronous)', color='cyan', linestyle=':', alpha=0.8)

    # Plot DP (Low start, hits a floor)
    mech_styles = {
        'DP: Laplace': {'color': 'red', 'ls': '--'},
        'DP: Gaussian': {'color': 'green', 'ls': '-.'},
        'DP: Uniform': {'color': 'purple', 'ls': '--', 'alpha': 0.6},
    }
    
    for label, data in results.items():
        if "DP" in label:
            style = mech_styles.get(label, {})
            plt.plot(data, label=label, **style)

    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.title(f"Privacy-Preserving Consensus: {topology_type.name} Topology\n(ASS Scrambling vs. DP Perturbation)")
    plt.xlabel("Iterations")
    plt.ylabel("Maximum Error (Log Scale)")
    plt.legend()
    
    # Text Annotation to explain the graph
    #plt.text(len(results["ASS Sync"])*0.05, 10**-5, 
    #         "ASS: High initial error\nbut converges to 0", color='blue', fontsize=10)
    #plt.text(len(results["ASS Sync"])*0.6, DP_NOISE_SCALE*0.1, 
    #         "DP: Low initial error\nbut hits noise floor", color='red', fontsize=10)

    plt.tight_layout()
    plt.show()

    print(f"Simulation Complete. True Average was: {true_avg:.4f}")