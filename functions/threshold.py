import networkx as nx
import numpy as np


class ThresholdModel:

    def __init__(self, graph, high=1.0, verbose=False):
        self.graph = graph
        self.iteration_dict = dict()
        self.iterations = 0
        self.node_counts = []
        self.verbose = verbose

        self.thresh_dict = {node: np.random.uniform(0.0, high) for node in graph.nodes}
        nx.set_node_attributes(graph, self.thresh_dict, "threshold")

    def run(self, n_iterations, use_int=False):
        self.iterations += 1
        graph = self.graph

        for i in range(n_iterations-1):
            update_nodes = []
            for node in graph.nodes:
                node_neighbors = graph[node]
                sum_total = 0.0
                sum_infected = 0.0
                for nbr in node_neighbors:
                    nbr_atts = graph.nodes[nbr]
                    sum_total += node_neighbors[nbr]["weight"]
                    if nbr_atts["label"] == 1:
                        sum_infected += node_neighbors[nbr]["weight"]

                thresh_check = 0.0
                if use_int or graph.nodes[node]["threshold"] > 1.0:
                    thresh_check = sum_infected
                else:
                    if sum_total > 0:
                        thresh_check = sum_infected / sum_total

                if thresh_check >= graph.nodes[node]["threshold"]:
                    update_nodes.append(node)

            for node in update_nodes:
                graph.nodes[node]["label"] = 1

            infect_nodes = [n for n in graph.nodes if graph.nodes[n]["label"] == 1]

            self.iteration_dict[self.iterations] = infect_nodes
            self.node_counts.append(len(infect_nodes))
            self.iterations += 1

            if self.verbose:
                print(f"Iteration {i + 1} complete.")

    def set_initial_config(self, infected_nodes, edge_weights=None, seed=724):
        np.random.seed(seed)
        graph = self.graph
        att_dict = {node: 0 for node in graph.nodes}

        for node in infected_nodes:
            att_dict[node] = 1

        nx.set_node_attributes(graph, att_dict, name='label')

        # TODO: allow edge weights
        weight_dict = {edge: 1.0 for edge in graph.edges}
        if edge_weights:
            pass

        nx.set_edge_attributes(graph, weight_dict, name="weight")

        self.node_counts.append(len(infected_nodes))
        self.iteration_dict[self.iterations] = infected_nodes

        # for node in graph.nodes:
        #     if node in infected_nodes:
        #         graph.nodes[node]["label"] = 1
        #     else:
        #         graph.nodes[node]["label"] = 0
        #
        #     # if node in thresholds:
        #     #     graph[node]["threshold"] = thresholds[node]
        #     # else:
        #     #     graph[node]["threshold"] = np.random.uniform(0, 1)
        #
        #     if edge_weights:
        #         pass
        #     else:
        #         for edge in graph.edges:
        #             graph.edges[edge]["weight"] = 1.0
        #
        # self.node_counts.append(len(infected_nodes))

    def set_thresholds(self, labeled_users, thresholds):
        graph = self.graph

        thresh_dict = {}
        for i, node in enumerate(labeled_users):
            thresh_dict[node] = thresholds[i]

        nx.set_node_attributes(graph, thresh_dict, "threshold")

        # for node in graph.nodes:
        #     # graph.nodes[l]["threshold"] = np.random.uniform(0.0, np.max(thresholds))
        #     graph.nodes[node] = np.random.uniform(0.0, 1.0)
        # for i, l in enumerate(labeled_users):
        #     if random:
        #         graph.nodes[l]["threshold"] = np.random.uniform(0.0, 1.0)
        #     else:
        #         graph.nodes[l]["threshold"] = thresholds[i]

    def set_thresholds_value(self, threshold):
        graph = self.graph

        thresh_dict = dict()
        for i, node in enumerate(graph.nodes):
            thresh_dict[node] = threshold

        nx.set_node_attributes(graph, thresh_dict, "threshold")

        return np.array(list(thresh_dict.values()))

    def set_thresholds_estimation(self, threshold_min, threshold_max):
        graph = self.graph

        thresh_dict = dict()
        for i, node in enumerate(graph.nodes):
            if threshold_min >= 1.0:
                thresh_dict[node] = np.random.randint(threshold_min, threshold_max)
            else:
                thresh_dict[node] = np.random.uniform(threshold_min, threshold_max)

        nx.set_node_attributes(graph, thresh_dict, "threshold")

        return np.array(list(thresh_dict.values()))


def get_basic_graph():
    graph = nx.Graph()

    graph.add_node("A")
    graph.add_node("B")
    graph.add_node("C")

    return graph
