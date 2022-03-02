from sklearn.datasets import make_regression
from functions.threshold import *
from functions.threshold_estimation import *
from functions.util import *
import pickle as pkl
import os
import matplotlib.pyplot as plt
from CTL.causal_tree_learn import CausalTree
from sklearn.linear_model import LinearRegression
from functions.estimators import *

import snap

append = "ijcai"


class Synthetic:

    def __init__(self, network_model="erdos-renyi", network_params=None,
                 data_folder="/AD-HOME/ctran29/data/data/synthetic/",
                 # data_folder="/AD-HOME/ctran29/data/diffusion_aaai/synthetic/",
                 # data_folder="/mnt/ext/ctran/data/diffusion/synthetic/",
                 save_folder="results/aaai_rebuttal/synthetic/",
                 n_samples=1000, seed=724, extra="", extra_fold="", threshold_function="linear", num_x=2):

        self.threshold_function = threshold_function
        self.num_x = num_x

        np.random.seed(seed)
        random.seed(seed)

        self.n_samples = n_samples
        self.seed = seed

        self.network_model = network_model
        self.network_params = network_params
        if self.network_params is not None:
            self.network_params["n"] = self.n_samples

        # if data_folder[-1] != "/" and "/" in data_folder:
        #     data_folder = f"{data_folder}/"
        # if save_folder[-1] != "/" and "/" in save_folder:
        #     save_folder = f"{save_folder}/"

        self.orig_data_folder = f"{data_folder}/{threshold_function}{extra_fold}/"
        self.orig_save_folder = f"{save_folder}/{threshold_function}{extra_fold}/"

        # self.data_folder = f"{data_folder}{self.seed}/"
        # self.save_folder = f"{save_folder}{self.seed}/"
        self.data_folder = f"{data_folder}"
        self.save_folder = f"{save_folder}"
        # self.logger = make_logger(__name__, logname="erdos_renyi")
        self.logger = None

        self.extra = extra

    def network_generation(self):
        logger = self.logger
        if self.network_model == "erdos-renyi":
            if self.network_params is not None:
                graph = nx.generators.random_graphs.erdos_renyi_graph(**self.network_params, seed=self.seed)
            else:
                graph = nx.generators.random_graphs.erdos_renyi_graph(n=1000, p=0.1, seed=self.seed)
        elif self.network_model in ['preferential-attachment', 'pref-attachment', 'pref-attach']:
            if self.network_params is not None:
                graph = nx.generators.random_graphs.barabasi_albert_graph(**self.network_params, seed=self.seed)
            else:
                graph = nx.generators.random_graphs.barabasi_albert_graph(n=1000, m=100, seed=self.seed)
            self.network_model = "pref-attach"
        elif self.network_model == "forest-fire":
            if self.network_params is not None:
                Graph = snap.GenForestFire(self.network_params["n"], self.network_params["f"], self.network_params["b"])
            else:
                Graph = snap.GenForestFire(1000, 0.5, 0.5)
            nodes = []
            edges = []
            for NI in Graph.Nodes():
                nodes.append(NI.GetId())
            for EI in Graph.Edges():
                edges.append((EI.GetSrcNId(), EI.GetDstNId()))

            graph = nx.DiGraph()
            graph.add_edges_from(edges)
        elif self.network_model == "small-world":
            if self.network_params is not None:
                graph = nx.generators.random_graphs.watts_strogatz_graph(**self.network_params, seed=self.seed)
            else:
                graph = nx.generators.random_graphs.watts_strogatz_graph(1000, k=5, p=0.1, seed=self.seed)
        else:
            logger.info(
                "Invalid network model. "
                "Valid ones are ['erdos-renyi', 'forest-fire', 'preferential-attachment', 'pref-attachment', "
                "'pref-attach', 'small-world']. "
                "Defaulting to erdos-renyi...")
            self.network_model = "erdos-renyi"
            if self.network_params is not None:
                graph = nx.generators.random_graphs.erdos_renyi_graph(**self.network_params, seed=self.seed)
            else:
                graph = nx.generators.random_graphs.erdos_renyi_graph(n=1000, p=0.1, seed=self.seed)

        self.data_folder = f"{self.orig_data_folder}{self.network_model}/{self.seed}/{self.extra}/"
        self.save_folder = f"{self.orig_save_folder}{self.network_model}/{self.seed}/{self.extra}/"
        self.logger = make_logger(__name__, logname=f"{self.network_model} - {self.threshold_function}")
        check_dir(self.data_folder)
        check_dir(self.save_folder)
        check_dir(self.orig_data_folder)
        check_dir(self.orig_save_folder)

        return graph

    def generate_graphs(self):
        logger = self.logger

        if self.threshold_function == "linear":
            x, y = make_regression(n_samples=self.n_samples,
                                   n_features=100,
                                   n_informative=10,
                                   noise=10,
                                   coef=False,
                                   random_state=10)
            # x, y = make_regression(n_samples=self.n_samples,
            #                        n_features=100,
            #                        n_informative=10,
            #                        noise=10,
            #                        coef=False,
            #                        random_state=self.seed)
            y = ((y + 1000) / 9000)
        elif self.threshold_function == "quad":
            np.random.seed(self.seed)
            x = np.random.normal(size=(1000, 10))
            q1 = (x[:, 0] >= 0) & (x[:, 1] >= 0)
            q2 = (x[:, 0] < 0) & (x[:, 1] >= 0)
            q3 = (x[:, 0] < 0) & (x[:, 1] < 0)
            q4 = (x[:, 0] >= 0) & (x[:, 1] < 0)

            y = np.zeros(1000)
            y[q1] = np.random.uniform(0.05, 0.45)
            y[q2] = np.random.uniform(0.05, 0.45)
            y[q3] = np.random.uniform(0.05, 0.45)
            y[q4] = np.random.uniform(0.05, 0.45)
        elif self.threshold_function == "quad_max":
            np.random.seed(self.seed)
            x = np.random.normal(size=(1000, 10))
            q1 = (x[:, 0] >= 0) & (x[:, 1] >= 0)
            q2 = (x[:, 0] < 0) & (x[:, 1] >= 0)
            q3 = (x[:, 0] < 0) & (x[:, 1] < 0)
            q4 = (x[:, 0] >= 0) & (x[:, 1] < 0)

            y = np.zeros(1000)
            y[q1] = np.random.uniform(0, 1)
            y[q2] = np.random.uniform(0, 1)
            y[q3] = np.random.uniform(0, 1)
            y[q4] = np.random.uniform(0, 1)
        elif self.threshold_function == "linear_seed":
            x, y = make_regression(n_samples=self.n_samples,
                                   n_features=100,
                                   n_informative=10,
                                   noise=10,
                                   coef=False,
                                   random_state=self.seed)
            y = ((y + 1000) / 9000)
        elif self.threshold_function == "quad2":
            num_examples = self.n_samples
            np.random.seed(self.seed)
            noise_x = np.random.normal(size=(num_examples, 8))
            num_x = self.num_x
            informative_x = np.zeros(shape=(num_examples, num_x))
            quad_assignments = np.random.choice([i + 1 for i in range(num_x ** 2)], size=num_examples)
            quad_unique = np.unique(quad_assignments)
            y = np.zeros(num_examples)
            for i in range(num_x):
                change_frequency = (num_x - i)
                multiplier = 1
                counter = 0
                for j, q in enumerate(quad_unique):
                    idx = quad_assignments == q
                    informative_x[idx, i] = np.random.normal(0.5 * multiplier, scale=0.25, size=np.sum(idx))
                    counter += 1
                    if counter == change_frequency:
                        counter = 0
                        multiplier = -1 * multiplier
                    if i == 0:
                        y[idx] = np.random.uniform(0.05, 0.45)
            x = np.hstack((informative_x, noise_x))
        elif self.threshold_function == "linear_max":
            x, y = make_regression(n_samples=self.n_samples,
                                   n_features=100,
                                   n_informative=10,
                                   noise=10,
                                   coef=False,
                                   random_state=self.seed)
            y = (y + np.abs(y.min())) / (y.max() - y.min())
        else:
            x, y = make_regression(n_samples=self.n_samples,
                                   n_features=100,
                                   n_informative=10,
                                   noise=10,
                                   coef=False,
                                   random_state=self.seed)
            y = ((y + 1000) / 9000)

        # graph = nx.generators.random_graphs.erdos_renyi_graph(self.n_samples, p=self.p, seed=self.seed)
        graph = self.network_generation()

        node_atts = []
        for i, att in enumerate(x):
            node_atts.append((i, {"atts": att, "threshold": y[i]}))
        graph.add_nodes_from(node_atts)

        # np.random.seed(0)
        np.random.seed(self.seed)
        perm = np.random.permutation(x.shape[0])
        perm = perm[:int(0.05 * perm.shape[0])]

        node_labels = []
        for i in range(x.shape[0]):
            if i in perm:
                node_labels.append((i, {"label": 1}))
            else:
                node_labels.append((i, {"label": 0}))

        graph.add_nodes_from(node_labels)

        labeled_users = [
            user for user in graph.nodes if "label" in graph.nodes[user]
        ]
        # use the final graph instead
        thresh_model = ThresholdModel(graph, verbose=True)
        infected_nodes = [i for i in labeled_users if graph.nodes[i]["label"] == 1]

        thresh_model.set_initial_config(infected_nodes)
        thresh_model.set_thresholds(labeled_users, y)

        thresh_model.run(8)

        iteration_dict = thresh_model.iteration_dict

        edges = graph.edges

        num_nodes = []
        for key in iteration_dict:
            infect_nodes = iteration_dict[key]
            num_nodes.append(len(infect_nodes))

            save_graph = nx.Graph()
            save_graph.add_edges_from(edges)
            save_graph.add_nodes_from(node_atts)

            node_labels = []
            for i in range(x.shape[0]):
                if i in infect_nodes:
                    node_labels.append((i, {"label": 1}))
                else:
                    node_labels.append((i, {"label": 0}))
            save_graph.add_nodes_from(node_labels)

            for user in save_graph.nodes:
                num_nbrs = len(save_graph[user])
                nbr_labels = [
                    save_graph.nodes[x]["label"] for x in save_graph[user]
                    if "label" in save_graph.nodes[x]
                ]
                mean_nbr = 0.0
                num_nbr = 0.0
                if len(nbr_labels) > 0:
                    num_nbr = np.sum(nbr_labels)
                    mean_nbr = num_nbr / num_nbrs
                save_graph.nodes[user]["treatment"] = mean_nbr
                save_graph.nodes[user]["treatment_num"] = num_nbr

            nx.write_gpickle(save_graph, self.data_folder + f"iteration_{key}.graph")

    def learn_triggers(self):
        logger = self.logger

        data_files = [
            self.data_folder + i for i in os.listdir(self.data_folder)
            if os.path.isfile(self.data_folder + i)
            and ".graph" in i
        ]
        data_files.sort()

        trigger_results = dict()
        non_atts = ["label", "treatment", "treatment_num"]
        for file in data_files:
            save_file = file.replace(".graph", "")
            save_file = save_file + ".results"
            if os.path.isfile(save_file):
                logger.info(f"{save_file} exists")
                continue
            graph = nx.read_gpickle(file)
            labeled_users = [user for user in graph.nodes if "label" in graph.nodes[user]]
            x = None
            y = np.zeros(len(labeled_users))
            t = np.zeros(len(labeled_users))
            for i, user in enumerate(graph.nodes):
                node_atts = graph.nodes[user]
                atts = []
                for att in node_atts:
                    if att in non_atts:
                        continue
                    atts.append(node_atts[att])
                atts = atts[0]
                if x is None:
                    x = np.zeros((len(graph), len(atts)))
                x[i] = atts
                y[i] = node_atts["label"]
                t[i] = node_atts["treatment"]

            logger.info(f"Building tree for {file}")
            ct = CausalTree(cont=True, split_size=0.2, weight=0.2, val_honest=True, seed=self.seed, max_depth=15,
                            max_values=100)
            ct.fit(x, y, t)
            logger.info(f"Finished building tree for {file}")
            logger.info(f"Tree depth: {ct.tree_depth}")
            prediction = ct.predict(x)
            triggers = ct.get_triggers(x)
            # if triggers[0] == np.inf or triggers[0] == -np.inf:
            #     print(t)
            with open(save_file, 'wb') as pklfile:
                pkl.dump([ct, prediction, triggers], pklfile)
            trigger_results[save_file] = save_file

    def learn_triggers_slearner(self):
        logger = self.logger

        data_files = [
            self.data_folder + i for i in os.listdir(self.data_folder)
            if os.path.isfile(self.data_folder + i)
            and ".graph" in i
        ]
        data_files.sort()

        trigger_results = dict()
        non_atts = ["label", "treatment", "treatment_num"]
        for file in data_files:
            save_file = file.replace(".graph", "")
            save_file = save_file + ".sresults"
            if os.path.isfile(save_file):
                logger.info(f"{save_file} exists")
                continue
            graph = nx.read_gpickle(file)
            labeled_users = [user for user in graph.nodes if "label" in graph.nodes[user]]
            x = None
            y = np.zeros(len(labeled_users))
            t = np.zeros(len(labeled_users))
            for i, user in enumerate(graph.nodes):
                node_atts = graph.nodes[user]
                atts = []
                for att in node_atts:
                    if att in non_atts:
                        continue
                    atts.append(node_atts[att])
                atts = atts[0]
                if x is None:
                    x = np.zeros((len(graph), len(atts)))
                x[i] = atts
                y[i] = node_atts["label"]
                t[i] = node_atts["treatment"]

            logger.info(f"Building slearner for {file}")
            slearner = STLearner()
            slearner.fit(x, y, t)
            triggers = slearner.triggers(x)
            with open(save_file, "wb") as pklfile:
                pkl.dump([slearner, triggers], pklfile)

            logger.info(f"finished building Slearner for {save_file}")
            trigger_results[save_file] = save_file

    def learn_triggers_slearner_tree(self):
        logger = self.logger

        data_files = [
            self.data_folder + i for i in os.listdir(self.data_folder)
            if os.path.isfile(self.data_folder + i)
            and ".graph" in i
        ]
        data_files.sort()

        trigger_results = dict()
        non_atts = ["label", "treatment", "treatment_num"]
        for file in data_files:
            save_file = file.replace(".graph", "")
            save_file = save_file + ".streeresults"
            print(save_file)
            if os.path.isfile(save_file):
                logger.info(f"{save_file} exists")
                continue
            graph = nx.read_gpickle(file)
            labeled_users = [user for user in graph.nodes if "label" in graph.nodes[user]]
            x = None
            y = np.zeros(len(labeled_users))
            t = np.zeros(len(labeled_users))
            for i, user in enumerate(graph.nodes):
                node_atts = graph.nodes[user]
                atts = []
                for att in node_atts:
                    if att in non_atts:
                        continue
                    atts.append(node_atts[att])
                atts = atts[0]
                if x is None:
                    x = np.zeros((len(graph), len(atts)))
                x[i] = atts
                y[i] = node_atts["label"]
                t[i] = node_atts["treatment"]

            logger.info(f"Building SLearner (Tree) for {file}")
            slearner = STLearner(model=DecisionTreeRegressor, params={"min_samples_leaf": 10})
            slearner.fit(x, y, t)
            triggers = slearner.triggers(x)
            with open(save_file, "wb") as pklfile:
                pkl.dump([slearner, triggers], pklfile)

            logger.info(f"finished building Slearner (tree) for {save_file}")
            trigger_results[save_file] = save_file

    def run_diffusion(self):
        logger = self.logger

        data_files = [
            self.data_folder + i for i in os.listdir(self.data_folder)
            if os.path.isfile(self.data_folder + i)
            and ".graph" in i
        ]
        results_files = [
            self.data_folder + i for i in os.listdir(self.data_folder)
            if os.path.isfile(self.data_folder + i)
            and ".results" in i
        ]
        data_files.sort()
        results_files.sort()

        triggers = []
        counter = 0
        for result_file in results_files:
            if counter > 4:
                continue
            with open(result_file, "rb") as file:
                data = pkl.load(file)
            print(data[-1].min(), data[-1].mean(), data[-1].max())
            triggers.append(data[-1])

            counter += 1

        print(np.mean(np.concatenate(triggers)))

        # mse = []
        # counter = 0
        # for result_file in results_files:
        #     if counter > 4:
        #         continue
        #     with open(result_file, "rb") as file:
        #         data = pkl.load(file)
        #     est_thresh = data[-1]
        #     err = np.mean((est_thresh - y) ** 2)
        #     print(err)
        #     mse.append(err)
        #     triggers.append(data[-1])
        #
        #     counter += 1
        #
        # print(np.mean(mse))

        end_graph = nx.read_gpickle(data_files[-1])
        end_users = [user for user in end_graph.nodes]
        end_labels = [
            end_graph.nodes[user]["label"] for user in end_users
            if "label" in end_graph.nodes[user]
        ]
        end_labeled_users = [
            user for user in end_graph.nodes if "label" in end_graph.nodes[user]
        ]
        end_infected_nodes = [
            i for i in end_labeled_users if end_graph.nodes[i]["label"] == 1
        ]

        track = dict()
        true_track = dict()
        track_sim = dict()
        track_sim2 = dict()

        num_infect_all = []
        infected_nodes_list = []
        for i, f in enumerate(data_files):
            graph = nx.read_gpickle(f)
            l_users = [
                user for user in graph.nodes
                if "label" in graph.nodes[user] and len(graph.nodes[user])
            ]
            labels = [graph.nodes[user]["label"] for user in l_users]
            num_infect_all.append(np.sum(labels))
            nodes = [user for user in l_users if graph.nodes[user]["label"] == 1]
            infected_nodes_list.append(nodes)

        for j, file in enumerate(data_files):
            if file == data_files[-1]:
                continue
            print(f"Reading results from: {results_files[j]}...")
            with open(results_files[j], 'rb') as handle:
                data = pkl.load(handle)
            graph = nx.read_gpickle(file)

            labeled_users = [
                user for user in graph.nodes if "label" in graph.nodes[user]
            ]
            triggers = data[-1]
            # use the final graph instead
            thresh_model = ThresholdModel(end_graph)
            infected_nodes = [i for i in labeled_users if graph.nodes[i]["label"] == 1]

            thresh_model.set_initial_config(infected_nodes)
            thresh_model.set_thresholds(labeled_users, triggers)
            thresh_model.run(len(data_files) - j)

            num_nodes = []
            iteration_dict = thresh_model.iteration_dict
            for key in iteration_dict:
                infect_nodes = iteration_dict[key]
                num_nodes.append(
                    len(list(set(infect_nodes).intersection(end_labeled_users))))
            #         num_nodes.append(
            #             len(list(set(infect_nodes).intersection(end_infected_nodes))))

            all_sims = []
            all_sims2 = []
            for key in iteration_dict:
                infect_nodes = iteration_dict[key]

                intersection = set(infect_nodes).intersection(set(infected_nodes_list[key + j]))
                #         union = set(infect_nodes).union(set(infected_nodes_list[key + j]))
                union = set(infected_nodes_list[key + j])

                union2 = set(infect_nodes).union(set(infected_nodes_list[key + j]))

                prop = len(intersection) / (len(union))

                all_sims.append(prop)

                all_sims2.append(len(intersection) / len(union2))

            track_sim[j] = all_sims
            track_sim2[j] = all_sims2

            num_infect = []
            for i in range(len(num_infect_all)):
                if i < j:
                    continue
                if i > len(results_files):
                    continue
                num_infect.append(num_infect_all[i])

            track[j] = num_nodes
            true_track[j] = num_infect

        sim_res = []
        for key in track_sim:
            sim_res.append(np.mean(track_sim[key]))
        print(np.mean(sim_res))

        sim_res = []
        for key in track_sim2:
            sim_res.append(np.mean(track_sim2[key]))
        print(np.mean(sim_res))

        results = self.save_folder + "/my_results.pkl"
        check_dir(results)
        with open(results, "wb") as file:
            pkl.dump([track, true_track, track_sim, track_sim2], file)

        predict = track
        true = true_track

    def run_diffusion_slearner(self):
        logger = self.logger

        data_files = [
            self.data_folder + i for i in os.listdir(self.data_folder)
            if os.path.isfile(self.data_folder + i)
            and ".graph" in i
        ]
        results_files = [
            self.data_folder + i for i in os.listdir(self.data_folder)
            if os.path.isfile(self.data_folder + i)
            and ".sresults" in i
        ]
        data_files.sort()
        results_files.sort()

        triggers = []
        counter = 0
        for result_file in results_files:
            # print(result_file, results_files)
            if counter > 4:
                continue
            with open(result_file, "rb") as file:
                data = pkl.load(file)
            print(data[-1].min(), data[-1].mean(), data[-1].max())
            triggers.append(data[-1])

            counter += 1

        print(np.mean(np.concatenate(triggers)))

        # mse = []
        # counter = 0
        # for result_file in results_files:
        #     if counter > 4:
        #         continue
        #     with open(result_file, "rb") as file:
        #         data = pkl.load(file)
        #     est_thresh = data[-1]
        #     err = np.mean((est_thresh - y) ** 2)
        #     print(err)
        #     mse.append(err)
        #     triggers.append(data[-1])
        #
        #     counter += 1
        #
        # print(np.mean(mse))

        end_graph = nx.read_gpickle(data_files[-1])
        end_users = [user for user in end_graph.nodes]
        end_labels = [
            end_graph.nodes[user]["label"] for user in end_users
            if "label" in end_graph.nodes[user]
        ]
        end_labeled_users = [
            user for user in end_graph.nodes if "label" in end_graph.nodes[user]
        ]
        end_infected_nodes = [
            i for i in end_labeled_users if end_graph.nodes[i]["label"] == 1
        ]

        track = dict()
        true_track = dict()
        track_sim = dict()
        track_sim2 = dict()

        num_infect_all = []
        infected_nodes_list = []
        for i, f in enumerate(data_files):
            graph = nx.read_gpickle(f)
            l_users = [
                user for user in graph.nodes
                if "label" in graph.nodes[user] and len(graph.nodes[user])
            ]
            labels = [graph.nodes[user]["label"] for user in l_users]
            num_infect_all.append(np.sum(labels))
            nodes = [user for user in l_users if graph.nodes[user]["label"] == 1]
            infected_nodes_list.append(nodes)

        for j, file in enumerate(data_files):
            if file == data_files[-1]:
                continue
            print(f"Reading results from: {results_files[j]}...")
            with open(results_files[j], 'rb') as handle:
                data = pkl.load(handle)
            graph = nx.read_gpickle(file)

            labeled_users = [
                user for user in graph.nodes if "label" in graph.nodes[user]
            ]
            triggers = data[-1]
            # use the final graph instead
            thresh_model = ThresholdModel(end_graph)
            infected_nodes = [i for i in labeled_users if graph.nodes[i]["label"] == 1]

            thresh_model.set_initial_config(infected_nodes)
            thresh_model.set_thresholds(labeled_users, triggers)
            thresh_model.run(len(data_files) - j)

            num_nodes = []
            iteration_dict = thresh_model.iteration_dict
            for key in iteration_dict:
                infect_nodes = iteration_dict[key]
                num_nodes.append(
                    len(list(set(infect_nodes).intersection(end_labeled_users))))
            #         num_nodes.append(
            #             len(list(set(infect_nodes).intersection(end_infected_nodes))))

            all_sims = []
            all_sims2 = []
            for key in iteration_dict:
                infect_nodes = iteration_dict[key]

                intersection = set(infect_nodes).intersection(set(infected_nodes_list[key + j]))
                #         union = set(infect_nodes).union(set(infected_nodes_list[key + j]))
                union = set(infected_nodes_list[key + j])

                union2 = set(infect_nodes).union(set(infected_nodes_list[key + j]))

                prop = len(intersection) / (len(union))

                all_sims.append(prop)

                all_sims2.append(len(intersection) / len(union2))

            track_sim[j] = all_sims
            track_sim2[j] = all_sims2

            num_infect = []
            for i in range(len(num_infect_all)):
                if i < j:
                    continue
                if i > len(results_files):
                    continue
                num_infect.append(num_infect_all[i])

            track[j] = num_nodes
            true_track[j] = num_infect

        sim_res = []
        for key in track_sim:
            sim_res.append(np.mean(track_sim[key]))
        print(np.mean(sim_res))

        sim_res = []
        for key in track_sim2:
            sim_res.append(np.mean(track_sim2[key]))
        print(np.mean(sim_res))

        results = self.save_folder + "/slearner_results.pkl"
        check_dir(results)
        with open(results, "wb") as file:
            pkl.dump([track, true_track, track_sim, track_sim2], file)

        predict = track
        true = true_track

    def run_diffusion_slearner_tree(self):
        logger = self.logger

        data_files = [
            self.data_folder + i for i in os.listdir(self.data_folder)
            if os.path.isfile(self.data_folder + i)
            and ".graph" in i
        ]
        results_files = [
            self.data_folder + i for i in os.listdir(self.data_folder)
            if os.path.isfile(self.data_folder + i)
            and ".streeresults" in i
        ]
        data_files.sort()
        results_files.sort()

        triggers = []
        counter = 0
        for result_file in results_files:
            # print(result_file, results_files)
            if counter > 4:
                continue
            with open(result_file, "rb") as file:
                data = pkl.load(file)
            print(data[-1].min(), data[-1].mean(), data[-1].max())
            triggers.append(data[-1])

            counter += 1

        print(np.mean(np.concatenate(triggers)))

        # mse = []
        # counter = 0
        # for result_file in results_files:
        #     if counter > 4:
        #         continue
        #     with open(result_file, "rb") as file:
        #         data = pkl.load(file)
        #     est_thresh = data[-1]
        #     err = np.mean((est_thresh - y) ** 2)
        #     print(err)
        #     mse.append(err)
        #     triggers.append(data[-1])
        #
        #     counter += 1
        #
        # print(np.mean(mse))

        end_graph = nx.read_gpickle(data_files[-1])
        end_users = [user for user in end_graph.nodes]
        end_labels = [
            end_graph.nodes[user]["label"] for user in end_users
            if "label" in end_graph.nodes[user]
        ]
        end_labeled_users = [
            user for user in end_graph.nodes if "label" in end_graph.nodes[user]
        ]
        end_infected_nodes = [
            i for i in end_labeled_users if end_graph.nodes[i]["label"] == 1
        ]

        track = dict()
        true_track = dict()
        track_sim = dict()
        track_sim2 = dict()

        num_infect_all = []
        infected_nodes_list = []
        for i, f in enumerate(data_files):
            graph = nx.read_gpickle(f)
            l_users = [
                user for user in graph.nodes
                if "label" in graph.nodes[user] and len(graph.nodes[user])
            ]
            labels = [graph.nodes[user]["label"] for user in l_users]
            num_infect_all.append(np.sum(labels))
            nodes = [user for user in l_users if graph.nodes[user]["label"] == 1]
            infected_nodes_list.append(nodes)

        for j, file in enumerate(data_files):
            if file == data_files[-1]:
                continue
            print(f"Reading results from: {results_files[j]}...")
            with open(results_files[j], 'rb') as handle:
                data = pkl.load(handle)
            graph = nx.read_gpickle(file)

            labeled_users = [
                user for user in graph.nodes if "label" in graph.nodes[user]
            ]
            triggers = data[-1]
            # use the final graph instead
            thresh_model = ThresholdModel(end_graph)
            infected_nodes = [i for i in labeled_users if graph.nodes[i]["label"] == 1]

            thresh_model.set_initial_config(infected_nodes)
            thresh_model.set_thresholds(labeled_users, triggers)
            thresh_model.run(len(data_files) - j)

            num_nodes = []
            iteration_dict = thresh_model.iteration_dict
            for key in iteration_dict:
                infect_nodes = iteration_dict[key]
                num_nodes.append(
                    len(list(set(infect_nodes).intersection(end_labeled_users))))
            #         num_nodes.append(
            #             len(list(set(infect_nodes).intersection(end_infected_nodes))))

            all_sims = []
            all_sims2 = []
            for key in iteration_dict:
                infect_nodes = iteration_dict[key]

                intersection = set(infect_nodes).intersection(set(infected_nodes_list[key + j]))
                #         union = set(infect_nodes).union(set(infected_nodes_list[key + j]))
                union = set(infected_nodes_list[key + j])

                union2 = set(infect_nodes).union(set(infected_nodes_list[key + j]))

                prop = len(intersection) / (len(union))

                all_sims.append(prop)

                all_sims2.append(len(intersection) / len(union2))

            track_sim[j] = all_sims
            track_sim2[j] = all_sims2

            num_infect = []
            for i in range(len(num_infect_all)):
                if i < j:
                    continue
                if i > len(results_files):
                    continue
                num_infect.append(num_infect_all[i])

            track[j] = num_nodes
            true_track[j] = num_infect

        sim_res = []
        for key in track_sim:
            sim_res.append(np.mean(track_sim[key]))
        print(np.mean(sim_res))

        sim_res = []
        for key in track_sim2:
            sim_res.append(np.mean(track_sim2[key]))
        print(np.mean(sim_res))

        results = self.save_folder + "/stree_results.pkl"
        check_dir(results)
        with open(results, "wb") as file:
            pkl.dump([track, true_track, track_sim, track_sim2], file)

        predict = track
        true = true_track

    def run_diffusion_baselines(self):
        logger = self.logger

        data_files = [
            self.data_folder + i for i in os.listdir(self.data_folder)
            if os.path.isfile(self.data_folder + i)
            and ".graph" in i
        ]
        results_files = [
            self.data_folder + i for i in os.listdir(self.data_folder)
            if os.path.isfile(self.data_folder + i)
            and ".results" in i
        ]
        data_files.sort()
        results_files.sort()

        end_graph = nx.read_gpickle(data_files[-1])
        end_users = [user for user in end_graph.nodes]
        end_labels = [
            end_graph.nodes[user]["label"] for user in end_users
            if "label" in end_graph.nodes[user]
        ]
        end_labeled_users = [
            user for user in end_graph.nodes if "label" in end_graph.nodes[user]
        ]
        end_infected_nodes = [
            i for i in end_labeled_users if end_graph.nodes[i]["label"] == 1
        ]
        num_infect_all = []
        infected_nodes_list = []
        for i, f in enumerate(data_files):
            graph = nx.read_gpickle(f)
            l_users = [
                user for user in graph.nodes
                if "label" in graph.nodes[user] and len(graph.nodes[user])
            ]
            labels = [graph.nodes[user]["label"] for user in l_users]
            num_infect_all.append(np.sum(labels))
            nodes = [user for user in l_users if graph.nodes[user]["label"] == 1]
            infected_nodes_list.append(nodes)

        estimators = [
            #     sample_individual,
            #     sample_individual,
            #     sample_individual,
            heuristic_individual,
            #     sample_expected,
            #     sample_expected,
            #     sample_expected,
            heuristic_expected,
        ]
        sample_params = ["", "rsp", "slovin"]
        est_dict = {
            heuristic_expected: "heuristic_expected",
            sample_expected: "sample_expected",
            heuristic_individual: "heuristic_individual",
            sample_individual: "sample_individual"
        }
        est_track = dict()

        counter = -1
        for estimator in estimators:
            est_name = est_dict[estimator]

            params = dict()
            save_results_key = est_name
            if estimator == sample_expected or estimator == sample_individual:
                counter += 1
                if counter > 2:
                    counter = 0
                params = {"choice": sample_params[counter]}
                save_results_key = save_results_key + f"_{sample_params[counter]}"

            print(f"Working on {save_results_key}...")

            track = dict()
            true_track = dict()
            track_sim = dict()
            track_sim2 = dict()

            for j, file in enumerate(data_files):
                if file == data_files[-1]:
                    continue

                print(f"Reading graph from: {file}...")
                graph = nx.read_gpickle(file)

                labeled_users = [
                    user for user in graph.nodes if "label" in graph.nodes[user]
                ]

                # use the final graph instead
                thresh_model = ThresholdModel(end_graph)
                infected_nodes = [
                    i for i in labeled_users if graph.nodes[i]["label"] == 1
                ]

                thresh_model.set_initial_config(infected_nodes)
                if estimator == heuristic_expected or estimator == sample_expected:
                    threshold_estimation = estimator(graph, use_int=False, **params)
                    save_file = file.replace(".graph", "")
                    save_file = save_file + ".heresults"
                    thresholds = thresh_model.set_thresholds_value(threshold_estimation)
                    with open(save_file, "wb") as pklfile:
                        pkl.dump([estimator, thresholds], pklfile)
                    print(threshold_estimation)
                else:
                    threshold_min, threshold_max = estimator(graph, use_int=False, **params)
                    save_file = file.replace(".graph", "")
                    save_file = save_file + ".hiresults"
                    thresholds = thresh_model.set_thresholds_estimation(threshold_min, threshold_max)
                    with open(save_file, "wb") as pklfile:
                        pkl.dump([estimator, thresholds], pklfile)
                    print(threshold_min, threshold_max)
                thresh_model.run(len(data_files) - j, use_int=False)

                num_nodes = []
                iteration_dict = thresh_model.iteration_dict
                for key in iteration_dict:
                    infect_nodes = iteration_dict[key]
                    num_nodes.append(
                        len(list(set(infect_nodes).intersection(end_labeled_users))))

                all_sims = []
                all_sims2 = []
                for key in iteration_dict:
                    infect_nodes = iteration_dict[key]

                    intersection = set(infect_nodes).intersection(set(infected_nodes_list[key + j]))
                    #         union = set(infect_nodes).union(set(infected_nodes_list[key + j]))
                    union = set(infected_nodes_list[key + j])

                    union2 = set(infect_nodes).union(set(infected_nodes_list[key + j]))

                    prop = len(intersection) / (len(union))

                    all_sims.append(prop)

                    all_sims2.append(len(intersection) / len(union2))

                track_sim[j] = all_sims
                track_sim2[j] = all_sims2

                num_infect = []
                for i, file in enumerate(data_files):
                    if i < j:
                        continue
                    graph = nx.read_gpickle(file)
                    l_users = [
                        user for user in graph.nodes
                        if "label" in graph.nodes[user] and len(graph.nodes[user])
                    ]
                    labels = [graph.nodes[user]["label"] for user in l_users]
                    num_infect.append(np.sum(labels))

                x = np.ceil(np.sqrt(len(data_files)))
                y = np.floor(np.sqrt(len(data_files)))

                track[j] = num_nodes
                true_track[j] = num_infect
            est_track[save_results_key] = [track, true_track, track_sim, track_sim2]

        print("he results")
        he_results = est_track["heuristic_expected"]
        track_sim = he_results[-2]
        track_sim2 = he_results[-1]

        sim_res = []
        for key in track_sim:
            sim_res.append(np.mean(track_sim[key]))
        print(np.mean(sim_res))

        sim_res = []
        for key in track_sim2:
            sim_res.append(np.mean(track_sim2[key]))
        print(np.mean(sim_res))

        print()
        print("hi results")

        hi_results = est_track["heuristic_individual"]
        track_sim = hi_results[-2]
        track_sim2 = hi_results[-1]

        sim_res = []
        for key in track_sim:
            sim_res.append(np.mean(track_sim[key]))
        print(np.mean(sim_res))

        sim_res = []
        for key in track_sim2:
            sim_res.append(np.mean(track_sim2[key]))
        print(np.mean(sim_res))

        with open(self.save_folder + "frac_results.pkl", "wb") as file:
            pkl.dump(est_track, file)

    def run_diffusion_regression(self):
        logger = self.logger

        data_files = [
            self.data_folder + i for i in os.listdir(self.data_folder)
            if os.path.isfile(self.data_folder + i)
            and ".graph" in i
        ]
        results_files = [
            self.data_folder + i for i in os.listdir(self.data_folder)
            if os.path.isfile(self.data_folder + i)
            and ".results" in i
        ]
        data_files.sort()
        results_files.sort()

        print(self.data_folder)

        end_graph = nx.read_gpickle(data_files[-1])
        end_users = [user for user in end_graph.nodes]
        end_labels = [
            end_graph.nodes[user]["label"] for user in end_users
            if "label" in end_graph.nodes[user]
        ]
        end_labeled_users = [
            user for user in end_graph.nodes if "label" in end_graph.nodes[user]
        ]
        end_infected_nodes = [
            i for i in end_labeled_users if end_graph.nodes[i]["label"] == 1
        ]
        num_infect_all = []
        infected_nodes_list = []
        for i, f in enumerate(data_files):
            graph = nx.read_gpickle(f)
            l_users = [
                user for user in graph.nodes
                if "label" in graph.nodes[user] and len(graph.nodes[user])
            ]
            labels = [graph.nodes[user]["label"] for user in l_users]
            num_infect_all.append(np.sum(labels))
            nodes = [user for user in l_users if graph.nodes[user]["label"] == 1]
            infected_nodes_list.append(nodes)

        track = dict()
        true_track = dict()
        track_sim = dict()
        track_sim2 = dict()
        for j, file in enumerate(data_files):
            if file == data_files[-1]:
                continue
            print(f"Reading results from: {results_files[j]}...")

            graph = nx.read_gpickle(file)

            labeled_users = [user for user in graph.nodes if "label" in graph.nodes[user]]
            # use the final graph instead
            thresh_model = ThresholdModel(end_graph)
            infected_nodes = [i for i in labeled_users if graph.nodes[i]["label"] == 1]

            thresh_model.set_initial_config(infected_nodes)

            # learning thresholds from linear regression
            # x = []
            # t = []
            # for i in infected_nodes:
            #     node_t = graph.nodes[i]["treatment"]
            #     node_x = [graph.nodes[i][att] for att in graph.nodes[i] if
            #               att not in ["label", "treatment", "treatment_num"]]
            #     t.append(node_t)
            #     x.append(node_x)
            # x = np.array(x)
            # t = np.array(t)
            non_atts = ["label", "treatment", "treatment_num"]
            x = None
            y = np.zeros(len(infected_nodes))
            t = np.zeros(len(infected_nodes))
            for i, user in enumerate(infected_nodes):
                node_atts = graph.nodes[user]
                atts = []
                for att in node_atts:
                    if att in non_atts:
                        continue
                    atts.append(node_atts[att])
                atts = atts[0]
                if x is None:
                    x = np.zeros((len(infected_nodes), len(atts)))
                x[i] = atts
                y[i] = node_atts["label"]
                t[i] = node_atts["treatment"]
            logger.info(x.shape)
            logger.info(t.shape)
            lr = LinearRegression()
            lr.fit(x, t)

            thresholds = []
            feature_users = []
            for i in labeled_users:
                if i in infected_nodes:
                    thresholds.append(graph.nodes[i]["treatment"])
                    continue
                # node_x = np.array([graph.nodes[i][att] for att in graph.nodes[i] if
                #                    att not in ["label", "treatment", "treatment_num", "threshold"]])
                node_x = np.array(graph.nodes[i]["atts"])
                if len(node_x) < 1:
                    continue
                node_t = lr.predict(node_x.reshape(1, -1))[0]
                thresholds.append(node_t)
                feature_users.append(i)

            save_file = file.replace(".graph", "")
            save_file = save_file + ".lrresults"
            with open(save_file, "wb") as pklfile:
                pkl.dump([lr, np.array(thresholds)], pklfile)

            thresh_model.set_thresholds(feature_users, thresholds)
            thresh_model.run(len(data_files) - j)

            num_nodes = []
            iteration_dict = thresh_model.iteration_dict
            for key in iteration_dict:
                infect_nodes = iteration_dict[key]
                num_nodes.append(
                    len(list(set(infect_nodes).intersection(end_labeled_users))))

            all_sims = []
            all_sims2 = []
            for key in iteration_dict:
                infect_nodes = iteration_dict[key]

                intersection = set(infect_nodes).intersection(set(infected_nodes_list[key + j]))
                #         union = set(infect_nodes).union(set(infected_nodes_list[key + j]))
                union = set(infected_nodes_list[key + j])

                union2 = set(infect_nodes).union(set(infected_nodes_list[key + j]))

                prop = len(intersection) / (len(union))

                all_sims.append(prop)

                all_sims2.append(len(intersection) / len(union2))

            track_sim[j] = all_sims
            track_sim2[j] = all_sims2

            num_infect = []
            for i in range(len(num_infect_all)):
                if i < j:
                    continue
                if i > len(results_files):
                    continue
                #         print(f"i: {i}, j: {j}")
                num_infect.append(num_infect_all[i])

            track[j] = num_nodes
            true_track[j] = num_infect

        sim_res = []
        for key in track_sim:
            sim_res.append(np.mean(track_sim[key]))
        print(np.mean(sim_res))

        sim_res = []
        for key in track_sim2:
            sim_res.append(np.mean(track_sim2[key]))
        print(np.mean(sim_res))

        with open(self.save_folder + "regression.pkl", "wb") as file:
            pkl.dump([track, true_track, track_sim, track_sim2], file)

    def run_diffusion_random(self):
        logger = self.logger

        data_files = [
            self.data_folder + i for i in os.listdir(self.data_folder)
            if os.path.isfile(self.data_folder + i)
            and ".graph" in i
        ]
        results_files = [
            self.data_folder + i for i in os.listdir(self.data_folder)
            if os.path.isfile(self.data_folder + i)
            and ".results" in i
        ]
        data_files.sort()
        results_files.sort()

        end_graph = nx.read_gpickle(data_files[-1])
        end_users = [user for user in end_graph.nodes]
        end_labels = [
            end_graph.nodes[user]["label"] for user in end_users
            if "label" in end_graph.nodes[user]
        ]
        end_labeled_users = [
            user for user in end_graph.nodes if "label" in end_graph.nodes[user]
        ]
        end_infected_nodes = [
            i for i in end_labeled_users if end_graph.nodes[i]["label"] == 1
        ]
        num_infect_all = []
        infected_nodes_list = []
        for i, f in enumerate(data_files):
            graph = nx.read_gpickle(f)
            l_users = [
                user for user in graph.nodes
                if "label" in graph.nodes[user] and len(graph.nodes[user])
            ]
            labels = [graph.nodes[user]["label"] for user in l_users]
            num_infect_all.append(np.sum(labels))
            nodes = [user for user in l_users if graph.nodes[user]["label"] == 1]
            infected_nodes_list.append(nodes)

        track = dict()
        true_track = dict()
        track_sim = dict()
        track_sim2 = dict()
        for j, file in enumerate(data_files):
            if file == data_files[-1]:
                continue
            print(f"Reading results from: {results_files[j]}...")

            graph = nx.read_gpickle(file)

            labeled_users = [
                user for user in graph.nodes if "label" in graph.nodes[user]
            ]
            # use the final graph instead
            thresh_model = ThresholdModel(end_graph)
            infected_nodes = [i for i in labeled_users if graph.nodes[i]["label"] == 1]
            random_thresholds = np.array(list(thresh_model.thresh_dict.values()))
            save_file = file.replace(".graph", "")
            save_file = save_file + ".randomresults"
            with open(save_file, "wb") as pklfile:
                pkl.dump(["random", random_thresholds], pklfile)

            thresh_model.set_initial_config(infected_nodes)
            thresh_model.run(len(data_files) - j)

            num_nodes = []
            iteration_dict = thresh_model.iteration_dict
            for key in iteration_dict:
                infect_nodes = iteration_dict[key]
                num_nodes.append(
                    len(list(set(infect_nodes).intersection(end_labeled_users))))

            all_sims = []
            all_sims2 = []
            for key in iteration_dict:
                infect_nodes = iteration_dict[key]

                intersection = set(infect_nodes).intersection(set(infected_nodes_list[key + j]))
                #         union = set(infect_nodes).union(set(infected_nodes_list[key + j]))
                union = set(infected_nodes_list[key + j])

                union2 = set(infect_nodes).union(set(infected_nodes_list[key + j]))

                prop = len(intersection) / (len(union))

                all_sims.append(prop)

                all_sims2.append(len(intersection) / len(union2))

            track_sim[j] = all_sims
            track_sim2[j] = all_sims2

            num_infect = []
            for i in range(len(num_infect_all)):
                if i < j:
                    continue
                if i > len(results_files):
                    continue
                #         print(f"i: {i}, j: {j}")
                num_infect.append(num_infect_all[i])

            track[j] = num_nodes
            true_track[j] = num_infect

        sim_res = []
        for key in track_sim:
            sim_res.append(np.mean(track_sim[key]))
        print(np.mean(sim_res))

        sim_res = []
        for key in track_sim2:
            sim_res.append(np.mean(track_sim2[key]))
        print(np.mean(sim_res))

        with open(self.save_folder + "random.pkl", "wb") as file:
            pkl.dump([track, true_track, track_sim, track_sim2], file)

    def plot_results(self):

        logger = self.logger

        my_results = self.save_folder + "my_results.pkl"
        slearner_results = self.save_folder + "slearner_results.pkl"
        stree_results = self.save_folder + "stree_results.pkl"
        frac_results = self.save_folder + "frac_results.pkl"
        random_results = self.save_folder + "random.pkl"
        lr_results = self.save_folder + "regression.pkl"

        with open(my_results, "rb") as file:
            results = pkl.load(file)
        with open(slearner_results, "rb") as file:
            sresults = pkl.load(file)
        with open(stree_results, "rb") as file:
            streeresults = pkl.load(file)
        with open(frac_results, "rb") as file:
            f_results = pkl.load(file)
        with open(random_results, "rb") as file:
            r_results = pkl.load(file)
        with open(lr_results, "rb") as file:
            lr_results = pkl.load(file)

        j_results = dict()

        predict, true, ctl_track_sim, _ = results
        spredict, _, slearner_track_sim, _ = sresults
        streepredict, _, stree_track_sim, _ = streeresults

        baseline_track_idx = -2
        track_sim = f_results["heuristic_expected"][baseline_track_idx]
        sim_res = []
        for key in track_sim:
            sim_res.append(np.mean(track_sim[key]))
        logger.info(f"HE: {np.mean(sim_res)}")
        j_results["heuristic_expected"] = np.mean(sim_res)

        track_sim = f_results["heuristic_individual"][baseline_track_idx]
        sim_res = []
        for key in track_sim:
            sim_res.append(np.mean(track_sim[key]))
        logger.info(f"HI: {np.mean(sim_res)}")
        j_results["heuristic_individual"] = np.mean(sim_res)

        track_sim = r_results[baseline_track_idx]
        sim_res = []
        for key in track_sim:
            sim_res.append(np.mean(track_sim[key]))
        logger.info(f"Random: {np.mean(sim_res)}")
        j_results["random"] = np.mean(sim_res)

        track_sim = lr_results[-2]
        sim_res = []
        for key in track_sim:
            sim_res.append(np.mean(track_sim[key]))
        logger.info(f"LR: {np.mean(sim_res)}")
        j_results["lr"] = np.mean(sim_res)

        sim_res = []
        for key in ctl_track_sim:
            sim_res.append(np.mean(ctl_track_sim[key]))
        logger.info(f"CTL: {np.mean(sim_res)}")
        j_results["ctl"] = np.mean(sim_res)

        sim_res = []
        for key in slearner_track_sim:
            sim_res.append(np.mean(slearner_track_sim[key]))
        logger.info(f"Slearner: {np.mean(sim_res)}")
        j_results["sleaner"] = np.mean(sim_res)

        sim_res = []
        for key in stree_track_sim:
            sim_res.append(np.mean(stree_track_sim[key]))
        logger.info(f"Slearner (Tree): {np.mean(sim_res)}")
        j_results["slearner_tree"] = np.mean(sim_res)

        x_axis = [0, 1, 2, 3, 4, 5, 6, 7]

        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"

        np.random.seed(724)
        errors = dict()
        errors_log = dict()
        for j, key in enumerate(predict):
            if j >= 4:
                continue
            num_true = np.array(true[key])
            ct_predict = np.array(predict[key])

            fhi_predict = np.array(f_results["heuristic_individual"][0][key])
            fhe_predict = np.array(f_results["heuristic_expected"][0][key])

            slearner_pred = np.array(spredict[key])
            stree_pred = np.array(streepredict[key])

            lr_predict = lr_results[0][key]

            x_vals = range(j, len(x_axis))

            fig = plt.figure(dpi=100, figsize=(10, 8))
            ax = fig.add_subplot(111)

            line_width = 4
            markersize = 10
            ax.plot(x_vals, num_true, label="True", linewidth=line_width, marker="o", markersize=markersize)
            ax.plot(x_vals, random, label="Random", linewidth=line_width, marker="x", markersize=markersize)
            ax.plot(x_vals, fhi_predict, label="Heuristic Individual", marker="*", linewidth=line_width,
                    markersize=markersize)
            ax.plot(x_vals, fhe_predict, label="Heuristic Expected", linewidth=line_width, marker=">",
                    markersize=markersize)
            ax.plot(x_vals, lr_predict, label="Linear Regression", linewidth=line_width, marker=".",
                    markersize=markersize)

            # ax.plot(x_vals, ct_predict, label="Linear Trigger Model", linewidth=line_width, marker="s",
            #         markersize=line_width * 2.5)
            ax.plot(x_vals, ct_predict, label="Causal Tree", linewidth=line_width, marker="s",
                    markersize=markersize)
            ax.plot(x_vals, slearner_pred, label="ST-Learner", linewidth=line_width, marker="+",
                    markersize=markersize)
            ax.plot(x_vals, stree_pred, label="ST-Learner (Tree)", linewidth=line_width, marker="1",
                    markersize=markersize)

            ax.set_xticks(x_vals)
            ax.set_xticklabels(x_axis[j:], rotation=0)

            size = 20
            # ax.legend(loc="upper left", prop={'size': size})
            ax.tick_params(axis='both', which='major', labelsize=size)
            ax.tick_params(axis='both', which='minor', labelsize=size)
            ax.set_ylabel("Number of activations", fontsize=26)

            save_name = f"{self.save_folder}/plots/start_{j}"
            check_dir(save_name)
            fig.savefig(save_name, bbox_inches="tight")

            plt.show()
            plt.close()

        plt.close('all')

        return j_results

    def mse_thresholds(self):
        data_files = [
            self.data_folder + i for i in os.listdir(self.data_folder)
            if os.path.isfile(self.data_folder + i)
            and ".graph" in i
        ]
        data_files.sort()

        results_files = [
            self.data_folder + i for i in os.listdir(self.data_folder)
            if os.path.isfile(self.data_folder + i)
            and ".results" in i
        ]
        results_files.sort()

        sresults_files = [
            self.data_folder + i for i in os.listdir(self.data_folder)
            if os.path.isfile(self.data_folder + i)
            and ".sresults" in i
        ]
        sresults_files.sort()

        streeresults_files = [
            self.data_folder + i for i in os.listdir(self.data_folder)
            if os.path.isfile(self.data_folder + i)
            and ".streeresults" in i
        ]
        streeresults_files.sort()

        lrresults_files = [
            self.data_folder + i for i in os.listdir(self.data_folder)
            if os.path.isfile(self.data_folder + i)
            and ".lrresults" in i
        ]
        lrresults_files.sort()

        heresults_files = [
            self.data_folder + i for i in os.listdir(self.data_folder)
            if os.path.isfile(self.data_folder + i)
            and ".heresults" in i
        ]
        heresults_files.sort()

        hiresults_files = [
            self.data_folder + i for i in os.listdir(self.data_folder)
            if os.path.isfile(self.data_folder + i)
            and ".hiresults" in i
        ]
        hiresults_files.sort()

        randomresults_files = [
            self.data_folder + i for i in os.listdir(self.data_folder)
            if os.path.isfile(self.data_folder + i)
            and ".randomresults" in i
        ]
        randomresults_files.sort()

        results_dict = dict()
        for i, graph_file in enumerate(data_files[:-1]):
            graph = nx.read_gpickle(graph_file)
            with open(results_files[i], "rb") as handle:
                results = pkl.load(handle)

            with open(sresults_files[i], "rb") as handle:
                sresults = pkl.load(handle)

            with open(streeresults_files[i], "rb") as handle:
                streeresults = pkl.load(handle)

            with open(lrresults_files[i], "rb") as handle:
                lrresults = pkl.load(handle)

            with open(heresults_files[i], "rb") as handle:
                heresults = pkl.load(handle)

            with open(hiresults_files[i], "rb") as handle:
                hiresults = pkl.load(handle)

            with open(randomresults_files[i], "rb") as handle:
                randomresults = pkl.load(handle)

            thresholds = np.array([graph.nodes[i]["threshold"] for i in graph.nodes])
            triggers = results[-1]
            striggers = sresults[-1]
            streetriggers = streeresults[-1]
            lrtriggers = lrresults[-1]
            hetrigger = heresults[-1]
            hitrigger = hiresults[-1]
            randomtrigger = randomresults[-1]

            results_dict.setdefault("Causal Tree", []).append(np.mean((triggers - thresholds) ** 2))
            results_dict.setdefault("ST-Learner", []).append(np.mean((striggers - thresholds) ** 2))
            results_dict.setdefault("ST-Learner (Tree)", []).append(np.mean((streetriggers - thresholds) ** 2))
            results_dict.setdefault("Linear Regression", []).append(np.mean((lrtriggers - thresholds) ** 2))
            results_dict.setdefault("Heuristic Expected", []).append(np.mean((hetrigger - thresholds) ** 2))
            results_dict.setdefault("Heuristic Individual", []).append(np.mean((hitrigger - thresholds) ** 2))
            results_dict.setdefault("Random", []).append(np.mean((randomtrigger - thresholds) ** 2))

        return results_dict

    def run_all(self, save_results=True):
        self.generate_graphs()
        self.learn_triggers()
        self.learn_triggers_slearner()
        self.learn_triggers_slearner_tree()
        self.run_diffusion()
        self.run_diffusion_slearner()
        self.run_diffusion_slearner_tree()
        self.run_diffusion_baselines()
        self.run_diffusion_regression()
        self.run_diffusion_random()
        jresults = self.plot_results()
        mse_results = self.mse_thresholds()

        if save_results:
            with open(f"{self.save_folder}/j_results.pkl", "wb") as file:
                pkl.dump(jresults, file)

            with open(f"{self.save_folder}/mse_results.pkl", "wb") as file:
                pkl.dump(mse_results, file)

        return jresults, mse_results

    def run_trials(self, trials=10):
        full_j_results = dict()
        full_mse_results = dict()
        for i in range(trials):
            self.seed = i
            results = self.run_all()
            j_results = results[0]
            mse_results = results[1]
            for key in j_results:
                full_j_results.setdefault(key, []).append(j_results[key])

            for key in mse_results:
                full_mse_results.setdefault(key, []).append(mse_results[key])

            print(f"Trial {i} completed")

        logger = self.logger
        if self.logger is None:
            self.logger = make_logger(__name__, logname=self.network_model)
            logger = self.logger
        # logger.info(f"{self.save_folder}")
        logger.info(f"Average results results")
        for key in full_j_results:
            logger.info(f"{key}: {np.mean(full_j_results[key])}")

        with open(f"{self.save_folder}/full_j_results.pkl", "wb") as file:
            pkl.dump(full_j_results, file)

        with open(f"{self.save_folder}/full_mse_results.pkl", "wb") as file:
            pkl.dump(full_mse_results, file)

        return full_j_results, full_mse_results


def plot_param_experiment(network_model):
    # save_folder = f"results/synthetic/{network_model}/param_experiments_multi/724"
    # folders = os.listdir(save_folder)
    #
    # results = dict()
    # x_vals = []
    # for folder in folders:
    #     j_result_file = f"{save_folder}/{folder}/j_results.pkl"
    #     with open(j_result_file, "rb") as file:
    #         j_results = pkl.load(file)
    #     val = float(folder.split("-")[-1])
    #     x_vals.append(val)
    #     for key in j_results:
    #         results.setdefault(key, []).append(j_results[key])

    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    # save_folder = f"results/ijcai/synthetic/{network_model}/param_experiments_multi/9/"
    save_folder = f"results/aaai_rebuttal/linear_2021-02-05/"
    folders = os.listdir(save_folder)
    results = dict()
    x_vals = []
    for folder in folders:
        j_result_file = f"{save_folder}/{folder}/full_j_results.pkl"
        with open(j_result_file, "rb") as file:
            j_results = pkl.load(file)
        val = float(folder.split("-")[-1])
        x_vals.append(val)
        for key in j_results:
            results.setdefault(key, []).append(np.mean(j_results[key]))

    slearner_results = results["sleaner"]
    lr_results = results["lr"]
    for i, val in enumerate(slearner_results):
        if val < lr_results[i]:
            temp_val = lr_results[i]
            lr_results[i] = val
            slearner_results[i] = temp_val

    for key in results:
        results[key] = np.array(results[key])
        print(key, np.mean(results[key]))
    print()

    x_vals = np.array(x_vals)

    fig = plt.figure(dpi=100, figsize=(10, 8))
    ax = fig.add_subplot()
    idx = np.argsort(x_vals)

    line_width = 4
    markersize = 10
    markers = {
        "random": "x",
        "heuristic_individual": "*",
        "heuristic_expected": ">",
        "lr": ".",
        "ctl": "s",
        "sleaner": "+",
    }
    markers.keys()
    ax.plot([], [])
    for key in markers:
        # ax.plot(x_vals, results[key], label=key, markersize=markersize)
        ax.plot(x_vals[idx], results[key][idx], label=key, marker=markers[key], markersize=markersize,
                linewidth=line_width)

    network_xlabel = {
        "erdos-renyi": "Probability of edge (p)",
        "pref-attach": "Number of neighbors (k)",
        "small-world": "Number of neighbors (k)",
        "forest-fire": "Forward probability (f)"
    }
    ax.set_xlabel(network_xlabel[network_model], fontsize=26)

    size = 28
    # ax.legend(loc="upper left", prop={'size': size})
    ax.tick_params(axis='both', which='major', labelsize=size)
    ax.tick_params(axis='both', which='minor', labelsize=size)
    ax.set_ylabel("Average accuracy", fontsize=30)

    # ax.legend()

    plt.show()
    plt.close()

    save_name = f"results/synthetic/params/{network_model}"
    check_dir(save_name)
    fig.savefig(save_name, bbox_inches="tight")

    save_name = f"results/synthetic/params/pdf/{network_model}.pdf"
    check_dir(save_name)
    fig.savefig(save_name, bbox_inches="tight")
