from functions.threshold import *
from functions.higgs.h02_triggers import *
import pprint as prettyprinter

pp = prettyprinter.PrettyPrinter()
pprint = pp.pprint


class HiggsDiffusion:

    def __init__(self, data_folder="/mnt/ext/ctran/data/higgs/", save_folder="/mnt/ext/ctran/higgs/save/", seed=724):

        self.data_folder = data_folder
        self.save_folder = save_folder
        self.seed = seed

    def run(self):

        LOGGER = make_logger(__name__, logname="higgs_diffusion")

        prior_results = self.save_folder + "trigger_names.pkl"
        if not os.path.isfile(prior_results):
            LOGGER.info("No graphs created. Running preprocessing now...")
            ht = HiggsTriggers(data_folder=self.data_folder, save_folder=self.save_folder, seed=self.seed)
            ht.run()
        with open(self.save_folder + "trigger_names.pkl", "rb") as file:
            results_dict = pkl.load(file)

        data_folder = self.save_folder
        data_files = [
            data_folder + i for i in os.listdir(data_folder)
            if os.path.isfile(data_folder + i) and ".graph" in i
        ]
        results_files = [
            data_folder + i for i in os.listdir(data_folder)
            if os.path.isfile(data_folder + i) and ".results" in i
        ]
        data_files.sort()
        results_files.sort()

        LOGGER.info("Reading the last graph file...")
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

        LOGGER.info("Reading all files to get number of infected...")

        num_infect_all = []
        infected_nodes_list = []
        for key in results_dict:
            graph_key = key.replace(".results", ".graph")
            LOGGER.info(f"Reading graph {graph_key}...")
            graph = nx.read_gpickle(graph_key)
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

        for j, key in enumerate(results_dict):
            LOGGER.info(f"Reading results from: {key}...")
            with open(key, "rb") as file:
                results = pkl.load(file)
            graph_key = key.replace(".results", ".graph")
            graph = nx.read_gpickle(graph_key)

            labeled_users = [
                user for user in graph.nodes if "label" in graph.nodes[user]
            ]
            triggers = results[-1]

            # use the final graph instead
            thresh_model = ThresholdModel(end_graph)
            infected_nodes = [i for i in labeled_users if graph.nodes[i]["label"] == 1]

            thresh_model.set_initial_config(infected_nodes)
            thresh_model.set_thresholds(labeled_users, triggers)

            LOGGER.info(f"Running diffusion for: {key}")
            thresh_model.run(len(results_dict) - j)
            LOGGER.info(f"Finished running diffusion for: {results_files[j]}")

            num_nodes = []
            iteration_dict = thresh_model.iteration_dict
            for key in iteration_dict:
                infect_nodes = iteration_dict[key]
                # num_nodes.append(
                #     len(list(set(infect_nodes).intersection(end_labeled_users))))
                num_nodes.append(
                    len(list(set(infect_nodes).intersection(end_infected_nodes))))

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

        self.save({"track": track, "true_track": true_track, "track_sim": track_sim,
                   "track_sim2": track_sim2})

    def save(self, d):
        with open(self.save_folder + "diffusion_results.pkl", "wb") as handle:
            pkl.dump(d, handle)

    def slearner(self):

        LOGGER = make_logger(__name__, logname="higgs_diffusion")

        prior_results = self.save_folder + "trigger_names.pkl"
        if not os.path.isfile(prior_results):
            LOGGER.info("No graphs created. Running preprocessing now...")
            ht = HiggsTriggers(data_folder=self.data_folder, save_folder=self.save_folder, seed=self.seed)
            ht.run()
        with open(self.save_folder + "trigger_names.pkl", "rb") as file:
            results_dict = pkl.load(file)

        data_folder = self.save_folder
        data_files = [
            data_folder + i for i in os.listdir(data_folder)
            if os.path.isfile(data_folder + i) and ".graph" in i
        ]
        results_files = [
            data_folder + i for i in os.listdir(data_folder)
            if os.path.isfile(data_folder + i) and ".sresults" in i
        ]
        data_files.sort()
        results_files.sort()

        LOGGER.info("Reading the last graph file...")
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

        LOGGER.info("Reading all files to get number of infected...")

        num_infect_all = []
        infected_nodes_list = []
        for key in results_dict:
            graph_key = key.replace(".results", ".graph")
            LOGGER.info(f"Reading graph {graph_key}...")
            graph = nx.read_gpickle(graph_key)
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

        for j, key in enumerate(results_dict):
            LOGGER.info(f"Reading results from: {key}...")
            with open(key, "rb") as file:
                results = pkl.load(file)
            graph_key = key.replace(".results", ".graph")
            graph = nx.read_gpickle(graph_key)

            labeled_users = [
                user for user in graph.nodes if "label" in graph.nodes[user]
            ]
            triggers = results[-1]

            LOGGER.info(f"Min: {np.min(triggers)}, Mean: {np.mean(triggers)}, Max: {np.max(triggers)}")

            # use the final graph instead
            thresh_model = ThresholdModel(end_graph)
            infected_nodes = [i for i in labeled_users if graph.nodes[i]["label"] == 1]

            thresh_model.set_initial_config(infected_nodes)
            thresh_model.set_thresholds(labeled_users, triggers)

            LOGGER.info(f"Running diffusion for: {key}")
            thresh_model.run(len(results_dict) - j)
            LOGGER.info(f"Finished running diffusion for: {results_files[j]}")

            num_nodes = []
            iteration_dict = thresh_model.iteration_dict
            for key in iteration_dict:
                infect_nodes = iteration_dict[key]
                # num_nodes.append(
                #     len(list(set(infect_nodes).intersection(end_labeled_users))))
                num_nodes.append(
                    len(list(set(infect_nodes).intersection(end_infected_nodes))))

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

        self.save_slearner({"track": track, "true_track": true_track, "track_sim": track_sim,
                   "track_sim2": track_sim2})

    def slearner_tree(self):

        LOGGER = make_logger(__name__, logname="higgs_diffusion")

        prior_results = self.save_folder + "trigger_names.pkl"
        if not os.path.isfile(prior_results):
            LOGGER.info("No graphs created. Running preprocessing now...")
            ht = HiggsTriggers(data_folder=self.data_folder, save_folder=self.save_folder, seed=self.seed)
            ht.run()
        with open(self.save_folder + "trigger_names.pkl", "rb") as file:
            results_dict = pkl.load(file)

        data_folder = self.save_folder
        data_files = [
            data_folder + i for i in os.listdir(data_folder)
            if os.path.isfile(data_folder + i) and ".graph" in i
        ]
        results_files = [
            data_folder + i for i in os.listdir(data_folder)
            if os.path.isfile(data_folder + i) and ".streeresults" in i
        ]
        data_files.sort()
        results_files.sort()

        LOGGER.info("Reading the last graph file...")
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

        LOGGER.info("Reading all files to get number of infected...")

        num_infect_all = []
        infected_nodes_list = []
        for key in results_dict:
            graph_key = key.replace(".results", ".graph")
            LOGGER.info(f"Reading graph {graph_key}...")
            graph = nx.read_gpickle(graph_key)
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

        for j, key in enumerate(results_dict):
            LOGGER.info(f"Reading results from: {key}...")
            with open(key, "rb") as file:
                results = pkl.load(file)
            graph_key = key.replace(".results", ".graph")
            graph = nx.read_gpickle(graph_key)

            labeled_users = [
                user for user in graph.nodes if "label" in graph.nodes[user]
            ]
            triggers = results[-1]

            LOGGER.info(f"Min: {np.min(triggers)}, Mean: {np.mean(triggers)}, Max: {np.max(triggers)}")

            # use the final graph instead
            thresh_model = ThresholdModel(end_graph)
            infected_nodes = [i for i in labeled_users if graph.nodes[i]["label"] == 1]

            thresh_model.set_initial_config(infected_nodes)
            thresh_model.set_thresholds(labeled_users, triggers)

            LOGGER.info(f"Running diffusion for: {key}")
            thresh_model.run(len(results_dict) - j)
            LOGGER.info(f"Finished running diffusion for: {results_files[j]}")

            num_nodes = []
            iteration_dict = thresh_model.iteration_dict
            for key in iteration_dict:
                infect_nodes = iteration_dict[key]
                # num_nodes.append(
                #     len(list(set(infect_nodes).intersection(end_labeled_users))))
                num_nodes.append(
                    len(list(set(infect_nodes).intersection(end_infected_nodes))))

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

        self.save_slearner({"track": track, "true_track": true_track, "track_sim": track_sim,
                   "track_sim2": track_sim2}, name="stree_diffusion_results.pkl")

    def save_slearner(self, d, name="slearner_diffusion_results.pkl"):
        with open(self.save_folder + name, "wb") as handle:
            pkl.dump(d, handle)
