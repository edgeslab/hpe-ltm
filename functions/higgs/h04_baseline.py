from functions.higgs.h03_diffusion import *
from functions.threshold import *
from functions.threshold_estimation import *
from sklearn.linear_model import LinearRegression

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


class HiggsBaselines:

    def __init__(self, data_folder="/AD-HOME/ctran29/data/data/higgs/", save_folder="/AD-HOME/ctran29/data/higgs/save/", seed=724,
                 save_results=True, skip_non_lr=False):
        self.data_folder = data_folder
        self.save_folder = save_folder
        self.seed = seed
        self.save_results = save_results
        self.skip_non_lr = skip_non_lr

    def run(self):
        logger = make_logger(__name__, logname="higgs_baselines")

        with open(self.save_folder + "diffusion_results.pkl", "rb") as file:
            results_dict = pkl.load(file)

        predict = results_dict["track"]
        true = results_dict["true_track"]

        skip = len(predict) / 8

        data_folder = self.save_folder
        flow_folder = self.save_folder

        data_files = [
            data_folder + i for i in os.listdir(flow_folder)
            if os.path.isfile(data_folder + i) and ".graph" in i
        ]
        results_files = [
            data_folder + i for i in os.listdir(flow_folder)
            if os.path.isfile(data_folder + i) and ".results" in i
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
            if i > len(results_files):
                continue
            logger.info(f"i: {i}, len(results_files): {len(results_files)}, file: {f}")
            graph = nx.read_gpickle(f)
            l_users = [
                user for user in graph.nodes
                if "label" in graph.nodes[user] and len(graph.nodes[user])
            ]
            labels = [graph.nodes[user]["label"] for user in l_users]
            num_infect_all.append(np.sum(labels))
            nodes = [user for user in l_users if graph.nodes[user]["label"] == 1]
            infected_nodes_list.append(nodes)

        est_track = dict()

        # ----------------------------------------------------------------
        # Heuristic baselines
        # ----------------------------------------------------------------

        track = dict()
        track_end = dict()
        true_track = dict()
        track_acc = dict()
        track_sims = dict()

        if not self.skip_non_lr:
            counter = -1
            logger.info("Running Heuristic baselines")
            for estimator in estimators:
                logger.info(f"Running {estimator}")
                est_name = est_dict[estimator]

                params = dict()
                save_results_key = est_name
                if estimator == sample_expected or estimator == sample_individual:
                    counter += 1
                    if counter > 2:
                        counter = 0
                    params = {"choice": sample_params[counter]}
                    save_results_key = save_results_key + f"_{sample_params[counter]}"

                # track = dict()
                # track_end = dict()
                # true_track = dict()
                # track_acc = dict()
                track_sims = dict()

                logger.info(f"Working on {save_results_key}...")
                for j, file in enumerate(data_files):
                    if j % skip != 0:
                        continue
                    if file == data_files[-1]:
                        continue

                    logger.info(f"Reading graph from: {file}...")
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
                        logger.info(threshold_estimation)
                    else:
                        threshold_min, threshold_max = estimator(graph, use_int=False, **params)
                        save_file = file.replace(".graph", "")
                        save_file = save_file + ".hiresults"
                        thresholds = thresh_model.set_thresholds_estimation(threshold_min, threshold_max)
                        with open(save_file, "wb") as pklfile:
                            pkl.dump([estimator, thresholds], pklfile)
                        logger.info(threshold_min, threshold_max)
                    thresh_model.run(len(data_files) - j, use_int=False)

                    num_nodes = []
                    intersect_nodes = []
                    iteration_dict = thresh_model.iteration_dict
                    for key in iteration_dict:
                        infect_nodes = iteration_dict[key]
                        num_nodes.append(
                            len(list(set(infect_nodes).intersection(end_labeled_users))))
                        intersect_nodes.append(
                            len(list(set(infect_nodes).intersection(end_infected_nodes))))

                    all_sims = []
                    for key in iteration_dict:
                        infect_nodes = iteration_dict[key]

                        intersection = set(infect_nodes).intersection(set(infected_nodes_list[key + j]))
                        union = set(infect_nodes).union(set(infected_nodes_list[key + j]))

                        prop = len(intersection) / len(union)

                        all_sims.append(prop)

                    track_sims[j] = all_sims

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
                    track_acc[j] = intersect_nodes
                est_track[save_results_key] = [track, true_track, track_acc, track_sims]

        # ----------------------------------------------------------------
        # Random Baseline
        # ----------------------------------------------------------------

        track = dict()
        true = dict()
        track_acc = dict()
        track_sims = dict()

        if not self.skip_non_lr:
            logger.info("Running random baseline")
            for j, file in enumerate(data_files):
                if j % skip != 0:
                    continue
                if file == data_files[-1]:
                    continue
                logger.info(f"Reading results from: {results_files[j]}...")

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
                intersect_nodes = []
                iteration_dict = thresh_model.iteration_dict
                for key in iteration_dict:
                    infect_nodes = iteration_dict[key]
                    num_nodes.append(
                        len(list(set(infect_nodes).intersection(end_labeled_users))))
                    intersect_nodes.append(
                        len(list(set(infect_nodes).intersection(end_infected_nodes))))

                all_sims = []
                for key in iteration_dict:
                    infect_nodes = iteration_dict[key]

                    intersection = set(infect_nodes).intersection(set(infected_nodes_list[key + j]))
                    union = set(infect_nodes).union(set(infected_nodes_list[key + j]))

                    prop = len(intersection) / len(union)

                    all_sims.append(prop)

                track_sims[j] = all_sims

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
                track_acc[j] = intersect_nodes

            est_track["random"] = [track, true_track, track_acc, track_sims]

        # ----------------------------------------------------------------
        # Linear regression baseline
        # ----------------------------------------------------------------

        track = dict()
        track_end = dict()
        true_track = dict()
        track_acc = dict()
        track_sims = dict()
        track_sim = dict()
        track_sim2 = dict()

        logger.info("Running linear regression baseline")
        for j, file in enumerate(data_files):
            if j % skip != 0:
                continue
            if file == data_files[-1]:
                continue
            graph = nx.read_gpickle(file)

            labeled_users = [
                user for user in graph.nodes if "label" in graph.nodes[user]
            ]
            # use the final graph instead
            thresh_model = ThresholdModel(end_graph)
            infected_nodes = [i for i in labeled_users if graph.nodes[i]["label"] == 1]

            thresh_model.set_initial_config(infected_nodes)

            # learning thresholds from linear regression
            x = []
            t = []
            for i in infected_nodes:
                node_t = graph.nodes[i]["treatment"]
                node_x = [graph.nodes[i][att] for att in graph.nodes[i] if
                          att not in ["label", "treatment", "treatment_num", "threshold_max", "threshold_min"]]
                t.append(node_t)
                x.append(node_x)
            lr = LinearRegression()
            lr.fit(x, t)

            thresholds = []
            feature_users = []
            for i in labeled_users:
                if i in infected_nodes:
                    thresholds.append(graph.nodes[i]["treatment"])
                    continue
                node_x = np.array([graph.nodes[i][att] for att in graph.nodes[i] if
                                   att not in ["label", "treatment", "treatment_num", "threshold_max",
                                               "threshold_min"]])
                if len(node_x) < 1:
                    continue
                node_t = lr.predict(node_x.reshape(1, -1))[0]
                thresholds.append(node_t)
                feature_users.append(i)
            save_file = file.replace(".graph", "")
            save_file = save_file + ".lrresults"
            with open(save_file, 'wb') as pklfile:
                pkl.dump([lr, thresholds], pklfile)
            logger.info(f"Min: {np.min(thresholds)}, Mean: {np.mean(thresholds)}, Max: {np.max(thresholds)}")
            if not self.save_results:
                continue

            thresh_model.set_thresholds(feature_users, thresholds)
            thresh_model.run(len(data_files) - j)
            num_nodes = []
            iteration_dict = thresh_model.iteration_dict
            for key in iteration_dict:
                infect_nodes = iteration_dict[key]
                num_nodes.append(
                    len(list(set(infect_nodes).intersection(end_labeled_users))))

            all_sims = []
            for key in iteration_dict:
                infect_nodes = iteration_dict[key]

                intersection = set(infect_nodes).intersection(set(infected_nodes_list[key + j]))
                union = set(infect_nodes).union(set(infected_nodes_list[key + j]))

                prop = len(intersection) / len(union)

                all_sims.append(prop)

            track_sims[j] = all_sims

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

            track[j] = num_nodes
            true_track[j] = num_infect

        est_track["regression"] = [track, true_track, track_sims]

        if self.save_results:
            self.save({"results": est_track})

    def save(self, d):
        with open(self.save_folder + "baseline_results.pkl", "wb") as handle:
            pkl.dump(d, handle)
