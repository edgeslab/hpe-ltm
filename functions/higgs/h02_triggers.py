from functions.higgs.h01_preprocess import *
from functions.util import *
from CTL.causal_tree_learn import CausalTree
from functions.estimators import *

np.random.seed(724)
random.seed(724)


class HiggsTriggers:

    def __init__(self, data_folder="/mnt/ext/ctran/data/higgs/", save_folder="/mnt/ext/ctran/higgs/save/", seed=724):

        self.data_folder = data_folder
        self.save_folder = save_folder
        self.seed = seed

        self.trigger_dict = None

    def run(self):
        if self.data_folder[-1] != "/":
            self.data_folder = self.data_folder + "/"
        if self.save_folder[-1] != "/":
            self.save_folder = self.save_folder + "/"

        LOGGER = make_logger(__name__, logname="higgs_triggers")

        prior_results = self.save_folder + "graph_names.pkl"
        if not os.path.isfile(prior_results):
            LOGGER.info("No graphs created. Running preprocessing now...")
            higgs_process = HiggsPreprocess(data_folder=self.data_folder, save_folder=self.save_folder)
            higgs_process.run()

        with open(prior_results, "rb") as file:
            graphs = pkl.load(file)

        trigger_results = dict()

        non_atts = ["label", "treatment", "treatment_num"]
        ignore_atts = ["userCreatedAt"]
        for key in graphs:
            file = graphs[key]
            save_file = file.replace(".graph", "")
            save_file = save_file + ".results"

            if os.path.isfile(save_file):
                # LOGGER.info(f"{save_file} already exists (tree already built)")
                LOGGER.info(f"{save_file} already exists (tree already built)")
                # with open(save_file, "rb") as pklfile:
                #     data = pkl.load(pklfile)
                trigger_results[save_file] = save_file
                continue

            LOGGER.info(f"Reading {file}...")
            graph = nx.read_gpickle(file)
            x = None
            y = np.zeros(len(graph))
            t = np.zeros(len(graph))
            for i, user in enumerate(graph.nodes):
                node_atts = graph.nodes[user]
                atts = []
                for att in node_atts:
                    if att in non_atts or att in ignore_atts:
                        continue
                    atts.append(node_atts[att])
                if len(atts) < 1:
                    continue
                if x is None:
                    x = np.zeros((len(graph), len(atts)))
                x[i] = np.array(atts)
                y[i] = node_atts["label"]
                t[i] = node_atts["treatment"]

            # subsample the graph
            np.random.seed(self.seed)
            perm = np.random.permutation(x.shape[0])
            perm = perm[:int(x.shape[0] * 0.01)]
            train_x = x[perm]
            y = y[perm]
            t = t[perm]

            LOGGER.info(f"{file}: {x.shape}, {train_x.shape}, {y.shape}, {t.shape}")

            LOGGER.info(f"Building tree...")
            ct = CausalTree(cont=True, split_size=0.2, weight=0.2, val_honest=True, seed=self.seed)
            ct.fit(train_x, y, t)
            # prediction, triggers = ct.predict(x)
            prediction = ct.predict(x)
            triggers = ct.get_triggers(x)
            with open(save_file, 'wb') as pklfile:
                pkl.dump([ct, prediction, triggers], pklfile)

            # LOGGER.info(f"Finished building tree for {save_file}")
            LOGGER.info(f"Finished building tree for {save_file}")

            # trigger_results[save_file] = [ct, prediction, triggers]
            trigger_results[save_file] = save_file

        self.trigger_dict = trigger_results
        self.save()

    def save(self):
        with open(self.save_folder + "trigger_names.pkl", "wb") as handle:
            pkl.dump(self.trigger_dict, handle)

    def slearner(self, rerun=False):
        if self.data_folder[-1] != "/":
            self.data_folder = self.data_folder + "/"
        if self.save_folder[-1] != "/":
            self.save_folder = self.save_folder + "/"

        LOGGER = make_logger(__name__, logname="higgs_triggers")

        prior_results = self.save_folder + "graph_names.pkl"
        if not os.path.isfile(prior_results):
            LOGGER.info("No graphs created. Running preprocessing now...")
            higgs_process = HiggsPreprocess(data_folder=self.data_folder, save_folder=self.save_folder)
            higgs_process.run()

        with open(prior_results, "rb") as file:
            graphs = pkl.load(file)

        trigger_results = dict()

        non_atts = ["label", "treatment", "treatment_num"]
        ignore_atts = ["userCreatedAt"]
        for key in graphs:
            file = graphs[key]
            save_file = file.replace(".graph", "")
            save_file = save_file + ".sresults"

            if os.path.isfile(save_file) and not rerun:
                # LOGGER.info(f"{save_file} already exists (tree already built)")
                LOGGER.info(f"{save_file} already exists (learner already built)")
                # with open(save_file, "rb") as pklfile:
                #     data = pkl.load(pklfile)
                trigger_results[save_file] = save_file
                continue

            LOGGER.info(f"Reading {file}...")
            graph = nx.read_gpickle(file)
            x = None
            y = np.zeros(len(graph))
            t = np.zeros(len(graph))
            for i, user in enumerate(graph.nodes):
                node_atts = graph.nodes[user]
                atts = []
                for att in node_atts:
                    if att in non_atts or att in ignore_atts:
                        continue
                    atts.append(node_atts[att])
                if len(atts) < 1:
                    continue
                if x is None:
                    x = np.zeros((len(graph), len(atts)))
                x[i] = np.array(atts)
                y[i] = node_atts["label"]
                t[i] = node_atts["treatment"]

            # subsample the graph
            np.random.seed(self.seed)
            perm = np.random.permutation(x.shape[0])
            perm = perm[:int(x.shape[0] * 0.01)]
            train_x = x[perm]
            y = y[perm]
            t = t[perm]

            # LOGGER.info(f"{file}: {x.shape}, {train_x.shape}, {y.shape}, {t.shape}")
            LOGGER.info(f"{file}: {x.shape}, {y.shape}, {t.shape}, {train_x.shape}")

            LOGGER.info(f"Building SLearner...")
            slearner = STLearner()
            slearner.fit(train_x, y, t)
            # slearner.fit(x, y, t)
            triggers = slearner.triggers(x, batch_size=10000, verbose=True)
            with open(save_file, "wb") as pklfile:
                pkl.dump([slearner, triggers], pklfile)

            LOGGER.info(f"finished building Slearner for {save_file}")
            trigger_results[save_file] = save_file

        self.save_slearner(trigger_results)

    def slearner_tree(self, rerun=False):
        if self.data_folder[-1] != "/":
            self.data_folder = self.data_folder + "/"
        if self.save_folder[-1] != "/":
            self.save_folder = self.save_folder + "/"

        LOGGER = make_logger(__name__, logname="higgs_triggers")

        prior_results = self.save_folder + "graph_names.pkl"
        if not os.path.isfile(prior_results):
            LOGGER.info("No graphs created. Running preprocessing now...")
            higgs_process = HiggsPreprocess(data_folder=self.data_folder, save_folder=self.save_folder)
            higgs_process.run()

        with open(prior_results, "rb") as file:
            graphs = pkl.load(file)

        trigger_results = dict()

        non_atts = ["label", "treatment", "treatment_num"]
        ignore_atts = ["userCreatedAt"]
        for key in graphs:
            file = graphs[key]
            save_file = file.replace(".graph", "")
            save_file = save_file + ".streeresults"

            if os.path.isfile(save_file) and not rerun:
                # LOGGER.info(f"{save_file} already exists (tree already built)")
                LOGGER.info(f"{save_file} already exists (learner already built)")
                # with open(save_file, "rb") as pklfile:
                #     data = pkl.load(pklfile)
                trigger_results[save_file] = save_file
                continue

            LOGGER.info(f"Reading {file}...")
            graph = nx.read_gpickle(file)
            x = None
            y = np.zeros(len(graph))
            t = np.zeros(len(graph))
            for i, user in enumerate(graph.nodes):
                node_atts = graph.nodes[user]
                atts = []
                for att in node_atts:
                    if att in non_atts or att in ignore_atts:
                        continue
                    atts.append(node_atts[att])
                if len(atts) < 1:
                    continue
                if x is None:
                    x = np.zeros((len(graph), len(atts)))
                x[i] = np.array(atts)
                y[i] = node_atts["label"]
                t[i] = node_atts["treatment"]

            # subsample the graph
            np.random.seed(self.seed)
            perm = np.random.permutation(x.shape[0])
            perm = perm[:int(x.shape[0] * 0.01)]
            train_x = x[perm]
            y = y[perm]
            t = t[perm]

            # LOGGER.info(f"{file}: {x.shape}, {train_x.shape}, {y.shape}, {t.shape}")
            LOGGER.info(f"{file}: {x.shape}, {y.shape}, {t.shape}, {train_x.shape}")

            LOGGER.info(f"Building SLearner...")
            slearner = STLearner(model=DecisionTreeRegressor, params={"min_samples_leaf": 10})
            slearner.fit(train_x, y, t)
            # slearner.fit(x, y, t)
            triggers = slearner.triggers(x, batch_size=10000, verbose=True)
            with open(save_file, "wb") as pklfile:
                pkl.dump([slearner, triggers], pklfile)

            LOGGER.info(f"finished building Slearner for {save_file}")
            trigger_results[save_file] = save_file

        self.save_slearner(trigger_results, name="stree_trigger_names.pkl")

    def save_slearner(self, trigger_results, name="slearner_trigger_names.pkl"):
        with open(self.save_folder + name, "wb") as handle:
            pkl.dump(trigger_results, handle)