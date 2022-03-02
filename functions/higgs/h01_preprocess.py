import pandas as pd
import networkx as nx
import pickle as pkl
from functions.util import *
import gc
import datetime
import random

random.seed(724)
np.random.seed(724)


class HiggsPreprocess:

    def __init__(self, data_folder="/mnt/ext/ctran/data/higgs/", save_folder="/mnt/ext/ctran/higgs/save/"):
        self.data_folder = data_folder
        self.save_folder = save_folder
        self.graph_dict = None

    def run(self):
        if self.data_folder[-1] != "/":
            self.data_folder = self.data_folder + "/"
        if self.save_folder[-1] != "/":
            self.save_folder = self.save_folder + "/"

        LOGGER = make_logger(__name__, logname="higgs_preprocess")
        data_folder = self.data_folder
        save_folder = self.save_folder

        LOGGER.info("Loading data...")
        higgs_network = pd.read_csv(f"{data_folder}higgs-social_network.edgelist", header=None, delimiter=" ")
        higgs_activity = pd.read_csv(f"{data_folder}higgs-activity_time.txt", header=None, delimiter=" ")

        higgs_activity["time"] = pd.to_datetime(higgs_activity[2], unit='s')
        # if self.date_only:
        #     higgs_activity["date"] = higgs_activity["time"].dt.date
        # else:
        #     higgs_activity["date"] = higgs_activity['time'].apply(
        #         lambda dt: datetime.datetime(dt.year, dt.month, dt.day, dt.hour))
        #     # gt_higgs = higgs_activity["date"] >= "2012-07-04"
        #     # higgs_activity = higgs_activity.loc[gt_higgs, :]
        higgs_activity["date"] = higgs_activity['time'].apply(
            lambda dt: datetime.datetime(dt.year, dt.month, dt.day, dt.hour))

        LOGGER.info("Building graph...")
        edges = higgs_network[[0, 1]].values
        graph = nx.DiGraph()
        graph.add_edges_from(edges)

        LOGGER.info("Adding initial node attributes...")
        use_features = [
            "centrality", "in_centrality", "out_centrality", "nbr_mean_num_treated",
            "nbr_num_treated", "nbr_mean_treatment_vals",
        ]

        nx.set_node_attributes(graph, 0.0, "label")
        nx.set_node_attributes(graph, 0.0, "treatment")
        nx.set_node_attributes(graph, 0.0, "treatment_num")
        nx.set_node_attributes(graph, 0.0, "tweet_count")
        nx.set_node_attributes(graph, 0.0, "nbr_tweet_count")
        nx.set_node_attributes(graph, 0.0, "nbr_avg_tweet_count")

        nx.set_node_attributes(graph, 0.0, "nbr_mean_num_treated")
        nx.set_node_attributes(graph, 0.0, "nbr_num_treated")
        nx.set_node_attributes(graph, 0.0, "nbr_mean_treatment_vals")

        # features
        centrality = nx.algorithms.centrality.degree_centrality(graph)
        nx.set_node_attributes(graph, centrality, 'centrality')

        in_centrality = nx.algorithms.centrality.in_degree_centrality(graph)
        nx.set_node_attributes(graph, in_centrality, 'in_centrality')

        out_centrality = nx.algorithms.centrality.out_degree_centrality(graph)
        nx.set_node_attributes(graph, out_centrality, 'out_centrality')

        prev_date = None
        graph_dict = dict()
        all_dates = higgs_activity["date"].unique()

        LOGGER.info("Building graphs for all time...")
        for date in all_dates:

            # getting only everything after 2014-07-04
            if date < np.datetime64("2012-07-04"):
                LOGGER.info(f"Skipping {date}")
                continue

            save_file = f"{save_folder}{date}.graph"
            if os.path.isfile(save_file):
                LOGGER.info(f"{save_file} already exists")
                # graph = nx.read_gpickle(save_file)
                graph_dict[date] = save_file
                prev_date = date
                gc.collect()
                continue

            LOGGER.info(f"Building graph for {save_file}")
            # if prev_date is not None:
            #     LOGGER.info(f"Loading previous graph: {prev_date}")
            #     graph = nx.read_gpickle(f"{save_folder}{prev_date}.graph")

            data_df = higgs_activity[higgs_activity["date"] < date]
            label_df = higgs_activity[higgs_activity["date"] <= date]

            tweeted_users = label_df[0].unique()
            rt_users = label_df.loc[label_df[3] == "RT", 1]
            higgs_users = set(tweeted_users).union(set(rt_users))

            data_df = data_df[data_df[0].isin(higgs_users)]

            for idx, row in data_df.groupby(by=0).count().iterrows():
                graph.nodes[idx]["tweet_count"] = row[1]

            for user in higgs_users:
                graph.nodes[user]["label"] = 1

            LOGGER.info(f"Getting treatment values...")
            for node in graph.nodes:
                out_edges = graph.out_edges(node)
                nbrs = [u for (v, u) in out_edges]
                nbr_labels = [
                    graph.nodes[x]["label"] for x in nbrs if "label" in graph.nodes[x]
                ]
                nbr_tweet_count = [
                    graph.nodes[x]["tweet_count"] for x in nbrs
                    if "tweet_count" in graph.nodes[x]
                ]
                mean_nbr = 0.0
                num_nbr = 0.0
                if len(nbr_labels) > 0:
                    num_nbr = np.sum(nbr_labels)
                    mean_nbr = num_nbr / len(nbrs)
                graph.nodes[node]["treatment"] = mean_nbr
                graph.nodes[node]["treatment_num"] = num_nbr

                num_count = 0.0
                mean_count = 0.0
                if len(nbr_labels) > 0:
                    num_count = np.sum(nbr_tweet_count)
                    mean_count = num_count / len(nbrs)
                graph.nodes[node]["nbr_tweet_count"] = num_count
                graph.nodes[node]["nbr_avg_tweet_count"] = mean_count

            LOGGER.info(f"Getting treatment from neighbors as features...")
            for node in graph.nodes:
                out_edges = graph.out_edges(node)
                nbrs = [u for (v, u) in out_edges]
                nbr_treatment = [
                    graph.nodes[x]["treatment_num"] for x in nbrs
                    if "treatment_num" in graph.nodes[x]
                ]
                nbr_treatment_vals = [
                    graph.nodes[x]["treatment"] for x in nbrs
                    if "treatment" in graph.nodes[x]
                ]

                mean_num_treat = 0.0
                num_treat = 0.0
                mean_val_treat = 0.0
                if len(nbr_treatment) > 0:
                    num_treat = np.sum(np.array(nbr_treatment) > 0)
                    mean_num_treat = np.mean(nbr_treatment)
                    mean_val_treat = np.mean(nbr_treatment_vals)
                graph.nodes[node]["nbr_num_treated"] = num_treat
                graph.nodes[node]["nbr_mean_num_treated"] = mean_num_treat
                graph.nodes[node]["nbr_mean_treatment_vals"] = mean_val_treat

            save_file = f"{save_folder}{date}.graph"
            check_dir(save_file)
            nx.write_gpickle(graph, save_file)

            LOGGER.info(f"Finished writing {save_file}")
            graph_dict[date] = save_file
            prev_date = date

        self.graph_dict = graph_dict
        self.save()

    def save(self):
        with open(self.save_folder + "graph_names.pkl", "wb") as handle:
            pkl.dump(self.graph_dict, handle)
