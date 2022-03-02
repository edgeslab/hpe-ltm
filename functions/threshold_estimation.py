import random
import networkx as nx
from functions.util import *


def heuristic_expected(graph, seed=724, use_int=True):
    np.random.seed(seed)
    random.seed(seed)
    all_iv = np.zeros(len(graph))
    for i, node in enumerate(graph.nodes):
        if nx.is_directed(graph):
            neighborhood = list(graph.out_edges(node))
        else:
            neighborhood = list(graph.edges(node))
        if len(neighborhood) == 0:
            continue
        num_samples = np.random.randint(1, len(neighborhood) + 1)
        sampled_edges = random.sample(neighborhood, k=num_samples)
        im_v = 0.0
        for edge in sampled_edges:
            if "weight" in graph.edges[edge]:
                pass
            else:
                if use_int:
                    im_v += 1
                else:
                    im_v += 1/len(neighborhood)
        all_iv[i] = im_v

    return np.mean(all_iv)


def sample_expected(graph, choice="", z=1.96, m=0.05, p=0.5, e=0.05, seed=724, use_int=True):
    np.random.seed(seed)
    random.seed(seed)
    N = len(graph)
    n = 30
    if choice == "rsp":
        n = (z / m) ** 2 * p * (1 - p)
    elif choice == "slovin":
        n = N / (1 + N * e ** 2)

    step_size = int(np.ceil(N / n))
    # print(step_size)
    all_iv = []
    for i, node in enumerate(graph.nodes):
        if i % step_size != 0:
            continue
        if nx.is_directed(graph):
            neighborhood = list(graph.out_edges(node))
        else:
            neighborhood = list(graph.edges(node))
        if len(neighborhood) == 0:
            all_iv.append(0)
            continue
        num_samples = np.random.randint(1, len(neighborhood) + 1)
        sampled_edges = random.sample(neighborhood, k=num_samples)
        im_v = 0.0
        for edge in sampled_edges:
            if "weight" in graph.edges[edge]:
                pass
            else:
                if use_int:
                    im_v += 1
                else:
                    im_v += 1/len(neighborhood)
        all_iv.append(im_v)

    return np.mean(all_iv)


def heuristic_individual(graph, seed=724, use_int=True):
    np.random.seed(seed)
    random.seed(seed)
    all_iv = np.zeros(len(graph))
    for i, node in enumerate(graph.nodes):
        if nx.is_directed(graph):
            neighborhood = list(graph.out_edges(node))
        else:
            neighborhood = list(graph.edges(node))
        if len(neighborhood) == 0:
            continue
        num_samples = np.random.randint(1, len(neighborhood) + 1)
        sampled_edges = random.sample(neighborhood, k=num_samples)
        im_v = 0.0
        # print(num_samples, len(neighborhood))
        for edge in sampled_edges:
            if "weight" in graph.edges[edge]:
                pass
            else:
                if use_int:
                    im_v += 1
                else:
                    im_v += 1/len(neighborhood)
        all_iv[i] = im_v

    a, b = np.percentile(all_iv, 25), np.percentile(all_iv, 75)

    if a == b and not use_int:
        a = b / 2
    elif a == b and use_int:
        b = a * 2

    return a, b


def sample_individual(graph, choice="", z=1.96, m=0.05, p=0.5, e=0.05, seed=724, use_int=True):
    np.random.seed(seed)
    random.seed(seed)
    N = len(graph)
    n = 30
    if choice == "rsp":
        n = (z / m) ** 2 * p * (1 - p)
    elif choice == "slovin":
        n = N / (1 + N * e ** 2)

    step_size = int(np.ceil(N / n))
    all_iv = []
    for i, node in enumerate(graph.nodes):
        if i % step_size != 0:
            continue
        if nx.is_directed(graph):
            neighborhood = list(graph.out_edges(node))
        else:
            neighborhood = list(graph.edges(node))
        if len(neighborhood) == 0:
            all_iv.append(0)
            continue
        num_samples = np.random.randint(1, len(neighborhood) + 1)
        sampled_edges = random.sample(neighborhood, k=num_samples)
        im_v = 0.0
        for edge in sampled_edges:
            if "weight" in graph.edges[edge]:
                pass
            else:
                if use_int:
                    im_v += 1
                else:
                    im_v += 1/len(neighborhood)
        all_iv.append(im_v)

    a, b = np.percentile(all_iv, 25), np.percentile(all_iv, 75)

    if a == b and not use_int:
        a = b / 2
    elif a == b and use_int:
        b = a * 2

    return a, b
