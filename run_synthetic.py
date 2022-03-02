from functions.synthetic.synthetic import Synthetic, plot_param_experiment
from functions.util import *
import argparse


logger = make_logger(__name__, logname="synthetic_experiments")
parser = argparse.ArgumentParser()
parser.add_argument('--threshold', default="linear")
args = parser.parse_args()
d = vars(args)
# s = Synthetic(network_model="erdos-renyi")
# s.run_all()

# ----------------------------------------------------------------
# Only generate graphs
# ----------------------------------------------------------------

# network_models = [
#     "erdos-renyi",
#     "pref-attachment",
#     "forest-fire",
#     "small-world",
# ]

# for model in network_models:
#     s = Synthetic(network_model=model, seed=7244)
#     # s.generate_graphs()
#     # s.run_all()
#     s.run_trials(10)

# network_models = [
#     "erdos-renyi",
#     "pref-attachment",
#     "forest-fire",
#     "small-world",
# ]
#
# for model in network_models:
#     s = Synthetic(network_model=model)
#     s.run_all()
#     # s.run_trials(10)

# ----------------------------------------------------------------
# Param experiments
# ----------------------------------------------------------------

params = {
    "erdos-renyi": {
        # "p": [0.05, 0.15, 0.25]
        "p": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    },
    "pref-attach": {
        # "m": [5, 15, 25],
        "m": [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    },
    "forest-fire": {
        # "f": [0.1, 0.2, 0.3]
        "f": [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    },
    "small-world": {
        # "k": [5, 10, 15],
        "k": [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    },
}

threshold_function = d["threshold"]

n_samples = 3000
extra_fold = "_2021-02-05"
print(threshold_function)

for network_model in params:
    param_dict = params[network_model]
    logger.info(f"Running experiments for {network_model}")
    for param_key in param_dict:
        param_list = param_dict[param_key]
        for param in param_list:
            logger.info(f"Running {network_model} for {param_key} = {param}")
            network_params = dict()
            network_params["n"] = 1000
            if network_model == "forest-fire":
                network_params["b"] = 0.1
            elif network_model == "small-world":
                network_params["p"] = 0.1
            network_params[param_key] = param
            print(network_model, network_params)
            s = Synthetic(network_model=network_model, network_params=network_params, extra=f"{param_key}-{param}",
                          threshold_function=threshold_function, n_samples=n_samples, extra_fold=extra_fold)
            # results = s.run_all()
            results = s.run_trials(10)

# ----------------------------------------------------------------
# Param results
# ----------------------------------------------------------------

# network_models = ["erdos-renyi", "pref-attach", "small-world", "forest-fire"]
#
# for network_model in network_models:
#     plot_param_experiment(network_model)
