import os
import pickle

import numpy as np
import pandas as pd
from scipy.special import softmax

import matplotlib.pyplot as plt
import seaborn as sns


def load_pkl(path):
    with open(path, "rb") as h:
        data = pickle.load(h)
    return data


def load_data(template_):
    dirs = os.listdir(
        os.path.join(
            "./logs/evaluations/runs",
            template_,
        )
    )
    dirs = sorted(dirs, reverse=True)

    y_true = load_pkl(
        os.path.join(
            "./logs/evaluations/runs",
            template_,
            dirs[0],
            "test_y_true.pkl",
        )
    )
    logits = load_pkl(
        os.path.join(
            "./logs/evaluations/runs",
            template_,
            dirs[0],
            "test_logits.pkl",
        )
    )

    return np.array(y_true), np.stack(logits)


class Accuracy:
    def __init__(self, domain=None):
        self.total = 0
        self.correct = 0
        self.domain = domain

    def update(self, logits, y_true, y_true_org, logits_org):
        self.total += len(y_true)
        self.correct += np.sum(np.array(y_true) == np.argmax(logits, axis=1))

    def get_score(self):
        return self.correct * 100 / self.total




class DeltaNew:
    def __init__(self, domain=None):
        self.domain = domain

    def update(self, logits, y_true, y_true_org, logits_org):
        y_prob_org = softmax(logits_org, axis=1)
        tmp_prob = [y_prob_org[m, y_true_org[m]] for m in range(y_true_org.shape[0])]
        p_real = np.mean(tmp_prob)

        y_prob = softmax(logits, axis=1)
        p_gen = [y_prob[m, y_true[m]] for m in range(y_true.shape[0])]
        self.score = np.array(p_gen) - p_real

    def get_score(self):
        return np.mean(self.score)



if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Define the training/evaluation settings."
    )
    parser.add_argument(
        "--gen_name",
        required=True,
        help="Generated logits run name.",
    )
    parser.add_argument(
        "--org_name",
        required=True,
        help="Original logits run name.",
    )
    parser.add_argument(
        "--output_dir",
        required=False,
        help="Path to store the results.",
        default="results"
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    dataset = {}
    store_path = args.output_dir
    metrics = {
        "accuracy": Accuracy(),
        "delta_new": DeltaNew(),
    }
    models = [
        args.org_name,
        args.gen_name
    ]

    for k, v in metrics.items():
        dataset[k] = []

    for model in models:
        y_true, logits = load_data(model)
        y_true_org, logits_org = load_data(args.org_name)

        for k, _ in metrics.items():
            metrics[k].update(logits, y_true, y_true_org, logits_org)
            dataset[k].append(metrics[k].get_score())

    df = pd.DataFrame(dataset)
    df.index = models
    df.to_csv(os.path.join(store_path, f"objects_{args.gen_name}_all.csv"))


    tex = df.to_latex()
    with open(os.path.join(store_path, f"objects_{args.gen_name}_overall.tex"), "w") as h:
        h.write(tex)
