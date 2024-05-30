import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from tqdm import tqdm
from collections import defaultdict
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json


import mltools.utils.cuda_tools as cuda_tools
from mltools.networks import networks

device = cuda_tools.get_freer_device()

import utils
import models
import importlib

from record_utils import record_activations


def test(base_dir):

    device = cuda_tools.get_freer_device(verbose=False)
    classifier = load_classifier(device)

    rep = 64
    _config = base_dir.split("/")[-1]

    seed = f"seed=0"

    curr_dir = os.path.join(base_dir, seed)
    yaml_path = os.path.join(curr_dir, f"{seed}.yaml")

    print(f"Running yaml {yaml_path}")

    config = utils.load_config(yaml_path)

    color_means = np.array(config["data_params"]["color"]["means"])
    size_means = np.array(config["data_params"]["size"]["means"])

    print(f"Color means: {color_means}")
    print(f"Size means: {size_means}")

    cs = []
    for i in range(4):
        l = [[0, 0], [0, 1], [1, 0], [1, 1]][i]  # first axis is color second is size
        c = torch.tensor(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                *color_means[l[1]],
                size_means[l[0]],
                0.0,
                0.0,
                0.0,
            ]
        )
        cs.append(c)

    # color_means: [2, 3]
    # size_means: [2]
    # [4, 11]
    cs = torch.stack(cs).to(dtype=torch.float32, device=device)
    cs = cs[-1].unsqueeze(0)

    ckpt_dir = os.path.join(curr_dir, "ckpts")
    ckpts = os.listdir(ckpt_dir)
    ckpts = sorted(
        [int(x.replace("ckpt_step=", "").replace(".pth", "")) for x in ckpts]
    )

    all_data = []
    for ckpt in tqdm(ckpts[40:80]):
        print(f"Checkpoint {ckpt}")

        ckpt_path = os.path.join(ckpt_dir, f"ckpt_step={ckpt}.pth")
        model = utils.get_model(config)
        model.load_state_dict(torch.load(ckpt_path))
        model = model.to(device)
        model = model.eval()

        # [4 * rep, 3, 32, 32]
        ims = model.generate(cs.repeat(rep, 1), gamma=1)

        color_acc, size_acc = classifier.classify(ims, return_probs=True)
        joint_acc = (color_acc[:, 1] * size_acc[:, 1]).mean()

        ims = model.generate(cs.repeat(rep, 1), gamma=None)
        color_acc, size_acc = classifier.classify(ims, return_probs=True)
        joint_acc2 = (color_acc[:, 1] * size_acc[:, 1]).mean()
        print(f"Joint acc: {joint_acc}, {joint_acc2}")
    breakpoint()


def plot(imgs, rep, output_path):
    # imgs = imgs.transpose(1, 0, 2, 3, 4)

    plt.figure(figsize=(12, 12))
    row = 0
    num_samples = imgs.shape[0]
    num_rows = 4
    num_cols = 4
    for row in range(num_rows):
        for col in range(num_cols):
            plt.subplot(num_rows, num_cols, row * num_cols + col + 1)
            plt.imshow(
                # imgs[row * num_cols + col, -1].transpose(1, 0, 2), origin="lower"
                imgs[row * num_cols + col].transpose(1, 0, 2),
                origin="lower",
            )
            plt.axis("off")

    plt.savefig(output_path)


def load_classifier(device):
    """
    Load classifier.
    """
    path = "classifier_combined.pth"
    n_classes = [2, 2]
    net = networks.CUNet(
        shape=(3, 32, 32), out_channels=64, chs=[32, 32, 32], norm_groups=4
    )
    classifier = models.Classifier(net=net, n_classes=n_classes)
    classifier = classifier.to(device)
    classifier.load_state_dict(torch.load(path))
    classifier = classifier.eval()
    return classifier


@torch.no_grad()
def linear_interv(base_dir):

    device = cuda_tools.get_freer_device(verbose=False)
    classifier = load_classifier(device)

    rep = 32

    _config = base_dir.split("/")[-1]
    output_dir = os.path.join(f"linear_interv_results/{_config}")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    seeds = [0]
    seed_runs = [f"seed={i}" for i in seeds]
    for seed in seed_runs:

        curr_dir = os.path.join(base_dir, seed)
        yaml_path = os.path.join(curr_dir, f"{seed}.yaml")

        print(f"Running yaml {yaml_path}")

        config = utils.load_config(yaml_path)

        color_means = np.array(config["data_params"]["color"]["means"])
        size_means = np.array(config["data_params"]["size"]["means"])

        print(f"Color means: {color_means}")
        print(f"Size means: {size_means}")

        cs = []
        for i in range(4):
            l = [[0, 0], [0, 1], [1, 0], [1, 1]][
                i
            ]  # first axis is color second is size
            c = torch.tensor(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    *color_means[l[1]],
                    size_means[l[0]],
                    0.0,
                    0.0,
                    0.0,
                ]
            )
            cs.append(c)

        # color_means: [2, 3]
        # size_means: [2]
        # [4, 11]
        cs = torch.stack(cs).to(dtype=torch.float32, device=device)
        cs = cs[-1].unsqueeze(0)

        ckpt_dir = os.path.join(curr_dir, "ckpts")
        ckpts = os.listdir(ckpt_dir)
        ckpts = sorted(
            [int(x.replace("ckpt_step=", "").replace(".pth", "")) for x in ckpts]
        )
        scales = [0.1, 1, 2]
        betas = [0.1]

        all_data = []
        for ckpt in tqdm(ckpts):
            print(f"Checkpoint {ckpt}")

            _per_ckpt_accs = []
            ckpt_path = os.path.join(ckpt_dir, f"ckpt_step={ckpt}.pth")
            model = utils.get_model(config)
            model.load_state_dict(torch.load(ckpt_path))
            model = model.to(device)
            model = model.eval()

            for scale in scales:
                for beta in betas:

                    # [4 * rep, 3, 32, 32]
                    ims = model.generate(cs.repeat(rep, 1), alpha=scale, beta=beta)

                    color_acc, size_acc = classifier.classify(ims, return_probs=True)

                    joint_acc = (color_acc[:, 1] * size_acc[:, 1]).mean()

                    _per_ckpt_accs.append(joint_acc.item())
                    all_data.append(
                        {
                            "acc": joint_acc.item(),
                            "ckpt": ckpt,
                            "scale": scale,
                            "beta": beta,
                        }
                    )

        output_filepath = f"{output_dir}/{seed}.json"
        with open(output_filepath, "w") as file_p:
            json.dump(all_data, file_p)


def plot_clf_results():
    all_data = []
    # seeds = [4, 0, 100, 200, 300, 400]
    seeds = [0, 100, 200, 300, 400]

    sns.set_theme(context="paper", style="ticks", rc={"lines.linewidth": 1})
    fig = plt.figure(figsize=(6.75, 6.75))
    gs = GridSpec(len(seeds), 1)
    all_max = {}

    for idx, seed in enumerate(seeds):
        with open(f"v3_clf_run_seed={seed}.json", "r") as file_p:
            seed_data = json.load(file_p)

        dict_form = {}
        max_samples = []
        for sample in seed_data:
            ckpt = sample["ckpt"]
            if ckpt not in dict_form:
                dict_form[sample["ckpt"]] = []
            dict_form[sample["ckpt"]].append(sample["acc"])
            sample["combined"] = f"{sample['scale']}_{sample['beta']}"

        max_samples = [
            {"scale": "max", "combined": "max", "acc": max(accs), "ckpt": ckpt}
            for ckpt, accs in dict_form.items()
        ]
        all_max[seed] = {sample["ckpt"]: sample["acc"] for sample in max_samples}

        # seed_data.extend(max_samples)

        ax = fig.add_subplot(gs[idx, 0])
        sns.lineplot(
            pd.DataFrame(max_samples),
            x="ckpt",
            y="acc",
            hue="combined",
            # style="scale",
            palette=sns.color_palette(),
            ax=ax,
        )

    fig.savefig("zxcv.png")

    # avged = []
    # for ckpt in all_max[0].keys():
    #    avged.append(
    #        {

    #        }
    #    )


def plot_clf_results_final():
    all_data = []
    seeds = [0, 4, 100, 200, 300, 400]

    sns.set_theme(context="paper", style="ticks", rc={"lines.linewidth": 1})
    fig = plt.figure(figsize=(6.75, 6.75))
    gs = GridSpec(1, 1)

    for idx, seed in enumerate(seeds):
        with open(f"v3_clf_run_seed={seed}.json", "r") as file_p:
            seed_data = json.load(file_p)

        dict_form = {}
        max_samples = []
        for sample in seed_data:
            ckpt = sample["ckpt"]
            if ckpt not in dict_form:
                dict_form[sample["ckpt"]] = []
            dict_form[sample["ckpt"]].append(sample["acc"])

        max_samples = [
            {"seed": seed, "acc": max(accs), "ckpt": ckpt}
            for ckpt, accs in dict_form.items()
        ]
        all_data.extend(max_samples)

    ax = fig.add_subplot(gs[0, 0])
    sns.lineplot(
        pd.DataFrame(all_data),
        x="ckpt",
        y="acc",
        hue="seed",
        palette=sns.color_palette(),
        ax=ax,
    )
    ax.legend_ = None

    fig.savefig("zxcv.png")


def plot_results(base_dirs):
    fig = plt.figure(figsize=(16, 16))
    gs = GridSpec(2, 3)

    for dir_idx, base_dir in enumerate(base_dirs):
        curr_row = dir_idx // 3
        curr_col = dir_idx % 3

        _config = base_dir.split("/")[-1]
        results_dir = os.path.join(f"linear_interv_results/{_config}")
        null_intv_results_dir = os.path.join(f"patching_results/{_config}")

        seeds = [0, 100, 200, 300, 400]
        _gs = GridSpecFromSubplotSpec(
            len(seeds), 1, subplot_spec=gs[curr_row, curr_col]
        )

        for idx, seed in enumerate(seeds):
            results_file = os.path.join(results_dir, f"seed_{seed}.json")
            ckpt_to_acc = defaultdict(list)
            with open(results_file, "r") as file_p:
                seed_data = json.load(file_p)
            for sample in seed_data:
                sample["linear_interv"] = "_".join(
                    [str(sample["alpha"]), str(sample["beta"])]
                )
                ckpt_to_acc[sample["ckpt"]].append(sample["acc"])

            results_file = os.path.join(
                null_intv_results_dir, f"null_intv_{seed}.json"
            )
            with open(results_file, "r") as file_p:
                no_intv_seed_data = json.load(file_p)
            for sample in no_intv_seed_data:
                sample["linear_interv"] = False

            data_to_plot = no_intv_seed_data
            data_to_plot.extend([
                {
                    "ckpt": ckpt,
                    "acc": max(accs),
                    "linear_interv": True
                } for ckpt, accs in ckpt_to_acc.items()
            ])

            ax = fig.add_subplot(_gs[idx, 0])
            sns.lineplot(
                #pd.DataFrame(seed_data),
                pd.DataFrame(data_to_plot),
                x="ckpt",
                y="acc",
                hue="linear_interv",
                ax=ax,
            )
            if idx == 0:
                ax.set_title(f"{_config}")
            else:
                ax.legend_ = None

    fig.savefig(f"plots/linear_interv_results_arxiv.png")


if __name__ == "__main__":
    base_dirs = [
        "/nfs/turbo/coe-mihalcea/ajyl/mirun_manyckpt5/sep_col=0.174_sep_size=0.4",
        # "/nfs/turbo/coe-mihalcea/ajyl/concept_learning/sep_col=0.175_sep_size=0.4",
        # "/nfs/turbo/coe-mihalcea/ajyl/concept_learning/sep_col=0.2_sep_size=0.4",
        # "/nfs/turbo/coe-mihalcea/ajyl/concept_learning/sep_col=0.225_sep_size=0.4",
        # "/nfs/turbo/coe-mihalcea/ajyl/concept_learning/sep_col=0.25_sep_size=0.4",
        # "/nfs/turbo/coe-mihalcea/ajyl/concept_learning/sep_col=0.3_sep_size=0.4",
    ]
    # test(base_dirs[-1])
    # for base_dir in base_dirs:
    #    linear_interv(base_dir)
    plot_results(base_dirs)
    # plot_clf_results_final()
