import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import seaborn as sns
import pandas as pd
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

import mltools.utils.cuda_tools as cuda_tools
from mltools.networks import networks

device = cuda_tools.get_freer_device()

import utils
import models
import importlib

from record_utils import get_module, record_activations

downs = [
    "model.score_model.downs.0.resnet_blocks.0.cond_projs.1",
    "model.score_model.downs.0.resnet_blocks.1.cond_projs.1",
    "model.score_model.downs.1.resnet_blocks.0.cond_projs.1",
    "model.score_model.downs.1.resnet_blocks.1.cond_projs.1",
    "model.score_model.downs.2.resnet_blocks.0.cond_projs.1",
    "model.score_model.downs.2.resnet_blocks.1.cond_projs.1",
]

mids = [
    "model.score_model.mid1.cond_projs.1",
    "model.score_model.mid2.cond_projs.1",
]
ups = [
    "model.score_model.ups.0.resnet_blocks.0.cond_projs.1",
    "model.score_model.ups.0.resnet_blocks.1.cond_projs.1",
    "model.score_model.ups.1.resnet_blocks.0.cond_projs.1",
    "model.score_model.ups.1.resnet_blocks.1.cond_projs.1",
    "model.score_model.ups.2.resnet_blocks.0.cond_projs.1",
    "model.score_model.ups.2.resnet_blocks.1.cond_projs.1",
]

naming = [
    "Down 0 Block 0",
    "Down 0 Block 1",
    "Down 1 Block 0",
    "Down 1 Block 1",
    "Down 2 Block 0",
    "Down 2 Block 1",
    "Mid 1",
    "Mid 2",
    "Up 0 Block 0",
    "Up 0 Block 1",
    "Up 1 Block 0",
    "Up 1 Block 1",
    "Up 2 Block 0",
    "Up 2 Block 1",
]


cos = F.cosine_similarity


@torch.no_grad()
def run():

    device = cuda_tools.get_freer_device(verbose=False)
    yaml_path = "./data/images_1/2x2_final2/detailed_run/seed=4.yaml"

    print("Running:", yaml_path)
    config = utils.load_config(yaml_path)

    color_means = np.array(config["data_params"]["color"]["means"])
    size_means = np.array(config["data_params"]["size"]["means"])

    cs = []
    for i in range(4):
        l = [[0, 0], [0, 1], [1, 0], [1, 1]][i]  # first axis is color second is size
        c = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, *color_means[l[1]], size_means[l[0]], 0.0, 0.0, 0.0]
        )
        cs.append(c)

    cs = torch.stack(cs).to(dtype=torch.float32, device=device)

    ckpt_dir = f"./data/images_1/2x2_final2/detailed_run/ckpts/"
    ckpts = os.listdir(ckpt_dir)
    ckpts = sorted(
        [int(x.replace("ckpt_step=", "").replace(".pth", "")) for x in ckpts]
    )
    rep = 4

    all_data = {}
    blue_vs_reds = []
    blue_vs_smalls = []
    blue_vs_larges = []

    for ckpt in tqdm(ckpts):

        print(f"Checkpoint {ckpt}")
        ckpt_path = (
            f"./data/images_1/2x2_final2/detailed_run/ckpts/ckpt_step={ckpt}.pth"
        )
        model = utils.get_model(config)
        model.load_state_dict(torch.load(ckpt_path))
        model = model.to(device)
        model = model.eval()

        all_data[ckpt] = {}

        for idx, module_name in enumerate(downs + mids + ups):
            all_data[ckpt][idx] = {}
            module = get_module(model, module_name)

            inputs = torch.zeros((4, 11))
            inputs[0, 4] = 0.4
            inputs[0, 5] = 0.4
            inputs[0, 6] = 0.6

            inputs[1, 4] = 0.6
            inputs[1, 5] = 0.4
            inputs[1, 6] = 0.4

            inputs[2, 7] = 0.3
            inputs[3, 7] = 0.6

            # [batch, dim]
            testing = module(inputs.to(device))

            blue_vs_red = cos(testing[0], testing[1], dim=0).item()
            blue_vs_small = cos(testing[0], testing[2], dim=0).item()
            blue_vs_large = cos(testing[0], testing[3], dim=0).item()

            blue_vs_reds.append(
                {
                    "ckpt": ckpt,
                    "module": idx,
                    "similarity": blue_vs_red,
                }
            )
            blue_vs_smalls.append(
                {
                    "ckpt": ckpt,
                    "module": idx,
                    "similarity": blue_vs_small,
                }
            )
            blue_vs_larges.append(
                {
                    "ckpt": ckpt,
                    "module": idx,
                    "similarity": blue_vs_large,
                }
            )

    return blue_vs_reds, blue_vs_smalls, blue_vs_larges


def blue_vs_concepts():

    # blue_vs_reds, blue_vs_smalls, blue_vs_larges = run()
    with open("blue_vs_red.json", "r") as file_p:
        blue_vs_reds = json.load(file_p)
    with open("blue_vs_smalls.json", "r") as file_p:
        blue_vs_smalls = json.load(file_p)
    with open("blue_vs_larges.json", "r") as file_p:
        blue_vs_larges = json.load(file_p)

    for x in blue_vs_reds:
        x["module"] = naming[x["module"]]
    for x in blue_vs_smalls:
        x["module"] = naming[x["module"]]
    for x in blue_vs_larges:
        x["module"] = naming[x["module"]]

    fig = plt.figure(figsize=(24, 9))
    # sns.set_theme(context="paper", style="ticks", rc={"lines.linewidth": 1})
    gs = GridSpec(3, 1)

    blue_vs_reds = pd.DataFrame(blue_vs_reds)
    blue_vs_smalls = pd.DataFrame(blue_vs_smalls)
    blue_vs_larges = pd.DataFrame(blue_vs_larges)

    ax = fig.add_subplot(gs[0, 0])
    sns.lineplot(
        blue_vs_reds,
        x="ckpt",
        y="similarity",
        hue="module",
        ax=ax,
        palette=sns.color_palette(),
    )
    # sns.boxplot(
    #    pd.DataFrame(blue_vs_reds),
    #    x="ckpt",
    #    y="similarity",
    #    hue=True,
    #    hue_order=[False, True],
    #    #split=True,
    #    ax=ax,
    #    palette="Blues",
    #    whis=(0, 100),
    # )
    # ax.legend_ = None
    ax.set(ylim=(-0.5, 1))
    ax.xaxis.set_visible(False)
    ax.set_title("Blue vs. Red")

    ax = fig.add_subplot(gs[1, 0])
    sns.lineplot(
        blue_vs_smalls,
        x="ckpt",
        y="similarity",
        hue="module",
        ax=ax,
        palette=sns.color_palette(),
    )
    # sns.boxplot(
    #    pd.DataFrame(blue_vs_smalls),
    #    x="ckpt",
    #    y="similarity",
    #    hue=True,
    #    hue_order=[False, True],
    #    #split=True,
    #    ax=ax,
    #    palette="Blues",
    #    whis=(0, 100),
    # )
    ax.legend_ = None
    ax.set(ylim=(-0.5, 1))
    ax.xaxis.set_visible(False)
    ax.set_title("Blue vs. Small")

    ax = fig.add_subplot(gs[2, 0])
    sns.lineplot(
        blue_vs_larges,
        x="ckpt",
        y="similarity",
        hue="module",
        ax=ax,
        palette=sns.color_palette(),
    )
    # b_vs_l_fig = sns.boxplot(
    #    pd.DataFrame(blue_vs_larges),
    #    x="ckpt",
    #    y="similarity",
    #    hue=True,
    #    hue_order=[False, True],
    #    #split=True,
    #    ax=ax,
    #    palette="Blues",
    #    whis=(0, 100),
    # )
    ax.legend_ = None
    ax.set(ylim=(-0.5, 1))
    ax.set_title("Blue vs. Large")

    fig.savefig("meow.png")
    breakpoint()
    print("z")


@torch.no_grad()
def vs_final_epoch():
    """
    device = cuda_tools.get_freer_device(verbose=False)
    yaml_path = "./data/images_1/2x2_final2/detailed_run/seed=4.yaml"

    print("Running:", yaml_path)
    config = utils.load_config(yaml_path)

    color_means = np.array(config["data_params"]["color"]["means"])
    size_means = np.array(config["data_params"]["size"]["means"])

    cs = []
    for i in range(4):
        l = [[0, 0], [0, 1], [1, 0], [1, 1]][i]  # first axis is color second is size
        c = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, *color_means[l[1]], size_means[l[0]], 0.0, 0.0, 0.0]
        )
        cs.append(c)

    cs = torch.stack(cs).to(dtype=torch.float32, device=device)
    cs = cs[-1].unsqueeze(0)

    ckpt_dir = f"./data/images_1/2x2_final2/detailed_run/ckpts/"
    ckpts = os.listdir(ckpt_dir)
    ckpts = sorted(
        [int(x.replace("ckpt_step=", "").replace(".pth", "")) for x in ckpts]
    )
    rep = 1

    all_data = {}
    blue_vs_reds = []
    blue_vs_smalls = []
    blue_vs_larges = []

    ckpt = ckpts[-1]
    ckpt_path = f"./data/images_1/2x2_final2/detailed_run/ckpts/ckpt_step={ckpt}.pth"
    model = utils.get_model(config)
    model.load_state_dict(torch.load(ckpt_path))
    model = model.to(device)
    model = model.eval()

    final_embeds = {}
    for idx, module_name in enumerate(downs + mids + ups):
        module = get_module(model, module_name)

        # [batch, dim]
        final_embeds[module_name] = module(cs)

    plot_data = []
    for ckpt in tqdm(ckpts):

        print(f"Checkpoint {ckpt}")
        ckpt_path = (
            f"./data/images_1/2x2_final2/detailed_run/ckpts/ckpt_step={ckpt}.pth"
        )
        model = utils.get_model(config)
        model.load_state_dict(torch.load(ckpt_path))
        model = model.to(device)
        model = model.eval()

        for idx, module_name in enumerate(downs + mids + ups):
            module = get_module(model, module_name)

            # [batch, dim]
            cond_emb = module(cs.to(device))

            similarity = cos(final_embeds[module_name], cond_emb, dim=1)
            plot_data.append(
                {
                    "ckpt": ckpt,
                    "module": naming[idx],
                    "similarity": similarity.item(),
                }
            )

    with open("vs_final_epoch.json", "w") as file_p:
        json.dump(plot_data, file_p, indent=2)
    """
    with open("vs_final_epoch.json", "r") as file_p:
        plot_data = json.load(file_p)

    fig = plt.figure(figsize=(24, 9))
    # sns.set_theme(context="paper", style="ticks", rc={"lines.linewidth": 1})
    gs = GridSpec(1, 1)

    for x in plot_data:
        x["module"] = naming[x["module"]]

    plot_data = pd.DataFrame(plot_data)

    ax = fig.add_subplot(gs[0, 0])
    sns.lineplot(
        plot_data,
        x="ckpt",
        y="similarity",
        hue="module",
        ax=ax,
        palette=sns.color_palette(),
    )
    ax.set(ylim=(0, 1))
    ax.set_title("vs. Final Epoch")

    fig.savefig("vs_final_epoch.png")
    breakpoint()
    print("z")


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
def patching():
    device = cuda_tools.get_freer_device(verbose=False)
    yaml_path = "./data/images_1/2x2_final2/detailed_run/seed=4.yaml"

    print("Running:", yaml_path)
    config = utils.load_config(yaml_path)

    color_means = np.array(config["data_params"]["color"]["means"])
    size_means = np.array(config["data_params"]["size"]["means"])

    classifier = load_classifier(device)

    cs = []
    for i in range(4):
        l = [[0, 0], [0, 1], [1, 0], [1, 1]][i]  # first axis is color second is size
        c = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, *color_means[l[1]], size_means[l[0]], 0.0, 0.0, 0.0]
        )
        cs.append(c)

    cs = torch.stack(cs).to(dtype=torch.float32, device=device)
    cs = cs[-1].unsqueeze(0)

    ckpt_dir = f"./data/images_1/2x2_final2/detailed_run/ckpts/"
    ckpts = os.listdir(ckpt_dir)
    ckpts = sorted(
        [int(x.replace("ckpt_step=", "").replace(".pth", "")) for x in ckpts]
    )
    rep = 64

    ckpt = ckpts[-1]
    ckpt_path = f"./data/images_1/2x2_final2/detailed_run/ckpts/ckpt_step={ckpt}.pth"

    model = utils.get_model(config)
    final_state_dict = torch.load(ckpt_path)

    plot_data = []
    for ckpt in tqdm(ckpts):

        print(f"Checkpoint {ckpt}")
        ckpt_path = (
            f"./data/images_1/2x2_final2/detailed_run/ckpts/ckpt_step={ckpt}.pth"
        )
        model = utils.get_model(config)
        curr_state_dict = torch.load(ckpt_path)
        for name in curr_state_dict.keys():
            for key in downs + mids + ups:
                if name.startswith(key):
                    print(key)
                    curr_state_dict[name] = final_state_dict[name]

        model.load_state_dict(curr_state_dict)
        model = model.to(device)
        model = model.eval()

        # [4 * rep, 3, 32, 32]
        ims = model.generate(cs.repeat(rep, 1), hack=False)

        color_acc, size_acc = classifier.classify(ims, return_probs=True)

        joint_acc = (color_acc[:, 1] * size_acc[:, 1]).mean()

        plot_data.append({"acc": joint_acc.item(), "ckpt": ckpt})

    fig = sns.relplot(pd.DataFrame(plot_data), x="ckpt", y="acc", kind="line")
    fig.set(ylim=(0, 1))
    fig.savefig(f"patching.png")

    with open("patching_results.json", "w") as file_p:
        json.dump(plot_data, file_p, indent=2)

    breakpoint()
    print("z")


def weight_shift():
    """
    Weight shift experiment.
    """
    device = cuda_tools.get_freer_device(verbose=False)
    yaml_path = "./data/images_1/2x2_final2/detailed_run/seed=4.yaml"

    print("Running:", yaml_path)
    config = utils.load_config(yaml_path)

    color_means = np.array(config["data_params"]["color"]["means"])
    size_means = np.array(config["data_params"]["size"]["means"])

    cs = []
    for i in range(4):
        l = [[0, 0], [0, 1], [1, 0], [1, 1]][i]  # first axis is color second is size
        c = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, *color_means[l[1]], size_means[l[0]], 0.0, 0.0, 0.0]
        )
        cs.append(c)

    cs = torch.stack(cs).to(dtype=torch.float32, device=device)
    cs = cs[-1].unsqueeze(0)

    ckpt_dir = f"./data/images_1/2x2_final2/detailed_run/ckpts/"
    ckpts = os.listdir(ckpt_dir)
    ckpts = sorted(
        [int(x.replace("ckpt_step=", "").replace(".pth", "")) for x in ckpts]
    )
    rep = 64

    ckpt = ckpts[0]
    ckpt_path = f"./data/images_1/2x2_final2/detailed_run/ckpts/ckpt_step={ckpt}.pth"

    prev_state_dict = torch.load(ckpt_path)

    plot_data = []
    diffs = {name: [] for name in prev_state_dict.keys() if name.endswith(".weight")}
    all_data = []
    for idx, ckpt in tqdm(enumerate(ckpts[1:])):
        print(f"Checkpoint {ckpt}")
        ckpt_path = (
            f"./data/images_1/2x2_final2/detailed_run/ckpts/ckpt_step={ckpt}.pth"
        )
        curr_state_dict = torch.load(ckpt_path)

        for name, prev_weights in prev_state_dict.items():
            if not name.endswith(".weight"):
                continue

            curr_weights = curr_state_dict[name]
            _diff = (curr_weights - prev_weights).norm().item()

            diffs[name].append(_diff)
            is_mlp = any(name.startswith(key) for key in ups + mids + downs)
            all_data.append(
                {
                    "idx": idx,
                    "component": name,
                    "diff": _diff,
                    "is_mlp": is_mlp,
                    "ckpt": ckpt,
                }
            )

        prev_state_dict = curr_state_dict

    plot_data = pd.DataFrame(all_data)
    fig = sns.relplot(plot_data, x="ckpt", y="diff", kind="line", hue="is_mlp")
    fig.savefig("weight_shift.png")
    breakpoint()
    print("z")


if __name__ == "__main__":
    # blue_vs_concepts()

    # vs_final_epoch()

    # patching()

    weight_shift()
