import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from tqdm import tqdm
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


@torch.no_grad()
def main():
    regen = True

    suffix = "_w_cfg_0.0"
    w_cfg = 0.0
    n_classes = [2, 2]

    ####
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

    # color_means: [2, 3]
    # size_means: [2]
    # [4, 11]
    cs = torch.stack(cs).to(dtype=torch.float32, device=device)

    # cs = cs[-1].unsqueeze(0)
    # Red
    # cs[:, 4] = 0.9
    # Green
    # cs[:, 5] = 0.9
    # Blue
    # cs[:, 6] = 0.9

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
    ckpt_dir = f"./data/images_1/2x2_final2/detailed_run/ckpts/"
    ckpts = os.listdir(ckpt_dir)
    ckpts = sorted(
        [int(x.replace("ckpt_step=", "").replace(".pth", "")) for x in ckpts]
    )

    # for ckpt in ckpts:
    for ckpt in [1950, 2446, 2999, 3000, 3035, 3070, 3105, 3140, 3175, 3527]:
        print(f"Checkpoint {ckpt}")
        ckpt_path = (
            f"./data/images_1/2x2_final2/detailed_run/ckpts/ckpt_step={ckpt}.pth"
        )
        model = utils.get_model(config)
        model.load_state_dict(torch.load(ckpt_path))
        model = model.to(device)
        model = model.eval()

        rep = 4

        outputs = {}
        for hack in [True, False]:
            ims = model.generate(cs.repeat(rep, 1), hack=hack).detach().cpu().numpy()

            ims = (
                np.clip(ims.transpose(0, 2, 3, 1), 0, 1)
                .reshape(4, rep, 32, 32, 3)
                .transpose(1, 0, 2, 3, 4)
            )

            outputs[str(hack)] = ims

        plt.figure(figsize=(16, 8))
        for row in range(4):
            for col in range(rep):
                plt.subplot(4, rep * 2, row * 2 * rep + col + 1)
                plt.imshow(
                    outputs["False"][row, col].transpose(1, 0, 2), origin="lower"
                )
                plt.axis("off")

                plt.subplot(4, rep * 2, row * 2 * rep + col + 1 + 4)
                plt.imshow(outputs["True"][row, col].transpose(1, 0, 2), origin="lower")
                plt.axis("off")

        plt.savefig(f"plots/{ckpt}.png")

    print("z")


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
def clf_exp():

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

    # color_means: [2, 3]
    # size_means: [2]
    # [4, 11]
    cs = torch.stack(cs).to(dtype=torch.float32, device=device)
    cs = cs[-1].unsqueeze(0)

    ckpt_dir = f"./data/images_1/2x2_final2/detailed_run/ckpts/"
    ckpts = os.listdir(ckpt_dir)
    ckpts = sorted(
        [int(x.replace("ckpt_step=", "").replace(".pth", "")) for x in ckpts]
    )

    # 4512
    # 5497
    # 6025
    # for ckpt in [4160]:
    # for ckpt in [9331]:

    rep = 64
    scales = [0.1, 1, 2, 4]
    all_data = []
    accs_to_plot = []
    for ckpt in tqdm(ckpts):
        print(f"Checkpoint {ckpt}")

        _per_ckpt_accs = []
        for scale in scales:
            print(f"Scale: {scale}")
            ckpt_path = (
                f"./data/images_1/2x2_final2/detailed_run/ckpts/ckpt_step={ckpt}.pth"
            )
            model = utils.get_model(config)
            model.load_state_dict(torch.load(ckpt_path))
            model = model.to(device)
            model = model.eval()

            # [4 * rep, 3, 32, 32]
            ims = model.generate(cs.repeat(rep, 1), hack=scale)

            color_acc, size_acc = classifier.classify(ims, return_probs=True)

            joint_acc = (color_acc[:, 1] * size_acc[:, 1]).mean()

            _per_ckpt_accs.append(joint_acc.item())
            all_data.append({"scale": scale, "acc": joint_acc.item(), "ckpt": ckpt})

            # [4, rep, 32, 32, 3]
            # ims = np.clip(ims.detach().cpu().numpy().transpose(0, 2, 3, 1), 0, 1).reshape(
            #    4, rep, 32, 32, 3
            # )
            # ims = np.clip(ims.detach().cpu().numpy().transpose(0, 2, 3, 1), 0, 1)
            # plot(ims, rep, f"plots/testing_{ckpt}.png")

        accs_to_plot.append({"ckpt": ckpt, "acc": max(_per_ckpt_accs), "scale": "max"})

    fig = sns.relplot(pd.DataFrame(accs_to_plot), x="ckpt", y="acc", kind="line")
    fig.set(ylim=(0, 1))
    fig.savefig(f"scrub_red_add_blue_max.png")


    fig = sns.relplot(pd.DataFrame(all_data + accs_to_plot), x="ckpt", y="acc", kind="line", hue="scale")
    fig.set(ylim=(0, 1))
    fig.savefig("scrub_red_add_blue_all.png")

    with open("results_all.json", "w") as file_p:
        json.dump(all_data, file_p, indent=2)

    with open("results_to_plot.json", "w") as file_p:
        json.dump(accs_to_plot, file_p, indent=2)
    breakpoint()
    print("z")


@torch.no_grad()
def clf_exp_5seed():

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

    # color_means: [2, 3]
    # size_means: [2]
    # [4, 11]
    cs = torch.stack(cs).to(dtype=torch.float32, device=device)
    cs = cs[-1].unsqueeze(0)

    ckpt_dir = f"./data/images_1/2x2_final2/detailed_run/ckpts/"
    ckpts = os.listdir(ckpt_dir)
    ckpts = sorted(
        [int(x.replace("ckpt_step=", "").replace(".pth", "")) for x in ckpts]
    )

    # 4512
    # 5497
    # 6025
    # for ckpt in [4160]:
    # for ckpt in [9331]:

    rep = 64
    scales = [0.1, 1, 2, 4]
    all_data = []
    accs_to_plot = []
    for ckpt in tqdm(ckpts):
        print(f"Checkpoint {ckpt}")

        _per_ckpt_accs = []
        for scale in scales:
            print(f"Scale: {scale}")
            ckpt_path = (
                f"./data/images_1/2x2_final2/detailed_run/ckpts/ckpt_step={ckpt}.pth"
            )
            model = utils.get_model(config)
            model.load_state_dict(torch.load(ckpt_path))
            model = model.to(device)
            model = model.eval()

            # [4 * rep, 3, 32, 32]
            ims = model.generate(cs.repeat(rep, 1), hack=scale)

            color_acc, size_acc = classifier.classify(ims, return_probs=True)

            joint_acc = (color_acc[:, 1] * size_acc[:, 1]).mean()

            _per_ckpt_accs.append(joint_acc.item())
            all_data.append({"scale": scale, "acc": joint_acc.item(), "ckpt": ckpt})

            # [4, rep, 32, 32, 3]
            # ims = np.clip(ims.detach().cpu().numpy().transpose(0, 2, 3, 1), 0, 1).reshape(
            #    4, rep, 32, 32, 3
            # )
            # ims = np.clip(ims.detach().cpu().numpy().transpose(0, 2, 3, 1), 0, 1)
            # plot(ims, rep, f"plots/testing_{ckpt}.png")

        accs_to_plot.append({"ckpt": ckpt, "acc": max(_per_ckpt_accs), "scale": "max"})

    fig = sns.relplot(pd.DataFrame(accs_to_plot), x="ckpt", y="acc", kind="line")
    fig.set(ylim=(0, 1))
    fig.savefig(f"scrub_red_add_blue_max.png")


    fig = sns.relplot(pd.DataFrame(all_data + accs_to_plot), x="ckpt", y="acc", kind="line", hue="scale")
    fig.set(ylim=(0, 1))
    fig.savefig("scrub_red_add_blue_all.png")

    with open("results_all.json", "w") as file_p:
        json.dump(all_data, file_p, indent=2)

    with open("results_to_plot.json", "w") as file_p:
        json.dump(accs_to_plot, file_p, indent=2)
    breakpoint()
    print("z")


if __name__ == "__main__":
    # main()
    #clf_exp()
    clf_exp_5seed()
