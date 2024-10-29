import os
import sys
import time

import numpy as np
import cv2
import glob
from PIL import Image
import scipy.ndimage as sim
import skimage.morphology as skmorph
import skimage.segmentation as skseg
import shutil
import torch
import warnings

clahe = cv2.createCLAHE(clipLimit=3).apply


def color_mask(mask, rgba_float=[1.0, 0.0, 1.0, 0.5]):
    rgba_float = np.array(rgba_float).astype(np.float32)
    return (mask / 255)[:, :, None] * rgba_float[None, None, :]


def float_to_int(im, dtype=np.uint8):
    return np.clip(im * np.iinfo(dtype).max, 0, np.iinfo(dtype).max).astype(dtype)


def int_to_float(im, dtype=np.float32):
    return im.astype(dtype) / np.iinfo(im.dtype).max


def get_logprob(logit):
    lse = torch.logsumexp(logit, dim=1, keepdim=True)
    return logit - lse


def get_prob(image, net, return_dtype=np.uint8):
    image_dtype = image.dtype
    if image_dtype == np.uint8 or image_dtype == np.uint16:
        image = int_to_float(image, dtype=np.float32)
    else:
        assert image_dtype == np.float32
    assert return_dtype == np.uint8 or return_dtype == np.float32
    res = torch.tensor(image)[None, None]
    with torch.no_grad():
        mask_logits = net(
            res.to(device=next(net.parameters()).device, dtype=torch.float32)
        )
    prob = torch.exp(get_logprob(mask_logits))[0].cpu().detach().numpy()[1]
    if return_dtype == np.uint8:
        return float_to_int(prob, dtype=return_dtype)
    else:
        return prob.astype(return_dtype)


def get_best_models(models_folpath, top=3):
    valaccs = []
    model_paths = glob.glob(os.path.join(models_folpath, "*.pth"))
    for model_path in model_paths:
        valaccs.append(
            float(os.path.split(model_path)[-1].split("_")[-1].split(".pth")[0])
        )
    return [str(fp) for fp in np.array(model_paths)[np.argsort(valaccs)[:top]]]


def load_im(im_path, do_clahe=False):
    im = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
    assert im.dtype == np.uint8 or im.dtype == np.uint16
    if do_clahe:
        return clahe(im)
    else:
        return im


def write_im(im_path, im):
    cv2.imwrite(im_path, im)


def import_matlab(lib_fol):  # "/n/home12/cfpark00/matlabpython/lib/"
    """
    Import matlab engine and matlab module from lib_fol

    lib_fol: str
        The path to the lib folder containing the matlab module and the matlab engine.
        Usually ends with "lib/".
    """
    warning.warn(
        "This function will create two global variables: matlab and eng,\
        respectively the matlab module and the matlab engine."
    )
    global matlab, eng
    import sys

    sys.path.append(lib_fol)
    import matlab
    import matlab.engine

    eng = matlab.engine.start_matlab()


def hmin(im, minsupp):
    im = im.astype(np.int32)
    return (255 - skmorph.reconstruction(255 - im - minsupp, 255 - im)).astype(np.uint8)


def hmin_matlab(im, minsupp):
    return np.array(eng.imhmin(matlab.uint8(im), minsupp))


def watershed_matlab(im):
    return np.array(eng.watershed(matlab.uint8(im), 8.0))


def get_seg(mb_prob, mb_thres=155, minsupp=77, use_matlab=False):
    if use_matlab:
        mb_hmin = hmin_matlab(mb_prob, minsupp)
        mb_wshed = watershed_matlab(mb_hmin)
    else:
        mb_hmin = hmin(mb_prob, minsupp)
        mb_wshed = skseg.watershed(mb_hmin, watershed_line=True)
    mb_wshed[mb_hmin > mb_thres] = 0
    return mb_wshed


def get_VI_from_seg(seg, seg_gt, gt_dilation=5):
    ker = np.ones((gt_dilation, gt_dilation))
    gt_mb_dilated = skmorph.binary_dilation(seg_gt == 0, ker)
    support = ~gt_mb_dilated
    labels = seg[support]
    labels_gt = seg_gt[support]
    return VI(labels, labels_gt)


def get_error_map(
    fm_prob,
    sm_prob,
    only_vi=False,
    return_all=False,
    use_matlab=False,
    rm_bounds=5,
    return_time=False,
    max_size=200000,
):
    t0 = time.perf_counter()
    assert (not use_matlab) or (
        ("eng" in globals()) and ("matlab" in globals())
    ), "call tools.import_matlab first"
    assert fm_prob.dtype == np.uint8 and sm_prob.dtype == np.uint8, "uint8 data needed"
    minsupp = int(0.3 * 255)
    mb_thres = 155
    mb_thres_low = 50
    vi_thres = 1e-5
    error_map_dilate = 1

    t1 = time.perf_counter()
    if use_matlab:
        sm_hmin = hmin_matlab(sm_prob, minsupp)
        fm_hmin = hmin_matlab(fm_prob, minsupp)
    else:
        sm_hmin = hmin(sm_prob, minsupp)
        fm_hmin = hmin(fm_prob, minsupp)

    t2 = time.perf_counter()
    if use_matlab:
        sm_wshed = watershed_matlab(sm_hmin)
        fm_wshed = watershed_matlab(fm_hmin)
    else:
        sm_wshed = skseg.watershed(sm_hmin, watershed_line=True)
        fm_wshed = skseg.watershed(fm_hmin, watershed_line=True)

    t3 = time.perf_counter()
    if rm_bounds > 0:
        ker = np.ones((rm_bounds, rm_bounds))
        not_boundary = np.logical_and(
            skmorph.binary_erosion(fm_wshed > 0, ker), sm_prob < mb_thres_low
        )
    sm_wshed[sm_hmin > mb_thres] = 0
    fm_wshed[fm_hmin > mb_thres] = 0
    out_labels = np.unique(sm_wshed[(sm_wshed > 0) * (fm_wshed == 0)])
    misses = out_labels[~np.isin(out_labels, sm_wshed[fm_wshed > 0])]
    out_labels = np.unique(fm_wshed[(fm_wshed > 0) * (sm_wshed == 0)])
    extras = out_labels[~np.isin(out_labels, fm_wshed[sm_wshed > 0])]

    u, c = np.unique(sm_wshed, return_counts=True)
    cellb = u[c > max_size]
    cellb = cellb[cellb != 0]
    toobig_sm = np.isin(sm_wshed, cellb)

    u, c = np.unique(fm_wshed, return_counts=True)
    cellb = u[c > max_size]
    cellb = cellb[cellb != 0]
    toobig_fm = np.isin(fm_wshed, cellb)

    support = ((sm_wshed * fm_wshed) > 0) * (~toobig_sm) * (~toobig_fm)
    # supp1_vol=support.sum()
    if rm_bounds > 0:
        support = np.logical_and(support, not_boundary)
    sm_labels = sm_wshed[support]
    fm_labels = fm_wshed[support]

    t4 = time.perf_counter()
    vi, vi_split, vi_merge, splitters, mergers = VI(fm_labels, sm_labels)
    t5 = time.perf_counter()
    if only_vi:
        if return_time:
            return (
                vi,
                vi_split,
                vi_merge,
                splitters,
                mergers,
                fm_wshed,
                sm_wshed,
                [t0, t1, t2, t3, t4, t5],
            )
        else:
            return vi, vi_split, vi_merge, splitters, mergers, fm_wshed, sm_wshed

    i_splits = splitters[splitters[:, 0] > vi_thres, 1].astype(np.uint32)
    i_merges = mergers[mergers[:, 0] > vi_thres, 1].astype(np.uint32)

    error_map = np.zeros(sm_prob.shape, dtype=bool)
    for i_split in i_splits:
        error_map[
            (sm_wshed == i_split) * (sm_hmin < mb_thres) * (fm_hmin > mb_thres)
        ] = True
    for i_merge in i_merges:
        error_map[
            (fm_wshed == i_merge) * (sm_hmin > mb_thres) * (fm_hmin < mb_thres)
        ] = True
    for miss in misses:
        error_map[sm_wshed == miss] = True
    for extra in extras:
        error_map[fm_wshed == extra] = True

    error_map = (
        skmorph.binary_dilation(error_map, np.ones((3, 3))).astype(np.uint8) * 255
    )

    if return_all:
        return (
            error_map,
            vi,
            vi_split,
            vi_merge,
            splitters,
            mergers,
            fm_wshed,
            sm_wshed,
            i_splits,
            i_merges,
            fm_wshed,
            sm_wshed,
            misses,
            extras,
        )
    return error_map


def get_VI_struct(m_prob, use_matlab=False, max_size=200000):
    assert (not use_matlab) or (
        ("eng" in globals()) and ("matlab" in globals())
    ), "call tools.import_matlab first"
    assert m_prob.dtype == np.uint8, "uint8 data needed"
    minsupp = int(0.3 * 255)
    mb_thres = 155
    if use_matlab:
        m_hmin = hmin_matlab(m_prob, minsupp)
    else:
        m_hmin = hmin(m_prob, minsupp)
    if use_matlab:
        m_wshed = watershed_matlab(m_hmin)
    else:
        m_wshed = skseg.watershed(m_hmin, watershed_line=True)
    m_wshed_orig = m_wshed.copy()
    m_wshed[m_hmin > mb_thres] = 0

    u, c = np.unique(m_wshed, return_counts=True)
    cellb = u[c > max_size]
    cellb = cellb[cellb != 0]
    toobig_m = np.isin(m_wshed, cellb)

    return (m_prob, m_wshed_orig, m_wshed, toobig_m)


def VI_from_struct(fm_struct, sm_struct, rm_bounds=5):
    mb_thres_low = 50

    fm_prob, fm_wshed_orig, fm_wshed, toobig_fm = fm_struct
    sm_prob, sm_wshed_orig, sm_wshed, toobig_sm = sm_struct
    if rm_bounds > 0:
        ker = np.ones((rm_bounds, rm_bounds))
        not_boundary = np.logical_and(
            skmorph.binary_erosion(fm_wshed_orig > 0, ker), sm_prob < mb_thres_low
        )

    support = ((sm_wshed * fm_wshed) > 0) * (~toobig_sm) * (~toobig_fm)
    if rm_bounds > 0:
        support = np.logical_and(support, not_boundary)

    sm_labels = sm_wshed[support]
    fm_labels = fm_wshed[support]
    vi, vi_split, vi_merge, splitters, mergers = VI(fm_labels, sm_labels)
    return vi, vi_split, vi_merge, splitters, mergers, fm_wshed, sm_wshed


def get_big_cells(m_prob, max_size=200000, use_matlab=False):
    minsupp = int(0.3 * 255)
    mb_thres = 155

    if use_matlab:
        m_hmin = hmin_matlab(m_prob, minsupp)
    else:
        m_hmin = hmin(m_prob, minsupp)

    if use_matlab:
        m_wshed = watershed_matlab(m_hmin)
    else:
        m_wshed = skseg.watershed(m_hmin, watershed_line=True)
    m_wshed[m_hmin > mb_thres] = 0

    u, c = np.unique(m_wshed, return_counts=True)
    cellb = u[c > max_size]
    cellb = cellb[cellb != 0]
    toobig_m = np.isin(m_wshed, cellb)

    return toobig_m


def VI(fm_labels, sm_labels):
    assert len(sm_labels) == len(fm_labels)
    size = len(sm_labels)

    mutual_labels = (fm_labels.astype(np.uint64) << 32) + sm_labels.astype(np.uint64)

    sm_unique, sm_inverse, sm_counts = np.unique(
        sm_labels, return_inverse=True, return_counts=True
    )
    fm_unique, fm_inverse, fm_counts = np.unique(
        fm_labels, return_inverse=True, return_counts=True
    )
    _, mutual_inverse, mutual_counts = np.unique(
        mutual_labels, return_inverse=True, return_counts=True
    )

    terms_mutual = -np.log(mutual_counts / size) * mutual_counts / size
    terms_mutual_per_count = (
        terms_mutual[mutual_inverse] / mutual_counts[mutual_inverse]
    )
    terms_sm = -np.log(sm_counts / size) * sm_counts / size
    terms_fm = -np.log(fm_counts / size) * fm_counts / size

    vi_split_each = np.zeros(len(sm_unique))
    np.add.at(vi_split_each, sm_inverse, terms_mutual_per_count)
    vi_split_each -= terms_sm
    vi_merge_each = np.zeros(len(fm_unique))
    np.add.at(vi_merge_each, fm_inverse, terms_mutual_per_count)
    vi_merge_each -= terms_fm

    vi_split = np.sum(vi_split_each)
    vi_merge = np.sum(vi_merge_each)
    vi = vi_split + vi_merge

    i_splitters = np.argsort(vi_split_each)[::-1]
    i_mergers = np.argsort(vi_merge_each)[::-1]

    vi_split_sorted = vi_split_each[i_splitters]
    vi_merge_sorted = vi_merge_each[i_mergers]

    splitters = np.stack([vi_split_sorted, sm_unique[i_splitters]], axis=1)
    mergers = np.stack([vi_merge_sorted, fm_unique[i_mergers]], axis=1)
    return vi, vi_split, vi_merge, splitters, mergers


def get_error_GT(
    prob_map,
    prob_map_ref,
    h1=0.3,
    h2=0.04,
    error_thres=1e-5,
    dilation=0,
    use_matlab=False,
    max_size=200000,
):
    assert (not use_matlab) or (
        ("eng" in globals()) and ("matlab" in globals())
    ), "call tools.import_matlab first"

    reduceMin1 = int(h1 * 255)
    reduceMin2 = int(h2 * 255)

    prob_map = np.pad(prob_map, (1, 1), mode="constant", constant_values=255)
    prob_map_ref = np.pad(prob_map_ref, (1, 1), mode="constant", constant_values=255)

    prob_map = np.pad(prob_map, (1, 1), mode="constant", constant_values=0)
    prob_map_ref = np.pad(prob_map_ref, (1, 1), mode="constant", constant_values=0)

    if use_matlab:
        prob_map_hmin1 = hmin_matlab(prob_map, reduceMin1)
        prob_map_ref_hmin1 = hmin_matlab(prob_map_ref, reduceMin2)
        prob_map_hmin2 = hmin_matlab(prob_map, reduceMin2)
        prob_map_ref_hmin2 = hmin_matlab(prob_map_ref, reduceMin1)
    else:
        prob_map_hmin1 = hmin(prob_map, reduceMin1)
        prob_map_ref_hmin1 = hmin(prob_map_ref, reduceMin2)
        prob_map_hmin2 = hmin(prob_map, reduceMin2)
        prob_map_ref_hmin2 = hmin(prob_map_ref, reduceMin1)

    if use_matlab:
        prob_map_hmin1_w = watershed_matlab(prob_map_hmin1)
        prob_map_ref_hmin1_w = watershed_matlab(prob_map_ref_hmin1)
        prob_map_hmin2_w = watershed_matlab(prob_map_hmin2)
        prob_map_ref_hmin2_w = watershed_matlab(prob_map_ref_hmin2)
    else:
        prob_map_hmin1_w = skseg.watershed(prob_map_hmin1, watershed_line=True)
        prob_map_ref_hmin1_w = skseg.watershed(prob_map_ref_hmin1, watershed_line=True)
        prob_map_hmin2_w = skseg.watershed(prob_map_hmin2, watershed_line=True)
        prob_map_ref_hmin2_w = skseg.watershed(prob_map_ref_hmin2, watershed_line=True)

    bordermask = prob_map > 150
    bordermask_ref = prob_map_ref > 150
    prob_map_hmin1_w[bordermask] = 0
    prob_map_ref_hmin1_w[bordermask_ref] = 0
    prob_map_hmin2_w[bordermask] = 0
    prob_map_ref_hmin2_w[bordermask_ref] = 0

    # prob_map=prob_map[2:-2,2:-2]
    # prob_map_ref=prob_map_ref[2:-2,2:-2]

    prob_map_hmin1_w = prob_map_hmin1_w[2:-2, 2:-2]
    prob_map_ref_hmin1_w = prob_map_ref_hmin1_w[2:-2, 2:-2]

    prob_map_hmin2_w = prob_map_hmin2_w[2:-2, 2:-2]
    prob_map_ref_hmin2_w = prob_map_ref_hmin2_w[2:-2, 2:-2]

    u, c = np.unique(prob_map_hmin1_w, return_counts=True)
    cellb = u[c > max_size]
    cellb = cellb[cellb != 0]
    toobig = np.isin(prob_map_hmin1_w, cellb)

    u, c = np.unique(prob_map_ref_hmin2_w, return_counts=True)
    cellb = u[c > max_size]
    cellb = cellb[cellb != 0]
    toobig_ref = np.isin(prob_map_ref_hmin2_w, cellb)

    not_cellb = ~np.logical_or(toobig, toobig_ref)

    range_1 = np.logical_and(prob_map_hmin1_w > 0, prob_map_ref_hmin1_w > 0)
    range_1 = np.logical_and(range_1, not_cellb)

    range_2 = np.logical_and(prob_map_hmin2_w > 0, prob_map_ref_hmin2_w > 0)
    range_2 = np.logical_and(range_2, not_cellb)
    # print("start vI")
    _, _, _, _, mergers = VI(prob_map_hmin1_w[range_1], prob_map_ref_hmin1_w[range_1])
    _, _, _, splitters, _ = VI(prob_map_hmin2_w[range_2], prob_map_ref_hmin2_w[range_2])
    # print("end vI")
    merge_errors = mergers[:, 0] > error_thres
    split_errors = splitters[:, 0] > error_thres

    DIL = 9

    E1 = prob_map_hmin1_w.copy()
    E1[~np.isin(prob_map_hmin1_w, mergers[merge_errors][:, 1])] = 0
    error1 = np.logical_and(prob_map_ref_hmin1_w == 0, E1)
    error1 = np.logical_and(
        error1, ~skmorph.binary_dilation(prob_map_hmin1_w == 0, np.ones((DIL, DIL)))
    ).astype(np.uint16)
    temp = error1 > 0
    error1[temp] = E1[temp]

    E2 = prob_map_ref_hmin2_w.copy()
    E2[~np.isin(prob_map_ref_hmin2_w, splitters[split_errors][:, 1])] = 0
    error2 = np.logical_and(prob_map_hmin2_w == 0, E2)
    error2 = np.logical_and(
        error2, ~skmorph.binary_dilation(prob_map_ref_hmin2_w == 0, np.ones((DIL, DIL)))
    ).astype(np.uint16)
    if dilation == 0:
        labels = np.logical_or(error1, error2)
    else:
        labels = sim.binary_dilation(
            np.logical_or(error1, error2), np.ones((dilation, dilation))
        )
    return labels


class EM2MBNet:
    defaultparams = {
        "em2mb_net": "",
        "flip_em2mb_output": False,
        "do_clahe": False,
        "em2mb_net_package": "UNet",
        "em2mb_net_params": {},
    }

    def __init__(self, params={}):
        sys.path.append(os.path.split(__file__)[0])
        from NNtools import UNet
        from NNtools import UNet2
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.params = self.defaultparams
        self.params.update(params)
        if self.params["em2mb_net_package"] == "UNet":
            self.em2mb_net = UNet.UNet(
                n_channels=1, n_classes=2, **self.params["em2mb_net_params"]
            )
        elif self.params["em2mb_net_package"] == "UNet2":
            self.em2mb_net = UNet2.UNet(
                n_channels=1, n_classes=2, **self.params["em2mb_net_params"]
            )
        else:
            assert False, "not an option"
        self.em2mb_net.load_state_dict(torch.load(self.params["em2mb_net"]))
        self.em2mb_net.to(device=self.device)

    def get_mb(self, em):
        import torch

        if not isinstance(em, np.ndarray):
            em = load_im(em, do_clahe=self.params["do_clahe"])

        with torch.no_grad():
            mb = get_prob(em, self.em2mb_net)
            if self.params["flip_em2mb_output"]:
                mb = 255 - mb
        return mb


class SmartEM:
    defaultparams = {
        "em2mb_net": "../data_perm/Berghia/em2mb_net.pth",
        "error_net": "../data_perm/Berghia/error_net.pth",
        "error_net_type": "cat",
        "pad": 0,
        "rescan_p_thres": 0.5,
        "flip_em2mb_output": False,
        "do_clahe": False,
        "rescan_prob": None,
        "em2mb_net_package": "UNet",
        "error_net_package": "UNet",
        "em2mb_net_params": {},
        "error_net_params": {},
    }

    def __init__(self, params={}):
        sys.path.append(os.path.split(__file__)[0])
        from NNtools import UNet
        from NNtools import UNet2
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.params = self.defaultparams
        self.params.update(params)

        assert self.params["error_net_type"] in [
            "cat",
            "prob",
        ], "Error Net Type should be cat or prob"

        if self.params["em2mb_net_package"] == "UNet":
            self.em2mb_net = UNet.UNet(
                n_channels=1, n_classes=2, **self.params["em2mb_net_params"]
            )
        elif self.params["em2mb_net_package"] == "UNet2":
            self.em2mb_net = UNet2.UNet(
                n_channels=1, n_classes=2, **self.params["em2mb_net_params"]
            )
        else:
            assert False, "not an option"
        self.em2mb_net.load_state_dict(torch.load(self.params["em2mb_net"]))
        self.em2mb_net.to(device=self.device)

        if self.params["error_net_package"] == "UNet":
            self.error_net = UNet.UNet(
                n_channels=1,
                n_classes=(2 if self.params["error_net_type"] == "cat" else 1),
                **self.params["error_net_params"],
            )
        elif self.params["error_net_package"] == "UNet2":
            self.error_net = UNet2.UNet(
                n_channels=1,
                n_classes=(2 if self.params["error_net_type"] == "cat" else 1),
                **self.params["error_net_params"],
            )
        else:
            assert False, "not an option"
        self.error_net.load_state_dict(torch.load(self.params["error_net"]))
        self.error_net.to(device=self.device)

    def get_mb(self, em):
        if not isinstance(em, np.ndarray):
            em = load_im(em, do_clahe=self.params["do_clahe"])
        with torch.no_grad():
            mb = get_prob(em, self.em2mb_net)
            if self.params["flip_em2mb_output"]:
                mb = 255 - mb
        return mb

    def get_error_prob(self, mb):
        if self.params["error_net_type"] == "cat":
            return get_prob(mb, self.error_net, return_dtype=np.float32)
        else:
            error_prob = self.error_net(
                torch.tensor(mb / 255)[None, None].to(
                    device=self.device, dtype=torch.float32
                )
            )[0, 0]
            error_prob = np.clip(error_prob.cpu().detach().numpy(), 0, 1)
            return error_prob

    def smart_mock(self, fast_em, slow_em, rescan_map=None):
        # t=time.perf_counter()
        if not isinstance(fast_em, np.ndarray):
            fast_em = load_im(fast_em, do_clahe=self.params["do_clahe"])
        if not isinstance(slow_em, np.ndarray):
            slow_em = load_im(slow_em, do_clahe=self.params["do_clahe"])
        # t1=time.perf_counter()-t

        # t=time.perf_counter()
        fast_mb = self.get_mb(fast_em)
        if rescan_map is None:
            with torch.no_grad():
                error_prob = self.get_error_prob(fast_mb)
            rescan_map = self.get_rescan_map(error_prob)
        else:
            error_prob = None
        fused_em = fast_em.copy()
        fused_em[rescan_map] = slow_em[rescan_map]
        # t2=time.perf_counter()-t

        fused_mb = self.get_mb(fused_em)
        slow_mb = self.get_mb(slow_em)
        return (
            fast_em,
            slow_em,
            fast_mb,
            slow_mb,
            error_prob,
            rescan_map,
            fused_em,
            fused_mb,
        )

    def pad(self, binim):
        if self.params["pad"] == 0:
            padded = binim
        else:
            padded = skmorph.binary_dilation(
                binim, np.ones((self.params["pad"], self.params["pad"]))
            )
        return padded

    def get_rescan_map(self, error_prob, search_int=0.01):
        if self.params["rescan_prob"] is None:
            rescan_map = self.pad(error_prob > self.params["rescan_p_thres"])
        else:
            rescan_prob = self.params["rescan_prob"]
            imsize = np.prod(error_prob.shape)
            n_tar = int(rescan_prob * imsize)
            thres = np.quantile(error_prob.flatten(), 1 - rescan_prob)
            rescan_map = self.pad(error_prob > thres)
            while rescan_map.sum() > n_tar:
                thres += search_int
                rescan_map = self.pad(error_prob > thres)

        return rescan_map
