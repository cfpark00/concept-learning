# Core Francisco Park implemented a wavelet package inspired from https://arxiv.org/abs/1203.1513

import torch
import torch.fft
import numpy as np

import scipy.interpolate as sintp
import scipy.ndimage as sim
import scipy.stats as sstats
import os
import glob

import time


def make_wavelets(
    N,
    NR=6,
    NT=8,
    twopi=False,
    dtype=torch.float64,
    add_z=False,
    NZ=6,
    return_bases=False,
    verbose=False,
    sqrt=True,
):
    if add_z:
        assert twopi
    k_arr = torch.fft.fftfreq(N, 1 / N).to(dtype=dtype)
    kx, ky = torch.meshgrid(k_arr, k_arr)
    k_abs = torch.sqrt(kx**2 + ky**2)
    k_theta = torch.remainder(torch.atan2(ky, kx), 2 * np.pi)
    rs = torch.logspace(1, np.log2(N // 2), NR + 1, base=2)
    rs = torch.cat([torch.zeros(1), rs], dim=0).to(dtype=dtype)
    r_dists = rs[1:] - rs[:-1]
    thetas = (
        torch.arange(-1, NT + 1).to(dtype=dtype) / NT * (2 * np.pi if twopi else np.pi)
    )
    t_dists = thetas[1:] - thetas[:-1]
    radials = []
    for r, prev_dist, post_dist in zip(rs[1:-1], r_dists[:-1], r_dists[1:]):
        diff = k_abs - r
        t = torch.abs(diff)
        t_prev = t / prev_dist
        t_post = t / post_dist
        prev = (2 * t_prev**3 - 3 * t_prev**2 + 1) * (t < prev_dist) * (diff <= 0)
        post = (2 * t_post**3 - 3 * t_post**2 + 1) * (t < post_dist) * (diff > 0)
        radials.append(post + prev)
    radials = torch.stack(radials)
    angulars = []
    for t, prev_dist, post_dist in zip(thetas[1:-1], t_dists[:-1], t_dists[1:]):
        diff = k_theta - t
        t = torch.abs(diff)
        t = torch.minimum(2 * np.pi - t, t)
        t_prev = t / prev_dist
        t_post = t / post_dist
        prev = (2 * t_prev**3 - 3 * t_prev**2 + 1) * (t < prev_dist) * (diff <= 0)
        post = (2 * t_post**3 - 3 * t_post**2 + 1) * (t < post_dist) * (diff > 0)
        angulars.append(post + prev)
    angulars = torch.stack(angulars)
    dcw = (k_abs <= rs[1]).to(dtype=dtype)
    dcw *= 1 - torch.sum(
        (radials[:, None] * angulars[None, :]).reshape(-1, N, N), dim=0
    )
    if add_z:
        rs = torch.logspace(1, np.log2(N // 2), NZ, base=2)
        rs = torch.cat([torch.zeros(1), rs], dim=0).to(dtype=dtype)
        r_dists = rs[1:] - rs[:-1]
        zs = []
        for r, prev_dist, post_dist in zip(rs[1:-1], r_dists[:-1], r_dists[1:]):
            diff = k_arr - r
            t = torch.abs(diff)
            t_prev = t / prev_dist
            t_post = t / post_dist
            prev = (
                (2 * t_prev**3 - 3 * t_prev**2 + 1) * (t < prev_dist) * (diff <= 0)
            )
            post = (
                (2 * t_post**3 - 3 * t_post**2 + 1) * (t < post_dist) * (diff > 0)
            )
            zs.append(post + prev)
        zs = torch.stack(zs)
        dcz = (torch.abs(k_arr) <= rs[1]).to(dtype=dtype)
        dcz *= 1 - torch.sum(zs, dim=0)
        zs = torch.cat([dcz[None, :], zs], dim=0)

    if return_bases:
        if add_z:
            return dcw, radials, angulars, zs
        else:
            return dcw, radials, angulars

    mms = []
    vals = []
    c = 0
    if add_z:
        tw = (NR * NT + 1) * NZ
        buffer = torch.ones(tuple(N for _ in range(3)), dtype=dtype)
    else:
        tw = NR * NT + 1

    # dcw first
    if add_z:
        for z in zs:
            if verbose:
                print("\r Making", c + 1, "over", tw, end="")
            buffer.fill_(1.0)
            buffer *= dcw[:, :, None]
            buffer *= z[None, None, :]
            mm, val = wavelet_to_mm_val(buffer)
            mms.append(mm)
            vals.append(val)
            c += 1
    else:
        if verbose:
            print("\r Making", c + 1, "over", tw, end="")
        mm, val = wavelet_to_mm_val(dcw)
        mms.append(mm)
        vals.append(val)
        c += 1
    for ri, r in enumerate(radials):
        for ai, angular in enumerate(angulars):
            if add_z:
                for zi, z in enumerate(zs):
                    if verbose:
                        print("\r Making", c + 1, "over", tw, end="")
                    buffer.fill_(1.0)
                    buffer *= (r * angular)[:, :, None]
                    buffer *= z[None, None, :]
                    mm, val = wavelet_to_mm_val(buffer)
                    mms.append(mm)
                    vals.append(val)
                    c += 1
            else:
                if verbose:
                    print("\r Making", c + 1, "over", tw, end="")
                mm, val = wavelet_to_mm_val(r * angular)
                mms.append(mm)
                vals.append(val)
                c += 1
    if sqrt:
        return mms, [torch.sqrt(val) for val in vals]
    return mms, vals


def wavelet_to_mm_val(wavelet):
    dim = len(wavelet.shape)
    assert dim == 2 or dim == 3, "2 or 3 dim"
    wavelet_shifted = torch.fft.fftshift(wavelet)
    logical = wavelet_shifted > 1e-13
    ms = []
    Ms = []
    for i in range(dim):
        fl = list(range(dim))
        fl.remove(i)
        inds = torch.nonzero(logical.sum(dim=tuple(fl)))[:, 0]
        ms.append(torch.min(inds).item())
        Ms.append(torch.max(inds).item() + 1)
    if dim == 3:
        return (
            np.array([ms, Ms]),
            wavelet_shifted[ms[0] : Ms[0], ms[1] : Ms[1], ms[2] : Ms[2]].clone(),
        )
    elif dim == 2:
        return np.array([ms, Ms]), wavelet_shifted[ms[0] : Ms[0], ms[1] : Ms[1]].clone()


def get_dense_wavelet(N, dim, wavelet_mm, wavelet_val):
    filler = torch.zeros([N for _ in range(dim)])
    ms = wavelet_mm[0]
    Ms = wavelet_mm[1]
    if dim == 2:
        filler[ms[0] : Ms[0], ms[1] : Ms[1]] = wavelet_val
    elif dim == 3:
        filler[ms[0] : Ms[0], ms[1] : Ms[1], ms[2] : Ms[2]] = wavelet_val
    return filler


def save_wavelets(wavelet_mms, wavelet_vals, name, wpath=None):
    if wpath is None:
        wpath = os.path.join(os.path.split(__file__)[0], "wavelets", name)
    os.mkdir(wpath)
    Nf = len(wavelet_mms)
    for i in range(Nf):
        np.save(os.path.join(wpath, "mm_" + str(i) + ".npy"), wavelet_mms[i])
        np.save(
            os.path.join(wpath, "val_" + str(i) + ".npy"),
            wavelet_vals[i].cpu().detach().numpy(),
        )


def load_wavelets(name, wpath=None, dtype=torch.float64, device="cpu"):
    if wpath is None:
        wpath = os.path.join(os.path.split(__file__)[0], "wavelets", name)
    fs = glob.glob(os.path.join(wpath, "mm_*"))
    Nf = len(fs)
    wavelet_mms = []
    wavelet_vals = []
    for i in range(Nf):
        wavelet_mms.append(np.load(os.path.join(wpath, "mm_" + str(i) + ".npy")))
        wavelet_vals.append(
            torch.tensor(
                np.load(os.path.join(wpath, "val_" + str(i) + ".npy")),
                dtype=dtype,
                device=device,
            )
        )
    return wavelet_mms, wavelet_vals


def get_apodizer(N, dim, apstart=3 / 4):
    cent = (N - 1) / 2
    arr = torch.arange(N)
    r = (
        torch.sqrt(
            (
                (torch.stack(torch.meshgrid(*[arr for _ in range(dim)]), dim=0) - cent)
                ** 2
            ).sum(0)
        )
        / cent
    )
    apodizer = torch.zeros_like(r)
    r_rec = (r - apstart) * 4
    apodizer = 2 * r_rec**3 - 3 * r_rec**2 + 1
    apodizer[r <= apstart] = 1
    apodizer[r > 1] = 0
    return apodizer


def WST_abs2(
    images,
    wavelet_mms,
    wavelet_vals,
    m=2,
    verbose=False,
    MAS_corrector=None,
    cross_correlate=False,
    cross_correlate_in_memory=True,
):
    assert m == 0 or m == 1 or m == 2
    dim = len(images.size()) - 1
    assert dim == 2 or dim == 3
    N = images.size(1)
    Nw = len(wavelet_mms)
    assert Nw == len(wavelet_vals)

    imdims = (1, 2, 3) if dim == 3 else (1, 2)

    coeffs = []
    std, mean = torch.std_mean(images, dim=imdims, unbiased=False)
    coeffs.append(mean)
    coeffs.append(std)
    if m == 0:
        return torch.stack(coeffs).T
    if dim == 3:
        images = (images - mean[:, None, None, None]) / (
            std[:, None, None, None] + 1e-8
        )
    else:
        images = (images - mean[:, None, None]) / (std[:, None, None] + 1e-8)
    image_k = torch.fft.fftn(images, dim=imdims)
    if MAS_corrector is not None:
        image_k *= MAS_corrector[None]
    image_k = torch.fft.fftshift(image_k, dim=imdims)

    if m == 2:
        buffer = torch.zeros_like(image_k)
        coeffs2 = []
    if cross_correlate:
        maps = []
    wavelet_sqs = [wavelet**2 for wavelet in wavelet_vals]
    for w1 in range(Nw):
        if verbose:
            print("Wavelet:", str(w1 + 1), "/", str(Nw))
        if m == 2:
            buffer.zero_()
        ms = wavelet_mms[w1][0]
        Ms = wavelet_mms[w1][1]
        if dim == 3:
            sub = (
                image_k[:, ms[0] : Ms[0], ms[1] : Ms[1], ms[2] : Ms[2]]
                * wavelet_vals[w1][None, :, :, :]
            )
            coeffs.append(torch.sum(sub.real**2 + sub.imag**2, dim=imdims))
            if m == 1 and not cross_correlate:
                continue
            buffer[:, ms[0] : Ms[0], ms[1] : Ms[1], ms[2] : Ms[2]] = sub
        elif dim == 2:
            sub = (
                image_k[:, ms[0] : Ms[0], ms[1] : Ms[1]] * wavelet_vals[w1][None, :, :]
            )
            coeffs.append(torch.sum(sub.real**2 + sub.imag**2, dim=imdims))
            if m == 1 and not cross_correlate:
                continue
            buffer[:, ms[0] : Ms[0], ms[1] : Ms[1]] = sub
        im1_r = torch.fft.ifftn(torch.fft.ifftshift(buffer, dim=imdims), dim=imdims)
        im1_r = torch.sqrt(im1_r.real**2 + im1_r.imag**2)
        if cross_correlate:
            maps.append(im1_r)
        if m == 1:
            continue
        im1_k = torch.fft.fftshift(torch.fft.fftn(im1_r, dim=imdims), dim=imdims)
        im1_k_abs2 = im1_k.real**2 + im1_k.imag**2
        for w2 in range(Nw):
            ms = wavelet_mms[w2][0]
            Ms = wavelet_mms[w2][1]

            if dim == 3:
                coeffs2.append(
                    torch.sum(
                        im1_k_abs2[:, ms[0] : Ms[0], ms[1] : Ms[1], ms[2] : Ms[2]]
                        * wavelet_sqs[w2][None, :, :, :],
                        dim=imdims,
                    )
                )
            elif dim == 2:
                coeffs2.append(
                    torch.sum(
                        im1_k_abs2[:, ms[0] : Ms[0], ms[1] : Ms[1]]
                        * wavelet_sqs[w2][None, :, :],
                        dim=imdims,
                    )
                )
    if m == 2:
        coeffs.extend(coeffs2)
    if cross_correlate:
        b = images.shape[0]
        if cross_correlate_in_memory:
            maps = torch.stack(maps, dim=1)
            maps = maps.reshape(b, Nw, -1)
            coeffs_cc = torch.bmm(maps, maps.permute(0, 2, 1)).reshape(b, -1).T
        else:
            coeffs_cc = torch.zeros(Nw, Nw, b, device=images.device, dtype=images.dtype)
            for i, map_1 in enumerate(maps):
                for j, map_2 in enumerate(maps):
                    if j > i:
                        continue
                    val = torch.sum(map_1 * map_2, dim=imdims)
                    coeffs_cc[i, j] = val
                    if i != j:
                        coeffs_cc[j, i] = val
            coeffs_cc = coeffs_cc.reshape(-1, b)
    if cross_correlate:
        coeffs.extend(coeffs_cc)
    return torch.stack(coeffs).T


def WST_abs(images, wavelet_mms, wavelet_vals, m=2, verbose=False, MAS_corrector=None):
    assert m == 0 or m == 1 or m == 2
    dim = len(images.size()) - 1
    assert dim == 2 or dim == 3
    N = images.size(1)
    Nw = len(wavelet_mms)
    assert Nw == len(wavelet_vals)

    imdims = (1, 2, 3) if dim == 3 else (1, 2)

    coeffs = []
    std, mean = torch.std_mean(images, dim=imdims, unbiased=False)
    coeffs.append(mean)
    coeffs.append(std)
    if m == 0:
        return torch.stack(coeffs).T
    if dim == 3:
        images = (images - mean[:, None, None, None]) / (
            std[:, None, None, None] + 1e-8
        )
    else:
        images = (images - mean[:, None, None]) / (std[:, None, None] + 1e-8)
    image_k = torch.fft.fftn(images, dim=imdims)
    if MAS_corrector is not None:
        image_k *= MAS_corrector[None]
    image_k = torch.fft.fftshift(image_k, dim=imdims)

    if m == 2:
        buffer = torch.zeros_like(image_k)
        coeffs2 = []

    wavelet_sqs = [wavelet**2 for wavelet in wavelet_vals]
    for w1 in range(Nw):
        if verbose:
            print("Wavelet:", str(w1 + 1), "/", str(Nw))
        if m == 2:
            buffer.zero_()
        ms = wavelet_mms[w1][0]
        Ms = wavelet_mms[w1][1]
        if dim == 3:
            buffer[:, ms[0] : Ms[0], ms[1] : Ms[1], ms[2] : Ms[2]] = (
                image_k[:, ms[0] : Ms[0], ms[1] : Ms[1], ms[2] : Ms[2]]
                * wavelet_vals[w1][None, :, :, :]
            )
        elif dim == 2:
            buffer[:, ms[0] : Ms[0], ms[1] : Ms[1]] = (
                image_k[:, ms[0] : Ms[0], ms[1] : Ms[1]] * wavelet_vals[w1][None, :, :]
            )
        im1_r = torch.fft.ifftn(torch.fft.ifftshift(buffer, dim=imdims), dim=imdims)
        im1_r = torch.sqrt(im1_r.real**2 + im1_r.imag**2)
        coeffs.append(torch.sum(im1_r, dim=imdims))
        if m == 1:
            continue
        im1_k = torch.fft.fftshift(torch.fft.fftn(im1_r, dim=imdims), dim=imdims)
        for w2 in range(Nw):
            buffer.zero_()
            ms = wavelet_mms[w2][0]
            Ms = wavelet_mms[w2][1]

            if dim == 3:
                buffer[:, ms[0] : Ms[0], ms[1] : Ms[1], ms[2] : Ms[2]] = (
                    im1_k[:, ms[0] : Ms[0], ms[1] : Ms[1], ms[2] : Ms[2]]
                    * wavelet_vals[w2][None, :, :, :]
                )
            elif dim == 2:
                buffer[:, ms[0] : Ms[0], ms[1] : Ms[1]] = (
                    im1_k[:, ms[0] : Ms[0], ms[1] : Ms[1]]
                    * wavelet_vals[w2][None, :, :]
                )
            im2_r = torch.fft.ifftn(torch.fft.ifftshift(buffer, dim=imdims), dim=imdims)
            coeffs2.append(
                torch.sum(torch.sqrt(im2_r.real**2 + im2_r.imag**2), dim=imdims)
            )
    if m == 2:
        coeffs.extend(coeffs2)
    return torch.stack(coeffs).T


def disp(wst, nf, m=2, each_norm=False, flip_rl=False, r=6, l=8):
    if type(wst) != np.ndarray:
        wst = wst.cpu().detach().numpy()
    assert m == 2, "only for m=2 for now"
    assert len(wst) == 2 + nf + nf**2, "length mismatch"
    feed = wst.copy()
    if flip_rl:
        feed[3 : 2 + nf] = feed[3 : 2 + nf].reshape(l, r).T.reshape(-1)

        temp = feed[2 + nf :].reshape(1 + r * l, 1 + r * l)
        temp[1:, 0] = temp[1:, 0].reshape(l, r).T.reshape(-1)
        temp[0, 1:] = temp[0, 1:].reshape(l, r).T.reshape(-1)
        temp[1:, 1:] = (
            temp[1:, 1:].reshape(l, r, l, r).transpose(1, 0, 3, 2).reshape(r * l, r * l)
        )
        feed[2 + nf :] = temp.reshape(-1)

    if each_norm:

        def normalize(vals):
            return (vals - np.mean(vals)) / np.std(vals)

    else:
        normalize = lambda x: x
    im = np.zeros((nf, 3 + nf))
    im[:, 0] = feed[0] if not each_norm else 0
    im[:, 1] = feed[1] if not each_norm else 0
    im[:, 2] = normalize(feed[2 : 2 + nf])
    im[:, 3:] = normalize(feed[2 + nf :].reshape(nf, nf))
    return im


def disploc(indice, nf):
    if indice == 0:
        return np.zeros(2, dtype=np.int16)
    if indice == 1:
        return np.array([0, 1], dtype=np.int16)
    if indice < (2 + nf):
        return np.array([indice - 2, 2])
    res = indice - 2 - nf
    return np.array(np.unravel_index(res, (nf, nf))) + np.array([0, 3])


def get_MAS_corrector(MAS_correction, N, dim, dtype=torch.float64):
    assert MAS_correction in [1, 2, 3, 4]
    assert dim in [2, 3]
    imdims = (1, 2, 3) if dim == 3 else (1, 2)
    k_arr = torch.fft.fftfreq(N, 1 / N, dtype=dtype) / N
    k_p_dims = torch.meshgrid(*(k_arr for _ in range(dim)))
    if dim == 3:
        MAS_corrector = 1 / (
            torch.sinc(k_p_dims[0]) * torch.sinc(k_p_dims[1]) * torch.sinc(k_p_dims[2])
        )
    else:
        MAS_corrector = 1 / (torch.sinc(k_p_dims[0]) * torch.sinc(k_p_dims[1]))
    return MAS_corrector**MAS_correction


def get_ks_pkop(N, dim, dtype=torch.float64, MAS_correction=0, broadcast_op=False):
    imdims = (1, 2, 3) if dim == 3 else (1, 2)
    k_arr = torch.fft.fftfreq(N, 1 / N, dtype=dtype)
    k_p_dims = torch.meshgrid(*(k_arr for _ in range(dim)))
    ksqs = torch.stack([k_p_dim**2 for k_p_dim in k_p_dims]).sum(dim=0)
    k_abs = torch.sqrt(ksqs)
    pk_len = int(torch.max(k_abs) + 0.5) + 1

    pkind = torch.floor(k_abs + 0.5).reshape(-1)
    _, counts = torch.unique(pkind, return_counts=True)
    norm = (
        get_MAS_corrector(MAS_correction, N, dim)
        if MAS_correction in [1, 2, 3, 4]
        else torch.ones_like(k_abs)
    )
    norm = norm.reshape(-1)

    ks = []
    k_abs_flat = k_abs.reshape(-1)
    for i in range(pk_len):
        where = pkind == i
        if not broadcast_op:
            norm[where] /= counts[i]
        ks.append(k_abs_flat[where].mean())
    ks = torch.tensor(ks, dtype=torch.float64)

    indarr = torch.arange(N)
    if dim == 3:
        indarr3d = (
            N * N * indarr[:, None, None]
            + N * indarr[None, :, None]
            + indarr[None, None, :]
        )
    else:
        indarr3d = N * indarr[:, None] + indarr[None, :]
    ind = torch.stack([pkind, indarr3d.reshape(-1)])

    pkop = torch.sparse_coo_tensor(ind, norm, (pk_len, N**dim))
    return ks, pkop


def get_k_kill(ks, kstart, kend, spline_func=None):
    if spline_func is None:
        spline_func = lambda t: 2 * t**3 - 3 * t**2 + 1
    k0 = ks[ks <= kstart].fill_(1.0)
    k1 = spline_func((ks[(ks > kstart) * (ks <= kend)] - kstart) / (kend - kstart))
    k2 = ks[ks > kend].zero_()
    return torch.cat([k0, k1, k2], dim=0)


def pk_rescale(images, pks, tpks, pkopT, rayleigh=False):
    N = images.shape[1]
    assert N % 2 == 0, "odd N not allowed yet"
    b = images.shape[0]
    dim = len(images.size()) - 1
    imdims = (1, 2, 3) if dim == 3 else (1, 2)
    assert dim == 2, "3D not implemented"
    fac = torch.sqrt(tpks / pks)
    fac[pks == 0] = 0
    res = (pkopT.t() @ fac.t()).t().reshape(b, N, N)
    res[:, 0, 0] = 0

    sh = (N, N)
    k_ny_ind = N // 2
    res = res[:, :, : k_ny_ind + 1]

    if rayleigh is True:
        mult = torch.sqrt(-torch.log(torch.rand(res.shape, device=res.device)))
        while torch.isnan(mult).any() or torch.isinf(mult).any():
            mult = torch.sqrt(-torch.log(torch.rand(res.shape, device=res.device)))
        mult[:, k_ny_ind + 1 :, 0] = mult[:, 1:k_ny_ind, 0]  # hermitianity
        res *= mult
    elif not (rayleigh is False):
        res *= rayleigh

    images_k = torch.fft.rfftn(images, dim=imdims)
    images_k *= res
    images_rescaled = torch.fft.irfftn(images_k, dim=imdims, s=sh)
    return images_rescaled


def get_gaussian_random_field(N, dim, pks, pkopT, rayleigh=False):
    assert N % 2 == 0, "odd N not allowed yet"
    b = pks.shape[0]
    imdims = (1, 2, 3) if dim == 3 else (1, 2)
    assert dim == 2, "3D not implemented"
    res = (pkopT.t() @ torch.sqrt(pks).t()).t().reshape(b, N, N)
    res[:, 0, 0] = torch.sqrt(pks[:, 0])

    sh = (N, N)
    k_ny_ind = N // 2
    res = res[:, :, : k_ny_ind + 1]

    if rayleigh is True:
        mult = torch.sqrt(-torch.log(torch.rand(res.shape, device=res.device)))
        while torch.isnan(mult).any() or torch.isinf(mult).any():
            mult = torch.sqrt(-torch.log(torch.rand(res.shape, device=res.device)))
        mult[:, k_ny_ind + 1 :, 0] = mult[:, 1:k_ny_ind, 0]  # hermitianity
        res *= mult
    elif not (rayleigh is False):
        res *= rayleigh

    phases = torch.rand(res.shape, device=res.device, dtype=res.dtype)
    phases[:, k_ny_ind + 1 :, 0] = phases[:, 1:k_ny_ind, 0]

    images_k = torch.exp(1j * 2 * np.pi * phases)
    images_k *= res
    images_rescaled = torch.fft.irfftn(images_k, dim=imdims, s=sh)
    return images_rescaled


def pk(images, pkop):
    dim = len(images.size()) - 1
    batch = images.size(0)
    imdims = (1, 2, 3) if dim == 3 else (1, 2)
    images_k = torch.fft.fftn(images, dim=imdims)
    images_k_abs = images_k.real**2 + images_k.imag**2
    pks = (pkop @ images_k_abs.reshape(batch, -1).T).T
    return pks


def make_spline_rings(
    N, dim, NR=16, rs_dimless=None, btype="lin", dtype=torch.float64, return_ks=False
):
    k_arr = torch.fft.fftfreq(N, 1 / N).to(dtype=dtype)
    k_p_dims = torch.meshgrid(*(k_arr for _ in range(dim)))
    ksqs = torch.stack([k_p_dim**2 for k_p_dim in k_p_dims]).sum(dim=0)
    k_abs = torch.sqrt(ksqs)

    if rs_dimless is not None:
        rs = rs_dimless
    else:
        if btype == "log":
            rs = torch.logspace(1, np.log2(N // 2), NR + 1, base=2)
        elif btype == "lin":
            rs = torch.linspace(2, N // 2, NR + 1)
        else:
            assert False, "Not implemented"
        rs = torch.cat([torch.zeros(1), rs], dim=0).to(dtype=dtype)
    r_dists = rs[1:] - rs[:-1]

    radials = []
    for r, prev_dist, post_dist in zip(rs[1:-1], r_dists[:-1], r_dists[1:]):
        diff = k_abs - r
        t = torch.abs(diff)
        t_prev = t / prev_dist
        t_post = t / post_dist
        prev = (2 * t_prev**3 - 3 * t_prev**2 + 1) * (t < prev_dist) * (diff <= 0)
        post = (2 * t_post**3 - 3 * t_post**2 + 1) * (t < post_dist) * (diff > 0)
        radials.append(post + prev)
    radials = torch.stack(radials)

    rings = radials
    if return_ks:
        return rings, rs[1:-1]
    return rings


def make_exp_rings(
    N,
    dim,
    NR=16,
    rs_dimless=None,
    sigmas_dimless=None,
    btype="lin",
    dtype=torch.float64,
    return_ks=False,
):
    k_arr = torch.fft.fftfreq(N, 1 / N).to(dtype=dtype)
    k_p_dims = torch.meshgrid(*(k_arr for _ in range(dim)))
    ksqs = torch.stack([k_p_dim**2 for k_p_dim in k_p_dims]).sum(dim=0)
    k_abs = torch.sqrt(ksqs)

    if rs_dimless is not None:
        rs = rs_dimless
    else:
        if btype == "log":
            rs = torch.logspace(1, np.log2(N // 2), NR - 1, base=2)
        elif btype == "lin":
            rs = torch.linspace(2, N // 2, NR - 1)
        else:
            assert False, "Not implemented"
        rs = torch.cat([torch.zeros(1), rs], dim=0).to(dtype=dtype)

    if sigmas_dimless is not None:
        sigmas = sigmas_dimless
    else:
        r_dists_ = torch.abs(rs[1:] - rs[:-1])
        r_distsavg = (r_dists_[1:] + r_dists_[:-1]) / 2
        sigmas = torch.cat([r_distsavg[[0]], r_distsavg, r_distsavg[[-1]]], dim=0).to(
            dtype=dtype
        )

    radials = []
    for r, sigma in zip(rs, sigmas):
        radials.append(
            torch.exp(-(((k_abs - r) ** 2) / (2 * sigma**2)))
            / (sigma * np.sqrt(2 * np.pi))
        )
    radials = torch.stack(radials)

    rings = radials
    if return_ks:
        return rings, rs
    return rings


def bke(images, rings, configs=None, MAS_corrector=None, inner_vec=False):
    dim = len(images.size()) - 1
    assert dim == 2 or dim == 3, "image should be 2D or 3D"
    N = images.size(1)
    batch = images.size(0)
    imdims = (1, 2, 3) if dim == 3 else (1, 2)
    image_k = torch.fft.fftn(images, dim=imdims)
    if MAS_corrector is not None:
        image_k *= MAS_corrector[None]
    temp = []
    for ring in rings:
        temp.append(torch.fft.ifftn(image_k * ring[None], dim=imdims).real)
    temp = torch.stack(temp)
    # TODO: triangular inequality

    Bkes = []
    if configs is None:
        if inner_vec:
            imdimsp1 = (2, 3, 4) if dim == 3 else (2, 3)
            for i in range(len(rings)):
                for j in range(len(rings)):
                    Bkes.append(torch.sum(temp[[i]] * temp[[j]] * temp, dim=imdimsp1))
            return torch.stack(Bkes).reshape(len(rings) ** 3, -1).T
        else:
            for i in range(len(rings)):
                for j in range(len(rings)):
                    for k in range(len(rings)):
                        Bkes.append(torch.sum(temp[i] * temp[j] * temp[k], dim=imdims))
            return torch.stack(Bkes).T

    for i, j, k in configs:
        Bkes.append(torch.sum(temp[i] * temp[j] * temp[k], dim=imdims))
    return torch.stack(Bkes).T


def bkn(rings, configs=None):
    Bkns = []
    temp = []
    for ring in rings:
        temp.append(torch.fft.ifftn(ring).real)
    # TODO: triangular inequality

    if configs is None:
        for i in range(len(rings)):
            for j in range(len(rings)):
                for k in range(len(rings)):
                    Bkns.append(torch.sum(temp[i] * temp[j] * temp[k]))

        return torch.stack(Bkns)
    for i, j, k in configs:
        Bkns.append(torch.sum(temp[i] * temp[j] * temp[k]))
    return torch.stack(Bkns)


def get_rwst(x, NR=6, NT=8, NZ=None):
    if NZ is None:  # 2d case
        NF = 1 + NR * NT
        assert x.shape[1] == (2 + NF + NF**2)
        b = x.shape[0]
        s0 = x[:, :2]
        s1dc = x[:, 2]
        s1 = x[:, 3 : 3 + NR * NT].reshape(b, NR, NT).sum(2)
        s2 = x[:, 2 + NF :].reshape(b, NF, NF)
        s2dcdc = s2[:, 0, 0][:, None]
        s2dc1 = s2[:, 0, 1:].reshape(b, NR, NT).sum(2)
        s2dc2 = s2[:, 1:, 0].reshape(b, NR, NT).sum(2)
        s2 = (
            s2[:, 1:, 1:]
            .reshape(b, NR, NT, NR, NT)
            .transpose(0, 1, 3, 2, 4)
            .reshape(b, NR * NR, NT, NT)
        )
        s2roll = []
        for l_ in range(NT):
            s2roll.append(np.roll(s2[:, :, l_, :], -l_, axis=2))
        s2roll = np.stack(s2roll).sum(0).reshape(b, -1)
        rwst = np.concatenate(
            [s0, s1dc[:, None], s1, s2dcdc, s2dc1, s2dc2, s2roll], axis=1
        )
    else:
        NF = (1 + NR * NT) * NZ
        assert x.shape[1] == (2 + NF + NF**2)
        b = x.shape[0]
        s0 = x[:, :2]
        s1dcz = x[:, 2 : 2 + NZ]
        s1 = (
            x[:, 2 + NZ : 2 + NZ + NR * NT * NZ]
            .reshape(b, NR, NT, NZ)
            .sum(2)
            .reshape(b, -1)
        )

        s2 = x[:, 2 + NF :].reshape(b, NF, NF)
        s2dczdcz = s2[:, :NZ, :NZ].reshape(b, -1)
        s2dcz1 = s2[:, :NZ, NZ:].reshape(b, NZ, NR, NT, NZ).sum(3).reshape(b, -1)
        s2dcz2 = s2[:, NZ:, :NZ].reshape(b, NR, NT, NZ, NZ).sum(2).reshape(b, -1)
        s2 = (
            s2[:, NZ:, NZ:]
            .reshape(b, NR, NT, NZ, NR, NT, NZ)
            .transpose(0, 1, 3, 4, 6, 2, 5)
            .reshape(b, NR * NZ * NR * NZ, NT, NT)
        )
        s2roll = []
        for l_ in range(NT):
            s2roll.append(np.roll(s2[:, :, l_, :], -l_, axis=2))
        s2roll = np.stack(s2roll).sum(0).reshape(b, -1)
        rwst = np.concatenate([s0, s1dcz, s1, s2dczdcz, s2dcz1, s2dcz2, s2roll], axis=1)
    return rwst


def get_rwst_cc(x, NR=6, NT=8):
    NF = 1 + NR * NT
    assert x.shape[1] == (NF**2)
    b = x.shape[0]
    cc = x.reshape(b, NF, NF)
    s2dcdc = cc[:, 0, 0][:, None]
    s2dc1 = cc[:, 0, 1:].reshape(b, NR, NT).sum(2)
    s2dc2 = cc[:, 1:, 0].reshape(b, NR, NT).sum(2)
    cc = (
        cc[:, 1:, 1:]
        .reshape(b, NR, NT, NR, NT)
        .transpose(0, 1, 3, 2, 4)
        .reshape(b, NR * NR, NT, NT)
    )
    s2roll = []
    for l_ in range(NT):
        s2roll.append(np.roll(cc[:, :, l_, :], -l_, axis=2))
    s2roll = np.stack(s2roll).sum(0).reshape(b, -1)
    rwst = np.concatenate([s2dcdc, s2dc1, s2dc2, s2roll], axis=1)
    return rwst


def batched_run(func, data, args, batch_size, outsel=None, verbose=False):
    N_tot = data.shape[0]
    jwsts_g = []
    N_rep = N_tot // batch_size
    if (N_tot % batch_size) != 0:
        N_rep += 1
    outlist = []
    for i in range(N_rep):
        if verbose:
            print("\r", i + 1, "/", N_rep, end="")
        ten = data[i * batch_size : min((i + 1) * batch_size, N_tot)]
        res = func(ten, *args)
        if outsel is not None:
            res = res[outsel]
        outlist.append(res)
    if verbose:
        print()
    return outlist


def gaussianize_pdf(data, target_ppf=sstats.norm(loc=0.0, scale=1.0).ppf):
    sh = data.shape
    data = data.copy().flatten()
    hist, bin_edges = np.histogram(data, bins="auto")
    r = sstats.rv_histogram((hist, bin_edges))

    min2i = np.argpartition(data, 2)[:2]
    max2i = np.argpartition(-data, 2)[:2]

    data[min2i[0]] = data[min2i[1]]
    data[max2i[0]] = data[max2i[1]]

    gdata = target_ppf(r.cdf(data))

    return gdata.reshape(sh)


######Visualization
def hsv2rgb(im_hsv):
    h = im_hsv[:, :, 0]
    s = im_hsv[:, :, 1]
    v = im_hsv[:, :, 2]
    h = np.mod(h, 2 * np.pi)
    h60 = h / (np.pi / 3)
    h60f = np.floor(h60)
    section = (h60f.astype(np.int16) % 6)[:, :, None]
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    res = np.stack([v, t, p], axis=2)
    res = np.where(section == 1, np.stack([q, v, p], axis=2), res)
    res = np.where(section == 2, np.stack([p, v, t], axis=2), res)
    res = np.where(section == 3, np.stack([p, q, v], axis=2), res)
    res = np.where(section == 4, np.stack([t, p, v], axis=2), res)
    res = np.where(section == 5, np.stack([v, p, q], axis=2), res)
    return res


def complex_image(dk, normfunc=None, mag=True):
    if normfunc is None:
        normfunc = lambda x: np.arcsinh(x / np.std(x))
    if mag:
        mag = normfunc(np.abs(dk))
        mag -= np.min(mag)
        mag /= np.max(mag)
        s = np.ones_like(dk, dtype=np.float64)
    else:
        s = mag = np.ones_like(dk, dtype=np.float64)
    phase = np.angle(dk) + np.pi
    arr = hsv2rgb(np.stack([phase, s, mag], axis=2))
    return arr


####################################


def make_healpix_wavelets(N, NR=8, nside=2, dtype=torch.float64, rough_mem_limit=4000):
    import healpy

    assert (
        dtype == torch.float64 or dtype == torch.float32
    ), "only torch.float64 or torch.float32 allowed"
    mem = NR * 12 * (nside**2) * (N**3) * (8 if dtype == torch.float64 else 4) / 1e6
    assert mem < rough_mem_limit, "This probably uses " + str(mem) + " MB"
    k_arr = torch.fft.fftfreq(N, 1 / N).to(dtype=dtype)
    kx, ky, kz = torch.meshgrid(k_arr, k_arr, k_arr)
    k_abs = torch.sqrt(kx**2 + ky**2 + kz**2)
    k_phi = torch.atan2(ky, kx)
    k_theta = torch.atan2(torch.sqrt(kx**2 + ky**2), kz)

    rs = torch.logspace(1, np.log2(N // 2), NR + 1, base=2)
    rs = torch.cat([torch.zeros(1), rs], dim=0).to(dtype=dtype)
    r_dists = rs[1:] - rs[:-1]
    radials = []
    for r, prev_dist, post_dist in zip(rs[1:-1], r_dists[:-1], r_dists[1:]):
        diff = k_abs - r
        t = torch.abs(diff)
        t_prev = t / prev_dist
        t_post = t / post_dist
        prev = (2 * t_prev**3 - 3 * t_prev**2 + 1) * (t < prev_dist) * (diff <= 0)
        post = (2 * t_post**3 - 3 * t_post**2 + 1) * (t < post_dist) * (diff > 0)
        radials.append(post + prev)
    radials = torch.stack(radials).to(dtype=dtype)

    inds, weights = healpy.pixelfunc.get_interp_weights(
        nside, k_theta.numpy(), k_phi.numpy()
    )
    indarr = np.arange(N)
    indarrx, indarry, indarrz = np.meshgrid(indarr, indarr, indarr, indexing="ij")

    angulars = torch.zeros(
        healpy.pixelfunc.nside2npix(nside), N, N, N, dtype=dtype
    )  # npix is 12*nside**2
    angulars[inds[0], indarrx, indarry, indarrz] += torch.tensor(
        weights[0], dtype=dtype
    )
    angulars[inds[1], indarrx, indarry, indarrz] += torch.tensor(
        weights[1], dtype=dtype
    )
    angulars[inds[2], indarrx, indarry, indarrz] += torch.tensor(
        weights[2], dtype=dtype
    )
    angulars[inds[3], indarrx, indarry, indarrz] += torch.tensor(
        weights[3], dtype=dtype
    )

    # make the DC part.. yes this is very inefficient.
    wavelets = (radials[None, :] * angulars[:, None]).reshape(-1, N, N, N)
    dcw = (k_abs <= rs[1]).to(dtype=dtype)
    dcw *= 1 - torch.sum(wavelets, dim=0)

    angulars = angulars[: healpy.pixelfunc.nside2npix(nside) // 2]
    wavelets = (radials[None, :] * angulars[:, None]).reshape(-1, N, N, N)
    wavelets = torch.cat([dcw[None, ...], wavelets], dim=0).to(dtype=dtype)
    return wavelets


def make_icosahedron_wavelets(N, NR=8, nside=1):
    assert False, "Not doing this"
    # from https://en.wikipedia.org/wiki/Regular_icosahedron#/media/File:Icosahedron-golden-rectangles.svg


def LIest(image, scales=None, NR=8, max_offset=10):
    N = image.size(0)
    if scales is None:
        scales = torch.logspace(0, np.log2(N // 2), NR, base=2).numpy()
    coeffs = []
    for i in range(len(scales) + 1):
        if i == 0:
            im = image
        else:
            im = torch.tensor(
                sim.gaussian_filter(image.numpy(), scales[i - 1], mode="wrap")
            )
        for offset in range(1, max_offset):
            coeffs.append(
                torch.sum(torch.sqrt(torch.abs(torch.roll(im, offset, 0) * im)))
                / (N * N)
            )
            coeffs.append(
                torch.sum(torch.sqrt(torch.abs(torch.roll(im, offset, 1) * im)))
                / (N * N)
            )
    return torch.stack(coeffs)
