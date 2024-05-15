import torch

def get_ks_pkop(N, dim, broadcast_op=False, dtype=torch.float32, device="cpu"):
    assert dim in [2,3], "dim must be 2 or 3"
    assert N%2==0, "N must be even"
    imdims = (1, 2, 3) if dim == 3 else (1, 2)
    k_arr = torch.fft.fftfreq(N, 1 / N, dtype=dtype)
    k_p_dims = torch.meshgrid(*(k_arr for _ in range(dim)),indexing='ij')
    ksqs = torch.stack([k_p_dim**2 for k_p_dim in k_p_dims]).sum(dim=0)
    k_abs = torch.sqrt(ksqs)
    pk_len = int(torch.max(k_abs) + 0.5) + 1

    pkind = torch.floor(k_abs + 0.5).reshape(-1)
    _, counts = torch.unique(pkind, return_counts=True)
    norm = torch.ones_like(k_abs)
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
    ks=ks.to(device)
    pkop=pkop.to(device)
    return ks, pkop

def get_pk(images, pkop):
    dim = len(images.size()) - 2
    b = images.size(0)
    c = images.size(1)
    imdims = (2, 3, 4) if dim == 3 else (2, 3)
    images_k = torch.fft.fftn(images, dim=imdims)
    images_k_abs = images_k.real**2 + images_k.imag**2
    pks = (pkop @ images_k_abs.reshape(b*c, -1).permute(1,0)).permute(1,0).reshape(b, c, -1)
    return pks

def pk_rescale(images, pks, target_pks, pkopT):
    N = images.shape[-1]
    assert N % 2 == 0, "N(image side) must be even"
    b = images.size(0)
    c = images.size(1)
    assert pks.size(0)==b and pks.size(1)==c, "pks must have shape (b, c, len_pk)"
    dim = len(images.size()) - 2
    imdims = (2, 3, 4) if dim == 3 else (2, 3)
    assert dim == 2, "3D not implemented"
    fac=torch.where(pks>0, torch.sqrt(target_pks / pks), torch.zeros_like(pks))
    rescaler = (pkopT.t() @ fac.reshape(b*c,-1).permute(1,0)).permute(1,0).reshape(b,c, N, N)
    rescaler[:, 0, 0] = 0

    sh = tuple([N]*dim)
    k_ny_ind = N // 2
    rescaler = rescaler[..., : k_ny_ind + 1]

    images_k = torch.fft.rfftn(images, dim=imdims)
    images_k *= rescaler
    images_rescaled = torch.fft.irfftn(images_k, dim=imdims, s=sh)
    return images_rescaled