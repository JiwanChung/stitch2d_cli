from __future__ import annotations
from typing import Iterable, Tuple, List, Optional, Literal
import numpy as np

RefMode = Literal["first", "global", "median"]


def match_global_mean_var_multi(
    images: Iterable[np.ndarray],
    *,
    ref: RefMode = "first",
    clip: Optional[Tuple[float, float]] = (0.0, 1.0),
    eps: float = 1e-12,
) -> Tuple[List[np.ndarray], List[Tuple[float, float]]]:
    """
    Affine-normalize a list of images so their mean/variance match a reference.

    For each image B_i: B'_i = a_i * B_i + b_i,
      where a_i = sigma_ref / (std(B_i) + eps),
            b_i = mu_ref   - a_i * mean(B_i).

    Args:
        images: iterable of 2D or 3D arrays. Dtypes may differ; converted to float32 internally.
        ref:    "first"  → use stats of the first image,
                "global" → use pixel-weighted global mean/std across all images,
                "median" → use median of per-image means/stds as the target.
        clip:   (lo, hi) to clip outputs after transform; set None to skip.
        eps:    numerical stabilizer to avoid divide-by-zero.

    Returns:
        imgs_out: list of transformed images (dtype preserved: float stays float32; integers cast back).
        params:   list of (a_i, b_i) per image.

    Notes:
        - If an image has ~zero std, we set a_i = 1 and b_i = mu_ref - mean_i (shift-only).
        - For integer inputs, clipping is applied (default [0,1]) before casting back.
    """
    imgs = [np.asarray(im) for im in images]
    if len(imgs) == 0:
        return [], []

    # gather basic stats per image
    means = []
    stds = []
    sizes = []
    for im in imgs:
        imf = im.astype(np.float32, copy=False)
        means.append(float(imf.mean()))
        stds.append(float(imf.std()))
        sizes.append(imf.size)

    means = np.array(means, dtype=np.float64)
    stds = np.array(stds, dtype=np.float64)
    sizes = np.array(sizes, dtype=np.float64)

    # choose reference stats
    if ref == "first":
        mu_ref = means[0]
        sig_ref = stds[0]
    elif ref == "global":
        # weighted global mean / std without concatenating
        N = sizes.sum()
        mu_ref = float(np.sum(means * sizes) / max(N, 1.0))
        # var = E[x^2] - mu^2; estimate E[x^2] from per-image stats
        ex2 = np.sum(((stds**2 + means**2) * sizes)) / max(N, 1.0)
        var = max(ex2 - mu_ref**2, 0.0)
        sig_ref = float(np.sqrt(var))
    elif ref == "median":
        mu_ref = float(np.median(means))
        sig_ref = float(np.median(stds))
    else:
        raise ValueError(f"Unknown ref mode: {ref}")

    # avoid degenerate reference std
    if sig_ref < eps:
        sig_ref = 1.0  # arbitrary scale; will become mostly shift alignment

    # transform each image
    out_imgs: List[np.ndarray] = []
    params: List[Tuple[float, float]] = []

    for im, mu_i, sig_i in zip(imgs, means, stds):
        imf = im.astype(np.float32, copy=False)

        if sig_i < eps:
            a = 1.0
            b = float(mu_ref - mu_i)  # shift-only if flat
        else:
            a = float(sig_ref / (sig_i + eps))
            b = float(mu_ref - a * mu_i)

        Bp = a * imf + b

        # clip & cast back if original is integer
        if clip is not None:
            lo, hi = clip
            Bp = np.clip(Bp, lo, hi)

        if np.issubdtype(im.dtype, np.integer):
            # assume clip in [0,1] for typical 8/16-bit; rescale to original range
            info = np.iinfo(im.dtype)
            # if inputs looked like 0..255, your upstream usually normalized to [0,1]
            # If not, you can adapt scaling here as needed.
            Bp = (Bp * info.max).round().astype(im.dtype, copy=False)
        else:
            Bp = Bp.astype(np.float32, copy=False)

        out_imgs.append(Bp)
        params.append((float(a), float(b)))

    return out_imgs, params
