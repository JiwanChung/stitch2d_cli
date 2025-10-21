import numpy as np
import cv2


def _hann2d(shape):
    H, W = shape
    wy = np.hanning(H)
    wx = np.hanning(W)
    return np.outer(wy, wx).astype(np.float32)


def _next_pow2(n):
    return 1 << (int(n - 1).bit_length())


def _subpix_quad2d_3x3(p):
    # p: 3x3 patch, peak at center
    # returns (dx, dy) in [-1,1]
    denom_x = (2 * p[1, 1] - p[1, 2] - p[1, 0]) + 1e-12
    denom_y = (2 * p[1, 1] - p[2, 1] - p[0, 1]) + 1e-12
    dx = 0.5 * (p[1, 2] - p[1, 0]) / denom_x
    dy = 0.5 * (p[2, 1] - p[0, 1]) / denom_y
    return float(np.clip(dx, -1, 1)), float(np.clip(dy, -1, 1))


def xcorr2d_linear_peak(A, B, window=True, subpixel=True, restrict_valid=True):
    """
    Linear (zero-padded, 'full') 2D cross-correlation of B over A via FFT.
    Returns (dx, dy, resp) meaning: move B by (dx,dy) to align with A.
    'restrict_valid' limits the peak search to the valid region
    [W2-1:W1-1] x [H2-1:H1-1], which lies 'between paddings'.
    """
    assert A.ndim == 2 and B.ndim == 2
    H1, W1 = A.shape
    H2, W2 = B.shape

    a = A.astype(np.float32, copy=False) - np.float32(A.mean())
    b = B.astype(np.float32, copy=False) - np.float32(B.mean())

    if window:
        a *= _hann2d(a.shape)
        b *= _hann2d(b.shape)

    # full linear convolution size
    Hp = H1 + H2 - 1
    Wp = W1 + W2 - 1

    Fa = np.fft.rfft2(a, s=(Hp, Wp))
    Fb = np.fft.rfft2(b, s=(Hp, Wp))
    corr = np.fft.irfft2(Fa * np.conj(Fb), s=(Hp, Wp))  # real

    # ---- 유효(valid) 영역만 검색 (패딩 사이) ----
    if restrict_valid:
        y_start, y_end = (H2 - 1), (H1 - 1)  # inclusive indices
        x_start, x_end = (W2 - 1), (W1 - 1)
        sub = corr[y_start : y_end + 1, x_start : x_end + 1]
        iy, ix = np.unravel_index(np.argmax(sub), sub.shape)
        y0 = y_start + iy
        x0 = x_start + ix
    else:
        y0, x0 = np.unravel_index(np.argmax(corr), corr.shape)

    # ---- corr-좌표 -> (dy,dx) 변환 (핵심 버그 픽스) ----
    dy_loc = y0 - (H2 - 1)
    dx_loc = x0 - (W2 - 1)

    # 서브픽셀 보정 (3x3 내부일 때만)
    if subpixel and 1 <= y0 < Hp - 1 and 1 <= x0 < Wp - 1:
        patch = corr[y0 - 1 : y0 + 2, x0 - 1 : x0 + 2]
        dx_sub, dy_sub = _subpix_quad2d_3x3(patch)
        dx_loc += dx_sub
        dy_loc += dy_sub

    # 간단 응답 점수 (정규화된 피크 에너지)
    resp = float(corr[y0, x0] / (np.sum(np.abs(corr)) + 1e-12))
    return float(dx_loc), float(dy_loc), resp


def fft_phase_correlation(
    I1, I2, *, direction="horizontal", overlap_ratio=0.5, window=True, subpixel=True
):
    """
    Robust ImageJ-like pairwise translation using linear (zero-padded) xcorr
    over the actual overlap strips only (prevents wrap-around).
    """
    H, W = I1.shape
    direction = direction.lower()
    overlap_ratio = float(np.clip(overlap_ratio, 1e-6, 1.0))

    if direction.startswith("h"):
        ov = max(2, int(round(W * overlap_ratio)))
        A = I1[:, W - ov : W]  # right strip of I1
        B = I2[:, :ov]  # left strip of I2
        dx_loc, dy_loc, resp = xcorr2d_linear_peak(
            A, B, window=window, subpixel=subpixel
        )
        # local->global: A is placed at x = W-ov, B at x = 0 in the crop frame
        dx = (W - ov) + dx_loc
        dy = dy_loc
    else:
        ov = max(2, int(round(H * overlap_ratio)))
        A = I1[H - ov : H, :]  # bottom strip of I1
        B = I2[:ov, :]  # top strip of I2
        dx_loc, dy_loc, resp = xcorr2d_linear_peak(
            A, B, window=window, subpixel=subpixel
        )
        dx = dx_loc
        dy = (H - ov) + dy_loc

    return dx, dy, resp
