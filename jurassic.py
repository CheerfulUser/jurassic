import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import pandas as pd
try:
    import stpsf
except ModuleNotFoundError:
    import webbpsf as stpsf
import os
import sep
import glob

sep.set_extract_pixstack(1_000_000)  # default 300000 overflows on frames with large contiguous excess regions

FORCED_PHOTOMETRY_APERTURE_HW = 1  # 3x3 -- shared between _amplitude_only_fit's
                                    # default and the box drawn around it in
                                    # plot_augmented_sw_lw's _draw_row, so the
                                    # two can't drift apart

from astropy.io import fits     
from astropy.convolution import convolve_fft
from joblib import Parallel, delayed
from pathlib import Path
from scipy.optimize import curve_fit
from astropy.stats import sigma_clipped_stats, sigma_clip
from lacosmic.core import lacosmic # func is apparently deprecated - will be 'remove_cosmics'
from sklearn.cluster import DBSCAN

import warnings
warnings.filterwarnings("ignore")

import logging, sys
logging.disable(sys.maxsize)

np.set_printoptions(legacy='1.25') # my environment is a bit wonky :/

def linear_fitting(coords,cube,n_int,n_group):
    """
    fits ramps piecewise with 2 straight lines if a jump is detected,
    otherwise fits a single line. If a piece is too short, fits only the longer one.
    """
    row, col = coords

    grads = [] # straight line gradient based on first data points
    intercepts = [] # straight line intercept based on first data points
    resids = [] # residuals from curve_fit of power law

    for int_num in range(n_int):
        x_dat = [i + (int_num)*n_group for i in range(n_group)]
        y_dat = [cube[x][row][col] for x in x_dat]
        x = np.asarray(x_dat, dtype=float) 
        y = np.asarray(y_dat, dtype=float)

        # removing first and last frames
        y[0] = np.nan
        y[-1] = np.nan

        mask = ~np.isnan(y)
        x = x[mask]
        y = y[mask]

        p,r,*_ = np.polyfit(x,y,1,full=True)
        m = p[0]
        c = p[1]
        grads.append(m)
        intercepts.append(c)
        resids.append(r[0] if len(r) > 0 else np.nan)
        
    return [row, col, grads, intercepts, resids]


def run_lacosmic(frame_data, mask, contrast=4, cr_threshold=2, neighbor_threshold=0.9):
    """
    running lacosmic so it can be done parallely on frames
    mask: boolean mask where True = science pixel, False = bad pixel

    contrast/cr_threshold/neighbor_threshold default to the original
    MIRI-validated values. _remove_cosmic passes looser NIRCam-specific
    values instead (see there) — these defaults never fit NIRCam data:
    on a real Cosmic Snake exposure they flagged 6.6% of the ENTIRE
    frame (vs. a physically-realistic cosmic-ray rate well under 0.5%)
    because the permissive neighbor_threshold=0.9 growth step runs away
    across ordinary extended structure (the bright cluster/arc light
    this field is full of), which is also why they took 23.8s/frame
    against ~12s for the looser values, and separately why the
    "cleaning" interpolation destroys so much real source flux (a real
    S/N=50 injected point source retained only ~22% of its flux with
    these defaults vs. ~56-100% with the looser NIRCam values, at every
    S/N tested).
    """
    from astropy import log
    log.setLevel('ERROR')

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

    masked_arr = np.where(mask, frame_data, np.nan)
    _, _, std = sigma_clipped_stats(masked_arr)
    error_arr = np.full(frame_data.shape, std)
    lacosmic_mask = ~mask # (True = masked/bad pixel)
    data_clean = np.nan_to_num(frame_data, nan=0.0, posinf=0.0, neginf=0.0) # replace nans which lacosmic doesn't like

    clean, crmask = lacosmic(data_clean,contrast=contrast,cr_threshold=cr_threshold,
                             neighbor_threshold=neighbor_threshold,mask=lacosmic_mask,error=error_arr)

    return clean, crmask

def _moment_elongation(img):
    """
    SEP-style elongation e = sqrt((Ixx-Iyy)^2 + 4*Ixy^2) / (Ixx+Iyy) of an
    image, from non-negative-weighted second moments about its own
    centroid. 0 = circularly symmetric, -> 1 = a line. Used to compare a
    candidate's actual shape against what the model PSF predicts at the
    same fit offset (see _psf_fit's elongation_excess).
    """
    ny, nx = img.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    w = np.clip(img, 0, None)
    wsum = w.sum()
    if wsum <= 0:
        return np.nan
    xc = (w * xx).sum() / wsum
    yc = (w * yy).sum() / wsum
    Ixx = (w * (xx - xc) ** 2).sum() / wsum
    Iyy = (w * (yy - yc) ** 2).sum() / wsum
    Ixy = (w * (xx - xc) * (yy - yc)).sum() / wsum
    denom = Ixx + Iyy
    if denom <= 0:
        return np.nan
    return float(np.sqrt((Ixx - Iyy) ** 2 + 4 * Ixy ** 2) / denom)


def _psf_fit(data_sub, x, y, kernel, bad_mask=None):
    """
    Fit PSF centroid by minimizing sum((norm_stamp - shifted_psf)**2) via Powell.
    Returns (psf_like, x_psf, y_psf, elongation_excess, symmetry180, psf_flux,
    psf_flux_err):
      psf_like           — Pearson r between stamp and best-fit shifted PSF.
                           A poor point-source discriminant on its own:
                           Pearson correlation is scale/shift invariant, so
                           an elongated cosmic-ray track can still correlate
                           well with a round PSF template as long as the
                           coarse "bright middle, faint edges" pattern lines
                           up — it doesn't penalize the actual shape mismatch.
      x_psf              — refined centroid x in image coordinates
      y_psf              — refined centroid y in image coordinates
      elongation_excess  — the core's own second-moment elongation (SEP-style
                           e, 0=round) minus the elongation the model PSF
                           shows at the same fit offset. ~0 for a real point
                           source, distinctly positive for an elongated
                           track — catches stretched-but-still-centered
                           features that fool both psf_like and symmetry180
                           (an elongated ellipse can still be inversion
                           symmetric).
      symmetry180        — Pearson r between the stamp core and its own
                           180-degree rotation about the fit center. Needs no
                           PSF model at all: a real (isotropic) PSF is
                           symmetric under 180-degree rotation by
                           construction, while a directional/comet-tail
                           feature generally isn't. Exact (no interpolation)
                           since the core is an odd-sized window centered on
                           an integer pixel. Complementary to
                           elongation_excess: catches lopsided/asymmetric
                           tracks that a symmetric-ellipse elongation measure
                           can miss.
      psf_flux           — PSF-fit flux: the optimal linear-least-squares
                           amplitude of the (already position-fit) PSF
                           template against the background-subtracted
                           stamp, A = sum(stamp*psf_unit)/sum(psf_unit**2)
                           with psf_unit peak-normalized. This is the
                           standard PSF-photometry flux estimator (weights
                           each pixel by how much the PSF itself puts there,
                           rather than a flat aperture sum), and doesn't
                           depend on an arbitrary aperture radius the way
                           the old sum-in-a-box light curve did — the same
                           box size choice was already shown (this session)
                           to change relative light-curve shape between
                           filters with different FWHM, not just its scale.
      psf_flux_err       — 1-sigma uncertainty on psf_flux, propagated from
                           the stamp's own local pixel noise (robust sigma-
                           clipped std, so the source's own core doesn't
                           bias it) through the same linear estimator:
                           sigma_A = sigma_pix / sqrt(sum(psf_unit**2)).
    bad_mask : ndarray(bool), same shape as `data_sub`, optional switch
        (default None, off) -- True = exclude this pixel (e.g. a
        JUMP_DET/cosmic-ray flag) from the fit entirely. See
        _amplitude_only_fit's own bad_mask for the motivating case.

    Returns (nan, x, y, nan, nan, nan, nan) if stamp is out of bounds or has
    no variance.
    """
    from scipy.optimize import minimize
    from scipy.ndimage import shift as ndshift

    h, w = data_sub.shape
    hs = kernel.shape[0] // 2
    xi, yi = int(round(x)), int(round(y))
    y0, y1 = yi - hs, yi + hs + 1
    x0, x1 = xi - hs, xi + hs + 1
    if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
        return np.nan, x, y, np.nan, np.nan, np.nan, np.nan

    stamp = data_sub[y0:y1, x0:x1].astype(np.float64)
    if bad_mask is not None:
        stamp = stamp.copy()
        stamp[bad_mask[y0:y1, x0:x1]] = np.nan
    stamp_sub = stamp - np.nanmedian(stamp)
    total = np.nansum(stamp_sub)
    if total == 0:
        return np.nan, x, y, np.nan, np.nan, np.nan, np.nan
    stamp_norm = stamp_sub / total

    psf_norm = kernel.astype(np.float64) / kernel.sum()

    # initial offset: fractional part of SEP centroid from rounded integer
    dx0 = x - xi
    dy0 = y - yi

    def residual(coeff):
        shifted = ndshift(psf_norm, (coeff[1], coeff[0]), order=3, mode='constant', cval=0)
        s = shifted.sum()
        if s > 0:
            shifted /= s
        return float(np.nansum((stamp_norm - shifted) ** 2)) * 1e6

    res = minimize(residual, [dx0, dy0], method='Powell',
                   bounds=[(-1.0, 1.0), (-1.0, 1.0)])
    dx_fit, dy_fit = res.x

    psf_shifted = ndshift(psf_norm, (dy_fit, dx_fit), order=3, mode='constant', cval=0)
    s = psf_shifted.sum()
    if s > 0:
        psf_shifted /= s

    # compute r on inner core only (~2×FWHM), not the full noisy stamp
    core_r = max(3, hs // 3)
    cy, cx = hs, hs
    core_sl = (slice(cy - core_r, cy + core_r + 1), slice(cx - core_r, cx + core_r + 1))
    stamp_f = stamp_norm[core_sl].flatten()
    psf_f = psf_shifted[core_sl].flatten()
    if stamp_f.std() == 0 or psf_f.std() == 0:
        return np.nan, x, y, np.nan, np.nan, np.nan, np.nan

    r = float(np.corrcoef(stamp_f, psf_f)[0, 1])

    # elongation excess: data's own shape vs. what the model PSF predicts
    # at this same fit offset, both measured identically (core window,
    # background-subtracted, non-negative-weighted moments)
    e_data = _moment_elongation(stamp_sub[core_sl])
    e_psf = _moment_elongation(total * psf_shifted[core_sl])
    elong_excess = e_data - e_psf if np.isfinite(e_data) and np.isfinite(e_psf) else np.nan

    # 180-degree self-symmetry: model-free, exact (odd-sized window, integer center)
    core = stamp_norm[core_sl]
    core_rot = core[::-1, ::-1]
    if core.std() == 0 or core_rot.std() == 0:
        sym180 = np.nan
    else:
        sym180 = float(np.corrcoef(core.flatten(), core_rot.flatten())[0, 1])

    # PSF-photometry flux: optimal linear-least-squares amplitude of the
    # (already position-fit) unit-sum PSF template, restricted to a narrow
    # aperture around the fitted centroid (FORCED_PHOTOMETRY_APERTURE_HW,
    # same as _amplitude_only_fit's forced-photometry method) rather than
    # the full kernel footprint used for the shape metrics above. This is
    # the same fix that _stacked_centroid_fit/_amplitude_only_fit already
    # apply downstream in the light-curve stage (found via objid57: a
    # nearby unrelated point source within the full footprint dragged
    # both the centroid and a full-footprint amplitude fit toward itself,
    # contaminating the flux). Folded into _psf_fit directly so every
    # caller -- including the initial detection-stage psf_flux/
    # psf_flux_err that source_extracting's S/N filters threshold on --
    # gets the same contamination resistance, not just the light-curve
    # stage.
    ap = min(FORCED_PHOTOMETRY_APERTURE_HW, hs)
    ap_sl = (slice(hs - ap, hs + ap + 1), slice(hs - ap, hs + ap + 1))
    ap_data = stamp_sub[ap_sl]
    ap_template = psf_shifted[ap_sl]
    denom = np.nansum(ap_template ** 2)
    psf_flux = float(np.nansum(ap_data * ap_template) / denom) if denom > 0 else np.nan

    # 1-sigma flux uncertainty, propagated through the same linear
    # estimator: sigma_A = sigma_pix / sqrt(sum(T**2)). sigma_pix comes
    # from a sigma-clipped std of the WHOLE stamp (not stamp_sub) --
    # clipping is what keeps the bright source core (a small fraction of
    # the stamp's pixels) from biasing the noise estimate high.
    if denom > 0:
        _, _, sigma_pix = sigma_clipped_stats(stamp, sigma=3.0)
        psf_flux_err = float(sigma_pix / np.sqrt(denom)) if np.isfinite(sigma_pix) else np.nan
    else:
        psf_flux_err = np.nan

    return r, float(xi + dx_fit), float(yi + dy_fit), elong_excess, sym180, psf_flux, psf_flux_err


def _stacked_centroid_fit(cube, x, y, kernel, bad_mask=None):
    """
    TESSELLATE-style position determination (see
    TESSELLATE/tessellate/detector.py's _Fit_psf), adopted here as the
    default centroid source for forced-photometry light curves (see
    _build_channel_lightcurve_at_xy): compares each individual frame's
    own _psf_fit against a fit on the straight (unweighted) stack of
    every valid frame, and keeps whichever gives the higher S/N
    (psf_flux/psf_flux_err) as the FIXED sub-pixel centroid used for
    every frame's flux measurement.

    Motivation: a single independently-re-optimized-per-frame centroid
    (the previous default) can still partially drift toward a nearby
    contaminating source even bounded to +/-1px, inflating that frame's
    flux without it being obvious from the shift alone (see the
    objid57 investigation — the drift was only ~0.3-0.5px, but the
    amplitude fit was still badly contaminated). A single, well-
    determined centroid — preferring the stack when no one frame has
    enough S/N to localize well on its own, but preferring a genuinely
    good single-frame detection when one exists — removes position as
    a free parameter from every subsequent per-frame flux fit
    entirely (see _amplitude_only_fit).

    Returns (x_psf, y_psf) — falls back to (x, y) unchanged if no frame
    or the stack produces a usable fit.

    Robust to a single-frame cosmic ray sitting in the fit window:
      - The stack itself is now a per-pixel sigma-clipped mean (rescaled
        by n_valid) rather than a plain sum — a real, persistent source
        contributes to every frame so survives clipping; a single-frame
        CR pixel is a 1-of-N outlier at that pixel and gets clipped out
        of the combined image before the stack's own _psf_fit ever sees
        it. Degrades gracefully to an unclipped mean for the N<3 case
        where sigma-clipping can't reject anything.
      - Any candidate (single-frame OR stack) whose Powell fit is PINNED
        at its own +/-1px search bound is dropped rather than trusted:
        a converged fit lands strictly inside its bounds, so landing
        exactly on the edge means the true optimum lies further out —
        i.e. something outside the fit's own trusted window (e.g. a
        nearby CR/hot pixel within the kernel-sized stamp but beyond the
        +/-1px centroid search) pulled it there. Caught directly on LW
        objid84 (snake arc region, dither 00003): the stacked fit landed
        at x_psf == xi-1.0 to machine precision while both real
        per-frame fits agreed with each other and the raw SEP position
        to ~0.1px.

    bad_mask : ndarray(bool), same shape as one cube frame, optional
        switch (default None, off). True = exclude this pixel (e.g. a
        JUMP_DET/cosmic-ray flag) from every per-frame and stacked fit.
        See _amplitude_only_fit's own bad_mask for the motivating case
        (a DQ-confirmed cosmic ray one pixel outside objid84's 3x3
        aperture, close enough to bias its centroid/flux).
    """
    from astropy.stats import sigma_clip

    n_frame = cube.shape[0]
    valid = [f for f in range(n_frame) if np.isfinite(cube[f]).any()]
    if not valid:
        return float(x), float(y)

    xi, yi = int(round(x)), int(round(y))

    def _pinned(x_psf, y_psf):
        # matches _psf_fit's own bounds=[(-1.0, 1.0), (-1.0, 1.0)] around
        # (xi, yi) -- landing within 1e-6 of either edge means the Powell
        # search terminated AT the bound, not at an interior optimum.
        return (abs(abs(x_psf - xi) - 1.0) < 1e-6) or (abs(abs(y_psf - yi) - 1.0) < 1e-6)

    candidates = []  # (snr, x_psf, y_psf)
    per_frame = []  # (snr, x_psf, y_psf), single-frame fits only -- used as a robust consensus check below
    for f in valid:
        frame_data = np.nan_to_num(cube[f])
        _r, x_psf, y_psf, *_junk, psf_flux, psf_flux_err = _psf_fit(
            frame_data, x, y, kernel, bad_mask=bad_mask)
        if (np.isfinite(psf_flux) and np.isfinite(psf_flux_err) and psf_flux_err > 0
                and not _pinned(x_psf, y_psf)):
            snr = psf_flux / psf_flux_err
            candidates.append((snr, x_psf, y_psf))
            per_frame.append((snr, x_psf, y_psf))

    frames = np.stack([np.nan_to_num(cube[f]) for f in valid], axis=0)
    if len(valid) >= 3:
        clipped = sigma_clip(frames, sigma=3.0, axis=0, masked=True)
        stacked = np.ma.mean(clipped, axis=0).filled(np.nan) * len(valid)
    else:
        stacked = np.nanmean(frames, axis=0) * len(valid)
    _r_s, x_psf_s, y_psf_s, *_junk, psf_flux_s, psf_flux_err_s = _psf_fit(
        np.nan_to_num(stacked), x, y, kernel, bad_mask=bad_mask)
    if (np.isfinite(psf_flux_s) and np.isfinite(psf_flux_err_s) and psf_flux_err_s > 0
            and not _pinned(x_psf_s, y_psf_s)):
        candidates.append((psf_flux_s / psf_flux_err_s, x_psf_s, y_psf_s))

    if not candidates:
        return float(x), float(y)
    best = max(candidates, key=lambda c: c[0])

    # Consensus guard: sigma-clipping the stack only rejects an outlier
    # AT A GIVEN PIXEL across frames, which doesn't catch every case --
    # e.g. a strong single-frame CR hit that clips out fine at ITS OWN
    # pixel can still be bright enough in the stamp to drag the stack
    # fit's *centroid* off by less than the +/-1px bound (so it isn't
    # caught by _pinned either), while >=2 independent per-frame fits
    # still agree tightly with each other. When that happens, trust the
    # per-frame consensus over the (possibly still-biased) winning
    # candidate rather than its raw S/N ranking.
    #
    # Only per-frame fits with S/N above MIN_CONSENSUS_SNR count toward
    # that consensus -- _pinned alone doesn't catch every noise fit (a
    # fit landing 1e-6 short of its own +/-1px bound still passes
    # _pinned but is pure noise). Caught directly on LW objid190 (snake
    # arc region, dither 00003): a S/N~0.3 empty-frame fit sat right at
    # the pinned-check's numerical edge, and being blindly averaged with
    # the one real S/N~50 per-frame detection dragged the reported
    # centroid ~0.8px off the true (visually obvious) source position.
    MIN_CONSENSUS_SNR = 3.0
    trusted = [(px, py) for snr, px, py in per_frame if snr >= MIN_CONSENSUS_SNR]
    if len(trusted) >= 2:
        med_x = float(np.median([p[0] for p in trusted]))
        med_y = float(np.median([p[1] for p in trusted]))
        if np.hypot(best[1] - med_x, best[2] - med_y) > 0.5:
            return med_x, med_y

    return float(best[1]), float(best[2])


def _amplitude_only_fit(data, x, y, dx_fixed, dy_fixed, kernel, aperture_hw=FORCED_PHOTOMETRY_APERTURE_HW,
                        bad_mask=None):
    """
    Forced photometry at a FIXED sub-pixel centroid (see
    _stacked_centroid_fit) — the only free parameter is the linear
    amplitude, restricted to a narrow (2*aperture_hw+1)-pixel aperture
    around the centroid rather than the full kernel-sized stamp.

    The narrow aperture (default 3x3, i.e. aperture_hw=1) is what
    actually matters for contamination — a fixed centroid alone does
    NOT exclude a nearby unrelated source from the fit, since the
    amplitude estimator sums stamp*template over the whole template
    footprint regardless of exact sub-pixel position, and a
    contaminant just 1-2px away still overlaps a full kernel-sized
    (e.g. 15x15) template's wings substantially (verified directly on
    objid57: fixing the centroid alone left the contaminated flux
    essentially unchanged, 7.0->6.5; narrowing to a 3x3 aperture fixed
    it, 7.0->-0.05). A 3x3 aperture at typical NIRCam pixel scales is
    still wide enough to contain the PSF core's main flux for a
    genuine point source.

    bad_mask : ndarray(bool), same shape as `data`, optional switch
        (default None, off). True = exclude this pixel (e.g. a
        JUMP_DET/cosmic-ray flag) from BOTH the local background
        estimate and the aperture sum -- a real, demonstrated fix for a
        cosmic ray sitting close enough to partially overlap the narrow
        aperture itself (snake arc region LW objid84, dither 00003: a
        DQ-confirmed JUMP_DET pixel one pixel outside the 3x3 aperture
        inflated frame 1's flux ~40% above the trend the other 3 frames
        agreed on). Off by default -- masked-out pixels become NaN
        within the stamp (nanmedian/nansum already used throughout
        already ignore them), so an aperture that's mostly bad pixels
        can still return NaN/degenerate results same as any other
        insufficient-data case.

    Returns (flux, flux_err) — matches _psf_fit's own linear-estimator
    convention (unit-sum shifted template, sigma-clipped stamp noise).
    """
    from scipy.ndimage import shift as ndshift

    hs = kernel.shape[0] // 2
    xi, yi = int(round(x)), int(round(y))
    y0, y1 = yi - hs, yi + hs + 1
    x0, x1 = xi - hs, xi + hs + 1
    h, w = data.shape
    if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
        return np.nan, np.nan

    stamp = data[y0:y1, x0:x1].astype(np.float64)
    if bad_mask is not None:
        stamp = stamp.copy()
        stamp[bad_mask[y0:y1, x0:x1]] = np.nan
    # Sigma-clipped median, not a plain np.nanmedian, over the FULL
    # kernel-sized stamp -- a bright cosmic ray/hot pixel landing
    # anywhere in this stamp (not just inside the narrow aperture below)
    # skews a plain median enough to shift the whole stamp's assumed
    # zero point, biasing the amplitude fit even though the CR's own
    # flux never enters the aperture sum. Narrowing the aperture (see
    # docstring) only protects against the CR's flux leaking into the
    # sum directly; this protects the background estimate it's measured
    # against. Same sigma_clipped_stats call also gives sigma_pix below,
    # so this isn't a second separate pass over the stamp.
    _, clipped_median, sigma_pix = sigma_clipped_stats(stamp, sigma=3.0)
    stamp_sub = stamp - clipped_median

    psf_norm = kernel.astype(np.float64) / kernel.sum()
    shifted = ndshift(psf_norm, (dy_fixed, dx_fixed), order=3, mode='constant', cval=0)
    s = shifted.sum()
    if s > 0:
        shifted /= s

    c = hs
    a = min(aperture_hw, hs)
    ap_data = stamp_sub[c - a:c + a + 1, c - a:c + a + 1]
    ap_template = shifted[c - a:c + a + 1, c - a:c + a + 1]

    denom = np.nansum(ap_template ** 2)
    if denom <= 0:
        return np.nan, np.nan
    amp = np.nansum(ap_data * ap_template) / denom
    flux = amp * np.nansum(ap_template)

    flux_err = float(np.nansum(ap_template) * sigma_pix / np.sqrt(denom)) if np.isfinite(sigma_pix) else np.nan

    return float(flux), flux_err


def _psf_fit_batch(owner, data_sub, indices, coords, kernel):
    """
    Runs _psf_fit for a batch of objects that all share one frame's
    data_sub — one task per batch (not per object) so data_sub (a full
    frame, several MB) is only pickled to the worker once per batch
    rather than once per individual object. owner identifies which
    frame's obj_df these results belong to, threaded through so results
    from many frames' batches can be flattened into one Parallel call
    (see source_extracting) and still be routed back correctly.
    """
    return [(owner, idx) + _psf_fit(data_sub, x, y, kernel) for idx, (x, y) in zip(indices, coords)]


def _sep_extract(frame, data, kernel, mask, n_group):
    """
    Cheap per-frame half of the old _run_sep: background subtraction +
    sep.extract only, no PSF-fitting. Split out so the expensive part
    (_psf_fit, ~1ms/object but called once per raw SEP detection — up to
    several thousand per frame) can be flattened across every frame's
    objects and parallelized with the full core count in one Parallel
    call, instead of being capped at frame-count parallelism: a short
    NIRCam exposure only has 3-7 usable frames, so the old per-frame-only
    parallelism left most cores idle while the frame with the most
    objects became the bottleneck (see source_extracting).
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

    data = np.array(data, dtype=np.float32, copy=True)
    mask = np.array(mask, dtype=bool, copy=True)
    bkg = sep.Background(data, mask=~mask)
    data_sub = data - bkg

    objects = sep.extract(data_sub, 1.5, filter_kernel=kernel, err=bkg.globalrms, mask=~mask)
    obj_df = pd.DataFrame(objects)

    if len(obj_df) > 0:
        obj_df['symmetry'] = (obj_df['a']/obj_df['b']).abs() - 1
        obj_df['frame'] = frame
        obj_df['group'] = (obj_df['frame'] % n_group) + 1  # adding the group number

    return obj_df, data_sub, float(bkg.globalrms)


def _finish_sep(frame, obj_df, data_sub, bkg_rms, mask, save, obs_dir, psf_fwhm=None, psf_corr_thresh=0.8,
                edge_margin_x=16, edge_margin_y=12, cr_mask=None, cr_frame_window=1, cr_px_window=1):
    """
    Second half of the old _run_sep: aperture photometry, npix_ratio,
    filtering, and plotting. Takes obj_df already carrying psf_like/
    x_psf/y_psf/elongation_excess/symmetry180 (filled in by _psf_fit,
    run separately — see source_extracting) plus the data_sub/bkg_rms
    _sep_extract produced for this same frame.

    cr_mask : ndarray (n_frame, ny, nx) bool, optional
        lacosmic's cosmic-ray mask (Jurassic.cr_mask_cube — NOT the
        pipeline's JUMP_DET flag). Shape-based cuts (psf_like,
        elongation_excess, symmetry180) cannot reliably separate cosmic
        rays from real point sources on their own — a single-particle hit
        near-normal-incidence produces a compact, genuinely PSF-like charge
        cloud indistinguishable by shape from a real source at this
        detector's pixel scale. lacosmic's Laplacian test is a better
        per-detection reject signal because it asks a physically different
        question than shape correlation alone: is this pixel-scale feature
        consistent with real, telescope-optics-broadened light, or is it an
        unresolved detector-level spike? A real transient must satisfy the
        former regardless of its time behavior, so — unlike the pipeline's
        JUMP_DET, which flags *any* sudden ramp discontinuity and can't
        distinguish a cosmic ray from a genuine fast brightening — this
        doesn't risk rejecting real discoveries (verified by injection
        testing on MIRI — see cosmic_ray_shapes/. NIRCam is different: its
        compact PSF fools lacosmic's sharpness test entirely, so this is
        always None there — see source_extracting). None disables the
        check (e.g. for the first-pass call, before the mask exists yet).
    cr_frame_window, cr_px_window : int
        Half-widths of the frame/pixel window checked around each
        candidate for a cr_mask hit.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

    # data_sub crossed a Parallel process boundary once already (from
    # _sep_extract, in source_extracting's stage 1) and is about to cross
    # another one here — joblib's loky backend automatically memory-maps
    # large arrays passed to worker processes, which come back read-only.
    # sep.sum_circle needs a writable buffer, so make a fresh copy.
    data_sub = np.array(data_sub, copy=True)

    if len(obj_df) == 0:
        filtered_df = pd.DataFrame(columns=obj_df.columns.tolist() + ['symmetry', 'psf_like', 'frame', 'sep_flux', 'sep_fluxerr', 'sep_s/n', 'sep_flag'])
        return obj_df, filtered_df

    # aperture photometry
    flux, fluxerr, flag = sep.sum_circle(data_sub, obj_df['x'], obj_df['y'], 3.0, err=bkg_rms, gain=1.0) # ap radius = 3.0
    obj_df['sep_flux'] = flux
    obj_df['sep_fluxerr'] = fluxerr
    obj_df['sep_s/n'] = flux/fluxerr
    obj_df['sep_flag'] = flag

    # npix/tnpix: ratio of the full detected footprint (npix, at the SEP
    # extraction threshold) to the deblended "core" area (tnpix). A real
    # point source's footprint and core area are nearly the same thing;
    # cosmic rays tend to run higher: even ones that pass the psf_like/
    # elongation shape checks tend to have irregular low-level structure
    # (secondary tracks, nearby associated hits from the same event)
    # attached to an otherwise compact core, inflating the total footprint
    # without inflating the core.
    #
    # The 2.0 cutoff and its ~97%-catch/~2.5%-false-reject numbers were
    # calibrated entirely on MIRI (miri/spt0346-52/processing/
    # cosmic_ray_shapes/) and never validated for NIRCam. A real
    # injection-recovery test on this NIRCam dataset (Level3-mosaic-as-
    # reference differencing, snake/processing/npix_ratio_injection_test.py)
    # found a 25% false-reject rate at 2.0 — 10x worse than assumed — with
    # the 97.5th percentile of npix_ratio among recovered real injected
    # point sources at 3.2. Loosened to 3.2 here to match that. For NIRCam,
    # cosmic-ray rejection leans on persistence (_check_persistence,
    # adjacent-frame confirmation) and LW/SW cross-detection
    # (crossmatch_sw_lw) as the primary discriminants, not this cut — it's
    # now a much looser backstop rather than doing the bulk of the work.
    NPIX_RATIO_MAX = 3.2
    npix_ratio = obj_df['npix'] / obj_df['tnpix'].replace(0, np.nan)
    obj_df['npix_ratio'] = npix_ratio

    # apply filtering: edge exclusion + PSF shape (correlation + fit didn't hit bound) + size
    ny, nx = data_sub.shape
    fit_dx = (obj_df['x_psf'] - obj_df['x']).abs()
    fit_dy = (obj_df['y_psf'] - obj_df['y']).abs()
    # npix_ratio cut disabled for now -- observed rejecting a clearly real,
    # high-S/N source (S/N=260.8, npix_ratio=3.32) by a razor-thin margin
    # over its near-identical, accepted neighbor (npix_ratio=3.08), and the
    # metric's own validation history already flags it as unreliable for
    # NIRCam (25% false-reject rate at the originally-assumed 2.0
    # threshold). npix_ratio is still computed/stored above for inspection.
    filter_mask = (obj_df['x'].between(edge_margin_x, nx - edge_margin_x) &
                   obj_df['y'].between(edge_margin_y, ny - edge_margin_y) &
                   (obj_df['psf_like'] >= psf_corr_thresh) &
                   (fit_dx < 0.9) & (fit_dy < 0.9))
    if psf_fwhm is not None:
        sigma_exp = psf_fwhm / (2 * np.sqrt(2 * np.log(2)))
        # SEP's isophotal 'a' for a genuine (fixed-shape) PSF isn't brightness
        # independent: extraction uses a fixed absolute threshold, so a
        # brighter source's isophote reaches further into the same Gaussian
        # wings before dropping below it. For a 2D Gaussian with std
        # sigma_exp, the radius at which it crosses a given threshold is
        # sigma_exp*sqrt(2*ln(peak/thresh)) -- growing with brightness. A
        # brightness-independent cutoff (the old `2.0*sigma_exp`) therefore
        # increasingly rejects real, genuinely round bright point sources
        # (confirmed via injection-recovery testing: real S/N~200 sources
        # were rejected ~70% of the time purely from this effect). Compare
        # against this brightness-aware expectation instead, with a 1.5x
        # margin for real PSF wings being a bit fatter than an ideal Gaussian
        # and for ordinary SEP measurement noise.
        ratio = np.maximum(obj_df['peak'] / obj_df['thresh'], 1.0001)
        a_expected = sigma_exp * np.sqrt(2 * np.log(ratio))
        a_expected = np.maximum(a_expected, sigma_exp)  # floor: never tighter than the old low-S/N limit
        filter_mask = filter_mask & (obj_df['a'] < 1.5 * a_expected)

    if cr_mask is not None:
        n_frame_total = cr_mask.shape[0]
        f0, f1 = max(0, frame - cr_frame_window), min(n_frame_total, frame + cr_frame_window + 1)
        cr_window = cr_mask[f0:f1]
        is_cr = np.zeros(len(obj_df), dtype=bool)
        for i, (xi, yi) in enumerate(zip(obj_df['x'].values, obj_df['y'].values)):
            xi, yi = int(round(xi)), int(round(yi))
            y0, y1 = max(0, yi - cr_px_window), min(ny, yi + cr_px_window + 1)
            x0, x1 = max(0, xi - cr_px_window), min(nx, xi + cr_px_window + 1)
            is_cr[i] = cr_window[:, y0:y1, x0:x1].any()
        obj_df['cr_flagged'] = is_cr
        filter_mask = filter_mask & ~is_cr
    else:
        obj_df['cr_flagged'] = False

    filtered_df = obj_df[filter_mask]

    # plotting -- restricted to frames with at least one high-confidence
    # (psf S/N >= 20) candidate. One of these is saved per frame with
    # any filtered detection at all, so on a typical exposure this was
    # producing far more per-frame diagnostic figures than were useful
    # to actually look at.
    snr = filtered_df['psf_flux'] / filtered_df['psf_flux_err']
    has_high_snr = (snr >= 20).any() if len(filtered_df) > 0 else False
    if save and has_high_snr:
        from matplotlib.patches import Ellipse
        matplotlib.use("Agg") # don't show em
        fig, ax = plt.subplots()
        m, s = np.mean(data_sub*mask), np.std(data_sub*mask)

        im = ax.imshow(data_sub*mask,interpolation='nearest',vmin=m-s,vmax=m+s,origin='lower')
        ax.set_title(f"Frame {frame}")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Pixel value")

        for i in filtered_df.index:
            e = Ellipse(
                xy=(filtered_df.at[i, 'x'], filtered_df.at[i, 'y']),
                width=6*filtered_df.at[i, 'a'],
                height=6*filtered_df.at[i, 'b'],
                angle=filtered_df.at[i, 'theta']*180./np.pi
            )
            e.set_facecolor('none')
            e.set_edgecolor('red')
            ax.add_artist(e)

        sep_dir = os.path.join(obs_dir, "sep_frames")
        os.makedirs(sep_dir, exist_ok=True)
        plt.savefig(os.path.join(sep_dir, f"frame_{frame:03d}.png"), bbox_inches="tight")
        plt.close(fig)

    return obj_df, filtered_df


def _measure_fwhm_px(psf_array):
    """
    Measures FWHM (in pixels) of a detector-sampled PSF array by linearly
    interpolating the half-max crossings on either side of the peak, along
    the central row. Used for instruments without a pre-tabulated FWHM
    lookup (e.g. NIRCam, whose FWHM varies strongly across its ~30 filters
    and two pixel scales). Sub-pixel interpolation matters here: NIRCam's
    SW channel is undersampled enough that sometimes only a single detector
    pixel sits above half-max, which a whole-pixel-counting measurement
    can't resolve.
    """
    ny, nx = psf_array.shape
    row = psf_array[ny // 2]
    peak = int(np.argmax(row))
    half = row[peak] / 2.0

    left = peak
    while left > 0 and row[left] > half:
        left -= 1
    right = peak
    while right < len(row) - 1 and row[right] > half:
        right += 1
    if left == peak or right == peak:
        return np.nan

    x_left = left + (half - row[left]) / (row[left + 1] - row[left])
    x_right = (right - 1) + (row[right - 1] - half) / (row[right - 1] - row[right])
    return float(x_right - x_left)


def _forward_diff(cube):
    """
    Forward (not centered) difference along axis 0: out[i] = cube[i+1] - cube[i].

    np.gradient's default centered difference, (cube[i+1]-cube[i-1])/2, is
    the wrong tool for discrete up-the-ramp reads: a single-group event
    (e.g. a cosmic ray) shows up smeared across two adjacent output frames,
    since the centered stencil at index i straddles it from both sides
    whichever group it lands in. A forward difference keeps a single-group
    event in exactly one output frame. Keeps the same shape as the input by
    setting the last frame to NaN (there's no cube[i+1] for it) -- that
    index is always in Jurassic.bad_frames anyway (last frame of the last
    integration), so nothing downstream relies on it having a value.
    """
    out = np.full_like(cube, np.nan)
    out[:-1] = np.diff(cube, axis=0)
    return out


def _integer_shift(arr, dy, dx, fill=0.0):
    """
    Shift a 2D array by an INTEGER (dy, dx) with no wraparound (unlike
    np.roll) -- pixels shifted in from outside the original footprint are
    set to `fill`. Same (dy, dx) sign convention as scipy.ndimage.shift:
    positive dy/dx moves content toward higher row/column indices.
    """
    out = np.full_like(arr, fill)
    ny, nx = arr.shape
    src_y0, src_y1 = max(0, -dy), ny - max(0, dy)
    dst_y0, dst_y1 = max(0, dy), ny - max(0, -dy)
    src_x0, src_x1 = max(0, -dx), nx - max(0, dx)
    dst_x0, dst_x1 = max(0, dx), nx - max(0, -dx)
    if src_y1 > src_y0 and src_x1 > src_x0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = arr[src_y0:src_y1, src_x0:src_x1]
    return out


def _source_matched_shift(moving, reference, thresh_sigma=20.0, match_radius_px=15.0,
                           min_matches=5):
    """
    Fallback registration for when phase_cross_correlation's global FFT
    approach fails (see _subpixel_align) -- e.g. aligning a master
    reference built from a differently-resampled mosaic (master_ref_path,
    Level 3 pipeline products) onto this exposure's own grad_cube median,
    where large-scale structure/noise differences between the two can
    confuse a global correlation even though real, matchable point
    sources are present in both.

    Detects bright, round, isolated sources independently in `moving` and
    `reference` via SEP, nearest-neighbor matches them (two rounds: a
    coarse pass with a generous radius, then a tighter pass re-centered
    on the coarse pass's median offset, discarding outlier matches via
    sigma-clipping), and returns the (dy, dx) shift that maps `moving`
    onto `reference`. Returns None if too few reliable matches are found
    to trust the result (safer to skip correction than apply a
    shift derived from too few/noisy matches).
    """
    def detect(img):
        data = np.nan_to_num(img.astype(np.float32))
        mask = ~np.isfinite(img)
        bkg = sep.Background(data, mask=mask)
        data_sub = data - bkg.back()
        objects = sep.extract(data_sub, thresh_sigma, err=bkg.globalrms, mask=mask)
        keep = []
        for obj in objects:
            a, b = float(obj['a']), float(obj['b'])
            if a <= 0 or b / a < 0.7:
                continue
            keep.append((float(obj['x']), float(obj['y'])))
        return np.array(keep)

    src_mov = detect(moving)
    src_ref = detect(reference)
    if len(src_mov) < min_matches or len(src_ref) < min_matches:
        return None

    def match(offset_dx, offset_dy, radius):
        shifted = src_mov + np.array([offset_dx, offset_dy])
        pairs = []
        for mx, my in shifted:
            d = np.hypot(src_ref[:, 0] - mx, src_ref[:, 1] - my)
            i = np.argmin(d)
            if d[i] < radius:
                pairs.append((mx - offset_dx, my - offset_dy, src_ref[i, 0], src_ref[i, 1]))
        return np.array(pairs)

    coarse = match(0.0, 0.0, match_radius_px)
    if len(coarse) < min_matches:
        return None
    dx0 = np.median(coarse[:, 2] - coarse[:, 0])
    dy0 = np.median(coarse[:, 3] - coarse[:, 1])

    fine = match(dx0, dy0, 3.0)
    if len(fine) < min_matches:
        return None
    dx_res = fine[:, 2] - fine[:, 0] - dx0
    dy_res = fine[:, 3] - fine[:, 1] - dy0
    keep = (np.abs(dx_res - np.median(dx_res)) < 3 * (np.std(dx_res) + 1e-3)) & \
           (np.abs(dy_res - np.median(dy_res)) < 3 * (np.std(dy_res) + 1e-3))
    if keep.sum() < min_matches:
        return None

    dx = dx0 + np.median(dx_res[keep])
    dy = dy0 + np.median(dy_res[keep])
    return np.array([dy, dx])


def _subpixel_align(moving, reference, upsample_factor=20):
    """
    Cross-correlation subpixel registration of `moving` onto `reference`.
    Used after WCS reprojection (see Jurassic._align_master_reference and
    build_master_refs.py) to correct residual offsets that a distortion
    WCS alone still leaves — typically a fraction of a pixel up to ~1 px,
    from guide-star pointing uncertainty and imperfect distortion models.
    Left uncorrected, that residual shows up as a spurious PSF-shaped
    dipole (bright/dark pair) in every differenced frame at every real
    source position. Falls back to returning `moving` unchanged if there
    isn't enough finite overlap to register reliably.

    Shifts in two steps rather than one FFT-based shift of the full
    (integer + fractional) offset: the integer part is applied first via
    plain array slicing (`_integer_shift`, no interpolation, no
    wraparound), and only the leftover sub-pixel remainder (always in
    [-0.5, 0.5]) goes through `scipy.ndimage.shift`'s spline
    interpolation. A single `fourier_shift` (global FFT) over the full
    offset introduced severe ringing artifacts here -- it treats the
    image as periodic, so real, sharp features (bright sources, the hard
    edge of the reprojected footprint itself) wrap and ring across the
    whole frame. `scipy.ndimage.shift`, applied only to a <=0.5px
    remainder on data already coarsely aligned, is a local, non-periodic
    interpolation with none of that failure mode.

    `moving` commonly has real NaN regions (e.g. reproject_interp fills
    pixels outside the source footprint with NaN — the ~200 px dither
    offsets here mean that's a wide strip, not just a few edge pixels).
    Neither shift method can take NaN input directly, so the validity
    mask is carried through both the integer and sub-pixel steps
    alongside the data and reapplied as NaN afterward, keeping the
    "no real data here" region honest post-shift.
    """
    from skimage.registration import phase_cross_correlation
    from scipy.ndimage import shift as ndshift

    valid = np.isfinite(moving) & np.isfinite(reference)
    if valid.sum() < 100:
        return moving

    ref_filled = np.nan_to_num(reference)
    mov_filled = np.nan_to_num(moving)
    mov_valid = np.isfinite(moving).astype(float)

    shift_yx, error, _ = phase_cross_correlation(ref_filled, mov_filled, upsample_factor=upsample_factor)
    # phase_cross_correlation is only meant to find the <=~1px residual
    # left after WCS reprojection (see docstring) -- if the two images
    # don't actually correlate (error near its max of 1.0) or it locks
    # onto a shift far larger than that residual could plausibly be, it's
    # found a spurious global offset, not the intended nudge. Applying
    # that anyway silently throws away most of the frame (as NaN, once
    # shifted_valid<0.5 below is applied) rather than correcting it.
    if error > 0.9 or np.any(np.abs(shift_yx) > 10):
        print(f'_subpixel_align: rejecting spurious phase-correlation shift {shift_yx} '
              f'(error={error:.3f}), trying source-matched fallback', flush=True)
        fallback_shift = _source_matched_shift(moving, reference)
        if fallback_shift is None:
            print('_subpixel_align: source-matched fallback also failed, returning unshifted',
                  flush=True)
            return moving
        print(f'_subpixel_align: source-matched fallback shift = {fallback_shift}', flush=True)
        shift_yx = fallback_shift
    int_shift = np.round(shift_yx).astype(int)
    frac_shift = shift_yx - int_shift

    data_int = _integer_shift(mov_filled, int_shift[0], int_shift[1], fill=0.0)
    valid_int = _integer_shift(mov_valid, int_shift[0], int_shift[1], fill=0.0)

    shifted = ndshift(data_int, frac_shift, order=3, mode='constant', cval=0.0)
    shifted_valid = ndshift(valid_int, frac_shift, order=3, mode='constant', cval=0.0)
    shifted[shifted_valid < 0.5] = np.nan
    return shifted


def make_reference_cube(pixel,grad_cube):
    """
    makes a reference cube and gets a median from it
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

    row,col = pixel
    pix = grad_cube[:,row,col]

    clipped_pix = sigma_clip(pix,sigma_upper=1,masked=False,axis=0).data

    return np.asarray(clipped_pix, dtype=float)


def crossmatch_sw_lw(base_dir, match_radius_arcsec=0.75, time_tol_s=30.0):
    """
    LW/SW cross-detection discriminator, run once across a full set of
    per-detector jurassic outputs (all 10 NIRCam detectors x N exposures
    under `base_dir`, dirs named dr_<exposure>_nrc{a,b}{1,2,3,4,long}).
    NIRCam images its SW (4 detectors/module) and LW (1 detector/module)
    channels simultaneously through the same dichroic, so a real
    astrophysical source detected on the LW detector should show a
    coincident detection on at least one of the four SW detectors
    covering the same module's field of view, at matching sky position
    and time -- unlike a single-detector cosmic ray or persistence
    residual, confined to one channel. This is a second, independent
    discriminator alongside the consecutive-frame persistence filter
    (`_check_persistence`, applied within a single detector's own frame
    sequence) -- together these are the two pathways for a candidate to
    be considered real rather than an artifact.

    Adds boolean columns in place to each exposure/module's
    grouped_filtered_sep.csv: `sw_match` on the LW catalog (True if a
    same-module SW detection falls within `match_radius_arcsec` and
    `time_tol_s`), `lw_match` on each SW catalog (the symmetric check).
    Requires `ra`/`dec`/`mjd` columns (i.e. these runs used
    compute_radec=True) -- catalogs missing them are left untouched
    apart from an `sw_match`/`lw_match` column of all False.
    """
    import re
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    dirs = sorted(glob.glob(os.path.join(base_dir, 'dr_*')))
    pat = re.compile(r'^dr_(.+)_nrc(a|b)(long|[1-4])$')
    exposures = {}
    for d in dirs:
        m = pat.match(os.path.basename(d))
        if not m:
            continue
        exp_id, module, det = m.groups()
        exposures.setdefault(exp_id, {}).setdefault(module, {})[det] = d

    def _load(dr_dir):
        csv_path = os.path.join(dr_dir, 'grouped_output', 'grouped_filtered_sep.csv')
        if not os.path.exists(csv_path):
            return None, csv_path
        df = pd.read_csv(csv_path)
        if 'ra' not in df.columns or 'dec' not in df.columns or 'mjd' not in df.columns or len(df) == 0:
            return None, csv_path
        return df, csv_path

    def _match_mask(query_df, ref_coord, ref_mjd):
        query_coord = SkyCoord(query_df['ra'].values * u.deg, query_df['dec'].values * u.deg)
        mask = np.zeros(len(query_df), dtype=bool)
        for i, c in enumerate(query_coord):
            sep_arcsec = c.separation(ref_coord).arcsec
            dt_s = np.abs(ref_mjd - query_df['mjd'].values[i]) * 86400.0
            mask[i] = bool(np.any((sep_arcsec < match_radius_arcsec) & (dt_s < time_tol_s)))
        return mask

    for exp_id, modules in exposures.items():
        for module, dets in modules.items():
            if 'long' not in dets:
                continue
            lw_df, lw_csv = _load(dets['long'])
            if lw_df is None:
                continue

            sw_dfs = {}
            for det_num in ('1', '2', '3', '4'):
                if det_num not in dets:
                    continue
                sw_df, sw_csv = _load(dets[det_num])
                if sw_df is not None:
                    sw_dfs[det_num] = (sw_df, sw_csv)

            if not sw_dfs:
                lw_df['sw_match'] = False
                lw_df.to_csv(lw_csv, index=False)
                continue

            sw_all = pd.concat([d for d, _ in sw_dfs.values()], ignore_index=True)
            sw_coord = SkyCoord(sw_all['ra'].values * u.deg, sw_all['dec'].values * u.deg)
            lw_df['sw_match'] = _match_mask(lw_df, sw_coord, sw_all['mjd'].values)
            lw_df.to_csv(lw_csv, index=False)
            print(f'{exp_id} module {module}: {lw_df["sw_match"].sum()}/{len(lw_df)} '
                  f'LW candidates have an SW match')

            lw_coord = SkyCoord(lw_df['ra'].values * u.deg, lw_df['dec'].values * u.deg)
            for det_num, (sw_df, sw_csv) in sw_dfs.items():
                sw_df['lw_match'] = _match_mask(sw_df, lw_coord, lw_df['mjd'].values)
                sw_df.to_csv(sw_csv, index=False)


def _build_channel_lightcurve(inst, objid, eventid):
    """
    Resolves (x, y, frame_start, frame_end) for a real candidate objid in
    `inst.events`, then delegates to _build_channel_lightcurve_at_xy.
    """
    obj_df = inst.events[inst.events['objid'] == objid].sort_values('frame')
    if 'event' in obj_df.columns and obj_df['event'].notna().any():
        event_labels = sorted(obj_df['event'].dropna().unique())
        event_label = event_labels[min(eventid - 1, len(event_labels) - 1)]
        ev_df = obj_df[obj_df['event'] == event_label]
    else:
        ev_df = obj_df

    x = int(round(ev_df['x'].mean()))
    y = int(round(ev_df['y'].mean()))
    frame_start = int(ev_df['frame'].min())
    frame_end = int(ev_df['frame'].max())
    return _build_channel_lightcurve_at_xy(inst, x, y, frame_start, frame_end)


def _build_channel_lightcurve_at_xy(inst, x, y, frame_start, frame_end, use_bad_mask=None):
    """
    Reconstructs the same light-curve/frame-selection quantities
    _plot_detection_nircam_grid consumes for a single already-run
    Jurassic instance (`inst.clean_cube`/`inst.kernel`/`inst.n_frame`/
    `inst.frame_mjd_df` all populated -- a normal images=True run, not
    reference_only=True), at an EXPLICIT (x, y, frame_start, frame_end)
    rather than looking one up by objid. Used directly (not via
    _build_channel_lightcurve) when the other channel has no real
    detection at all -- see plot_augmented_sw_lw's other_objid=None path:
    forced PSF photometry at the geometrically-correct position (this
    channel's own WCS applied to the matched channel's RA/Dec) is the
    right fallback, NOT borrowing some unrelated nearby candidate's
    objid, which would show the wrong pixels entirely.

    Forced PSF photometry at (x, y), one fit per frame across the WHOLE
    exposure (same reasoning as plot_detection's NIRCam branch: this
    gives a continuous light curve, not just the detection-catalog rows
    for one particular objid/event). Factored out here (duplicated from
    plot_detection/_plot_detection_nircam_grid rather than refactoring
    those) so plot_augmented_sw_lw can call it identically for both the
    SW and LW instance of a matched (or forced-position) candidate.

    use_bad_mask : bool or None, default None, which falls back to
        inst.mask_jump_det (False by default now -- see its own
        mask_jump_det docstring for why: JUMP_DET can't tell a cosmic ray
        from a real fast transient, so it's off at the detection stage).
        When True (an explicit opt-in for a forced-photometry context,
        e.g. plot_augmented_sw_lw's forced-position fallback), excludes
        any JUMP_DET-flagged pixel (inst.mask_tot's own exclusions) from
        the centroid fit and per-frame flux -- but ONLY when that pixel
        does NOT overlap the source's own photometry aperture (checked
        below, per (x, y), before every fit). A jump landing ON the
        source itself is left unmasked: masking it would risk erasing a
        real transient's own signal, which is exactly the failure mode
        blanket JUMP_DET masking has no way to avoid. A jump landing
        elsewhere in the wider fit stamp (contaminating the local
        background estimate, not the source) is still masked -- see
        _amplitude_only_fit's own bad_mask docstring for the motivating
        case (snake arc region LW objid84, a nearby-but-not-on-source
        cosmic ray). A stand-in `inst` built outside the real __init__
        (e.g. a Jurassic.__new__ shortcut for one-off scripts) has no
        mask_jump_det attribute and no mask_tot, so this safely falls
        back to off unless both are set explicitly.
    """
    if use_bad_mask is None:
        use_bad_mask = getattr(inst, 'mask_jump_det', False)
    bad_mask = ~inst.mask_tot if (use_bad_mask and getattr(inst, 'mask_tot', None) is not None) else None
    if bad_mask is not None:
        xi, yi = int(round(x)), int(round(y))
        a = FORCED_PHOTOMETRY_APERTURE_HW
        ny, nx = bad_mask.shape
        y0c, y1c = max(0, yi - a), min(ny, yi + a + 1)
        x0c, x1c = max(0, xi - a), min(nx, xi + a + 1)
        if bad_mask[y0c:y1c, x0c:x1c].any():
            # the flagged pixel(s) sit ON the source's own aperture --
            # don't mask; trust the shape/persistence/S-N filters
            # downstream to judge real-vs-artifact instead.
            bad_mask = None

    mjd_arr = inst.frame_mjd_df.set_index('frame')['mjd'].values
    time = mjd_arr - mjd_arr[0]
    cadence = np.median(np.diff(time))

    calibrated = getattr(inst, 'cube_units', None) == 'mjy/sr'
    has_pixar = calibrated and getattr(inst, 'pixar_sr', None) is not None
    ylabel = (r'$\mu$Jy' if has_pixar else (r'MJy sr$^{-1}$' if calibrated else 'DN/group'))

    # TESSELLATE-style fixed centroid (see _stacked_centroid_fit) + a
    # narrow 3x3-pixel amplitude-only fit per frame (see
    # _amplitude_only_fit) -- now the default forced-photometry method.
    # Replaces the previous per-frame independent _psf_fit (position
    # re-optimized every frame, full kernel-sized fit window): that
    # approach let a nearby unrelated source contaminate a frame's flux
    # even with the fitted centroid barely moving, since the amplitude
    # estimator summed stamp*template over the FULL kernel-sized
    # footprint regardless of exact sub-pixel position (confirmed
    # directly on objid78-field objid57 -- a contaminating source ~2px
    # from the target inflated psf_flux ~2.5x in the frames it appeared
    # in, and fixing the centroid alone left that essentially
    # unchanged; only narrowing the aperture fixed it).
    x_psf_fixed, y_psf_fixed = _stacked_centroid_fit(inst.clean_cube, x, y, inst.kernel, bad_mask=bad_mask)
    dx_fixed = x_psf_fixed - int(round(x))
    dy_fixed = y_psf_fixed - int(round(y))

    raw_flux = np.full(inst.n_frame, np.nan)
    raw_flux_err = np.full(inst.n_frame, np.nan)
    for frame in range(inst.n_frame):
        frame_data = inst.clean_cube[frame]
        if np.all(np.isnan(frame_data)):
            continue
        psf_flux, psf_flux_err = _amplitude_only_fit(
            np.nan_to_num(frame_data), x, y, dx_fixed, dy_fixed, inst.kernel, bad_mask=bad_mask)
        raw_flux[frame] = psf_flux
        raw_flux_err[frame] = psf_flux_err

    if has_pixar:
        f = raw_flux * inst.pixar_sr * 1e12
        f_err = raw_flux_err * inst.pixar_sr * 1e12
    else:
        f = raw_flux
        f_err = raw_flux_err

    valid_lc = ~np.isnan(f) & ~np.isnan(time)
    time_c = time[valid_lc]
    f_c = f[valid_lc]
    ferr_c = f_err[valid_lc]

    med_c = np.nanmedian(np.diff(time_c)) if len(time_c) > 1 else cadence
    brk = np.where(np.diff(time_c) > med_c * 1.5)[0]
    brk = np.insert(np.append(brk + 1, len(time_c)), 0, 0)

    if frame_end - frame_start >= 2:
        seg = np.abs(f[frame_start:frame_end + 1])
        brightestframe = frame_start + int(np.nanargmax(seg)) if np.any(np.isfinite(seg)) else frame_start
    else:
        brightestframe = frame_start
    if brightestframe >= inst.clean_cube.shape[0]:
        brightestframe = inst.clean_cube.shape[0] - 1

    n_frame = inst.clean_cube.shape[0]
    valid_frames = [i for i in range(n_frame) if not np.all(np.isnan(inst.clean_cube[i]))]
    event_frames = set(range(frame_start, frame_end + 1)) & set(valid_frames)
    span = len(event_frames)
    if len(valid_frames) <= 4:
        frame_indices = valid_frames
    else:
        sorted_valid = sorted(valid_frames)
        if span >= 4:
            frame_indices = sorted([f for f in sorted_valid if f in event_frames][:4])
        else:
            anchor = frame_start if frame_start in sorted_valid else min(
                sorted_valid, key=lambda f: abs(f - frame_start))
            anchor_idx = sorted_valid.index(anchor)
            start = max(0, anchor_idx - (4 - span) // 2)
            start = min(start, len(sorted_valid) - 4)
            frame_indices = sorted_valid[start:start + 4]

    return dict(x=x, y=y, x_fit=x_psf_fixed, y_fit=y_psf_fixed,
                frame_start=frame_start, frame_end=frame_end,
                time=time, time_c=time_c, f_c=f_c, ferr_c=ferr_c, brk=brk,
                cadence=cadence, mjd_arr=mjd_arr, ylabel=ylabel,
                brightestframe=brightestframe, event_frames=event_frames,
                frame_indices=frame_indices)


def _plot_one_detection_object(objid, obj_df, clean_cube, second_ref_frame, kernel, n_frame,
                               fwhm, instrument, pixar_sr, calibrated, has_pixar, lc_units,
                               ylabel, group_time_s, time, mjd_arr, cadence, show_inset, save_dir):
    """
    One object's worth of plot_detection's former per-object loop body,
    pulled out to a top-level function (explicit args instead of self.X)
    so it can run under joblib.Parallel(prefer="processes") -- matching
    _sep_extract/_finish_sep's established pattern in this file.
    """
    if obj_df.empty:
        return
    _ylabel = ylabel

    # Build per-object event list from DBSCAN 'event' column
    if 'event' in obj_df.columns and obj_df['event'].notna().any():
        event_labels = sorted(obj_df['event'].dropna().unique())
    else:
        event_labels = [None]
    total_events = len(event_labels)

    for eventid, event_label in enumerate(event_labels, start=1):
        ev_df = obj_df if event_label is None else obj_df[obj_df['event'] == event_label]
        if ev_df.empty:
            continue

        # Restrict per-event diagnostic figures to high-confidence
        # (psf S/N >= 20) events -- a crowded field can carry thousands
        # of S/N>=8 detections, each producing its own multi-panel
        # figure; this keeps only the events worth actually looking at.
        if 'psf_flux' in ev_df.columns and 'psf_flux_err' in ev_df.columns:
            ev_snr = ev_df['psf_flux'] / ev_df['psf_flux_err']
            if not (ev_snr.notna() & (ev_snr >= 20)).any():
                continue

        # Prefer the PSF-fit refined centroid (x_psf/y_psf) over
        # the raw SEP first-moment centroid (x/y) for where the
        # box is actually drawn -- the raw SEP centroid is more
        # easily biased (background/segmentation asymmetry), and
        # this box is purely a visualization aid; falls back to
        # the raw centroid if x_psf is missing/NaN for this event.
        if 'x_psf' in ev_df.columns and ev_df['x_psf'].notna().any():
            x = int(round(ev_df['x_psf'].mean()))
            y = int(round(ev_df['y_psf'].mean()))
        else:
            x = int(round(ev_df['x'].mean()))
            y = int(round(ev_df['y'].mean()))
        frame_start = int(ev_df['frame'].min())
        frame_end = int(ev_df['frame'].max())

        if instrument == 'NIRCAM':
            # Forced PSF photometry, one fit per frame at this
            # object's mean position -- not aperture summation.
            # _psf_fit's returned amplitude is the optimal linear-
            # least-squares scaling of the (position-fit) PSF
            # template against the stamp, weighting each pixel by
            # how much the model itself puts there rather than a
            # flat box sum. This also sidesteps the aperture-size
            # dependence that made buf=floor(fwhm) change the
            # light curve's actual SHAPE (not just its scale)
            # between filters with different FWHM -- confirmed
            # comparing F090W (fwhm~1.2px) against F410M
            # (fwhm~2.4px) test cases this session.
            raw_flux = np.full(n_frame, np.nan)
            raw_flux_err = np.full(n_frame, np.nan)
            n_pix_equiv = (2 * (kernel.shape[0] // 2) + 1) ** 2
            for frame in range(n_frame):
                frame_data = clean_cube[frame]
                if np.all(np.isnan(frame_data)):
                    continue
                *_junk, psf_flux, psf_flux_err = _psf_fit(np.nan_to_num(frame_data), x, y, kernel)
                raw_flux[frame] = psf_flux
                raw_flux_err[frame] = psf_flux_err
            all_nan = np.isnan(raw_flux)
            n_pix = np.full(n_frame, float(n_pix_equiv))
            n_pix[all_nan] = np.nan
        else:
            # Aperture LC using filter FWHM as radius (buffer = floor(fwhm))
            buf = int(np.floor(fwhm))

            aperture = clean_cube[:,
                                  max(0, y - buf):min(clean_cube.shape[1], y + buf + 1),
                                  max(0, x - buf):min(clean_cube.shape[2], x + buf + 1)]
            all_nan = np.all(np.isnan(aperture), axis=(1, 2))
            raw_flux = np.where(all_nan, np.nan, np.nansum(aperture, axis=(1, 2)))
            raw_flux_err = np.full(n_frame, np.nan)  # not computed for the MIRI aperture path
            n_pix = np.sum(~np.isnan(aperture), axis=(1, 2)).astype(float)
            n_pix[n_pix == 0] = np.nan

        if calibrated:
            if lc_units == 'uJy' and has_pixar:
                # MJy -> uJy is *1e12 (MJy->Jy is 1e6, Jy->uJy is
                # another 1e6), not *1e6 -- fixed a pre-existing
                # factor-of-1e6 unit bug here.
                f = raw_flux * pixar_sr * 1e12
                f_err = raw_flux_err * pixar_sr * 1e12
            elif lc_units == 'MJy' and has_pixar:
                f = raw_flux * pixar_sr
                f_err = raw_flux_err * pixar_sr
            else:
                f = raw_flux / n_pix
                f_err = raw_flux_err / n_pix
        else:
            f = raw_flux
            f_err = raw_flux_err
            if lc_units == 'dn/s':
                f = f / group_time_s
                f_err = f_err / group_time_s

        # Strip NaN frames from both time and f; recompute breaks on clean arrays
        valid_lc = ~np.isnan(f) & ~np.isnan(time)
        time_c = time[valid_lc]
        f_c = f[valid_lc]
        ferr_c = f_err[valid_lc]
        # A multiplicative threshold (not med+std) so a nearly-
        # uniform cadence -- where std collapses toward 0 -- can't
        # have a diff spuriously exceed it from ordinary MJD
        # floating-point jitter between consecutive timestamps;
        # a real gap only ever needs to be a real multiple of the
        # typical cadence to register.
        med_c = np.nanmedian(np.diff(time_c))
        brk = np.where(np.diff(time_c) > med_c * 1.5)[0]
        brk = np.insert(np.append(brk + 1, len(time_c)), 0, 0)

        # Brightest frame within detection span
        if frame_end - frame_start >= 2:
            brightestframe = frame_start + int(np.where(
                np.abs(f[frame_start:frame_end]) == np.nanmax(np.abs(f[frame_start:frame_end]))
            )[0][0])
        else:
            brightestframe = frame_start
        try:
            brightestframe = int(brightestframe)
        except TypeError:
            brightestframe = int(brightestframe[0])
        if brightestframe >= clean_cube.shape[0]:
            brightestframe -= 1
        if frame_end >= clean_cube.shape[0]:
            frame_end -= 1

        if instrument == 'NIRCAM':
            _plot_detection_nircam_grid(
                clean_cube, second_ref_frame, save_dir, objid, eventid, total_events, ev_df, x, y,
                frame_start, frame_end, time, time_c, f_c, brk, cadence,
                mjd_arr, _ylabel, brightestframe, ferr_c=ferr_c)
            continue

        fstart = frame_start - 20
        if fstart < 0:
            fstart = 0

        fig, ax = plt.subplot_mosaic([[1, 1, 1, 2, 2], [1, 1, 1, 3, 3]],
                                     figsize=(7 * 1.1, 5.5 * 1.1), constrained_layout=True)

        if show_inset:
            # Ghost plot to fix inset ylims
            zoom_valid = valid_lc[fstart:frame_end + 20]
            zoom_t = time[fstart:frame_end + 20][zoom_valid]
            zoom_f = f[fstart:frame_end + 20][zoom_valid]
            ax[1].plot(zoom_t, zoom_f, 'k', alpha=0)
            insert_ylims = ax[1].get_ylim()

        # Full light curve — NaN frames already removed in time_c/f_c
        for seg in range(len(brk) - 1):
            ax[1].plot(time_c[brk[seg]:brk[seg + 1]],
                       f_c[brk[seg]:brk[seg + 1]], 'k', alpha=0.8)

        if not show_inset:
            # Event span highlighted directly on the main light curve
            # instead of in a separate zoom inset.
            ax[1].axvspan(time[frame_start] - cadence / 2,
                          time[frame_end] + cadence / 2, color='C1', alpha=0.4)

        ylims = ax[1].get_ylim()
        ax[1].set_ylim(ylims[0], ylims[1] + abs(ylims[0] - ylims[1]))
        ax[1].set_xlim(np.min(time_c), np.max(time_c))
        ax[1].set_title(f'ObjID: {objid}', fontsize=15)
        ax[1].set_ylabel(_ylabel, fontsize=15, labelpad=10)
        ax[1].set_xlabel(f'Time (MJD - {np.round(mjd_arr[0], 3)})', fontsize=15)

        if show_inset:
            axins = ax[1].inset_axes([0.1, 0.55, 0.86, 0.43])
            axins.axvspan(time[frame_start] - cadence / 2,
                          time[frame_end] + cadence / 2, color='C1', alpha=0.4)
            for seg in range(len(brk) - 1):
                axins.plot(time_c[brk[seg]:brk[seg + 1]],
                           f_c[brk[seg]:brk[seg + 1]], 'k', alpha=0.8, marker='.')

            duration = frame_end - frame_start
            if duration < 4:
                duration = 4
            fe = frame_end + 20
            if fe >= len(time):
                fe = len(time) - 1
            xmin_z = time[frame_start] - (3 * duration * cadence)
            xmax_z = time[frame_end] + (3 * duration * cadence)
            if xmin_z <= 0:
                xmin_z = 0
            if xmax_z >= np.nanmax(time):
                xmax_z = np.nanmax(time)
            axins.set_xlim(xmin_z, xmax_z)
            axins.set_ylim(insert_ylims[0], insert_ylims[1])
            mark_inset(ax[1], axins, loc1=3, loc2=4, fc="none", ec="r", lw=2)
            plt.setp(axins.spines.values(), color='r', lw=2)
            plt.setp([axins.get_xticklines(), axins.get_yticklines()], color='C3')

        # Colour stretch from 3x3 patch at brightest frame. Uses
        # nanpercentile, not percentile: any NaN in the frame
        # (e.g. reference footprint edges) makes plain percentile
        # return NaN for the whole computation, silently breaking
        # the color scale and rendering the image blank.
        bright_frame = clean_cube[brightestframe, max(0, y - 1):y + 2, max(0, x - 1):x + 2]
        vmin = np.nanpercentile(clean_cube[brightestframe], 16)
        try:
            vmax = np.nanpercentile(bright_frame, 80)
        except Exception:
            vmax = vmin + 20
        if vmin >= vmax:
            vmin = vmax - 5

        # 19x19 px cutout
        ymin = y - 9
        if ymin < 0:
            ymin = 0
        xmin = x - 9
        if xmin < 0:
            xmin = 0
        cutout_image = clean_cube[:, ymin:y + 10, xmin:x + 10]

        ax[2].imshow(cutout_image[brightestframe], cmap='gray', origin='lower',
                     vmin=vmin, vmax=vmax)
        ax[2].scatter(ev_df['x'].mean() - xmin, ev_df['y'].mean() - ymin,
                      color='r', s=50, marker='x', lw=2)
        ax[2].set_title('Brightest frame', fontsize=15)
        ax[2].get_xaxis().set_visible(False)
        ax[2].get_yaxis().set_visible(False)
        ax[3].get_xaxis().set_visible(False)
        ax[3].get_yaxis().set_visible(False)

        # 20 non-NaN frames after; fall back to 20 non-NaN frames before
        valid_after = [i for i in range(brightestframe + 1, len(cutout_image))
                       if not np.all(np.isnan(cutout_image[i]))]
        if len(valid_after) >= 20:
            after = valid_after[19]
        else:
            valid_before = [i for i in range(brightestframe)
                            if not np.all(np.isnan(cutout_image[i]))]
            after = valid_before[-20] if len(valid_before) >= 20 else (valid_before[0] if valid_before else brightestframe)

        offset = after - brightestframe
        after_label = f'+{offset}' if offset >= 0 else str(offset)

        ax[3].imshow(cutout_image[after], cmap='gray', origin='lower',
                     vmin=vmin, vmax=vmax)
        ax[3].set_title(f'Frame {after_label}', fontsize=15)
        ax[3].annotate('', xy=(0.2, 1.15), xycoords='axes fraction', xytext=(0.2, 1.),
                       arrowprops=dict(arrowstyle="<|-", color='r', lw=3))
        ax[3].annotate('', xy=(0.8, 1.15), xycoords='axes fraction', xytext=(0.8, 1.),
                       arrowprops=dict(arrowstyle="<|-", color='r', lw=3))

        # 5x5 red/cyan detection box — separate patches per panel (matching tessellate)
        rect = patches.Rectangle((x - 2.5 - xmin, y - 2.5 - ymin), 5, 5,
                                 linewidth=3, edgecolor='r', facecolor='none')
        ax[2].add_patch(rect)
        ax[2].add_line(Line2D([x - 2.5 - xmin, x + 2.5 - xmin],
                              [y + 2.5 - ymin, y + 2.5 - ymin], color='c', linewidth=3))
        ax[2].add_line(Line2D([x + 2.5 - xmin, x + 2.5 - xmin],
                              [y - 2.5 - ymin, y + 2.5 - ymin], color='c', linewidth=3))

        rect = patches.Rectangle((x - 2.5 - xmin, y - 2.5 - ymin), 5, 5,
                                 linewidth=3, edgecolor='r', facecolor='none')
        ax[3].add_patch(rect)
        ax[3].add_line(Line2D([x - 2.5 - xmin, x + 2.5 - xmin],
                              [y + 2.5 - ymin, y + 2.5 - ymin], color='c', linewidth=3))
        ax[3].add_line(Line2D([x + 2.5 - xmin, x + 2.5 - xmin],
                              [y - 2.5 - ymin, y + 2.5 - ymin], color='c', linewidth=3))

        plt.savefig(os.path.join(save_dir,
                                 f'object{objid:04d}_event{eventid}of{total_events}.png'),
                    bbox_inches='tight')
        plt.close(fig)


def _plot_detection_nircam_grid(clean_cube, second_ref_frame, save_dir, objid, eventid, total_events,
                                ev_df, x, y, frame_start, frame_end, time, time_c, f_c, brk, cadence,
                                mjd_arr, ylabel, brightestframe, ref_zoom_halfwidth=40, ferr_c=None):
    """
    NIRCam detection figure, replacing the MIRI-style 3-panel layout
    with a 3x4 grid (per direct request):
        AABB
        AABB
        cdef
    A: light curve, individual points marked (not just a connecting
       line -- NIRCam ramps are only a handful of points, so every
       one matters and should be visible on its own).
    B: reference frame (the one actually subtracted to make this
       detection), zoomed OUT relative to the tight per-frame cutouts
       below, for spatial context (nearby sources, cluster/arc light,
       crowding) that a 19x19 px cutout can't show.
    cdef: up to 4 sequential frames spanning the event, centered on
       it when the exposure has more than 4 usable frames. Event
       frame(s) get an orange border + "(event)" label so they're
       unambiguous at a glance. If the exposure has fewer than 4
       usable (non-all-NaN) frames, only that many panels are drawn
       -- no blank placeholder axes.

    Top-level (not a method) so it can run under joblib's
    Parallel(..., prefer="processes") from plot_detection -- takes
    clean_cube/second_ref_frame explicitly instead of via self.
    """
    import matplotlib.patches as patches
    from matplotlib.gridspec import GridSpec

    os.makedirs(save_dir, exist_ok=True)

    # up to 4 valid (non-all-NaN) frames, centered on the event span
    n_frame = clean_cube.shape[0]
    valid_frames = [i for i in range(n_frame) if not np.all(np.isnan(clean_cube[i]))]
    event_frames = set(range(frame_start, frame_end + 1)) & set(valid_frames)
    span = len(event_frames)
    if len(valid_frames) <= 4:
        frame_indices = valid_frames
    else:
        sorted_valid = sorted(valid_frames)
        if span >= 4:
            frame_indices = sorted([f for f in sorted_valid if f in event_frames][:4])
        else:
            anchor = frame_start if frame_start in sorted_valid else min(
                sorted_valid, key=lambda f: abs(f - frame_start))
            anchor_idx = sorted_valid.index(anchor)
            start = max(0, anchor_idx - (4 - span) // 2)
            start = min(start, len(sorted_valid) - 4)
            frame_indices = sorted_valid[start:start + 4]
    n_bottom = max(1, len(frame_indices))

    # One shared color stretch for every cdef panel (not per-frame),
    # fixed from the brightest frame: vmin from that frame's global
    # background level, vmax set to the event's actual peak
    # brightness there (not a percentile approximation) -- so the
    # brightest frame uses its full dynamic range and every other
    # frame is shown on that exact same scale, making relative
    # brightness across frames directly comparable at a glance.
    # nanpercentile/nanmax, not their non-nan counterparts: any NaN
    # in the frame (e.g. reference footprint edges) makes the plain
    # versions return NaN for the whole computation, silently
    # breaking the color scale and rendering every panel blank.
    bright_patch = clean_cube[brightestframe, max(0, y - 1):y + 2, max(0, x - 1):x + 2]
    vmin = np.nanpercentile(clean_cube[brightestframe], 16)
    try:
        vmax = np.nanmax(bright_patch)
    except Exception:
        vmax = vmin + 20
    if vmin >= vmax:
        vmin = vmax - 5

    ny, nx = clean_cube.shape[1], clean_cube.shape[2]
    hs_tight = 9
    ymin_t, ymax_t = max(0, y - hs_tight), min(ny, y + hs_tight + 1)
    xmin_t, xmax_t = max(0, x - hs_tight), min(nx, x + hs_tight + 1)
    hs_wide = ref_zoom_halfwidth
    ymin_w, ymax_w = max(0, y - hs_wide), min(ny, y + hs_wide + 1)
    xmin_w, xmax_w = max(0, x - hs_wide), min(nx, x + hs_wide + 1)

    fig = plt.figure(figsize=(10, 7.5), constrained_layout=True)
    # 5th column reserved only for rows 0-1 (B's colorbar) -- narrow
    # enough (6% width) to be visually negligible, but a genuinely
    # separate column rather than carved out of ax_B's own box
    # (make_axes_locatable/append_axes would shrink ax_B itself,
    # making it narrower than ax_A). The bottom (cdef) row only ever
    # spans the first 4 columns, so it's unaffected either way.
    gs_top = GridSpec(3, 5, figure=fig, width_ratios=[1, 1, 1, 1, 0.06])
    ax_A = fig.add_subplot(gs_top[0:2, 0:2])
    ax_B = fig.add_subplot(gs_top[0:2, 2:4])
    cax_B = fig.add_subplot(gs_top[0:2, 4])
    # Direct columns, not a subgridspec with an extra column added
    # for the colorbar -- that compressed all n_bottom frame panels
    # horizontally to make room. cax_last instead reuses column 4
    # (row 2 of it, cax_B already uses rows 0-1), the SAME dedicated
    # column as the reference colorbar -- same width, vertically
    # aligned with it, and never steals width from the frame panels
    # since it isn't part of their own column span at all.
    bottom_axes = [fig.add_subplot(gs_top[2, i]) for i in range(n_bottom)]
    cax_last = fig.add_subplot(gs_top[2, 4])

    # A: light curve, individual points clearly marked. NIRCam ramps
    # span seconds to minutes, not days -- MJD-day units (as used for
    # MIRI, whose light curves can span hours) render as unreadable
    # ~0.0000-0.0050 fractions here, so convert to seconds instead.
    time_s = time * 86400
    time_c_s = time_c * 86400
    cadence_s = cadence * 86400
    for seg in range(len(brk) - 1):
        sl = slice(brk[seg], brk[seg + 1])
        ax_A.plot(time_c_s[sl], f_c[sl], 'k-', alpha=0.8, zorder=1)
        if ferr_c is not None:
            ax_A.errorbar(time_c_s[sl], f_c[sl], yerr=ferr_c[sl], fmt='o',
                          color='k', ms=6, ecolor='k', elinewidth=1.2,
                          capsize=3, zorder=2)
        else:
            ax_A.plot(time_c_s[sl], f_c[sl], 'o', color='k', ms=6, zorder=2)
    ax_A.axvspan(time_s[frame_start] - cadence_s / 2, time_s[frame_end] + cadence_s / 2,
                 color='C1', alpha=0.4)
    ax_A.set_xlim(np.min(time_c_s) - cadence_s / 2, np.max(time_c_s) + cadence_s / 2)
    ax_A.set_title(f'ObjID: {objid}', fontsize=15)
    ax_A.set_ylabel(ylabel, fontsize=15, labelpad=10)
    ax_A.set_xlabel(f'Time (s) - MJD {np.round(mjd_arr[0], 5)}', fontsize=15)

    # B: reference, zoomed out for spatial context -- extent shifted
    # by -0.5 so pixel centers land on their real detector coordinate
    # (imshow's extent otherwise places pixel EDGES at the given
    # bounds, offsetting every pixel center by half a pixel). Needs
    # its OWN color stretch, not vmin/vmax from the diff-cube patch:
    # second_ref_frame is an absolute sky image (median flux across
    # dithers), on a completely different scale than the small-
    # amplitude group-to-group differences in clean_cube, so reusing
    # that stretch here saturates the whole panel to white.
    ref_cutout = second_ref_frame[ymin_w:ymax_w, xmin_w:xmax_w]
    ref_valid = ref_cutout[np.isfinite(ref_cutout)]
    if ref_valid.size:
        ref_vmin, ref_vmax = np.percentile(ref_valid, [16, 99])
        if ref_vmin >= ref_vmax:
            ref_vmax = ref_vmin + 1
    else:
        ref_vmin, ref_vmax = 0, 1
    im_ref = ax_B.imshow(ref_cutout, cmap='gray', origin='lower', vmin=ref_vmin, vmax=ref_vmax,
                         extent=[xmin_w - 0.5, xmax_w - 0.5, ymin_w - 0.5, ymax_w - 0.5])
    # aspect='auto', not imshow's default 'equal' (square pixels):
    # 'equal' shrinks the actual displayed image within its allocated
    # GridSpec box whenever that box isn't precisely square, while
    # ax_A (a line plot, no aspect constraint) fills its box fully --
    # left ax_B visibly smaller than ax_A, and cax_B's height (sized
    # to the box, not the shrunk image) then didn't match either.
    ax_B.set_aspect('auto')
    ax_B.set_title('Reference', fontsize=15)
    ax_B.set_xlabel('x (px)', fontsize=10)
    ax_B.set_ylabel('y (px)', fontsize=10)
    ax_B.add_patch(patches.Rectangle((x - 2.5, y - 2.5), 5, 5,
                                      linewidth=2, edgecolor='r', facecolor='none'))
    # cax_B is its own dedicated GridSpec column (added above), not
    # carved out of ax_B via make_axes_locatable/append_axes -- that
    # would shrink ax_B itself, leaving it narrower than ax_A.
    # second_ref_frame is built from grad_cube pre-_flux_calibrate
    # (see that method's docstring: "Convert self.clean_cube from
    # DN/group to MJy/sr"), so it's still in raw DN/group here.
    cbar = fig.colorbar(im_ref, cax=cax_B)
    cbar.set_label('DN/group', fontsize=10)

    # cdef: sequential frames, event frame(s) flagged
    for panel_ax, fi in zip(bottom_axes, frame_indices):
        frame_img = clean_cube[fi, ymin_t:ymax_t, xmin_t:xmax_t]
        im_frame = panel_ax.imshow(frame_img, cmap='gray', origin='lower', vmin=vmin, vmax=vmax,
                                   extent=[xmin_t - 0.5, xmax_t - 0.5, ymin_t - 0.5, ymax_t - 0.5])
        panel_ax.set_aspect('auto')  # see ax_B above -- keeps this panel the same size as the others
        panel_ax.set_xlabel('x (px)', fontsize=9)
        panel_ax.set_ylabel('y (px)', fontsize=9)
        panel_ax.tick_params(labelsize=8)
        panel_ax.add_patch(patches.Rectangle((x - 2.5, y - 2.5), 5, 5,
                                              linewidth=2, edgecolor='r', facecolor='none'))
        is_event = fi in event_frames
        panel_ax.set_title(f'Frame {fi}' + (' (event)' if is_event else ''),
                           fontsize=11, color='C1' if is_event else 'black',
                           fontweight='bold' if is_event else 'normal')
        if is_event:
            for spine in panel_ax.spines.values():
                spine.set_edgecolor('C1')
                spine.set_linewidth(3)
        if panel_ax is bottom_axes[-1]:
            # cax_last is its own dedicated subgridspec column (see
            # above), not carved out of this panel -- same reasoning
            # as cax_B, so this last frame panel stays the same size
            # as the others.
            cbar_last = fig.colorbar(im_frame, cax=cax_last)
            cbar_last.set_label('DN/group', fontsize=9)
            cbar_last.ax.tick_params(labelsize=8)

    plt.savefig(os.path.join(save_dir, f'object{objid:04d}_event{eventid}of{total_events}.png'),
                bbox_inches='tight')
    plt.close(fig)


class Jurassic():
    """
        Class for searching the ramps of full array MIRI/NIRCam images for fast transients

        JURASSIC: JWST Up the Ramp Analysis Searching the Sky for Infrared Transients
    """

    def __init__(self,file=None,num_cores=35,run=True,method='mega',ramps=None,images=True,
                 significance=False,mask_correction=True,plot=True,no_sat_mask=False,
                 base_dir=None,data_dir=None,correct_ramps=True,master_ref_path=None,
                 compute_radec=True,reference_only=False,save_detection_figures=True,
                 mask_jump_det=False,snr_thresh=8):
        """
        Initialise or whatevs

        Parameters
        ----------
        file : str
                File name of the observation

        method : str
                either 'ramp' or 'mega'

        master_ref_path : str or None
                Path to a pre-built master reference FITS file (SCI data +
                WCS header) to use as first_ref_frame instead of computing
                it from this exposure's own grad_cube. For short NIRCam
                ramps a per-exposure median (built from only a handful of
                groups) can itself contain a transient that persists across
                the whole exposure, silently diluting or hiding it in the
                difference cube. A master reference built from many
                independent dithers/epochs (see build_master_refs.py)
                doesn't have that problem, since any one transient
                contributes to only one of many epochs going into its
                median. Aligned onto this exposure's own distortion-
                corrected WCS via reproject at use time, so one master ref
                per (filter, detector) covers all its epochs regardless of
                small pointing differences between them.

        other stuff I guess - will update at some point
        """
        self.master_ref_path = master_ref_path
        self.file = file
        parts = self.file.replace('\\', '/').split('/')
        self.name = parts[-2] if len(parts) >= 2 else '.'
        self.obs_id = parts[-1]
        self.method = method
        self.plot = plot
        # gates the per-object/per-frame detection diagnostic figures
        # (sep_frames/, detection_figures_sep/) built inside
        # source_extracting's save_plot=True pass -- separate from
        # `plot`, which controls significance-mode output only. Set
        # False for a run whose figures aren't going to be looked at
        # (e.g. batch reprocessing many dithers where only the CSV
        # candidate catalog matters, with figures generated separately
        # afterward on demand from the cached clean_cube/events instead).
        self.save_detection_figures = save_detection_figures
        self.base_dir = base_dir
        self.data_dir = data_dir
        self.mask_correction = mask_correction
        # compute_radec=False skips the AssignWcsStep/CRDS call in
        # assign_radec entirely -- for a throwaway reference-only run
        # (e.g. reduce_nircam_stage1.py, which only needs ref_frame_1.npy
        # and never looks at per-candidate positions) that call is pure
        # waste, and unlike photmjsr (constant per filter/detector, see
        # flux_calibrate's local cache) WCS genuinely does depend on this
        # exposure's own pointing, so it can't be cached the same way.
        self.compute_radec = compute_radec
        # reference_only=True stops right after ref_frame_1.npy is built
        # and saved (_reference_frame(), inside the images/significance
        # block below) -- skips differencing, PSF-kernel/SEP extraction,
        # cosmic-ray removal, clean_cube flux calibration, and
        # save_outputs() entirely. For a stage1-style run (see
        # reduce_nircam_stage1.py) none of that downstream work is ever
        # looked at -- only the per-exposure reference median feeds
        # build_master_refs.py -- so running it was pure wasted time.
        self.reference_only = reference_only
        self.no_sat_mask = no_sat_mask
        self.mask_jump_det = mask_jump_det
        self.snr_thresh = snr_thresh
        self.correct_ramps = correct_ramps
        self.num_cores = num_cores # number of cores to use when running functions (sep, 1st order polyfit, lacosmic) in parallel
        self.psf_fwhm_px = { # taken from JDOX
            "F560W": 1.882,
            "F770W": 2.445,
            "F1000W": 2.982,
            "F1130W": 3.409,
            "F1280W": 3.818,
            "F1500W": 4.436,
            "F1800W": 5.373,
            "F2100W": 6.127,
            "F2550W": 7.300,
        }
        self.stpsf_class = { # instrument name -> stpsf class name
            "MIRI": "MIRI",
            "NIRCAM": "NIRCam",
        }

        if run:
            self._assign_data()
            if self.correct_ramps and self.instrument != 'MIRI':
                print(f'{self.instrument}: BFE/RCD ramp correction is only validated for '
                      f'MIRI — skipping ramp_correction() and using uncorrected ramps.')
                self.correct_ramps = False
            if self.correct_ramps:
                self.ramp_correction(cube=self.data)
            else:
                self.data_cor = self.data
                ny, nx = self.data.shape[2], self.data.shape[3]
                self.gen_mask = self._get_gen_mask(ny, nx)
            self.flux_calibrate(cube=self.data_cor)
            self._make_cubes()
            del self.data, self.data_cor, self.flux_data
            self._mask_pixels(mask_jump_det=self.mask_jump_det)
            del self.rampy_cube_dn, self.dq_cube

            if ramps: # search on ramp level
                print('ramps')
                self.parallel_fit_df(self.rampy_cube) # only fitting rampy_cube not mega

            if self.method == 'mega':
                if images or significance:
                    print('images')
                    self.mega_inator(self.rampy_cube)
                    del self.rampy_cube
                    self._cube_gradient(self.mega_cube_masked, save=True)
                    del self.mega_cube, self.mega_cube_masked, self.fakey_cube
                    self._reference_frame()
                    if self.reference_only:
                        return
                    self._cube_differenced(self.grad_cube, self.first_ref_frame, save=False, first=None)
                    self._psf_kernel()
                    self.source_extracting(self.diff_cube, save_plot=False, save_csv=True)

                    print('re-difference')
                    self._masked_reference(self.mask_correction)
                    self._cube_differenced(self.grad_cube, self.second_ref_frame, save=True)
                    del self.grad_cube
                    self._remove_cosmic(self.diff_cube)
                    del self.diff_cube
                    self._make_ref_cr_mask()
                    self.source_extracting(self.clean_cube, save_plot=self.save_detection_figures, save_csv=False)

                if significance:
                    print('significance')
                    self._cube_significance()
                    self._cube_threshold()
                    del self.sig_cube, self.conv_sig_cube
                    self._cube_rolling_sum()
                    del self.bool_threshold_cube
                    self._significance_output()
            
            if self.method == 'ramp':
                if images or significance: # search on image level
                    print('images')
                    self._cube_gradient(self.rampy_cube,save=True)
                    self._reference_frame()
                    if self.reference_only:
                        return
                    self._cube_differenced(self.grad_cube,self.first_ref_frame,save=False,first=None) # first saves first iteration of difference cube
                    self._psf_kernel()
                    self.source_extracting(self.diff_cube,save_plot=False,save_csv=False)

                    print('re-difference')
                    self._masked_reference(self.mask_correction) # creating new mask and doing differencing again
                    self._cube_differenced(self.grad_cube,self.second_ref_frame,save=True)
                    self._remove_cosmic(self.diff_cube)
                    self._make_ref_cr_mask()
                    self.source_extracting(self.clean_cube,save_plot=self.save_detection_figures,save_csv=False)

                if significance:
                    print('significance')
                    self._cube_significance()
                    self._cube_threshold() 
                    self._cube_rolling_sum()
                    self._significance_output()

            if not hasattr(self, 'significance_df'):
                self.significance_df = pd.DataFrame()
            self._time_mjd()
            self._flux_calibrate()
            self.save_outputs()


    def _assign_data(self):
        """
        Opens the fits file and assigns the data to the class
        """
        # base outputs folder — defaults to outputs/ relative to cwd
        if self.base_dir is None:
            self.base_dir = os.path.join(os.getcwd(), 'outputs')
        os.makedirs(self.base_dir, exist_ok=True)

        # data folder — defaults to the directory containing the ramp file
        if self.data_dir is None:
            self.data_dir = os.path.dirname(os.path.abspath(self.file))

        # Remove the filename suffix (instrument/detector token + '_ramp.fits',
        # e.g. '..._mirimage_ramp.fits' or '..._nrcb1_ramp.fits')
        suffix1 = '_ramp.fits'
        suffix2 = '_cal.fits'
        if self.obs_id.endswith(suffix1):
            obs_n = self.obs_id[:-len(suffix1)]
        else:
            obs_n = self.obs_id  # fallback

        obs_name = f"dr_{obs_n}" 

        # directory for specific observation/segment
        self.obs_dir = os.path.join(self.base_dir, obs_name)
        os.makedirs(self.obs_dir, exist_ok=True)

        # get level 2a (ramp.fits) data
        self.stage1_filepath = os.path.abspath(self.file)
        if not os.path.exists(self.stage1_filepath):
            print(f"Cannot find the Stage 1 file: {self.stage1_filepath}")

        # get level 2b (cal.fits) data
        try:
            self.stage2_filepath = os.path.join(self.data_dir, obs_n + suffix2)
            self.do_flux_cal = os.path.exists(self.stage2_filepath)
            if not self.do_flux_cal:
                print("Cannot find the Stage 2 file --- No flux calibration will be performed")
        except OSError:
            print("Cannot find the Stage 2 file --- No flux calibration will be performed")
            self.do_flux_cal = False

        # assigning data from ramp.fits file
        with fits.open(self.stage1_filepath, ignore_missing_end=True) as hdul:
            self.data = np.array(hdul[1].data)
            self.dq_2d_arr = np.array(hdul[2].data)
            try:
                self.dq_3d_arr = np.array(hdul[3].data)
            except (TypeError, OSError, ValueError):
                print('GROUPDQ truncated — using zero DQ array')
                self.dq_3d_arr = np.zeros(self.data.shape, dtype=np.uint8)
            phdr = hdul['PRIMARY'].header
            self.instrument = phdr['INSTRUME'].strip().upper()
            self.detector   = phdr.get('DETECTOR', '').strip().upper()
            self.pupil      = phdr.get('PUPIL', None)
            self.tgroup    = phdr['TGROUP']
            self.filename  = phdr.get('FILENAME', self.obs_id)
            self.filter    = phdr['FILTER']
            self.subarray  = phdr['SUBARRAY']
            self.targname  = phdr['TARGNAME']
            self.substrt1  = phdr.get('SUBSTRT1', 1)   # 1-indexed FITS column start
            self.substrt2  = phdr.get('SUBSTRT2', 1)   # 1-indexed FITS row start
            try:
                self.time_df = pd.DataFrame(hdul[7].data)
            except (IndexError, KeyError):
                n_i = len(self.data)
                effinttm_days = phdr.get('EFFINTTM', self.tgroup * phdr.get('NGROUPS', 1)) / 86400.0
                t0 = phdr.get('EXPSTART', 0.0)
                starts = t0 + np.arange(n_i) * effinttm_days
                self.time_df = pd.DataFrame({
                    'integration_number': np.arange(1, n_i + 1),
                    'int_start_MJD_UTC':  starts,
                    'int_mid_MJD_UTC':    starts + effinttm_days / 2,
                    'int_end_MJD_UTC':    starts + effinttm_days,
                    'int_start_BJD_TDB':  starts,
                    'int_mid_BJD_TDB':    starts + effinttm_days / 2,
                    'int_end_BJD_TDB':    starts + effinttm_days,
                })
        # assigning data from cal.fits file. flux_calibrate() (called
        # earlier, at ramp-cube construction time) may already have set
        # self.pixar_sr as a fallback from the PHOTOM reference file when
        # no Stage2 product exists -- don't clobber that back to None.
        self.pixar_sr = getattr(self, 'pixar_sr', None)
        if self.do_flux_cal:
            with fits.open(self.stage2_filepath) as hdul:
                self.cal_data = hdul[1].data
                self.pixar_sr = hdul[1].header.get('PIXAR_SR', None)
            m, s = np.nanmedian(self.cal_data), np.nanstd(self.cal_data)
            plt.figure()
            plt.imshow(self.cal_data,origin='lower',vmin=m-s,vmax=m+s)
            plt.savefig(os.path.join(self.obs_dir, 'cal_image.png'), bbox_inches="tight")


        self.n_int = len(self.data) # number of integrations (ramps) in file
        self.n_group = len(self.data[0]) # number of groups per integration
        self.n_frame = self.n_int * self.n_group # number of frames in file
        self.frames = list(range(self.n_frame)) # list of all frame indices

        # Last frame of each integration is always excluded: the forward
        # difference there would need the next integration's first group,
        # which isn't the same physical quantity (a fresh ramp after
        # reset), so _forward_diff leaves it NaN structurally for every
        # instrument.
        #
        # First frame of each integration (group1-group0) is additionally
        # excluded only for MIRI: that difference is contaminated by RCD
        # (reset charge decay), a physical effect specific to MIRI's Si:As
        # detectors (see ramp_correction docstring). NIRCam's HgCdTe
        # detectors don't exhibit it -- confirmed empirically on this
        # data: group1-group0's median/std/outlier-fraction in a
        # background region are statistically indistinguishable from the
        # other (uncontested) group-to-group differences in the same
        # exposure, so there's no basis to throw it away here. Excluding
        # it anyway would mean losing 1 of only 3-6 usable frames per
        # short NIRCam exposure for no real reason.
        bad_frames = []
        for integration in list(range(self.n_int)):
            if self.instrument == 'MIRI':
                bad_frames.append(integration*self.n_group)
            bad_frames.append(((integration+1)*self.n_group)-1)
        self.bad_frames = bad_frames

        self.fwhm = self._get_psf_fwhm_px()


    def _make_stpsf_instrument(self):
        """
        Builds the stpsf instrument object for self.instrument/self.filter,
        setting the detector too where relevant (e.g. NIRCam, which has
        multiple SCAs with different pixel scales).
        """
        cls_name = self.stpsf_class.get(self.instrument)
        if cls_name is None:
            raise ValueError(f'No stpsf model mapping for instrument {self.instrument}')
        inst = getattr(stpsf, cls_name)()
        inst.filter = self.filter
        if self.instrument == 'NIRCAM' and self.detector:
            # FITS DETECTOR keyword uses NRCA/BLONG for the LW channel;
            # stpsf/webbpsf names that SCA NRCA/B5 instead.
            inst.detector = self.detector.replace('LONG', '5')
        return inst


    def _get_psf_fwhm_px(self):
        """
        Returns the PSF FWHM in pixels for self.filter. MIRI uses the
        pre-tabulated JDOX values (self.psf_fwhm_px); other instruments
        (e.g. NIRCam, whose FWHM varies strongly across ~30 filters and
        two pixel scales) measure it directly from an stpsf model instead
        of relying on a hand-maintained table.
        """
        if self.instrument == 'MIRI':
            if self.filter in self.psf_fwhm_px:
                return self.psf_fwhm_px[self.filter]
            raise ValueError(f'Unknown MIRI filter {self.filter}: no FWHM available')

        inst = self._make_stpsf_instrument()
        psf = inst.calc_psf(fov_pixels=41)
        fwhm = _measure_fwhm_px(psf[3].data)
        if np.isnan(fwhm):
            raise ValueError(f'Could not measure PSF FWHM for {self.instrument}/{self.filter}')
        return fwhm


    def _get_gen_mask(self, ny, nx):
        """
        Returns the (ny, nx) boolean science-pixel mask (True = good).
        MIRI uses the bundled full-frame bad-pixel mask, cropped to the
        subarray (SUBSTRT are 1-indexed FITS coords). Other instruments have
        no bundled mask, so fall back to a DQ-derived mask from the PIXELDQ
        extension (DO_NOT_USE bit, bit 0 of the JWST DQ flag scheme).
        """
        if self.instrument == 'MIRI':
            _mask_path = os.path.join(os.path.dirname(__file__), 'full_MIRI_mask.npy')
            full_mask = np.load(_mask_path)
            if full_mask.shape == (ny, nx):
                return full_mask
            r0 = self.substrt2 - 1
            c0 = self.substrt1 - 1
            return full_mask[r0:r0+ny, c0:c0+nx]

        return (self.dq_2d_arr & 1) == 0


    def ramp_correction(self,cube):
        """
        Uses rampdoctor to correct both the brighter-fatter effect
        and the reset switch charge decay effects.

        RCD (reset charge decay) is a physical effect specific to MIRI's
        Si:As detectors; NIRCam's HgCdTe detectors don't exhibit it, and BFE
        hasn't been separately characterized/validated for NIRCam with this
        tool yet, so this should only be called for MIRI (see __init__).
        """
        from rampdoctor import RampDoctor
        ny, nx = cube.shape[2], cube.shape[3]
        self.gen_mask = self._get_gen_mask(ny, nx)
        rd = RampDoctor(cube=cube,bg_mask=self.gen_mask,sci_mask=self.gen_mask,verbose=True)

        self.data_cor = rd.correct(diagnostics=True, charge_adaptive=False)


    def flux_calibrate(self,cube):
        """
        Calibrates the 4-dimensional ramp data (self.data)
        Using the information from the reference files, via the
        instrument-appropriate photom data model (e.g. MirImgPhotomModel,
        NrcImgPhotomModel).
        """
        photom_model_cls = { # instrument -> imaging PHOTOM datamodel class name
            'MIRI': 'MirImgPhotomModel',
            'NIRCAM': 'NrcImgPhotomModel',
        }

        photmjsr = None
        uncertainty = None
        filt = self.filter

        # photmjsr/pixar_sr depend only on (instrument, filter, pupil,
        # detector) -- NOT on the specific exposure -- unlike WCS
        # assignment, which genuinely does vary per exposure. Every
        # (filter, detector) combination in a real dataset repeats across
        # many exposures (e.g. all 4 dithers x however many epochs of the
        # same filter/detector), so a live CRDS round-trip on every single
        # one is pure repeat work -- and, given CRDS's own reliability
        # problems (see snake/CONTEXT.md item 15), pure repeat RISK.
        # Cache successful resolutions locally and check that first.
        import json
        cache_path = os.path.join(os.path.dirname(__file__), 'photmjsr_cache.json')
        cache_key = f"{self.instrument}_{self.filter}_{self.pupil}_{getattr(self, 'detector', None)}"
        cache = {}
        if os.path.exists(cache_path):
            try:
                with open(cache_path) as f:
                    cache = json.load(f)
            except Exception:
                cache = {}

        if cache_key in cache:
            entry = cache[cache_key]
            photmjsr = entry['photmjsr']
            uncertainty = entry['uncertainty']
            if getattr(self, 'pixar_sr', None) is None:
                self.pixar_sr = entry.get('pixar_sr')
            filt = entry.get('filt', filt)
            print(f"Using cached PHOTOM value for {cache_key} (no CRDS call)")
        else:
            try:
                from jwst import datamodels
                from stpipe import crds_client

                with datamodels.open(self.stage1_filepath) as model:
                    crds_params = model.get_crds_parameters()
                    filt = model.meta.instrument.filter
                    pupil = model.meta.instrument.pupil

                photom_file = crds_client.get_reference_file(crds_params, 'photom', 'jwst')
                print(f"Using PHOTOM ref: {photom_file}")

                # PIXAR_SR (pixel solid angle, sr) is normally only picked up
                # from a Stage2 cal.fits product (see __init__), which this
                # ramp/rate-only pipeline usually doesn't have -- leaving
                # self.pixar_sr permanently None and silently blocking the
                # MJy/uJy light-curve unit options in plot_detection. The
                # PHOTOM reference file itself already carries the same
                # keyword in its primary header, so fall back to that when
                # Stage2 wasn't available.
                if getattr(self, 'pixar_sr', None) is None:
                    with fits.open(photom_file) as phot_hdul:
                        self.pixar_sr = phot_hdul[0].header.get('PIXAR_SR', None)

                model_cls_name = photom_model_cls.get(self.instrument)
                if model_cls_name is None:
                    raise ValueError(f'No PHOTOM datamodel mapping for instrument {self.instrument}')

                with getattr(datamodels, model_cls_name)(photom_file) as phot:
                    table = phot.phot_table
                    row_mask = table['filter'] == filt
                    if 'pupil' in table.dtype.names and pupil is not None:
                        row_mask = row_mask & (table['pupil'] == pupil)
                    row = table[row_mask]
                    photmjsr = float(row['photmjsr'][0])
                    uncertainty = float(row['uncertainty'][0])

                cache[cache_key] = dict(photmjsr=photmjsr, uncertainty=uncertainty,
                                        pixar_sr=getattr(self, 'pixar_sr', None), filt=filt)
                try:
                    with open(cache_path, 'w') as f:
                        json.dump(cache, f, indent=2)
                except Exception:
                    pass
            except Exception as e:
                print(f"CRDS flux cal failed ({e})", end='')
                if self.instrument == 'MIRI':
                    print(" — falling back to miri_photom.csv")
                else:
                    print(f" — no bundled fallback table for {self.instrument}")
                    raise

        if photmjsr is None:
            csv_path = os.path.join(os.path.dirname(__file__), 'miri_photom.csv')
            phot_df = pd.read_csv(csv_path)
            mask = (phot_df['filter'] == filt) & (phot_df['subarray'] == self.subarray)
            if not mask.any():
                mask = phot_df['filter'] == filt
            row = phot_df[mask].iloc[0]
            photmjsr = float(row['photmjsr'])
            uncertainty = float(row['uncertainty'])

        print(f"filter={filt}  PHOTMJSR={photmjsr:.4f} MJy/sr per DN/s  +/- {uncertainty:.4f}")
        self.flux_conv = photmjsr
        self.flux_uncert = uncertainty

        # Not applied here — self.rampy_cube stays in raw DN and feeds
        # mega_inator / _cube_gradient / clean_cube unscaled. Only the final
        # gradient image (self.clean_cube) is flux-calibrated, in
        # _flux_calibrate(), to avoid double-calibrating and to keep frame 0
        # of every integration a legitimate DN value for mega_inator's
        # zero-point/extrapolation arithmetic.
        self.flux_data = cube


    def _make_cubes(self):
        """
        makes cube from 4d uncal file, also jump detected cube
        """
        # DN cube (for saturation masking)
        ramps_dn = np.array_split(self.data_cor, self.n_int, axis=0)
        self.rampy_cube_dn = np.squeeze(np.concatenate(ramps_dn, axis=1))

        # MJy/sr cube (for science)
        ramps_flux = np.array_split(self.flux_data, self.n_int, axis=0)
        self.rampy_cube = np.squeeze(np.concatenate(ramps_flux, axis=1))

        # make reference cube for jumps detected with calwebb_detector1
        dq_ints = np.array_split(self.dq_3d_arr,len(self.dq_3d_arr),axis=0)
        dq_cube = np.concatenate(dq_ints,axis=1)
        self.dq_cube = np.squeeze(dq_cube) # bitwise cube with all the dq flags

        self.jump_cube = (self.dq_cube & 4) == 4
        

    def _circle_app(self,rad):
        """
        Makes a kinda circular aperture, probably not worth using. - from ryan
        """
        mask = np.zeros((int(rad*2+.5)+1, int(rad*2+.5)+1))
        c = rad
        x,y = np.where(mask==0)
        dist = np.sqrt((x-c)**2 + (y-c)**2)

        ind = (dist) < rad + .2
        mask[y[ind],x[ind]] = 1

        return mask
    

    def _mask_pixels(self,threshold = 45000,mask_jump_det=False): # could be udated w/ quality flags from JWST
        """
        returns a list of tuples that are pixel (row,col) coordinates
        that have masked out the non-science and saturated pixels
        threshold used to be 47000 but that let things pass through that we didn't want

        MIRI uses the calibrated DN threshold above (validated on MIRI data).
        Other instruments have no such calibrated threshold, so they use the
        SATURATED DQ flag (bit 1) from calwebb_detector1's own saturation
        step instead.

        mask_jump_det : bool, default False. When True, also excludes any
            pixel flagged JUMP_DET (cosmic ray, bit 2) in ANY group of the
            ramp (self.jump_cube, built in _make_cubes). Deliberately OFF
            by default for the DETECTION stage: JUMP_DET fires on any
            sudden ramp discontinuity, and has no way to distinguish a
            cosmic ray from a genuine fast transient turning on or off
            within one group -- exactly the signal this pipeline exists
            to find. Masking it here would silently discard real
            candidates before they ever reach the shape/persistence/S-N
            filters that are actually designed to make that call.
            Verified directly on a real case (snake arc region LW,
            dither 00003, pixel (769,815)): JUMP_DET flagged at 2
            consecutive groups with a DECAYING amplitude (+2084 DN then
            +1284 DN, ~62% falloff) -- consistent with HgCdTe charge-trap
            persistence/afterglow following one cosmic-ray hit, not two
            independent events, but this was determined by inspecting the
            decay shape, not something a blanket pixel mask can know.
            Still useful in a narrower, targeted context: forced
            photometry at an ALREADY-CONFIRMED candidate position, to
            protect against a nearby (not on-source) contaminating cosmic
            ray -- see _build_channel_lightcurve_at_xy's per-object
            on-source check, which only applies masking when the flagged
            pixel does NOT overlap the source's own aperture.
        """
        # load general mask (bad pixels / non-science)
        mask = self.gen_mask

        # mask out saturated pixels
        if self.instrument == 'MIRI':
            mask_sat = self.rampy_cube_dn[-1] < threshold
        else:
            mask_sat = (self.dq_cube[-1] & 2) == 0  # SATURATED = bit 1
            if mask_jump_det:
                mask_sat = mask_sat & ~self.jump_cube.any(axis=0)
        mask_sat = mask_sat.astype(int) # to convolve with aperture

        kernel = self._circle_app(10)

        mask_sat = convolve_fft(mask_sat, kernel)
        mask_sat = mask_sat >= 0.99 # boolean

        # creating a list of tuples which are the (row,column) coords of each science pixel
        rows = list(range(self.rampy_cube.shape[1]))
        cols = list(range(self.rampy_cube.shape[2]))

        pixels = []

        for i in rows:
            row_num = [i] * len(cols)
            pixel_row = list(zip(row_num,cols)) # tuples of a single row's (i's) pixel coordinates
            pixels.extend(pixel_row)

        if self.subarray == 'FULL':
            self.mask_tot = mask_sat & mask
            if self.no_sat_mask:
                self.mask_tot = mask
        else:
            self.mask_tot = mask_sat # need to add option here

        nan_mask = self.mask_tot * 1.0 
        nan_mask[nan_mask < 1] = np.nan
        self.nan_mask = nan_mask

        pixel_mask = self.mask_tot.flatten(order='C').tolist() # flattening mask to make same size/dimensions as the list of pixel coords
        self.masked_pixels = [pixel for pixel, m in zip(pixels, pixel_mask) if m]


    def _pixel_integration(self,cube,int_num,row,col):
        """
        Gets the x and y data of a specified integration for a specific pixel in a specified cube
        """
        integration_length = list(range(0,self.n_group))
        x_dat = [i + (int_num)*self.n_group for i in integration_length]
    
        ramp = []
        for x in x_dat:
            ramp.append(cube[x][row][col])

        return x_dat, ramp   


    def parallel_fit_df(self,cube,save_df=False):
        """
        fits all ramps of specified cube parallely
        """
        fitting = Parallel(n_jobs=self.num_cores, verbose=0)(
            delayed(linear_fitting)(pixel,cube,self.n_int,self.n_group) for pixel in self.masked_pixels)

        obj_df = pd.DataFrame(fitting, columns=["row","col","gradients","intercepts","residuals"])
        obj_df['max_residual'] = obj_df['residuals'].apply(max)
        obj_df['mean_residual'] = obj_df['residuals'].apply(np.mean)

        self.obj_df = obj_df
        if save_df:
            filepath = os.path.join(self.obs_dir, 'ramp_fittings.csv')
            self.obj_df.to_csv(filepath, index=False)


    def _line(self,m,c,x):
        """
        straight line eqn
        """
        return [i*m + c for i in x]


    def _check_jump(self,coords):
        """
        checking if jump was detected in the dq cube
        """
        row, col = coords
        # check through all frames for jumps  
        vals = self.jump_cube[:, row, col]
        if vals.any():
            return 1, int(np.argmax(vals))  # 1 and first z index
        else:
            return 0, None


# --------------------- Image Search -----------------------


    def mega_inator(self,cube):
        """
        makes a mega cube out of a rampy one
        """
        ng = self.n_group
        ni = self.n_int
        mega_cube = np.zeros((self.n_frame, cube.shape[1], cube.shape[2]))

        # Integration 0: zero relative to its first frame
        mega_cube[:ng] = cube[:ng] - cube[0]

        # Subsequent integrations: extrapolate the ramp value at the boundary
        for i in range(1, ni):
            i0 = i * ng
            difference = mega_cube[i0-2] + 2*(mega_cube[i0-2] - mega_cube[i0-3])
            mega_cube[i0:i0+ng] = cube[i0:i0+ng] - cube[i0] + difference

        # mask first and last frame of each integration
        bad = [i * ng for i in range(ni)] + [(i+1) * ng - 1 for i in range(ni)]
        mega_cube_masked = mega_cube.copy()
        mega_cube_masked[bad] = np.nan

        self.mega_cube = mega_cube
        self.mega_cube_masked = mega_cube_masked


    def _cube_gradient(self,cube,save=None):
        """
        make a gradient cube with the fakey fake frames for mega method
        for ramp method just takes the gradient then masks out bad frames

        Uses a forward difference (grad[i] = cube[i+1] - cube[i]), not
        np.gradient's centered difference. np.gradient approximates a
        continuous derivative from samples — for discrete up-the-ramp reads
        it has a real cost: a single-group event (a cosmic ray, a genuine
        one-group jump) gets smeared across *two* output frames, since the
        centered stencil at index i pulls in both cube[i-1] and cube[i+1].
        That makes a single-frame event look like two consecutive frames of
        signal, which is misleading for both visual inspection and any
        multi-frame persistence reasoning. A forward difference keeps a
        single-group event in exactly one output frame. The last frame has
        no cube[i+1] to diff against and is set to NaN — this is already
        always in self.bad_frames (the last frame of the last integration),
        so nothing already relied on it having a value.
        """
        if self.method == 'mega':
            fakeified_cube = cube.copy()
            vals = np.arange(1, self.n_int) * self.n_group
            if len(vals) > 0:
                fakeified_cube[vals - 1] = 2*fakeified_cube[vals - 2] - fakeified_cube[vals - 3]
                fakeified_cube[vals]     = 3*fakeified_cube[vals - 2] - 2*fakeified_cube[vals - 3]
            # Fakeify the absolute edge frames so the forward difference at
            # frame n_frame-2 doesn't need an out-of-range cube[n_frame], and
            # so small n_group (<=4 with n_int=1) still leaves valid frames.
            if self.n_group >= 3:
                fakeified_cube[0]  = 2*fakeified_cube[1]  - fakeified_cube[2]
                fakeified_cube[-1] = 2*fakeified_cube[-2] - fakeified_cube[-3]
            self.fakey_cube = fakeified_cube
            self.grad_cube = _forward_diff(fakeified_cube)

        if self.method == 'ramp':
            grad_cube = _forward_diff(cube)
            grad_cube[self.bad_frames] = np.nan
            self.grad_cube = grad_cube

        if save:
            filepath = os.path.join(self.obs_dir, "grad_cube.npy")
            np.save(filepath, self.grad_cube)


    def _reference_frame(self):
        """
        Making a reference frame but more complicated to counteract smearing
        of bright asteroids. - Armin's suggestion (starts with original, then masks)
        """
        if self.master_ref_path:
            own_median = self._own_median_reference()
            aligned = self._align_master_reference()
            self.first_ref_frame = _subpixel_align(aligned, own_median)
        else:
            self.first_ref_frame = self._own_median_reference()

        filepath = os.path.join(self.obs_dir, "ref_frame_1.npy")
        np.save(filepath, self.first_ref_frame)

    def _own_median_reference(self):
        """
        This exposure's own median-of-groups frame — jurassic's original
        per-exposure reference. Used directly when no master_ref_path is
        given, and otherwise as the registration target for subpixel-
        aligning the master reference onto this exposure (see
        _reference_frame / _subpixel_align): it's a real image of this
        exposure's own field, however transient-contaminated, so it's a
        valid target for cross-correlation even though it isn't trusted as
        the reference itself in that case.
        """
        # Use NaN fraction to make sure not a mostly NaN frame
        nan_fraction = np.isnan(self.grad_cube).reshape(self.n_frame, -1).mean(axis=1)
        not_nans = nan_fraction < 0.1
        not_nans[self.bad_frames] = False

        good_slices = self.grad_cube.copy()[not_nans]

        if len(good_slices) == 0:
            raise RuntimeError("No valid frames found for reference frame — "
                            "check grad_cube for all-NaN output.")

        return np.nanmedian(good_slices, axis=0)

    def _align_master_reference(self):
        """
        Reprojects the pre-built master reference (see master_ref_path in
        __init__) onto this exposure's own distortion-corrected WCS. rate.fits
        WCS alone (no assign_wcs) is only a crude TAN projection about the
        detector center, not accurate enough at pixel scale for this — see
        the reproject_to_ref/AssignWcsStep pattern already used for RGB
        alignment in snake/processing/plot_snake_rgb.py. Subpixel residuals
        left over after this (WCS/distortion-model imperfections, typically
        a fraction of a pixel to ~1 px) are corrected separately in
        _reference_frame via cross-correlation against this exposure's own
        median frame — otherwise they show up as a spurious PSF-shaped
        dipole in every differenced frame.
        """
        from astropy.wcs import WCS
        from reproject import reproject_interp
        from jwst.assign_wcs import AssignWcsStep

        rate_path = self.file.replace('_ramp.fits', '_rate.fits')
        model = AssignWcsStep.call(rate_path)
        target_wcs = model.meta.wcs
        target_shape = model.data.shape

        with fits.open(self.master_ref_path) as h:
            ref_data = h['SCI'].data
            ref_wcs = WCS(h['SCI'].header)

        aligned, _ = reproject_interp((ref_data, ref_wcs), target_wcs, shape_out=target_shape)
        return aligned  # NaN outside the master ref's footprint kept as-is; see _subpixel_align


    def _cube_differenced(self,cube,reference,save=None,first=None):
        """
        make a differenced cube from gradient cube using a median frame as reference
        """
        diff_cube = cube.copy() - reference[np.newaxis,:,:]
        diff_cube[self.bad_frames] = np.nan

        self.diff_cube = diff_cube
        self.diff_cube_masked = self.diff_cube.copy() * self.mask_tot

        if save:
            filepath = os.path.join(self.obs_dir, "diff_cube.npy")
            np.save(filepath, self.diff_cube)

        if first:
            filepath = os.path.join(self.obs_dir, "diff_cube_1.npy")
            np.save(filepath, self.diff_cube)


    def _psf_kernel(self):
        """
        creates kernel based on filter using stpsf; size scales with PSF FWHM
        """
        fwhm = self.fwhm
        size = max(11, int(round(6 * fwhm)) | 1)  # odd, at least 11, ~3 FWHM radius
        inst = self._make_stpsf_instrument()
        psf = inst.calc_psf(fov_pixels=size)
        self.kernel = psf[3].data

    
    def _check_persistence(self, cube, df, sigma_thresh=5.0):
        """
        NIRCam-only temporal-persistence check, added as a replacement
        signal for lacosmic's cr_flagged veto (disabled for NIRCam in
        source_extracting — see its docstring). Used as a HARD FILTER:
        a candidate must show elevated signal in at least one adjacent
        frame (previous or next) to survive — single-frame-only
        detections are dropped, full stop, per explicit direction (no
        exception carved out for a hypothetical single-group-only real
        transient; that protection was tried as a reason to keep this
        informational-only and was judged not worth the false-accept
        cost it left in place).

        Physical basis: a cosmic-ray hit deposits charge in one group
        read (an instantaneous step in the cumulative ramp), so it
        elevates exactly the one group-to-group difference spanning that
        group — both the difference before and after look normal, since
        the added charge appears in both halves of those differences and
        cancels. NIRCam's HgCdTe detectors don't have MIRI's RCD (reset
        charge decay), so there's no known mechanism for that spike to
        persist. A genuine sustained brightening (this pipeline's actual
        science target — a microlensing event should stay bright for the
        rest of the exposure, far longer than one ~107s group here)
        instead adds a roughly constant excess to every subsequent
        group-to-group difference for as long as it stays bright, so it
        should show up as elevated in an adjacent independent frame too,
        not just the one it was detected in. Checking both directions
        (not just forward) means a detection in the last usable frame is
        still confirmable via the previous frame.

        Verified by injection-recovery + real-CR testing on this data:
        an injected sustained brightening shows up in >=2 consecutive
        frames ~100% of the time at S/N>=8 (any onset group, including
        the very first usable frame — confirmed via direct serial
        reproduction of the injection test after finding the original
        parallel version had raced multiple worker processes against a
        shared scratch directory, corrupting exactly the onset=0 case);
        real JUMP_DET-flagged cosmic rays show elevated signal in the
        adjacent frame only ~16% of the time.
        """
        if df.empty:
            df = df.copy()
            df['persists_next_frame'] = pd.Series(dtype=bool)
            df['persists_prev_frame'] = pd.Series(dtype=bool)
            df['persists'] = pd.Series(dtype=bool)
            return df

        frame_stats = {}

        def _elevated(frame_idx, x, y):
            if frame_idx < 0 or frame_idx >= cube.shape[0] or not np.isfinite(cube[frame_idx]).any():
                return False
            if frame_idx not in frame_stats:
                frame_stats[frame_idx] = sigma_clipped_stats(cube[frame_idx], sigma=3.0)
            _, med, std = frame_stats[frame_idx]
            if not std or np.isnan(std):
                return False
            xi, yi = int(round(x)), int(round(y))
            if not (0 <= yi < cube.shape[1] and 0 <= xi < cube.shape[2]):
                return False
            val = cube[frame_idx, yi, xi]
            return bool(np.isfinite(val) and (val - med) / std > sigma_thresh)

        persists_next, persists_prev = [], []
        for _, row in df.iterrows():
            frame = int(row['frame'])
            persists_next.append(_elevated(frame + 1, row['x'], row['y']))
            persists_prev.append(_elevated(frame - 1, row['x'], row['y']))

        df = df.copy()
        df['persists_next_frame'] = persists_next
        df['persists_prev_frame'] = persists_prev
        df['persists'] = df['persists_next_frame'] | df['persists_prev_frame']
        return df


    def source_extracting(self,cube,save_plot,save_csv):
        """
        using source extractor (sep) instead of StarFinder
        """
        psf_fwhm = getattr(self, 'fwhm', None)
        # lacosmic's mask only, NOT jump_cube: JUMP_DET is a pure ramp-level
        # statistical discontinuity test with no way to distinguish a cosmic
        # ray from a real fast transient brightening -- using it here would
        # systematically reject genuine discoveries. lacosmic instead tests
        # spatial PSF-consistency (is this sharp/unresolved vs. properly
        # optics-broadened), which a real transient satisfies regardless of
        # its time behavior, so it doesn't have that failure mode (verified
        # by injection testing — see cosmic_ray_shapes/).
        #
        # That was verified on MIRI only. Injection-recovery testing on
        # this NIRCam data found lacosmic's Laplacian sharpness test
        # doesn't hold here: NIRCam's PSF is compact enough (FWHM as low
        # as ~1.2 px) that a bright real point source is itself about as
        # "sharp" as a cosmic-ray hit, and the veto rejected ~100% of
        # real injected sources at every S/N tested, for every one of the
        # 10 filters in this dataset. So for NIRCam the veto is disabled
        # entirely; _check_persistence below is used instead (see its
        # docstring).
        cr_mask = getattr(self, 'cr_mask_cube', None) if self.instrument == 'MIRI' else None

        # Stage 1: cheap per-frame extraction (background + sep.extract),
        # parallel across frames.
        extract_tasks = (delayed(_sep_extract)(frame, cube[frame], self.kernel, self.mask_tot, self.n_group)
                          for frame in range(self.n_frame))
        extract_results = Parallel(n_jobs=self.num_cores, prefer="processes")(extract_tasks)

        # Stage 2: PSF-fitting -- the expensive part (~1ms/object, but up
        # to several thousand raw SEP objects per frame). Flattened across
        # every frame's objects and chunked (not one task per object:
        # data_sub is a full frame, several MB, and passing it to
        # thousands of individual single-object tasks would mean
        # thousands of copies pickled to worker processes) into one
        # Parallel call, so the full core count is used regardless of how
        # many usable frames there are -- as few as 3 for a short NIRCam
        # ramp, well under num_cores, which left most cores idle under
        # the old frame-only parallelism while the single largest frame
        # became the bottleneck.
        CHUNK_SIZE = 300
        psf_tasks = []
        for fi, (obj_df, data_sub, bkg_rms) in enumerate(extract_results):
            if len(obj_df) == 0:
                continue
            idx_arr = obj_df.index.to_numpy()
            xs, ys = obj_df['x'].to_numpy(), obj_df['y'].to_numpy()
            for start in range(0, len(idx_arr), CHUNK_SIZE):
                chunk_idx = idx_arr[start:start + CHUNK_SIZE]
                chunk_xy = list(zip(xs[start:start + CHUNK_SIZE], ys[start:start + CHUNK_SIZE]))
                psf_tasks.append(delayed(_psf_fit_batch)(fi, data_sub, chunk_idx, chunk_xy, self.kernel))

        psf_batches = Parallel(n_jobs=self.num_cores, prefer="processes")(psf_tasks) if psf_tasks else []

        for obj_df, _, _ in extract_results:
            if len(obj_df) > 0:
                for col in ['psf_like', 'x_psf', 'y_psf', 'elongation_excess', 'symmetry180', 'psf_flux', 'psf_flux_err']:
                    obj_df[col] = np.nan
        for batch in psf_batches:
            for owner, idx, psf_like, x_psf, y_psf, elong_excess, sym180, psf_flux, psf_flux_err in batch:
                obj_df = extract_results[owner][0]
                obj_df.loc[idx, ['psf_like', 'x_psf', 'y_psf', 'elongation_excess', 'symmetry180', 'psf_flux', 'psf_flux_err']] = \
                    [psf_like, x_psf, y_psf, elong_excess, sym180, psf_flux, psf_flux_err]

        # Stage 3: aperture photometry, npix_ratio, filtering, plotting --
        # cheap, parallel across frames same as the old single-stage version.
        finish_tasks = (delayed(_finish_sep)(fi, obj_df, data_sub, bkg_rms, self.mask_tot, save_plot,
                                              self.obs_dir, psf_fwhm, cr_mask=cr_mask)
                         for fi, (obj_df, data_sub, bkg_rms) in enumerate(extract_results))
        results = Parallel(n_jobs=self.num_cores, prefer="processes")(finish_tasks)
        obj_dfs, filtered_dfs = zip(*results)

        # keep only non empty dfs
        non_empty_obj = [df for df in obj_dfs if not df.empty]
        non_empty_filt = [df for df in filtered_dfs if not df.empty]

        # total sep detections
        if len(non_empty_obj) == 0:
            self.total_df = pd.DataFrame(columns=obj_dfs[0].columns)
        else:
            self.total_df = pd.concat(non_empty_obj, ignore_index=True)

        # filtered SEP detections
        if len(non_empty_filt) == 0:
            self.filtered_sep_df = pd.DataFrame(columns=filtered_dfs[0].columns)
        else:
            self.filtered_sep_df = pd.concat(non_empty_filt, ignore_index=True)

        if self.instrument == 'NIRCAM':
            self.filtered_sep_df = self._check_persistence(cube, self.filtered_sep_df)
            # Hard filter: a candidate seen in only one frame is dropped,
            # no exceptions -- see _check_persistence docstring.
            self.filtered_sep_df = self.filtered_sep_df[
                self.filtered_sep_df['persists']
            ].reset_index(drop=True)

            # Hard filter: require psf_flux/psf_flux_err >= self.snr_thresh
            # (default 8) -- see injection-recovery testing (snake/
            # processing/injection_recovery_thresh8.py): 8 is the
            # effective S/N floor the persistence filter above was
            # already calibrated against, made explicit rather than left
            # implicit. Same numeric cut for both LW and SW -- SW's more
            # compact PSF means this corresponds to a fainter physical
            # depth there, but SW is used for LW-candidate confirmation,
            # not primary detection, so that asymmetry is fine.
            # self.snr_thresh is adjustable (default 8) for exploratory
            # lower-threshold reruns -- shape (psf_like/npix_ratio) and
            # persistence stay at their own validated defaults regardless.
            snr = self.filtered_sep_df['psf_flux'] / self.filtered_sep_df['psf_flux_err']
            self.filtered_sep_df = self.filtered_sep_df[
                snr.notna() & (snr >= self.snr_thresh)
            ].reset_index(drop=True)

        # printing detection stats
        print(f"SEP: {len(non_empty_filt)} / {self.n_frame} frames with filtered detections "
              f"({len(self.filtered_sep_df)} total sources)")

        # save csv's of the filtered and unfiltered dfs
        if save_csv:
            if len(self.filtered_sep_df) > 0:
                filepath = os.path.join(self.obs_dir, "filtered_sources.csv")
                self.filtered_sep_df.to_csv(filepath, index=False)

            filepath = os.path.join(self.obs_dir, "all_sources.csv")
            self.total_df.to_csv(filepath, index=False)


    def _masked_reference(self,mask_correction,mask_radius=10,max_gap_frames=20):
        """
        Makes a reference frame (median) but masks out any variable sources.
        Masks detected source positions and takes nanmedian of remaining pixels.

        Two failure modes let a source leak into its own "background"
        reference, discovered investigating a slow-moving (~0.02 px/frame)
        bright MIRI asteroid that left a real, ~10+ sigma positive trace in
        ref_frame_2 along its own track, which then self-subtracted into a
        spurious negative dip whenever a given frame's true flux fell below
        that contaminated reference:

        1. mask_radius=10 px is smaller than the PSF model's own assumed
           extent (jurassic's WebbPSF kernel stamp half-size, ~6xFWHM/2 —
           13 px for F1500W). Masking a circle smaller than the PSF
           template itself guarantees real wing flux lands just outside
           the mask on every frame, biasing the reference high right where
           the source sits. Fixed by flooring mask_radius at the kernel's
           own half-size when self.kernel is available.
        2. The mask only covers frames where SEP actually reported a
           detection that frame. A frame where the source's per-frame
           significance dips below threshold — including, self-reinforcingly,
           frames already suffering from this exact contamination — gets
           no mask at all, so its source-containing pixels flow straight
           into the median uncorrected. Fixed by running this table through
           the pipeline's own trajectory linker (_spatial_group +
           _tag_asteroids) and, only for frames inside an asteroid-tagged
           track's own span, filling gaps of up to max_gap_frames by linear
           interpolation along that specific track's fitted trajectory.
           Scoping the fill to a single linked, classified track (rather
           than interpolating blindly between whatever SEP found in the
           nearest bracketing frames) avoids bridging two unrelated sources
           if the field has more than one — a real risk this early in the
           pipeline, since this runs on the first-pass, per-frame catalog,
           before the final grouping/classification later in the pipeline.
        """
        def _use_first_ref():
            self.second_ref_frame = self.first_ref_frame.copy()
            np.save(os.path.join(self.obs_dir, "ref_frame_2.npy"), self.second_ref_frame)

        if self.filtered_sep_df.empty:
            _use_first_ref()
            return

        if mask_correction == False:
            _use_first_ref()
            return

        if getattr(self, 'master_ref_path', None):
            # This whole self-masking rebuild exists to stop a source from
            # contaminating its own reference when that reference is a
            # median of THIS exposure's own 3-7 frames -- easy for one
            # bright, persistent source to bias. A master reference
            # (median across 4 independent dither epochs, each
            # contributing only ~1/4 weight) doesn't have that problem
            # structurally, so masking here would just be discarding the
            # master ref and rebuilding the old self-referenced median
            # from grad_cube instead -- exactly backwards. Use the
            # (already aligned) master reference as-is.
            _use_first_ref()
            return

        # mask_radius derives from the actual PSF kernel size for this
        # instrument/filter/detector (self.kernel already scales with
        # FWHM per _psf_kernel -- as low as ~11px for NIRCam's most
        # compact SW filters, much larger for MIRI or NIRCam LW), rather
        # than a fixed default floored by it -- a single hardcoded radius
        # would under-mask a broad PSF or over-mask a compact one.
        kernel_obj = getattr(self, 'kernel', None)
        fwhm = getattr(self, 'fwhm', None)
        if kernel_obj is not None:
            mask_radius = kernel_obj.shape[0] // 2
        elif fwhm is not None:
            mask_radius = max(mask_radius, int(np.ceil(3 * fwhm)))

        # gap-fill positions: only a fallback for frames with NO real
        # detection at all, so there's no multi-source ambiguity to resolve
        # here -- the mean per detected frame is just the interpolation
        # endpoint, not a replacement for that frame's own (possibly
        # multi-row) mask below.
        #
        # Scoped to single linked tracks: run the pipeline's own DBSCAN
        # spatial linker + trajectory classifier on this frame's detections
        # (using 'frame' as a monotonic stand-in for 'mjd' -- _tag_asteroids
        # only needs relative spacing for the linear fit, and real mjd
        # isn't assigned yet at this point in the pipeline) so a gap only
        # gets bridged when both bracketing detections have already been
        # identified, by that same linking logic, as the same moving object.
        det = self.filtered_sep_df
        linked = self._spatial_group(det[['x', 'y', 'frame']].copy())
        linked['mjd'] = linked['frame'].astype(float)
        tagged = self._tag_asteroids(linked)

        # Only confirmed moving objects (asteroid_id > 0) mask the
        # reference -- a static candidate (this survey's actual target:
        # a microlensing event doesn't move) should NOT have its own
        # position excluded from the per-pixel median just for having
        # been detected. Masking every detection indiscriminately would
        # also, given how crowded NIRCam's raw catalog is with lacosmic
        # disabled, poke far more holes in the reference than necessary.
        asteroid_df = tagged[tagged['asteroid_id'] > 0]

        gap_fill = {}  # frame -> (x, y), only for frames absent from an asteroid track
        for ast_id in sorted(tagged.loc[tagged['asteroid_id'] > 0, 'asteroid_id'].unique()):
            track = tagged[tagged['asteroid_id'] == ast_id].sort_values('frame')
            track_frames = track['frame'].values
            track_x = track['x'].values
            track_y = track['y'].values
            for f0, x0, y0, f1, x1, y1 in zip(track_frames[:-1], track_x[:-1], track_y[:-1],
                                               track_frames[1:], track_x[1:], track_y[1:]):
                gap = f1 - f0
                if 1 < gap <= max_gap_frames:
                    for f in range(f0 + 1, f1):
                        t = (f - f0) / gap
                        gap_fill[f] = (x0 + t * (x1 - x0), y0 + t * (y1 - y0))

        # source masks: real per-frame detections keep their original,
        # possibly-multi-source handling; frames with no detection at all
        # fall back to the interpolated position, so a transient
        # non-detection (including one caused by this same self-subtraction
        # effect) no longer leaves an unmasked window
        reference_cube = np.zeros_like(self.grad_cube)
        kernel = self._circle_app(mask_radius)

        for frame in self.frames:
            has_detection = frame in asteroid_df['frame'].values
            has_fill = frame in gap_fill
            if not (has_detection or has_fill):
                continue

            mask = np.zeros_like(self.grad_cube[0])

            if has_detection:
                frame_df = asteroid_df[asteroid_df['frame'] == frame]
                x_int = [round(x) for x in frame_df['x'].values]
                y_int = [round(y) for y in frame_df['y'].values]
            else:
                gx, gy = gap_fill[frame]
                x_int, y_int = [round(gx)], [round(gy)]

            for i in range(len(x_int)):
                mask[y_int[i], x_int[i]] = 1

            reference_cube[frame] = convolve_fft(mask, kernel)

        source_mask = reference_cube >= 0.00001  # boolean: True = source pixel to exclude
        self.source_mask = source_mask

        # Only use good frames (not bad/all-NaN)
        nan_fraction = np.isnan(self.grad_cube).reshape(self.n_frame, -1).mean(axis=1)
        not_nans = nan_fraction < 0.1
        not_nans[self.bad_frames] = False

        good_slices = self.grad_cube.copy()[not_nans]
        mask_slices = self.source_mask[not_nans]

        # NaN out source pixels, then make reference from median
        masked_slices = np.where(mask_slices, np.nan, good_slices)
        self.second_ref_frame = np.nanmedian(masked_slices, axis=0)

        # A pixel goes NaN here only if the source's mask covered it in
        # every single valid frame -- true whenever the source's own
        # trajectory over the segment is smaller than the mask footprint
        # (exactly this slow-moving asteroid: ~12-16 px of total drift
        # across the segment vs a ~26 px mask diameter), so there's no time
        # in the whole segment this pixel is ever clear. Left as NaN, that
        # hole poisons diff_cube = grad_cube - reference at that exact
        # detector position for every frame in the segment, not just some
        # -- silently erasing the source's own detectability across
        # whatever portion of the track sits deep enough in its own mask
        # footprint to starve every frame (this is what turned a modest,
        # ~10 sigma reference contamination into a ~280-frame dead zone
        # with zero detections, found testing on a single segment before
        # running the full 7-segment set). Fill any such holes from the
        # local background via Gaussian interpolation instead of leaving
        # them empty -- a locally-smooth fallback beats an outright hole,
        # even though it's a coarser estimate than a real temporal median.
        nan_holes = np.isnan(self.second_ref_frame)
        if nan_holes.any():
            from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
            # interpolate_replace_nans defaults to a real-space convolution,
            # whose cost scales with kernel area (O(N*K^2)) -- fine for a
            # small kernel, but with only a handful of scattered NaN
            # holes needing an 81x81 kernel to bridge (this pipeline's
            # actual case: ~1600 holes, 0.04% of the frame), that's ~40s
            # spent convolving the other 99.96% of the image for no
            # reason. convolve_fft's cost is O(N log N) regardless of
            # kernel size, so it doesn't care how large the kernel needs
            # to get to close the holes — same result, ~150x faster on
            # this data (37.5s -> 0.26s, confirmed identical output).
            filled = self.second_ref_frame.copy()
            stddev = mask_radius
            for _ in range(4):  # widen the kernel until every hole is bridged
                fill_kernel = Gaussian2DKernel(x_stddev=stddev)
                filled = interpolate_replace_nans(filled, fill_kernel, convolve=convolve_fft)
                if not np.isnan(filled).any():
                    break
                stddev *= 2
            self.second_ref_frame = filled

        filepath = os.path.join(self.obs_dir, "ref_frame_2.npy")
        np.save(filepath, self.second_ref_frame)


    def _remove_cosmic(self,cube):
        """
        uses lacosmic to remove the cosmic rays in each frame

        Not run for NIRCam at all — confirmed by injection testing that
        lacosmic's cleaning actively erases real injected source flux via
        interpolation (a real S/N=50 source retained only ~16% of its
        aperture flux, ~1.6% of its peak pixel), independent of whether
        its flag is used as a hard veto (already disabled for NIRCam in
        source_extracting). Two earlier attempts to keep some form of
        this step (skip entirely with no replacement; loosen its
        contrast/cr_threshold/neighbor_threshold) were reverted after
        each caused the second SEP pass's candidate count to explode
        (27->2961, then 27->4194) — but erasing real signal isn't an
        acceptable tradeoff to avoid that regardless of the cost, so
        lacosmic is off for NIRCam and the resulting candidate-count
        problem is handled downstream instead (see
        _check_persistence / source_extracting) rather than by keeping a
        tool known to destroy real flux around as a noise-suppression
        crutch. clean_cube = cube unchanged, cr_mask_cube = all-False,
        for NIRCam.
        """
        if self.instrument != 'MIRI':
            self.clean_cube = cube.copy()
            self.cr_mask_cube = np.zeros(cube.shape, dtype=bool)
            filepath = os.path.join(self.obs_dir, "clean_cube.npy")
            np.save(filepath, self.clean_cube)
            return

        # run lacosmic on each frame in parallel
        results = Parallel(n_jobs=self.num_cores,verbose=0)(delayed(run_lacosmic)(cube[i],self.mask_tot) for i in range(len(cube)))
        clean_cube = np.array([r[0] for r in results])
        cr_mask_cube = np.array([r[1] for r in results])

        # run_lacosmic zeroes NaN input (lacosmic can't accept NaN) and never
        # restores it, so the first/last frame of every integration — masked
        # NaN upstream in diff_cube — would otherwise silently become 0 here.
        clean_cube[self.bad_frames] = np.nan

        self.cr_mask_cube = cr_mask_cube
        self.clean_cube = clean_cube

        filepath = os.path.join(self.obs_dir, "clean_cube.npy")
        np.save(filepath, self.clean_cube)


    def _make_ref_cr_mask(self):
        """
        makes a cosmic ray mask that is a union of the lacosmic cr_mask
        and the JWST pipeline jump detections from the dq array
        """
        ref_cr_mask = self.cr_mask_cube | self.jump_cube
        self.ref_cr_mask = ref_cr_mask


# --------------------- Significance Functions -----------------------


    def _cube_significance(self,magic_number=3):
        """
        making a significance cube - dividing the differenced cube by the
        standard deviation of the background of each frame
        Then making a cut based on a ~magic number~ which at this point is just 3
        """
        dat = np.where(self.mask_tot[None, :, :], self.clean_cube, np.nan)

        def _frame_stats(frame_data):
            _, med, std = sigma_clipped_stats(frame_data)
            return med, std

        stats = Parallel(n_jobs=self.num_cores)(
            delayed(_frame_stats)(dat[f]) for f in range(self.n_frame))
        meds = np.array([s[0] for s in stats])
        stds = np.array([s[1] for s in stats])

        sig_cube = (dat - meds[:, None, None]) / stds[:, None, None]
        sig_cube[self.ref_cr_mask] = 0

        self.sig_cube = sig_cube
        self.bool_sig_cube = sig_cube > magic_number


    def _cube_threshold(self,rad=2,threshold=9):
        """
        convolves the significance cube with a circle and identifies
        the bits above a threshold, above which should be psf-like sources
        and below are cosmic ray junk stuffs (ideally)
        """
        kernel = self._circle_app(rad)
        results = Parallel(n_jobs=self.num_cores)(
            delayed(convolve_fft)(self.bool_sig_cube[f], kernel, normalize_kernel=False)
            for f in range(self.n_frame))
        conv_sig_cube = np.array(results)
        self.conv_sig_cube = conv_sig_cube
        self.bool_threshold_cube = conv_sig_cube > threshold


    def _cube_rolling_sum(self,num_frames=4,threshold=3):
        """
        rolling sum over (num_frames) frames of threshold cube, cut for >= threshold
        to identify 'significant' flux changes
        """
        good_frames_cube = np.delete(self.bool_threshold_cube, self.bad_frames, axis=0)
        n_good = good_frames_cube.shape[0]
        rows = good_frames_cube.shape[1]
        cols = good_frames_cube.shape[2]

        rolling_sum_cube = np.zeros((n_good,rows,cols), dtype=int)

        for frame in range(n_good):
            rolling_sum_cube[frame] = np.sum(good_frames_cube[frame:frame+num_frames], axis=0)

        # make cut for >= threshold
        bool_rolling_sum_cube = rolling_sum_cube >= threshold

        # reinsert bad frames at their original positions by pre-allocating
        # the full arrays and index-assigning the good frame values
        n_total = self.bool_threshold_cube.shape[0]
        good_idx = [i for i in range(n_total) if i not in set(self.bad_frames)]

        rolling_sum_full = np.full((n_total, rows, cols), np.nan)
        bool_rolling_sum_full = np.zeros((n_total, rows, cols), dtype=bool)
        rolling_sum_full[good_idx] = rolling_sum_cube
        bool_rolling_sum_full[good_idx] = bool_rolling_sum_cube

        rolling_sum_cube = rolling_sum_full
        bool_rolling_sum_cube = bool_rolling_sum_full

        self.rolling_sum_cube = rolling_sum_cube
        self.bool_rolling_sum_cube = bool_rolling_sum_cube

        filepath = os.path.join(self.obs_dir, "rolling_sum_cube.npy")
        np.save(filepath, self.rolling_sum_cube)

    
    def _significance_output(self):
        """
        Making the output for the significance way of things
        For now makes (and saves?) a dataframe containing the pixel coords and frame
        where something has passed the multiple signicance thresholds.
        """
        frames,rows,cols = np.where(self.bool_rolling_sum_cube==True)
        data_dict = {'frame': frames,
                     'x': cols,
                     'y': rows}
        
        significance_df = pd.DataFrame(data_dict)
        self.significance_df = significance_df
        
        if len(self.significance_df) > 0:
            filepath = os.path.join(self.obs_dir, 'significance.csv')
            significance_df.to_csv(filepath,index=False)


# ------------------------------ Output stuff! ----------------------------
    

    def _spatial_group(self, df, min_samples=1, distance=1):
        """
        Groups events based on proximity w/ dbscan
        """
        if df.empty:
            df['objid'] = pd.Series(dtype=int)
            return df
        
        output = df.copy()

        pos = np.column_stack([output['x'].values, output['y'].values])
        cluster = DBSCAN(eps=distance, min_samples=min_samples, n_jobs=self.num_cores).fit(pos)
        labels = cluster.labels_
        objid = labels + 1
        objid[objid < 0] = 0 
        output['objid'] = objid.astype(int)

        return output
    

    def _temporal_group(self,df,min_samples=5,distance=2):
        """
        Groups events based on time w/ dbscan
        """
        if df.empty:
            return df

        output = pd.DataFrame()

        ids_list = sorted(df['objid'].unique())

        for id in ids_list:
            obj_df = df[df['objid']==id]
            # loop through each grouped object to find events
            if len(obj_df) >= 3:
                data = obj_df['frame'].values
                data = data.reshape(-1, 1)
                db = DBSCAN(eps=distance, min_samples=min_samples,n_jobs=self.num_cores).fit(data)
                labels = db.labels_
                obj_df['event'] = labels.astype(int)

            output = pd.concat([output, obj_df], ignore_index=True)

        # see if can clean up number of events    
        if len(output) > 0:
            filepath = os.path.join(self.obs_dir, f'ms-{min_samples}_d-{distance}_events.csv')
            output.to_csv(filepath,index=False)

        return output


    def asteroid_candidate(self, df, threshold_1=10, threshold_2=5, threshold_3=2):
        """
        Determines if in the grouped detections there are any potential asteroids.
        Uses trajectory-based classification from _tag_asteroids when available,
        otherwise falls back to displacement grading.
        """
        ids = []

        if 'asteroid_id' in df.columns and (df['asteroid_id'] > 0).any():
            for ast_id in sorted(df[df['asteroid_id'] > 0]['asteroid_id'].unique()):
                obj = df[df['asteroid_id'] == ast_id].sort_values('frame')
                objid = int(obj['objid'].iloc[0])
                x0, y0 = obj.iloc[0]['x'], obj.iloc[0]['y']
                ids.append(f'Asteroid ID {ast_id}, Object: {objid}, '
                           f'Start Coords: ({x0:.2f},{y0:.2f})')
            return len(ids), ids

        # Fallback: displacement grading
        num_candidates_1 = 0
        num_candidates_2 = 0
        num_candidates_3 = 0

        for id in range(1, df['objid'].max() + 1):
            df_obj = df[df['objid'] == id]
            if len(df_obj) < 2:
                continue
            try:
                idx_min = df_obj['frame'].idxmin()
                idx_max = df_obj['frame'].idxmax()
            except ValueError:
                continue
            row_min = df_obj.loc[idx_min]
            row_max = df_obj.loc[idx_max]
            dist = np.sqrt((row_max['x'] - row_min['x'])**2 +
                           (row_max['y'] - row_min['y'])**2)
            if dist > threshold_1:
                num_candidates_1 += 1
                ids.append(f'Grade 1, Object: {id}, Start Coords: ({row_min["x"]:.2f},{row_min["y"]:.2f})')
            elif dist > threshold_2:
                num_candidates_2 += 1
                ids.append(f'Grade 2, Object: {id}, Start Coords: ({row_min["x"]:.2f},{row_min["y"]:.2f})')
            elif dist > threshold_3:
                num_candidates_3 += 1
                ids.append(f'Grade 3, Object: {id}, Start Coords: ({row_min["x"]:.2f},{row_min["y"]:.2f})')

        return num_candidates_1 + num_candidates_2 + num_candidates_3, ids


    def _tag_asteroids(self, df, min_frames=5, min_displacement_px=2.0,
                       max_residual_px=3.0, link_eps_px=5.0, min_track_frames=10):
        """
        Classifies objects as asteroids by fitting a linear trajectory (x, y vs mjd).
        Objects with significant displacement and low trajectory residuals are flagged.
        Nearby objects that fall on the same trajectory are linked with a shared asteroid_id.
        Adds 'classification' and 'asteroid_id' columns.
        """
        from scipy import stats as scipy_stats

        df = df.copy()
        df['classification'] = 'Unknown'
        df['asteroid_id'] = -1

        objids = sorted([oid for oid in df['objid'].unique() if oid > 0])
        tracks = {}

        for objid in objids:
            obj = df[df['objid'] == objid].sort_values('mjd')
            if len(obj) < min_frames:
                continue
            x_vals = obj['x'].values
            y_vals = obj['y'].values
            mjd_vals = obj['mjd'].values
            dist = np.sqrt((x_vals[-1] - x_vals[0])**2 + (y_vals[-1] - y_vals[0])**2)
            if dist < min_displacement_px:
                continue
            t_ref = mjd_vals.mean()
            t = mjd_vals - t_ref
            slope_x, intercept_x, *_ = scipy_stats.linregress(t, x_vals)
            slope_y, intercept_y, *_ = scipy_stats.linregress(t, y_vals)
            pred_x = intercept_x + slope_x * t
            pred_y = intercept_y + slope_y * t
            rms = np.sqrt(np.mean((x_vals - pred_x)**2 + (y_vals - pred_y)**2))
            if rms < max_residual_px:
                tracks[objid] = dict(slope_x=slope_x, intercept_x=intercept_x,
                                     slope_y=slope_y, intercept_y=intercept_y,
                                     t_ref=t_ref, rms=rms)

        if not tracks:
            return df

        # Link nearby non-asteroid objids onto existing trajectories
        asteroid_objids = set(tracks.keys())
        other_objids = set(objids) - asteroid_objids
        assigned = {oid: oid for oid in asteroid_objids}

        for other_oid in other_objids:
            obj = df[df['objid'] == other_oid]
            x_c = obj['x'].mean()
            y_c = obj['y'].mean()
            t_c = obj['mjd'].mean()
            for leader_oid, tr in tracks.items():
                dt = t_c - tr['t_ref']
                pred_x = tr['intercept_x'] + tr['slope_x'] * dt
                pred_y = tr['intercept_y'] + tr['slope_y'] * dt
                if np.sqrt((x_c - pred_x)**2 + (y_c - pred_y)**2) < link_eps_px:
                    assigned[other_oid] = leader_oid
                    break

        leaders = sorted(set(assigned.values()))
        id_map = {leader: i + 1 for i, leader in enumerate(leaders)}

        for oid, leader in assigned.items():
            mask = df['objid'] == oid
            if mask.sum() >= min_track_frames:
                df.loc[mask, 'classification'] = 'Asteroid'
                df.loc[mask, 'asteroid_id'] = id_map[leader]

        return df


    def _time_mjd(self):
        """
        takes df with col 'frame' and adds a mjd col
        the time of the frames in mjd
        """
        df = self.time_df
        df = df.apply(lambda s: s.astype(s.dtype.newbyteorder('=')))

        frames = self.frames
        times = []

        for i in range(len(df)):
            start = df.loc[i, "int_start_MJD_UTC"]
            end = df.loc[i, "int_end_MJD_UTC"]
            times.extend(np.linspace(start,end,self.n_group))

        data = {'frame': frames, 'mjd': times}

        self.frame_mjd_df = pd.DataFrame(data) 


    def assign_mjd(self,df):
        """
        for a given pd dataframe with a column 'frame' will assign a mjd column
        """
        df = df.merge(self.frame_mjd_df, on="frame", how="left")

        return df


    def assign_radec(self, df):
        """
        Adds 'ra'/'dec' columns (deg) from 'x'/'y' via this exposure's own
        distortion-corrected WCS (AssignWcsStep on rate.fits -- rate.fits'
        own header WCS alone is only a crude undistorted TAN projection,
        not accurate at pixel scale, same reasoning as
        _align_master_reference). Computed once per grouped output (not
        per candidate lookup after the fact) so ra/dec are always present
        in the saved catalog -- previously this required a separate,
        repeated AssignWcsStep call per ad-hoc investigation script.
        """
        if df.empty:
            df = df.copy()
            df['ra'] = pd.Series(dtype=float)
            df['dec'] = pd.Series(dtype=float)
            return df

        from jwst.assign_wcs import AssignWcsStep
        rate_path = self.file.replace('_ramp.fits', '_rate.fits')
        model = AssignWcsStep.call(rate_path)
        ra, dec = model.meta.wcs(df['x'].values, df['y'].values)
        df = df.copy()
        df['ra'] = ra
        df['dec'] = dec
        return df


    def _psf_correlation(self, df, cutout_half=5):
        """
        Compute Pearson correlation between each detection cutout and a 2D Gaussian
        PSF model (sigma = FWHM/2.355, sub-pixel shifted to the source centroid).
        Adds 'psf_like' column; 1.0 = perfect PSF match, lower = extended/noise/CR.
        """
        from scipy.ndimage import shift as nd_shift

        sigma = self.fwhm / (2 * np.sqrt(2 * np.log(2)))
        size = 2 * cutout_half + 1
        yg, xg = np.mgrid[-cutout_half:cutout_half + 1, -cutout_half:cutout_half + 1]
        psf_base = np.exp(-(xg**2 + yg**2) / (2 * sigma**2))
        psf_base /= psf_base.sum()

        cube = self.diff_cube if hasattr(self, 'diff_cube') else self.clean_cube
        ny, nx = cube.shape[1], cube.shape[2]
        n_frames = cube.shape[0]

        psf_like = np.full(len(df), np.nan)

        for i, (_, row) in enumerate(df.iterrows()):
            frame = int(row['frame'])
            cx, cy = row['x'], row['y']
            xi, yi = int(round(cx)), int(round(cy))

            y0, y1 = yi - cutout_half, yi + cutout_half + 1
            x0, x1 = xi - cutout_half, xi + cutout_half + 1

            if frame >= n_frames or y0 < 0 or y1 > ny or x0 < 0 or x1 > nx:
                continue

            cutout = cube[frame, y0:y1, x0:x1].copy()
            if cutout.shape != (size, size) or np.all(np.isnan(cutout)):
                continue

            # Shift PSF to sub-pixel centroid position
            psf = nd_shift(psf_base, (cy - yi, cx - xi), mode='constant', cval=0)
            psf_sum = psf.sum()
            if psf_sum > 0:
                psf /= psf_sum

            valid = ~np.isnan(cutout)
            if valid.sum() < 4:
                continue

            r = np.corrcoef(cutout[valid].flatten(), psf[valid].flatten())[0, 1]
            psf_like[i] = r

        df = df.copy()
        df['psf_like'] = psf_like
        return df


    def make_video(self, objid, save_path, half=50, fps=20, dpi=100):
        """
        Save an MP4 video of a 2*half x 2*half px cutout centered on objid.
        Color range is set from the brightest event frame.
        Event frames are highlighted with a red border and annotated in the title.
        Colorbar is matched to the image height.
        """
        import matplotlib.animation as animation
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        obj = self.events[self.events['objid'] == objid].sort_values('frame')
        if obj.empty:
            raise ValueError(f'objid {objid} not found in self.events')

        cx = int(round(obj['x'].mean()))
        cy = int(round(obj['y'].mean()))
        event_frames = set(obj['frame'].astype(int).tolist())

        x0 = max(0, cx - half)
        x1 = min(self.clean_cube.shape[2], cx + half)
        y0 = max(0, cy - half)
        y1 = min(self.clean_cube.shape[1], cy + half)
        cutout = self.clean_cube[:, y0:y1, x0:x1]

        # Skip NaN/zero frames
        valid_frames = [i for i in range(cutout.shape[0])
                        if not np.all(np.isnan(cutout[i])) and not np.all(cutout[i] == 0)]

        # Color range from the brightest event frame
        bright_frame = int(obj.loc[obj['sep_flux'].idxmax(), 'frame'])
        bright_data = cutout[bright_frame]
        vmin = np.nanpercentile(cutout[valid_frames], 1)
        vmax = np.nanpercentile(bright_data[~np.isnan(bright_data)], 99) if not np.all(np.isnan(bright_data)) else vmin + 1

        fig, ax = plt.subplots(figsize=(5, 5))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)

        im = ax.imshow(cutout[valid_frames[0]], origin='lower', cmap='gray', vmin=vmin, vmax=vmax, animated=True)
        plt.colorbar(im, cax=cax, label='DN/group')
        ax.axvline(cx - x0, color='r', lw=0.5, alpha=0.4)
        ax.axhline(cy - y0, color='r', lw=0.5, alpha=0.4)
        title = ax.set_title(f'Frame {valid_frames[0]}', fontsize=12)

        for spine in ax.spines.values():
            spine.set_linewidth(2)

        def update(i):
            frame_idx = valid_frames[i]
            im.set_data(cutout[frame_idx])
            is_event = frame_idx in event_frames
            color = 'red' if is_event else 'black'
            label = '  [EVENT]' if is_event else ''
            title.set_text(f'Frame {frame_idx}{label}')
            title.set_color(color)
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
            return im, title

        fig.tight_layout()
        ani = animation.FuncAnimation(fig, update, frames=len(valid_frames),
                                      interval=1000 / fps, blit=False)
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        ani.save(save_path, writer='ffmpeg', fps=fps, dpi=dpi)
        plt.close(fig)
        print(f'Video saved to {save_path}')


    def plot_detection(self, save_dir, show_inset=True, lc_units='dn/s'):
        """
        Tessellate-identical 3-panel figure for each detected event:
          left   — light curve with event span highlighted (plus a zoom
                   inset when show_inset=True — off by default for NIRCam,
                   whose light curves are already short enough that a zoom
                   panel adds nothing)
          middle — 19x19 px cutout at the brightest frame ('Brightest
                   frame' for MIRI, 'Diff (brightest frame)' for NIRCam)
          right  — same cutout ~1 hour later (MIRI, whose long ramps make
                   a fixed frame offset meaningful) or the reference that
                   was actually subtracted to make this detection (NIRCam,
                   whose exposures are too short for a frame offset to
                   mean anything — showing the reference instead makes it
                   obvious whether a detection is a residual already
                   visible there or something genuinely absent from it)
        Saves as object{objid:04d}_event{eventid}of{total_events}.png
        """
        import matplotlib.patches as patches
        from matplotlib.lines import Line2D
        from mpl_toolkits.axes_grid1.inset_locator import mark_inset

        os.makedirs(save_dir, exist_ok=True)

        mjd_arr = self.frame_mjd_df.set_index('frame')['mjd'].values
        time = mjd_arr - mjd_arr[0]
        cadence = np.median(np.diff(time))
        group_time_s = cadence * 86400  # MJD days → seconds, for DN/s scaling

        calibrated = getattr(self, 'cube_units', None) == 'mjy/sr'
        has_pixar = calibrated and getattr(self, 'pixar_sr', None) is not None
        if calibrated:
            if lc_units == 'uJy' and has_pixar:
                _ylabel = r'$\mu$Jy'
            elif lc_units == 'MJy' and has_pixar:
                _ylabel = 'MJy'
            else:
                _ylabel = r'MJy sr$^{-1}$'
        else:
            _ylabel = 'DN/s' if lc_units == 'dn/s' else 'DN/group'

        # Detect gaps in the time series
        med = np.nanmedian(np.diff(time))
        std = np.nanstd(np.diff(time))
        break_ind = np.where(np.diff(time) > med + 1 * std)[0]
        break_ind = np.append(break_ind, len(time))
        break_ind += 1
        break_ind = np.insert(break_ind, 0, 0)

        obj_ids = sorted([oid for oid in self.events['objid'].unique() if oid > 0])

        # Parallelized across objects (prefer="processes", matching
        # _sep_extract/_finish_sep's pattern) -- plot_detection was
        # previously a fully sequential loop over every object/event, one
        # matplotlib figure at a time, with nothing else in the pipeline
        # to justify that cost.
        Parallel(n_jobs=self.num_cores, prefer="processes")(
            delayed(_plot_one_detection_object)(
                objid, self.events[self.events['objid'] == objid].sort_values('frame'),
                self.clean_cube, self.second_ref_frame, self.kernel, self.n_frame,
                self.fwhm, self.instrument, getattr(self, 'pixar_sr', None),
                calibrated, has_pixar, lc_units, _ylabel, group_time_s,
                time, mjd_arr, cadence, show_inset, save_dir)
            for objid in obj_ids)


    def plot_decam_finder(self, objid, df=None, save_dir=None, join_with_pipeline=True,
                          pipeline_fig_dir=None, size_arcsec=20.0, pixscale=0.262,
                          show_error_ellipse=True, footprint_size_arcsec=None):
        """
        DECam (Legacy Survey DR10) color finder chart for one candidate,
        in the style of TESSELLATE's event_cutout()/_DESI_phot()
        (external_photometry.py) -- WCS-projected grz RGB cutout, DELVE
        DR2 catalog sources overplotted (via NOIRLab Astro Data Lab SQL,
        same access pattern as TESSELLATE's _delve_objects; the Legacy
        Survey viewer's own "cat.json" endpoint returns the generic
        viewer HTML page to a direct request, not real per-query data),
        candidate position + centroid-precision error ellipse, optionally
        joined to the right of the matching pipeline detection figure
        (join_finder_chart_right, TESSELLATE navigator.py
        save_combined_path pattern).

        objid : the candidate's objid in `df` (default self.events, which
                has 'ra'/'dec' from assign_radec and 'psf_flux'/
                'psf_flux_err' from _psf_fit already).
        Saves to `{save_dir}/object{objid:04d}_decam_finder.png`, plus
        `_combined.png` if join_with_pipeline. save_dir defaults to
        `{self.grouped_dir}/combined`; pipeline_fig_dir (where the
        matching object{objid:04d}_event*.png already lives) defaults to
        `{self.grouped_dir}/detection_figures_sep`.
        """
        import io
        import requests
        from astropy.wcs import WCS as AstropyWCS
        from matplotlib.patches import Ellipse, Rectangle
        from PIL import Image

        df = self.events if df is None else df
        save_dir = os.path.join(self.grouped_dir, 'combined') if save_dir is None else save_dir
        # only fall back to self.grouped_dir when actually needed -- a
        # caller with join_with_pipeline=False (e.g. plot_augmented_sw_lw,
        # whose "pipeline figure" isn't in the usual detection_figures_sep
        # layout at all) may not have self.grouped_dir set, and shouldn't
        # need to for a call that never uses this value.
        if join_with_pipeline and pipeline_fig_dir is None:
            pipeline_fig_dir = os.path.join(self.grouped_dir, 'detection_figures_sep')
        os.makedirs(save_dir, exist_ok=True)

        obj_df = df[df['objid'] == objid]
        if obj_df.empty or 'ra' not in obj_df or obj_df['ra'].isna().all():
            raise ValueError(f'objid {objid} not found in df, or has no ra/dec (assign_radec not run?)')
        ra, dec = float(obj_df['ra'].mean()), float(obj_df['dec'].mean())
        snr_row = obj_df.loc[obj_df['psf_flux'].abs().idxmax()]
        snr = abs(snr_row['psf_flux'] / snr_row['psf_flux_err']) if snr_row['psf_flux_err'] else np.nan

        # --- DECam grz cutout (image + WCS) ---
        size_px = int(round(size_arcsec / pixscale))
        jpg_url = (f"https://www.legacysurvey.org/viewer/cutout.jpg"
                   f"?ra={ra}&dec={dec}&size={size_px}&layer=ls-dr10&pixscale={pixscale}")
        fits_url = (f"https://www.legacysurvey.org/viewer/cutout.fits"
                    f"?ra={ra}&dec={dec}&size={size_px}&layer=ls-dr10&pixscale={pixscale}")
        image = np.array(Image.open(io.BytesIO(requests.get(jpg_url, timeout=30).content)))
        with fits.open(io.BytesIO(requests.get(fits_url, timeout=30).content)) as hdul:
            wcs = AstropyWCS(hdul[0].header)
            if wcs.naxis > 2:
                wcs = wcs.dropaxis(2)

        # --- DELVE DR2 catalog sources (NOIRLab Astro Data Lab SQL) ---
        # Cone-search radius must cover the SQUARE cutout's corners, not
        # just its half-width -- a circle of radius size_arcsec/2 misses
        # the corners of a size_arcsec x size_arcsec box (which extend out
        # to size_arcsec/2 * sqrt(2)), silently dropping real sources near
        # the edges rather than raising an error, which looked like
        # "wrong" marker placement rather than incomplete coverage.
        from dl import queryClient as qc
        size_deg = (size_arcsec / 2.0 * np.sqrt(2)) / 3600.0
        query = f"""
            SELECT o.ra, o.dec,
            o.extended_class_g, o.extended_class_r, o.extended_class_i, o.extended_class_z
            FROM delve_dr2.objects AS o
            WHERE q3c_radial_query(ra, dec, {ra}, {dec}, {size_deg})
            """
        cat = qc.query(sql=query, fmt='pandas')
        marker_style = {
            'star': dict(marker='*', color='cyan', label='DELVE point source'),
            'maybe star': dict(marker='*', color='lightblue', label='DELVE maybe point source'),
            'galaxy': dict(marker='o', color='orange', label='DELVE galaxy'),
            'maybe galaxy': dict(marker='o', color='navajowhite', label='DELVE maybe galaxy'),
        }

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection=wcs)
        # WCSAxes requires origin='lower' (raises ValueError otherwise) --
        # it assumes the array's row 0 is the bottom, FITS convention,
        # matching the WCS's own pixel indexing. cutout.jpg is a rendered
        # raster image instead (row 0 = top, standard image convention),
        # so the fix is flipping the ARRAY itself before display, not the
        # origin parameter: without this, the displayed image is
        # vertically flipped relative to the WCS grid the catalog/
        # candidate markers are correctly projected onto -- small (easy
        # to miss) for a source near the cutout center, glaring for
        # anything off-center.
        ax.imshow(image[::-1], origin='lower')

        # The DELVE cone-search radius covers the square cutout's corners
        # (a circle circumscribing the square, see size_deg above), so it
        # can return real sources that sit in that circle but OUTSIDE the
        # actual displayed square image -- drop those rather than plot
        # markers beyond the image bounds.
        ny_img, nx_img = image.shape[0], image.shape[1]
        seen_types = set()
        for _, row in cat.iterrows():
            px, py = wcs.all_world2pix(row['ra'], row['dec'], 0)
            if not (0 <= px < nx_img and 0 <= py < ny_img):
                continue
            ext = [row[f'extended_class_{b}'] for b in 'griz']
            src_type = 'star' if 0 in ext else 'maybe star' if 1 in ext else 'galaxy' if 3 in ext else 'maybe galaxy'
            style = marker_style[src_type]
            first = src_type not in seen_types
            seen_types.add(src_type)
            ax.scatter(row['ra'], row['dec'], transform=ax.get_transform('fk5'),
                       marker=style['marker'], s=100, facecolors='none',
                       edgecolors=style['color'], linewidths=1.5,
                       label=style['label'] if first else None)

        # centroid-precision-only error ellipse (see NIRCAM_FWHM_PX/
        # PIXSCALE below) -- does NOT include JWST's absolute astrometric
        # uncertainty, no reliably sourced number for that available here
        pixscale_nircam = self._nircam_pixscale_arcsec()
        sigma_centroid_px = (self.fwhm / 2.355) / snr if np.isfinite(snr) and snr > 0 else np.nan
        sigma_centroid_arcsec = sigma_centroid_px * pixscale_nircam
        ellipse_deg = (3 * sigma_centroid_arcsec) / 3600.0 if np.isfinite(sigma_centroid_arcsec) else 0.0

        ax.scatter(ra, dec, transform=ax.get_transform('fk5'), marker='+',
                   s=300, color='red', linewidths=2, label='NIRCam candidate position', zorder=10)
        if show_error_ellipse and ellipse_deg > 0:
            ell = Ellipse((ra, dec), width=2 * ellipse_deg / np.cos(np.deg2rad(dec)), height=2 * ellipse_deg,
                          transform=ax.get_transform('fk5'), edgecolor='red', facecolor='none',
                          linewidth=1.5, linestyle='--', zorder=9, label=r'3$\sigma$ centroid')
            ax.add_patch(ell)

        # NIRCam reference-cutout footprint (e.g. plot_augmented_sw_lw's
        # own SW+LW composite reference panel), drawn as a box the same
        # angular size centered on this candidate -- shows how that tiny
        # NIRCam cutout maps onto the wider DECam field.
        if footprint_size_arcsec is not None and footprint_size_arcsec > 0:
            fp_deg = footprint_size_arcsec / 3600.0
            fp_rect = Rectangle((ra - fp_deg / (2 * np.cos(np.deg2rad(dec))), dec - fp_deg / 2),
                                width=fp_deg / np.cos(np.deg2rad(dec)), height=fp_deg,
                                transform=ax.get_transform('fk5'), edgecolor='lime', facecolor='none',
                                linewidth=1.5, zorder=9, label='NIRCam reference footprint')
            ax.add_patch(fp_rect)

        ax.set_xlabel('Right Ascension', fontsize=15)
        ax.set_ylabel('Declination', fontsize=15)
        ax.set_title(f"objid{objid}  (RA={ra:.4f}, Dec={dec:.4f})  --  DECam grz, DELVE DR2 catalog",
                     fontsize=12)
        ax.coords[0].set_major_formatter('hh:mm:ss')
        ax.coords[1].set_major_formatter('dd:mm:ss')
        ax.invert_xaxis()
        ax.legend(loc='upper right', fontsize=13, framealpha=0.7)

        out_path = os.path.join(save_dir, f'object{objid:04d}_decam_finder.png')
        plt.savefig(out_path, bbox_inches='tight', dpi=130)
        plt.close(fig)

        if join_with_pipeline:
            pipeline_matches = glob.glob(os.path.join(pipeline_fig_dir, f'object{objid:04d}_event*.png'))
            if pipeline_matches:
                combined_path = os.path.join(save_dir, f'object{objid:04d}_combined.png')
                self._join_finder_chart_right(pipeline_matches[0], out_path, combined_path)
                return combined_path
        return out_path


    def _nircam_pixscale_arcsec(self):
        """
        NIRCam pixel scale (arcsec/px), measured directly from this
        exposure's own WCS (two adjacent-pixel separations near the
        detector center) rather than an assumed nominal value.
        """
        from astropy.wcs import WCS as AstropyWCS
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        rate_path = self.file.replace('_ramp.fits', '_rate.fits')
        with fits.open(rate_path) as h:
            wcs = AstropyWCS(h[1].header)
        ny, nx = wcs.pixel_shape[1], wcs.pixel_shape[0]
        cx, cy = nx // 2, ny // 2
        ra0, dec0 = wcs.all_pix2world(cx, cy, 0)
        ra1, dec1 = wcs.all_pix2world(cx + 1, cy, 0)
        c0 = SkyCoord(float(ra0) * u.deg, float(dec0) * u.deg)
        c1 = SkyCoord(float(ra1) * u.deg, float(dec1) * u.deg)
        return c0.separation(c1).arcsec


    def plot_augmented_sw_lw(self, other, save_dir, objid, other_objid=None,
                              eventid=1, total_events=1, ref_zoom_halfwidth=20,
                              include_decam=False, decam_size_arcsec=20.0, decam_pixscale=0.262,
                              shared_lc_axis=False):
        """
        Combined SW+LW detection figure for a candidate matched via
        crossmatch_sw_lw's sw_match/lw_match tagging -- NIRCam images SW
        and LW simultaneously through the same dichroic, so a real
        source should show up in both channels at once; this figure
        makes that checkable at a glance instead of comparing two
        separate per-channel figures (the LW/SW cross-detection
        discriminator, alongside consecutive-frame persistence, is the
        other pathway for a candidate to be considered real).

        Call on either channel's already fully-run Jurassic instance
        (`self`), passing the OTHER channel's already fully-run instance
        as `other` -- both need self.clean_cube / self.second_ref_frame
        / self.events (with ra/dec, i.e. compute_radec=True) /
        self.kernel / self.n_frame / self.frame_mjd_df populated, i.e. a
        normal images=True run, NOT reference_only=True. Which is SW vs
        LW is read off each instance's own self.detector (nrc{a,b}LONG
        = LW).

        `other_objid=None` (no real matching detection in the other
        channel -- e.g. crossmatch_sw_lw found no sw_match/lw_match, or
        the caller only checked one of the 4 SW quadrants and it wasn't
        the right one): if the candidate still passes the duration/
        persistence check on its own detected channel, do NOT borrow
        some unrelated nearby objid there (shows the wrong pixels
        entirely, and makes the composite reference/frame rows
        meaningless) -- instead do forced PSF photometry at the
        geometrically-correct position: this candidate's own RA/Dec
        transformed into the other channel's own pixel grid via its own
        WCS, over the same absolute time span (nearest frames by MJD,
        since SW/LW cadence differs).

        Layout (direct instruction):
            top-left   : LW + SW light curves overlaid, LW orange (C1),
                         SW blue (C0)
            top-right  : colour SW+LW composite reference (R=LW, B=SW,
                         G=mean of both -- a source common to both
                         channels renders white/gray, a single-channel-
                         only artifact renders strongly tinted),
                         reprojected onto a common small tangent-plane
                         grid at the LW pixel scale via each channel's
                         own distortion-corrected WCS (reproject_interp,
                         same pattern as _align_master_reference)
            middle row : up to 4 sequential SW frames spanning its own
                         event span
            bottom row : up to 4 sequential LW frames spanning its own
                         event span (frame COUNTS need not match, since
                         SW/LW cadence differs -- SW reads out faster)

        include_decam=True: appends a DECam grz + DELVE DR2 finder chart
        (plot_decam_finder) to the right of this figure, TESSELLATE-
        style (same _join_finder_chart_right pattern the single-channel
        figures use). Centered on whichever channel has a REAL detection
        (lw_objid if not None, else sw_objid) -- same candidate identity
        already used for the composite reference panel's RA/Dec.
        """
        import matplotlib.patches as patches
        from matplotlib.gridspec import GridSpec
        from astropy.wcs import WCS as AstropyWCS
        from reproject import reproject_interp
        from jwst.assign_wcs import AssignWcsStep

        os.makedirs(save_dir, exist_ok=True)

        lw, sw = (self, other) if self.detector.endswith('LONG') else (other, self)
        lw_objid, sw_objid = (objid, other_objid) if self.detector.endswith('LONG') else (other_objid, objid)

        if lw_objid is None and sw_objid is None:
            raise ValueError('plot_augmented_sw_lw needs a real objid on at least one channel')

        # Cached on the instance itself: calling this repeatedly against
        # the SAME lw/sw pair for many different candidates (a batch over
        # every objid in an exposure) would otherwise redo the identical
        # AssignWcsStep call every single time.
        if getattr(lw, '_assign_wcs_model', None) is None:
            lw._assign_wcs_model = AssignWcsStep.call(lw.file.replace('_ramp.fits', '_rate.fits'))
        if getattr(sw, '_assign_wcs_model', None) is None:
            sw._assign_wcs_model = AssignWcsStep.call(sw.file.replace('_ramp.fits', '_rate.fits'))
        lw_model = lw._assign_wcs_model
        sw_model = sw._assign_wcs_model

        def _forced(anchor_inst, anchor_objid, anchor_dict, target_inst, target_model):
            # no real detection on target_inst -- force photometry at the
            # anchor's own RA/Dec, transformed into the target's pixel
            # grid, over the anchor's own event time span mapped onto the
            # target's own frame indices via nearest MJD (cadences
            # differ, so frame COUNTS aren't comparable 1:1 -- absolute
            # time is)
            ev_pos = anchor_inst.events[anchor_inst.events['objid'] == anchor_objid]
            cand_ra = float(ev_pos['ra'].mean())
            cand_dec = float(ev_pos['dec'].mean())
            t_x, t_y = target_model.meta.wcs.world_to_pixel_values(cand_ra, cand_dec)
            t_x, t_y = float(t_x), float(t_y)
            if not (np.isfinite(t_x) and np.isfinite(t_y)):
                # candidate's RA/Dec falls outside the target channel's
                # gwcs bounding_box (a real edge case near a detector
                # boundary, or the target detector simply doesn't cover
                # this part of the sky at all -- e.g. only one of the 4
                # SW quadrants images a given LW candidate) --
                # world_to_pixel_values returns NaN rather than an
                # out-of-range pixel value there. Rather than failing the
                # whole combined figure, return an empty/blank result:
                # _draw_row's per-frame loop (zip against frame_indices)
                # simply does nothing when frame_indices is empty, so
                # this channel's row renders as blank panels instead of
                # a real cutout -- makes clear at a glance that this
                # candidate has no coverage there, rather than silently
                # missing from the batch.
                mjd_arr = target_inst.frame_mjd_df.set_index('frame')['mjd'].values
                time = mjd_arr - mjd_arr[0]
                cadence = np.median(np.diff(time)) if len(time) > 1 else np.nan
                calibrated = getattr(target_inst, 'cube_units', None) == 'mjy/sr'
                has_pixar = calibrated and getattr(target_inst, 'pixar_sr', None) is not None
                ylabel = (r'$\mu$Jy' if has_pixar else (r'MJy sr$^{-1}$' if calibrated else 'DN/group'))
                return dict(x=0, y=0, x_fit=0.0, y_fit=0.0, frame_start=0, frame_end=0,
                            time=time, time_c=np.array([]), f_c=np.array([]), ferr_c=np.array([]),
                            brk=np.array([0]), cadence=cadence, mjd_arr=mjd_arr, ylabel=ylabel,
                            brightestframe=0, event_frames=set(), frame_indices=[],
                            no_coverage=True)
            t_x, t_y = int(round(t_x)), int(round(t_y))
            anchor_mjd = anchor_inst.frame_mjd_df.set_index('frame')['mjd']
            mjd_start = anchor_mjd.loc[anchor_dict['frame_start']]
            mjd_end = anchor_mjd.loc[anchor_dict['frame_end']]
            target_mjd = target_inst.frame_mjd_df.set_index('frame')['mjd']
            t_frame_start = int((target_mjd - mjd_start).abs().idxmin())
            t_frame_end = int((target_mjd - mjd_end).abs().idxmin())
            return _build_channel_lightcurve_at_xy(target_inst, t_x, t_y, t_frame_start, t_frame_end)

        if lw_objid is not None:
            lwd = _build_channel_lightcurve(lw, lw_objid, eventid)
            swd = (_build_channel_lightcurve(sw, sw_objid, eventid) if sw_objid is not None
                   else _forced(lw, lw_objid, lwd, sw, sw_model))
        else:
            swd = _build_channel_lightcurve(sw, sw_objid, eventid)
            lwd = _forced(sw, sw_objid, swd, lw, lw_model)

        fig = plt.figure(figsize=(10, 7.5), constrained_layout=True)
        # frame rows squished relative to the top (lc/reference) row --
        # direct instruction ("squish the frames")
        gs = GridSpec(3, 5, figure=fig, height_ratios=[1.5, 1, 1], width_ratios=[1, 1, 1, 1, 0.06])
        ax_lc = fig.add_subplot(gs[0, 0:2])
        ax_ref = fig.add_subplot(gs[0, 2:4])
        n_sw = max(1, len(swd['frame_indices']))
        n_lw = max(1, len(lwd['frame_indices']))
        sw_axes = [fig.add_subplot(gs[1, i]) for i in range(n_sw)]
        lw_axes = [fig.add_subplot(gs[2, i]) for i in range(n_lw)]
        cax_sw = fig.add_subplot(gs[1, 4])
        cax_lw = fig.add_subplot(gs[2, 4])

        # top-left: both light curves, LW orange / SW blue (direct
        # instruction), each with a low-alpha axvspan marking its own
        # REAL detection time span -- only for the channel that actually
        # has one; a forced-photometry channel wasn't detected, so
        # nothing to span. LW and SW flux scales commonly differ by a
        # large factor (different filters/sources), so SW normally gets
        # its own twin y-axis on the right rather than sharing LW's --
        # otherwise a much fainter channel flattens to an invisible line
        # against the brighter one's scale. shared_lc_axis=True (direct
        # instruction, per-call) puts both on the same left axis instead
        # -- useful when the two channels' scales are actually
        # comparable for this particular candidate.
        ax_lc_sw = ax_lc if shared_lc_axis else ax_lc.twinx()
        axes_by_channel = {'LW': (ax_lc, 'C1'), 'SW': (ax_lc_sw, 'C0')}
        for label, chd, real_objid in (('LW', lwd, lw_objid), ('SW', swd, sw_objid)):
            ax, color = axes_by_channel[label]
            time_c_s = chd['time_c'] * 86400
            brk = chd['brk']
            for seg in range(len(brk) - 1):
                sl = slice(brk[seg], brk[seg + 1])
                ax.plot(time_c_s[sl], chd['f_c'][sl], '-', color=color, alpha=0.6, zorder=1)
                ax.errorbar(time_c_s[sl], chd['f_c'][sl], yerr=chd['ferr_c'][sl], fmt='o',
                            color=color, ms=5, ecolor=color, elinewidth=1, capsize=2,
                            zorder=2, label=label if seg == 0 else None)
            if real_objid is not None:
                time_s = chd['time'] * 86400
                cadence_s = chd['cadence'] * 86400
                span_start = time_s[chd['frame_start']] - cadence_s / 2
                span_end = time_s[chd['frame_end']] + cadence_s / 2
                ax.axvspan(span_start, span_end, color=color, alpha=0.15, zorder=0)
        if shared_lc_axis:
            ax_lc.legend(fontsize=9, loc='best')
        else:
            h1, l1 = ax_lc.get_legend_handles_labels()
            h2, l2 = ax_lc_sw.get_legend_handles_labels()
            ax_lc.legend(h1 + h2, l1 + l2, fontsize=9, loc='best')
        lw_label = f'LW ObjID {lw_objid}' if lw_objid is not None else 'LW forced photometry (no real detection)'
        sw_label = f'SW ObjID {sw_objid}' if sw_objid is not None else 'SW forced photometry (no real detection)'
        ax_lc.set_title(f'{lw_label} / {sw_label}', fontsize=13)
        ax_lc.set_xlabel(f"Time (s) - MJD {np.round(lwd['mjd_arr'][0], 5)}", fontsize=11)
        if shared_lc_axis:
            ax_lc.set_ylabel(f"{lwd['ylabel']} (LW) / {swd['ylabel']} (SW)", fontsize=10)
        else:
            ax_lc.set_ylabel(f"{lwd['ylabel']} (LW)", fontsize=10, color='C1')
            ax_lc.tick_params(axis='y', labelcolor='C1')
            ax_lc_sw.set_ylabel(f"{swd['ylabel']} (SW)", fontsize=10, color='C0')
            ax_lc_sw.tick_params(axis='y', labelcolor='C0')

        # top-right: colour SW+LW composite reference, reprojected onto a
        # common small tangent grid centered on the REAL candidate's
        # RA/Dec (whichever channel actually has one)
        pixscale_deg = lw._nircam_pixscale_arcsec() / 3600.0
        if lw_objid is not None:
            ev_pos = lw.events[lw.events['objid'] == lw_objid]
        else:
            ev_pos = sw.events[sw.events['objid'] == sw_objid]
        cand_ra = float(ev_pos['ra'].mean())
        cand_dec = float(ev_pos['dec'].mean())

        n_px = 2 * ref_zoom_halfwidth + 1
        target_wcs = AstropyWCS(naxis=2)
        target_wcs.wcs.crpix = [ref_zoom_halfwidth + 1, ref_zoom_halfwidth + 1]
        target_wcs.wcs.crval = [cand_ra, cand_dec]
        target_wcs.wcs.cdelt = [-pixscale_deg, pixscale_deg]
        target_wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        shape_out = (n_px, n_px)

        lw_cut, _ = reproject_interp((lw.second_ref_frame, lw_model.meta.wcs), target_wcs, shape_out=shape_out)
        sw_cut, _ = reproject_interp((sw.second_ref_frame, sw_model.meta.wcs), target_wcs, shape_out=shape_out)

        def _norm(a):
            v = a[np.isfinite(a)]
            if v.size == 0:
                return np.zeros_like(a)
            lo, hi = np.percentile(v, [16, 99.5])
            if hi <= lo:
                hi = lo + 1
            return np.clip((np.nan_to_num(a, nan=lo) - lo) / (hi - lo), 0, 1)

        r_ch, b_ch = _norm(lw_cut), _norm(sw_cut)
        rgb = np.dstack([r_ch, (r_ch + b_ch) / 2, b_ch])
        ax_ref.imshow(rgb, origin='lower', extent=[-0.5, n_px - 0.5, -0.5, n_px - 0.5])
        ax_ref.set_aspect('auto')
        lw_filt = getattr(lw, 'filter', None) or 'LW'
        sw_filt = getattr(sw, 'filter', None) or 'SW'
        ax_ref.set_title(f'SW+LW reference (B={sw_filt}, R={lw_filt})', fontsize=12)
        # same 5x5-pixel red box style as the frame panels below, on this
        # panel's own grid (built at the LW pixel scale, so this box is
        # the same real angular size as the LW frame panels' box)
        ax_ref.add_patch(patches.Rectangle((ref_zoom_halfwidth - 2.5, ref_zoom_halfwidth - 2.5), 5, 5,
                                            linewidth=2, edgecolor='r', facecolor='none'))
        ax_ref.get_xaxis().set_visible(False)
        ax_ref.get_yaxis().set_visible(False)

        # middle/bottom rows: SW on top, LW below (direct instruction).
        # Event-frame highlight color matches the LC span colors (SW
        # blue/C0, LW orange/C1) -- and only appears at all when this
        # channel has a REAL detection; a forced-photometry channel
        # wasn't actually detected in any frame, so nothing gets
        # highlighted there (direct instruction: "do not highlight
        # frames that it wasn't detected in").
        def _draw_row(inst, chd, axes, cax, cbar_label, highlight_color, is_real):
            clean_cube = inst.clean_cube
            x, y = chd['x'], chd['y']
            ny, nx = clean_cube.shape[1], clean_cube.shape[2]
            hs = 9
            ymin_t, ymax_t = max(0, y - hs), min(ny, y + hs + 1)
            xmin_t, xmax_t = max(0, x - hs), min(nx, x + hs + 1)
            bf = chd['brightestframe']
            if chd.get('no_coverage'):
                for panel_ax in axes:
                    panel_ax.set_facecolor('0.9')
                    panel_ax.set_xticks([]); panel_ax.set_yticks([])
                    panel_ax.text(0.5, 0.5, 'no coverage', ha='center', va='center',
                                  fontsize=9, color='0.4', transform=panel_ax.transAxes)
                return

            # vmax from the tight 3x3 core (the actual photometry
            # aperture) around the brightest frame's peak; vmin from a
            # wider 15x15 region around the source, not the whole frame
            # -- these are difference images, so a negative value is
            # physically meaningful (fainter than the reference), not
            # just noise, and the previous vmin (16th percentile of the
            # WHOLE frame) paired with the 3x3 peak made an enormous,
            # peak-dominated range that visually flattened any real
            # negative structure near the source down to
            # indistinguishable-from-background.
            bright_patch = clean_cube[bf, max(0, y - 1):y + 2, max(0, x - 1):x + 2]
            local_region = clean_cube[bf, max(0, y - 7):y + 8, max(0, x - 7):x + 8]
            try:
                vmax = float(np.nanmax(bright_patch))
            except (ValueError, TypeError):
                vmax = 1.0
            try:
                vmin = float(np.nanmin(local_region))
            except (ValueError, TypeError):
                vmin = -1.0
            if not np.isfinite(vmax):
                vmax = 1.0
            if not np.isfinite(vmin):
                vmin = -1.0
            if vmin >= vmax:
                vmin = vmax - 5

            for panel_ax, fi in zip(axes, chd['frame_indices']):
                frame_img = clean_cube[fi, ymin_t:ymax_t, xmin_t:xmax_t]
                im = panel_ax.imshow(frame_img, cmap='cividis', origin='lower', vmin=vmin, vmax=vmax,
                                     extent=[xmin_t - 0.5, xmax_t - 0.5, ymin_t - 0.5, ymax_t - 0.5])
                panel_ax.set_aspect('auto')
                panel_ax.tick_params(labelsize=7)
                panel_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                # box centered on the FITTED sub-pixel centroid (chd['x_fit']/
                # ['y_fit'], see _stacked_centroid_fit), not the raw integer
                # guess -- and sized to match the actual forced-photometry
                # aperture (FORCED_PHOTOMETRY_APERTURE_HW), not an unrelated
                # fixed 5x5 marker
                box_size = 2 * FORCED_PHOTOMETRY_APERTURE_HW + 1
                x_fit, y_fit = chd.get('x_fit', x), chd.get('y_fit', y)
                panel_ax.add_patch(patches.Rectangle((x_fit - box_size / 2, y_fit - box_size / 2),
                                                      box_size, box_size,
                                                      linewidth=2, edgecolor='r', facecolor='none'))
                is_event = is_real and fi in chd['event_frames']
                panel_ax.set_title(f'Frame {fi}' + (' (event)' if is_event else ''),
                                   fontsize=9, color=highlight_color if is_event else 'black',
                                   fontweight='bold' if is_event else 'normal')
                if is_event:
                    for spine in panel_ax.spines.values():
                        spine.set_edgecolor(highlight_color)
                        spine.set_linewidth(2.5)
                if panel_ax is axes[-1]:
                    cbar = fig.colorbar(im, cax=cax)
                    cbar.set_label(cbar_label, fontsize=8)
                    cbar.ax.tick_params(labelsize=7)

        _draw_row(sw, swd, sw_axes, cax_sw, 'SW DN/group', 'C0', sw_objid is not None)
        _draw_row(lw, lwd, lw_axes, cax_lw, 'LW DN/group', 'C1', lw_objid is not None)
        sw_axes[0].set_ylabel('SW', fontsize=11)
        lw_axes[0].set_ylabel('LW', fontsize=11)

        fig.suptitle(f'{os.path.basename(lw.file)}  /  {os.path.basename(sw.file)}', fontsize=9)

        # detector name included in both tags -- objid numbering is only
        # unique WITHIN one detector's own catalog (e.g. nrcb1 objid 57
        # and nrcb4 objid 57 are different physical candidates), so a
        # filename keyed on objid alone would collide across detectors
        # and silently skip real candidates in a resumable batch.
        lw_tag = f'{lw.detector.lower()}{lw_objid:04d}' if lw_objid is not None else f'{lw.detector.lower()}forced'
        sw_tag = f'{sw.detector.lower()}{sw_objid:04d}' if sw_objid is not None else f'{sw.detector.lower()}forced'
        out_path = os.path.join(save_dir, f'augmented_sw_lw_lw{lw_tag}_sw{sw_tag}.png')
        plt.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {out_path}')

        if include_decam:
            anchor_inst, anchor_objid = (lw, lw_objid) if lw_objid is not None else (sw, sw_objid)
            footprint_size_arcsec = n_px * pixscale_deg * 3600.0
            anchor_inst.plot_decam_finder(anchor_objid, save_dir=save_dir, join_with_pipeline=False,
                                           size_arcsec=decam_size_arcsec, pixscale=decam_pixscale,
                                           show_error_ellipse=False,
                                           footprint_size_arcsec=footprint_size_arcsec)
            decam_path = os.path.join(save_dir, f'object{anchor_objid:04d}_decam_finder.png')
            combined_path = out_path.replace('.png', '_decam.png')
            self._join_finder_chart_right(out_path, decam_path, combined_path)
            # only the final combined figure is wanted -- the plain
            # augmented figure and standalone DECam finder chart are
            # just intermediates on the way to it
            for intermediate in (out_path, decam_path):
                if os.path.exists(intermediate):
                    os.remove(intermediate)
            print(f'Saved {combined_path}')
            return combined_path

        return out_path


    @staticmethod
    def _join_finder_chart_right(pipeline_fig_path, finder_chart_path, out_path):
        """
        Appends the finder chart to the right of the pipeline detection
        figure -- TESSELLATE navigator.py save_combined_path pattern
        (Navigator.plot_lc): match heights (not widths) via Lanczos
        resize, then paste side by side with a small gap.
        """
        from PIL import Image

        img1 = Image.open(pipeline_fig_path)
        img2 = Image.open(finder_chart_path)

        max_height = max(img1.height, img2.height)
        if img1.height != max_height:
            img1 = img1.resize((int(img1.width * max_height / img1.height), max_height), Image.LANCZOS)
        if img2.height != max_height:
            img2 = img2.resize((int(img2.width * max_height / img2.height), max_height), Image.LANCZOS)

        combined_width = int((img1.width + img2.width) * 1.01)
        combined_img = Image.new('RGB', (combined_width, max_height), (255, 255, 255))
        combined_img.paste(img1, (0, 0))
        combined_img.paste(img2, (int(img1.width + combined_width / 100), 0))
        combined_img.save(out_path, dpi=(150, 150))


    def _flux_calibrate(self):
        """
        Convert self.clean_cube from DN/group to MJy/sr, then divide by the
        group time so the result is in MJy/sr (surface brightness per pixel).

        MIRI uses the bundled miri_photom.csv (derived from
        jwst_miri_photom_0230.fits) with its exponential + linear
        time-dependent sensitivity-loss correction applied at the median
        observation MJD. No such drift model is available for other
        instruments, so they use the static CRDS photmjsr from
        flux_calibrate() (set on self.flux_conv) with no time correction.

        Sets:
            self.photmjsr : effective conversion factor [MJy/sr per DN/s]
            self.cube_units : 'mjy/sr' (used by plot_detection for unit handling)
        """
        mjd_arr = self.frame_mjd_df['mjd'].values
        group_time_s = np.nanmedian(np.diff(mjd_arr)) * 86400

        if self.instrument == 'MIRI':
            mjd = np.nanmedian(mjd_arr)

            csv_path = os.path.join(os.path.dirname(__file__), 'miri_photom.csv')
            phot = pd.read_csv(csv_path)

            filt = self.filter.strip()
            sub = self.subarray.strip()
            row = phot[(phot['filter'] == filt) & (phot['subarray'] == sub)]
            if row.empty:
                row = phot[(phot['filter'] == filt) & (phot['subarray'] == 'FULL')]
            if row.empty:
                raise ValueError(f'No photom entry for {filt}/{sub} in miri_photom.csv')
            row = row.iloc[0]

            dt_days = mjd - row['t0']
            exp_corr = row['const'] + row['amplitude'] * np.exp(-dt_days / row['tau'])
            lin_corr = 1.0 + row['lossperyear'] * (dt_days / 365.25)
            self.photmjsr = row['photmjsr'] * exp_corr * lin_corr

            print(f'Flux calibration: filter={filt}, subarray={sub}, '
                  f'MJD={mjd:.3f}, photmjsr={self.photmjsr:.4f} MJy/sr per DN/s')
        else:
            self.photmjsr = self.flux_conv

            print(f'Flux calibration: filter={self.filter}, instrument={self.instrument}, '
                  f'photmjsr={self.photmjsr:.4f} MJy/sr per DN/s (no time-dependent correction)')

        self.clean_cube = self.clean_cube * (self.photmjsr / group_time_s)
        self.cube_units = 'mjy/sr'


    def save_outputs(self):
        """
        outputs i wanna save (from themselves)
        """
        def safe_max(series):
            """
            Returns max if there is one, returns 0 otherwise
            """
            if series is None:
                return 0
            if not hasattr(series, "__len__"):
                return 0
            if len(series) == 0:
                return 0
            arr = series.to_numpy()
            if arr.size == 0:
                return 0
            return np.nanmax(arr)

        # make a folder for the grouped output
        self.grouped_dir = os.path.join(self.obs_dir, 'grouped_output')
        os.makedirs(self.grouped_dir, exist_ok=True)
        
        if len(self.significance_df) > 0:
            g_sig_df = self._spatial_group(self.significance_df)
            g_sig_df = g_sig_df.sort_values(by=['objid', 'frame'], ascending=[True, True])
            g_sig_df = self._temporal_group(g_sig_df)
            g_sig_df = self.assign_mjd(g_sig_df)
            if self.compute_radec:
                g_sig_df = self.assign_radec(g_sig_df)
            filepath = os.path.join(self.grouped_dir, 'grouped_significance.csv')
            g_sig_df.to_csv(filepath, index=False)
            if self.plot and len(g_sig_df) > 0:
                if not hasattr(self, 'events'):
                    self.events = g_sig_df
                self.plot_detection(os.path.join(self.grouped_dir, 'detection_figures_sig'),
                                     show_inset=self.instrument != 'NIRCAM',
                                     lc_units='uJy' if self.instrument == 'NIRCAM' else 'dn/s')

        # filtered sep sources (grouped)
        num_candidates, data = 0, []
        if len(self.filtered_sep_df) > 0:
            self.events = self._spatial_group(self.filtered_sep_df)
            self.events = self.events.sort_values(by=['objid', 'frame'], ascending=[True, True])
            self.events = self._temporal_group(self.events)
            self.events = self.assign_mjd(self.events)
            self.events = self._tag_asteroids(self.events)
            self.events = self._psf_correlation(self.events)
            if self.compute_radec:
                self.events = self.assign_radec(self.events)
            self.events.to_csv(os.path.join(self.grouped_dir, 'grouped_filtered_sep.csv'), index=False)
            if self.plot and len(self.events) > 0:
                self.plot_detection(os.path.join(self.grouped_dir, 'detection_figures_sep'),
                                     show_inset=self.instrument != 'NIRCAM',
                                     lc_units='uJy' if self.instrument == 'NIRCAM' else 'dn/s')
            num_candidates, data = self.asteroid_candidate(self.events)

        if len(self.total_df) > 0:
            g_tot_sep_df = self._spatial_group(self.total_df)
            g_tot_sep_df = g_tot_sep_df.sort_values(by=['objid', 'frame'], ascending=[True, True])
            g_tot_sep_df = self._temporal_group(g_tot_sep_df)
            g_tot_sep_df = self.assign_mjd(g_tot_sep_df)
        else:
            g_tot_sep_df = self.total_df.copy()
            g_tot_sep_df['objid'] = pd.Series(dtype=int)
        filepath = os.path.join(self.grouped_dir, 'grouped_total_sep.csv')
        g_tot_sep_df.to_csv(filepath, index=False)

        full_file_path = os.path.join(self.grouped_dir, 'objects_summary.txt')
        summary_path = os.path.join(self.base_dir, 'interesting_findings.txt')

        with open(summary_path, "a") as summary:
            if len(self.filtered_sep_df) > 0 and len(data) > 0:
                print(f"------------------------------------------------------", file=summary)
                print(f"{self.filename}", file=summary)
                print(f"------------------------------------------------------", file=summary)
                print(f"{num_candidates} asteroid candidates in filtered objects", file=summary)
                if len(data) >= 1:
                    for i in range(len(data)):
                        print(f"----- {data[i]}", file=summary)
                print(" ", file=summary)

        with open(full_file_path, "w") as f:
            print(f"{safe_max(g_tot_sep_df['objid'])} total objects identified by sep", file=f)
            if len(self.filtered_sep_df) > 0:
                print(f"{safe_max(self.events['objid'])} filtered objects identified by sep", file=f)
                print(f"----- {num_candidates} asteroid candidates in filtered objects", file=f)
                if len(data) >= 1:
                    for i in range(len(data)):
                        print(f"---------- {data[i]}", file=f)
            else:
                print('0 objects passed through filtering', file=f)
            if len(self.significance_df) > 0:
                print(f"{safe_max(g_sig_df['objid'])} objects identified by significance", file=f)
            else:
                print('0 objects identified by significance', file=f)

        print(f'{safe_max(g_tot_sep_df["objid"])} total objects identified by sep')
        if len(self.filtered_sep_df) > 0:
            print(f'{safe_max(self.events["objid"])} filtered objects identified by sep')
            print(f"----- {num_candidates} asteroid candidates in filtered objects")
            if len(data) >= 1:
                for i in range(len(data)):
                    print(f"---------- {data[i]}")
        else:
            print('0 objects passed through filtering')
        if len(self.significance_df) > 0:
            print(f'{safe_max(g_sig_df["objid"])} objects identified by significance')
        else:
            print('0 objects identified by significance')


for file in glob.glob('/home/phys/astronomy/jlu69/Masters/jurassic/pipeline_data/Obs/stage1/ast6/*ramp.fits'):
    Jurassic(file, method='mega', num_cores=55)