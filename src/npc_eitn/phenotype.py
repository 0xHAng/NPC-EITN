"""EITN extraction from six MRI volumes already registered to planning CT.

Computational definitions are retained from the as-run feature extractor. This
module accepts registered arrays; it does not register raw DICOM examinations.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter
CFG=dict(smooth_sigma_vox=1.0,ct_hu_low=-300.,ct_hu_high=300.,min_valid_voxels=10,
         coverage_threshold=.8,affine_tolerance=.001,spacing_tolerance_mm=.001)
ROIS=('core','ring5','ring5_10','ring10_15','ring10','ring15')
def ball(spacing, radius):
    r = [int(np.ceil(radius / s)) for s in spacing]
    grid = np.mgrid[-r[0]:r[0]+1, -r[1]:r[1]+1, -r[2]:r[2]+1]
    dist = np.sqrt(sum((grid[i] * spacing[i]) ** 2 for i in range(3)))
    return dist <= radius


def smooth_masked(z, mask):
    sigma = CFG["smooth_sigma_vox"]
    sm = gaussian_filter(z * mask, sigma=sigma)
    w = gaussian_filter(mask.astype(np.float32), sigma=sigma)
    out = np.zeros_like(sm, dtype=np.float32)
    np.divide(sm, w, out=out, where=w > 0.1)
    return out


def crop_slices(gtv, spacing):
    """Crop to GTV + 15 mm analysis radius + 6-voxel Gaussian safety margin."""
    pos = np.where(gtv)
    slices = []
    for i in range(3):
        pad = int(np.ceil(15.0 / spacing[i])) + 6
        lo = max(0, int(pos[i].min()) - pad)
        hi = min(gtv.shape[i], int(pos[i].max()) + pad + 1)
        slices.append(slice(lo, hi))
    return tuple(slices)


def exact_histogram_map_crop(pre, post, tissue, slc):
    """Project13/skimage empirical-CDF mapping, evaluated only in the ROI crop.

    Avoids the huge return_inverse array and full-volume mapped image while
    preserving the same unique-value cumulative-distribution transform.
    """
    tissue_mask = tissue & (pre > 0) & np.isfinite(pre)
    pcrop = post[slc].copy()
    mcrop = tissue_mask[slc]
    if tissue_mask.sum() < 20:
        return pcrop
    source = post[tissue_mask]
    template = pre[tissue_mask]
    src_values, src_counts = np.unique(source, return_counts=True)
    tmpl_values, tmpl_counts = np.unique(template, return_counts=True)
    src_q = np.cumsum(src_counts, dtype=np.float64) / source.size
    tmpl_q = np.cumsum(tmpl_counts, dtype=np.float64) / template.size
    mapped_values = np.interp(src_q, tmpl_q, tmpl_values).astype(np.float32)
    idx = np.searchsorted(src_values, pcrop[mcrop])
    idx = np.clip(idx, 0, len(mapped_values) - 1)
    pcrop[mcrop] = mapped_values[idx]
    return pcrop


def zscore_pair_crop(pre, post, gtv, tissue, slc):
    post_crop = exact_histogram_map_crop(pre, post, tissue, slc)
    ref = pre[(pre > 0) & np.isfinite(pre) & ~gtv]
    if ref.size < 20:
        ref = pre[(pre > 0) & np.isfinite(pre)]
    if ref.size < 20 or float(ref.std()) < 1e-6:
        raise ValueError("reference_std_or_size_invalid")
    mu, sd = float(ref.mean()), float(ref.std())
    pre_crop = pre[slc]
    tissue_crop = tissue[slc]
    pre_mask = tissue_crop & (pre_crop > 0) & np.isfinite(pre_crop)
    post_mask = tissue_crop & (post_crop > 0) & np.isfinite(post_crop)
    return smooth_masked((pre_crop - mu) / sd, pre_mask), smooth_masked((post_crop - mu) / sd, post_mask)


def features(mask, arrays, geom_n, voxel_volume, gtv_volume, prefix=""):
    n = int(mask.sum())
    base = {
        "valid_nvox": n, "geometric_nvox": int(geom_n),
        "coverage": n / geom_n if geom_n else np.nan,
        "valid_volume_cm3": n * voxel_volume / 1000.0,
        "geometric_volume_cm3": geom_n * voxel_volume / 1000.0,
        "gtv_volume_cm3": gtv_volume,
    }
    if n < CFG["min_valid_voxels"]:
        for k in ["q_hh", "q_hl", "q_lh", "q_ll", "enh_up_prop", "t2_nonup_prop",
                  "mean_delta_enh", "median_delta_enh", "mean_delta_t2", "median_delta_t2",
                  "mean_pre_enh", "mean_post_enh", "mean_pre_t2", "mean_post_t2"]:
            base[k] = np.nan
        return base
    de, dt = arrays["delta_enh"][mask], arrays["delta_t2"][mask]
    pe, poe = arrays["pre_enh"][mask], arrays["post_enh"][mask]
    pt, pot = arrays["pre_t2"][mask], arrays["post_t2"][mask]
    base.update({
        "q_hh": np.mean((de > 0) & (dt > 0)), "q_hl": np.mean((de > 0) & (dt <= 0)),
        "q_lh": np.mean((de <= 0) & (dt > 0)), "q_ll": np.mean((de <= 0) & (dt <= 0)),
        "enh_up_prop": np.mean(de > 0), "t2_nonup_prop": np.mean(dt <= 0),
        "mean_delta_enh": np.mean(de), "median_delta_enh": np.median(de),
        "mean_delta_t2": np.mean(dt), "median_delta_t2": np.median(dt),
        "mean_pre_enh": np.mean(pe), "mean_post_enh": np.mean(poe),
        "mean_pre_t2": np.mean(pt), "mean_post_t2": np.mean(pot),
    })
    sd_de, sd_dt = float(np.std(de)), float(np.std(dt))
    for a in [-0.1, 0.0, 0.1]:
        for b in [-0.1, 0.0, 0.1]:
            an = str(a).replace("-", "m").replace(".", "p")
            bn = str(b).replace("-", "m").replace(".", "p")
            base[f"q_hl_thr_de_{an}_dt_{bn}"] = np.mean((de > a * sd_de) & (dt <= b * sd_dt))
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in base.items()}



def extract_arrays(images, spacing_mm):
    """Return six ROI summaries and automatic QC; physician QC is separate.

    images must contain pre_t1, pre_t1ce, pre_t2, post_t1, post_t1ce,
    post_t2, gtv and ct, all on the same native CT grid. Axis order must match
    spacing_mm. Geometric validation of affine matrices is performed by the CLI.
    """
    required=('pre_t1','pre_t1ce','pre_t2','post_t1','post_t1ce','post_t2','gtv','ct')
    if set(required)-set(images):
        raise ValueError('Missing registered volumes: '+str(sorted(set(required)-set(images))))
    ims={k:np.asarray(images[k],dtype=np.float32) for k in required}
    shape=ims['ct'].shape
    if len(shape)!=3 or any(x.shape!=shape for x in ims.values()):
        raise ValueError('All registered images must have the same 3D shape')
    spacing=np.asarray(spacing_mm,float)
    if spacing.shape!=(3,) or not np.isfinite(spacing).all() or (spacing<=0).any():
        raise ValueError('Three finite positive spacings are required')
    gtv=np.isfinite(ims['gtv'])&(ims['gtv']>0)
    if not gtv.any():raise ValueError('Empty GTVnx')
    volume=float(gtv.sum()*np.prod(spacing)/1000.)
    slc=crop_slices(gtv,spacing);g=gtv[slc]
    pre,post=ims['pre_t1'],ims['post_t1']
    tissue=((pre>0)|(post>0))&np.isfinite(pre)&np.isfinite(post)
    common=np.ones(g.shape,bool)
    for k in required[:6]:common &= np.isfinite(ims[k][slc])&(ims[k][slc]!=0)
    ct=ims['ct'][slc]
    valid_ct=np.isfinite(ct)&(ct>CFG['ct_hu_low'])&(ct<CFG['ct_hu_high'])
    a,b=zscore_pair_crop(pre,post,gtv,tissue,slc)
    c,d=zscore_pair_crop(ims['pre_t1ce'],ims['post_t1ce'],gtv,tissue,slc)
    e,f=zscore_pair_crop(ims['pre_t2'],ims['post_t2'],gtv,tissue,slc)
    arrays=dict(pre_enh=c-a,post_enh=d-b,pre_t2=e,post_t2=f)
    arrays['delta_enh']=arrays['post_enh']-arrays['pre_enh']
    arrays['delta_t2']=arrays['post_t2']-arrays['pre_t2']
    dist=distance_transform_edt(~g,sampling=spacing)
    geom=dict(core=g,ring5=(~g)&(dist>0)&(dist<=5),ring5_10=(dist>5)&(dist<=10),
              ring10_15=(dist>10)&(dist<=15))
    geom['ring10']=geom['ring5']|geom['ring5_10'];geom['ring15']=geom['ring10']|geom['ring10_15']
    geom={k:(v if k=='core' else v&valid_ct) for k,v in geom.items()}
    rows=[dict(roi=r,**features(geom[r]&common,arrays,int(geom[r].sum()),float(np.prod(spacing)),volume)) for r in ROIS]
    result=pd.DataFrame(rows)
    enough=result.valid_nvox.ge(CFG['min_valid_voxels']).all()
    coverage=result.set_index('roi').coverage
    qc=dict(automatic_qc_pass=bool(enough and coverage['core']>=.8 and coverage['ring15']>=.8),
            gtv_volume_cm3=volume,spacing_mm=spacing.tolist(),shape=list(shape),
            core_common_coverage=float(coverage['core']),ring15_common_coverage=float(coverage['ring15']))
    return result,qc
