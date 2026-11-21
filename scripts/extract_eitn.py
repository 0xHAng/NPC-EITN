"""Extract six EITN ROI summaries from one registered examination pair."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

import argparse,json
import nibabel as nib
import numpy as np
from npc_eitn.io import output_dir
from npc_eitn.phenotype import extract_arrays,CFG

FILES=dict(pre_t1='mr_pre_t1.nii.gz',pre_t1ce='mr_pre_t1ce.nii.gz',pre_t2='mr_pre_t2.nii.gz',
           post_t1='mr_post_t1.nii.gz',post_t1ce='mr_post_t1ce.nii.gz',post_t2='mr_post_t2.nii.gz',
           gtv='gtv_mr_pre_gtvnx.nii.gz',ct='ct.nii.gz')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();images={k:nib.load(a.input_dir/v) for k,v in FILES.items()};ref=images['ct']
    for name,image in images.items():
        if image.shape!=ref.shape or len(image.shape)!=3:raise ValueError('Registered grid shape mismatch: '+name)
        if not np.isfinite(image.affine).all() or np.max(abs(image.affine-ref.affine))>CFG['affine_tolerance']:
            raise ValueError('Registered affine mismatch: '+name)
        if np.max(abs(np.array(image.header.get_zooms()[:3])-np.array(ref.header.get_zooms()[:3])))>CFG['spacing_tolerance_mm']:
            raise ValueError('Registered voxel spacing mismatch: '+name)
    data={k:v.get_fdata(dtype=np.float32) for k,v in images.items()}
    result,qc=extract_arrays(data,ref.header.get_zooms()[:3]);out=output_dir(a.output)
    result.to_csv(out/'eitn_features.csv',index=False);(out/'automatic_qc.json').write_text(json.dumps(qc,indent=2))
