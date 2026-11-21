"""Check the distributed code/aggregate inventory and sealed file hashes."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

import argparse,hashlib,json,re
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
def inventory():
    result={}
    for p in ROOT.rglob('*'):
        rel=p.relative_to(ROOT)
        if any(x in {'__pycache__','.git','.venv','build','dist'} or x.endswith('.egg-info') for x in rel.parts):continue
        if p.name.startswith('._') or p.name=='.DS_Store':continue
        if p.is_symlink():raise ValueError('Symlink in distribution')
        if not p.is_file() or p.name=='FILE_MANIFEST.json':continue
        if p.suffix not in {'.py','.md','.toml','.txt','.json','.csv','.yml','.cff'} and p.name not in {'.gitignore','.gitattributes'}:
            raise ValueError('Unreviewed file type: '+str(rel))
        text=p.read_text()
        if re.search(r'/(?:Volumes|Users|home)/',text):raise ValueError('Local absolute path: '+str(rel))
        if re.search(r'(?<![A-Za-z])(?:KY|KS|ZJ|JZ)[:_-]?[0-9]{4,}',text):raise ValueError('Potential study identifier: '+str(rel))
        if p.suffix=='.csv':
            allowed={'prediction_results.csv','paired_comparisons.csv','broad_qc_results.csv','associations.csv','cumulative_incidence_fixed_grid.csv','broader_association_sensitivity.csv','sex_adjusted.csv','external_distributions.csv'}
            if rel.parts[0]!='results' or p.name not in allowed:raise ValueError('Unregistered CSV: '+str(rel))
            columns=set(pd.read_csv(p,nrows=0).columns)
            if columns&{'patient_id','local_id','email','name'}:raise ValueError('Participant-level field in deposited aggregates')
        result[rel.as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    return result

def check():
    expected=json.loads((ROOT/'FILE_MANIFEST.json').read_text())['files'];actual=inventory()
    if expected!=actual:raise ValueError('Sealed file inventory mismatch')
    d=pd.read_csv(ROOT/'results/prediction_results.csv')
    if len(d)!=56 or set(d.model)!={'M1','M2','M3','M4'}:raise ValueError('Current four-model aggregate coverage changed')
    if d.duplicated(['cohort','horizon_months','repeat','model']).any():raise ValueError('Duplicate aggregate groups')
    target={(36,'M4'):.8479651434498274,(60,'M4'):.8101224888260657,
            (36,'M2'):.82469419519245,(60,'M2'):.8012755985679677}
    for (h,m),v in target.items():
        row=d.loc[d.cohort.eq('external')&d.horizon_months.eq(h)&d.model.eq(m)].iloc[0]
        if abs(row.ipcw_auc-v)>1e-6:raise ValueError('Aggregate target no longer matches manuscript')
    a=pd.read_csv(ROOT/'results/associations.csv')
    row=a.loc[a.center.eq('Pooled external')&a.roi.eq('ring15')&a.endpoint.eq('lrr')&a.landmark.eq(0)&a.feature.eq('eitn_z')].iloc[0]
    if row['n']!=203 or row.events!=16 or abs(row.hr-1.75796022452331)>1e-10:raise ValueError('Current main association target changed')
    counts=json.loads((ROOT/'results/cohort_counts.json').read_text())
    required_counts={'phenotype_internal':517,'phenotype_external':203,
        'association_internal_complete':517,'association_external_complete':203,
        'prediction_internal':372,'prediction_external':197,'sensitivity_external_pre_restriction':206}
    if any(counts.get(k)!=v for k,v in required_counts.items()):raise ValueError('Current population reference changed')
    tables=json.loads((ROOT/'results/manuscript_tables.json').read_text())
    if len(tables)!=14 or sum(t['document']=='main.tex' for t in tables)!=4:raise ValueError('Manuscript table coverage changed')
    if 'Analysis cohort, n & 517 & 124 & 79' not in tables[0]['latex']:raise ValueError('Main phenotype table snapshot is stale')
    import npc_eitn
    version=json.loads((ROOT/'FILE_MANIFEST.json').read_text())['version']
    if npc_eitn.__version__!=version or f'version = "{version}"' not in (ROOT/'pyproject.toml').read_text():raise ValueError('Software version metadata mismatch')
    print(json.dumps(dict(status='PASS',files=len(actual),current_prediction_metric_groups=len(d),
        real_participant_rows_in_package=False,scope='Code and aggregate inventory; clinical inputs remain controlled'),indent=2))
if __name__=='__main__':check()
