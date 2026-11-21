"""Prepare analysis-ready matrices from approved frozen clinical/feature files."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

import argparse,json
from npc_eitn.prepare import prepare
from npc_eitn.io import output_dir

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();cohorts,features=prepare(a.config);out=output_dir(a.output)
    for n,d in cohorts.items():d.to_csv(out/(n+'.csv'),index=False)
    cfg=dict(conventional_features=features,expected_counts={k:len(d) for k,d in cohorts.items()},cohorts={k:k+'.csv' for k in cohorts})
    (out/'inputs.json').write_text(json.dumps(cfg,indent=2))
    print('Prepared explicit frozen cohorts: '+str({k:len(d) for k,d in cohorts.items()}))
