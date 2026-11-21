"""Run EITN extraction and all four model pipelines on invented inputs."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

import argparse,json
from npc_eitn.io import output_dir
from npc_eitn.synthetic import cohort,registered_images
from npc_eitn.phenotype import extract_arrays
from run_analysis import run

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=output_dir(a.output)
    internal,features=cohort(240,'INT',20261115)
    external,_=cohort(100,'EXT',20261116)
    internal.to_csv(out/'synthetic_internal.csv',index=False)
    external.to_csv(out/'synthetic_external.csv',index=False)
    config=dict(conventional_features=features,cohorts=dict(internal='synthetic_internal.csv',external='synthetic_external.csv'))
    (out/'inputs.json').write_text(json.dumps(config,indent=2))
    images,spacing=registered_images();results,qc=extract_arrays(images,spacing)
    results.to_csv(out/'synthetic_eitn.csv',index=False);(out/'synthetic_image_qc.json').write_text(json.dumps(qc,indent=2))
    run(out/'inputs.json',out/'analysis',demo=True,bootstrap=30)
    print('Synthetic extraction and four-model analysis completed. These are not study results.',flush=True)
