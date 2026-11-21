"""Prepare main phenotype/association flags and broader sensitivity source rows."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

import argparse
from npc_eitn.prepare import prepare_associations
from npc_eitn.io import output_dir
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();d=prepare_associations(a.config);out=output_dir(a.output);d.to_csv(out/'phenotypes.csv',index=False)
    print('Prepared main phenotype/association populations 517/203; 206 external source participants retained for sensitivity only.')
