"""Prepare an isolated relocated training copy; historical inputs stay intact."""
from pathlib import Path
import argparse,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[2]
def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--destination',type=Path,required=True);a=p.parse_args()
    dest=a.destination.resolve();dest.mkdir(parents=True,exist_ok=False)
    original=ROOT/'configs/ndfa_revision_followups_20260918.json';plan=json.loads(original.read_text())
    names=set(plan['source_sha256'])
    for directory in ['infogeo','experiments']:
        names.update(str(f.relative_to(ROOT)) for f in (ROOT/directory).glob('*.py'))
    for name in names:
        target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    cfg=dest/'configs/ndfa_bn_forward_decorrelation_20260915.json';base=json.loads(cfg.read_text());base['shared_args']['data_dir']=str(a.data_dir.resolve());cfg.write_text(json.dumps(base,indent=2)+'\n')
    plan['reproduction_parent_plan_sha256']=hashlib.sha256(original.read_bytes()).hexdigest();plan['output_root']=str(dest/'results/rerun');plan['status']='relocated_reproduction_plan_not_original_execution'
    plan['source_sha256']={name:hashlib.sha256((dest/name).read_bytes()).hexdigest() for name in plan['source_sha256']}
    target=dest/'configs/reproduction_plan.json';target.write_text(json.dumps(plan,indent=2)+'\n');print(target)
if __name__=='__main__':main()
