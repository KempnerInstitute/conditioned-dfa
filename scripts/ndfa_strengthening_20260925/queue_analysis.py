"""Freeze analyses and run them after all declared study arrays terminate."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
from scripts.ndfa_strengthening_20260925.freeze_round2 import ROOT,sha


def submit(study):
    out=ROOT/'results/ndfa_strengthening_20260925'/study
    receipt_path=out/'submission.json';receipt=json.loads(receipt_path.read_text())
    if receipt.get('summary_job'):return receipt['summary_job']
    kind=json.loads((out/'config.json').read_text()).get('followup_kind')
    name='summarize_alignment.py' if kind.startswith('alignment') else ('summarize_development.py' if kind=='digits' else 'summarize_round2.py')
    frozen=out/'summary_dispatch_v1';frozen.mkdir(exist_ok=False)
    hashes={}
    for file in ['summarize_development.py','summarize_round2.py','summarize_alignment.py']:
        dest=frozen/file;shutil.copy2(ROOT/'scripts/ndfa_strengthening_20260925'/file,dest);hashes[file]=sha(dest)
    batch=frozen/'summary.sbatch';shutil.copy2(ROOT/'slurm/ndfa_strengthening_summary_20260925.sbatch',batch);hashes[batch.name]=sha(batch)
    jobs=[receipt[k] for k in ['job','verification_job','remaining_job'] if k in receipt]
    assert jobs
    output=out/'followup_summary.json'
    env=os.environ.copy();env.update(NDFA_SUMMARY_SCRIPT=str(frozen/name),NDFA_SUMMARY_CONFIG=str(out/'config.json'),NDFA_SUMMARY_OUTPUT=str(output))
    cmd=['sbatch','--parsable','--dependency=afterany:'+':'.join(jobs),str(batch)]
    r=subprocess.run(cmd,env=env,text=True,capture_output=True)
    if r.returncode:raise RuntimeError(r.stderr)
    receipt.update(summary_job=r.stdout.strip(),summary=str(output),summary_source_sha256=hashes,summary_command=cmd,summary_submitted_utc=datetime.now(timezone.utc).isoformat())
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt['summary_job']


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('studies',nargs='+')
    args=p.parse_args()
    for study in args.studies:print(json.dumps(dict(study=study,summary_job=submit(study))),flush=True)
