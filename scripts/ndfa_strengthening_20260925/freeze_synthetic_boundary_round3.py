"""One bounded extension of the two remaining synthetic damping boundaries."""
import copy
from datetime import datetime,timezone
import json
import shutil
from scripts.ndfa_strengthening_20260925.freeze_round2 import ROOT,sha,extension_cases,submit
from scripts.ndfa_strengthening_20260925.summarize_round2 import pool
from scripts.ndfa_strengthening_20260925.queue_analysis import submit as queue_summary


def freeze():
    root=ROOT/'results/ndfa_strengthening_20260925'
    first=root/'synthetic_development_v1';second=root/'synthetic_boundaries_v1'
    output=root/'synthetic_boundaries_v2'
    if (output/'submission.json').exists():return json.loads((output/'submission.json').read_text())
    old=json.loads((first/'development_summary.json').read_text())
    extension=json.loads((second/'local_audit_summary.json').read_text())
    assert old['complete'] and extension['complete']
    assert old['config_sha256']==sha(first/'config.json') and extension['config_sha256']==sha(second/'config.json')
    merged=copy.deepcopy(extension);merged['candidates']=old['candidates']+extension['candidates']
    merged['selections']=pool(old,extension)
    merged['cohorts']=[dict(config_sha256=sha(p/'config.json'),summary_sha256=sha(p/s)) for p,s in [(first,'development_summary.json'),(second,'local_audit_summary.json')]]
    config=json.loads((first/'config.json').read_text())
    for rel,digest in config['source_sha256'].items():assert sha(first/'source'/rel)==digest
    config['cases']=[r['case'] for r in merged['candidates']]
    cases,ledger=extension_cases(config,merged)
    assert len(cases)==12 and len(ledger)==2, 'Review scope if more than the two audited boundary families need extension'
    output.mkdir(exist_ok=False);shutil.copytree(first/'source',output/'source')
    shutil.copy2(__file__,output/'freeze_synthetic_boundary_round3.py')
    parent=output/'parent_summary.json';parent.write_text(json.dumps(merged,indent=2)+'\n')
    config.update(cases=cases,output_root=str(output/'runs'),frozen_utc=datetime.now(timezone.utc).isoformat(),
                  parent_summary_sha256=sha(parent),parent_config_sha256=sha(first/'config.json'),selection_cohorts=merged['cohorts'],
                  followup_kind='boundaries',extension_ledger=ledger,
                  selection='Pooled original, first extension and second extension candidates; lowest mean final validation CE, accuracy then unique case ID',
                  scope='One additional bounded application of the same outward 3x/10x boundary policy to the two unresolved task-aligned activity families. No optional seed extension; further boundaries require review, not automatic resubmission.')
    path=output/'config.json';path.write_text(json.dumps(config,indent=2)+'\n')
    receipt=submit(path,'kempner_eng',12)
    queue_summary(output.name)
    return receipt


if __name__=='__main__':print(json.dumps(freeze()),flush=True)
