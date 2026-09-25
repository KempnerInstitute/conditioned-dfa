"""Development-only pooled tuning, horizon, and matched-geometry summaries."""
from collections import defaultdict
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from summarize_development import summarize,choose


def group(record):
    c=record["case"];return f"{c.get('cell','cifar10')}/{c['family']}"


def pool(parent,current):
    """Retain the original candidates; a follow-up cannot discard an old winner."""
    assert parent["complete"]
    records=copy.deepcopy(parent["candidates"])+copy.deepcopy(current["candidates"])
    ids=[r["case"]["id"] for r in records]
    assert len(set(ids))==len(ids),"Cohort case IDs collide"
    groups=defaultdict(list)
    for record in records:groups[group(record)].append(record)
    return {key:choose(values,[r["case"] for r in values]) for key,values in groups.items()}


def deltas(reference,comparison):
    before={r["seed"]:r for r in reference["runs"]}
    after={r["seed"]:r for r in comparison["runs"]}
    assert before.keys()==after.keys()
    values=[]
    for seed in sorted(before):
        a,b=before[seed],after[seed]
        entry=dict(seed=seed,reference_status=a["status"],comparison_status=b["status"])
        if a["status"]==b["status"]=="complete":
            entry.update(accuracy_delta=b["final_validation"]["accuracy"]-a["final_validation"]["accuracy"],
                         ce_delta=b["final_validation"]["loss"]-a["final_validation"]["loss"])
        values.append(entry)
    return values


def report(config_path):
    config_path=Path(config_path);cfg=json.loads(config_path.read_text())
    parent_path=config_path.parent/"parent_summary.json"
    assert hashlib.sha256(parent_path.read_bytes()).hexdigest()==cfg["parent_summary_sha256"]
    parent=json.loads(parent_path.read_text())
    current=summarize(config_path)
    current["interpretation"]="Two paired development seeds, descriptive contrasts only; no confirmation claim"
    kind=cfg.get("followup_kind","geometry")
    if kind=="boundaries":
        current["extension_only_selections"]=current.pop("selections")
        current["selections"]=pool(parent,current)
        current["selection_cohorts"]=[parent["config_sha256"],current["config_sha256"]]
    elif kind=="horizon":
        original={group(r):r for r in parent["candidates"] if r["case"]["id"]==parent["selections"][group(r)]["selected"]["id"]}
        current["horizon_contrasts"]={group(r):dict(parent_case=original[group(r)]["case"],extended_case=r["case"],
                 seed_deltas=deltas(original[group(r)],r)) for r in current["candidates"]}
        current["interpretation"]+="; longer horizon restarts initialization and stretches the schedule"
    else:
        records=current["candidates"]
        coordinate=lambda c:c.get("rho") if c.get("rho") is not None else c["damping"]
        full={(r["case"]["cell"],r["case"]["normalization"],coordinate(r["case"])):r for r in records if r["case"].get("statistic")=="full"}
        contrasts=[]
        for r in records:
            c=r["case"]
            if c.get("statistic") not in {None,"full"}:
                ref=full[c["cell"],c["normalization"],coordinate(c)]
                contrasts.append(dict(cell=c["cell"],normalization=c["normalization"],damping=c["damping"],rho=c.get("rho"),statistic=c["statistic"],seed_deltas=deltas(ref,r)))
        current["matched_damping_contrasts"]=contrasts
        current["geometry_probe_histories"]=[str(Path(cfg["output_root"])/r["case"]["id"]/f"seed_{seed}"/"history.json") for r in records for seed in cfg["development_seeds"]]
        current["interpretation"]+="; provisional first-grid optimizer/rate anchors; five equally searched operators, plus matched raw and batch-space references"
    return current


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();result=report(args.config)
    args.output.parent.mkdir(exist_ok=True,parents=True)
    temporary=args.output.with_suffix(".pending");temporary.write_text(json.dumps(result,indent=2)+"\n");temporary.replace(args.output)
    print(json.dumps({k:result[k] for k in ["endpoint_counts","complete","confirmation_ready"]}),flush=True)
