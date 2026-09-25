"""Audit frozen validation grids and flag settings needing further development.

This does not evaluate test data or authorize a confirmation experiment.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()


def choose(records,case_specs):
    """Only complete a family's selection after all declared candidates end."""
    if any(r["status"]=="incomplete" for r in records):return dict(status="incomplete",selected=None)
    eligible=[r for r in records if r["status"]=="complete"]
    if not eligible:return dict(status="all_candidates_failed",selected=None)
    best=min(eligible,key=lambda r:(r["mean_validation_loss"],-r["mean_validation_accuracy"],r["case"]["id"]))
    spec=best["case"]
    comparable=[c for c in case_specs if c["optimizer"]==spec["optimizer"]]
    boundaries={}
    for name in ["lr","rho","damping","decor_lr"]:
        if spec.get(name) is None:continue
        values=sorted({c[name] for c in comparable if c.get(name) is not None})
        if len(values)>1 and spec[name] in {values[0],values[-1]}:
            boundaries[name]=dict(selected=spec[name],range=[values[0],values[-1]])
    return dict(status="development_selected",selected=spec,mean_validation_loss=best["mean_validation_loss"],
                mean_validation_accuracy=best["mean_validation_accuracy"],boundary_flags=boundaries,
                seed_values=best["runs"],confirmation_ready=False)


def summarize(config_path):
    config_path=Path(config_path);cfg=json.loads(config_path.read_text())
    for name,expected in cfg["source_sha256"].items():
        assert sha(config_path.parent/"source"/name)==expected,name
    config_hash=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    records=[];counts=Counter();groups=defaultdict(list)
    for case in cfg["cases"]:
        runs=[]
        for seed in cfg["development_seeds"]:
            folder=Path(cfg["output_root"])/case["id"]/f"seed_{seed}"
            if not (folder/"endpoint.json").exists():
                runs.append(dict(seed=seed,status="incomplete"));counts["incomplete"]+=1
                continue
            e=json.loads((folder/"endpoint.json").read_text());m=json.loads((folder/"manifest.json").read_text())
            assert e["case"]==m["case"]==case and e["seed"]==seed
            assert m["protocol"]==cfg
            assert e["official_test_loaded"] is False and m["official_test_loaded"] is False
            assert e["status"] in {"complete","numerical_failure"}
            if "manifest_sha256" in e:assert sha(folder/"manifest.json")==e["manifest_sha256"]
            if "config_hash" in e:assert e["config_hash"]==config_hash
            assert sha(folder/"final.pt")==e.get("checkpoint_sha256",e.get("final_checkpoint_sha256"))
            row=dict(seed=seed,status=e["status"],updates=e["completed_updates"],error=e["error"],
                     training_seconds=e["training_seconds"],final_validation=e["final_validation"])
            if e["status"]=="complete":
                n=cfg["cells"][case["cell"]]["n_train"] if cfg["dataset"]=="synthetic" else 45000
                assert e["completed_updates"]==math.ceil(n/cfg["batch_size"])*cfg["epochs"]
                history=json.loads((folder/"history.json").read_text())
                assert history[-1]["epoch"]==cfg["epochs"] and history[-1]["validation"]==e["final_validation"]
                assert math.isfinite(e["final_validation"]["loss"])
                late=[h for h in history if h["epoch"]>=.75*cfg["epochs"]]
                row["late_curve"]=[{k:h.get(k) for k in ["epoch","training_loss","validation"]} for h in late]
            runs.append(row);counts[e["status"]]+=1
        status="incomplete" if any(r["status"]=="incomplete" for r in runs) else (
            "numerical_failure" if any(r["status"]=="numerical_failure" for r in runs) else "complete")
        record=dict(case=case,status=status,runs=runs)
        if status=="complete":
            record.update(mean_validation_loss=statistics.mean(r["final_validation"]["loss"] for r in runs),
                          mean_validation_accuracy=statistics.mean(r["final_validation"]["accuracy"] for r in runs))
        records.append(record);groups[(case.get("cell",cfg["dataset"]),case["family"])].append(record)
    selections={f"{cell}/{family}":choose(values,[r["case"] for r in values]) for (cell,family),values in groups.items()}
    return dict(created_utc=datetime.now(timezone.utc).isoformat(),config_sha256=sha(config_path),
                official_test_loaded=False,replication_unit="paired development seed; per-seed values, no inferential intervals",
                selection_rule=cfg["selection"],endpoint_counts=dict(counts),
                complete=not counts["incomplete"],confirmation_ready=False,selections=selections,candidates=records)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();summary=summarize(args.config)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_suffix(".pending")
    temporary.write_text(json.dumps(summary,indent=2)+"\n");temporary.replace(args.output)
    print(json.dumps({k:summary[k] for k in ["endpoint_counts","complete","confirmation_ready"]}))
