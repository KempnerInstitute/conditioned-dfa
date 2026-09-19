"""All-candidate/all-seed report; confirmation inference requires the complete cohort."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
import statistics

from common import read_config, require, sha256, write_json


def moments(values):
    from scipy.stats import t
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values)>1 else None
    half = float(t.ppf(.975,len(values)-1))*sd/len(values)**.5 if sd is not None else None
    return {"n":len(values),"mean":mean,"sd":sd,"ci95":None if half is None else [mean-half,mean+half],
            "values":values,"positive":sum(v>0 for v in values),"negative":sum(v<0 for v in values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    parser.add_argument("--config-sha256",required=True)
    args = parser.parse_args()
    config = read_config(args.config,args.config_sha256)
    out = Path(config["output_root"])
    workflow = json.loads((out/"workflow.json").read_text())
    audit = json.loads((out/"audit.json").read_text())
    selection = json.loads((out/"selection.json").read_text())
    require(sha256(out/"selection.json") == workflow["selection_sha256"],"selection changed")
    require(audit["config_sha256"] == args.config_sha256 and audit["source_sha256"] == config["source_sha256"],"audit config/source mismatch")
    require(audit["selection_sha256"] == workflow["selection_sha256"] and audit["bound_case_inventory"] == workflow["cases"],"audit selection/case binding mismatch")
    for name in ("summary.json","report.md","histories.csv","curves.pdf","curves.png"):
        require(not (out/name).exists(),"refusing existing report artifact: "+name)
    histories=[]
    for case in workflow["cases"]:
        if case["status"] in ("complete","numerical_failed"):
            for name,digest in case["artifacts_sha256"].items():
                require(sha256(out/case["case_id"]/name) == digest,"post-audit artifact changed")
            rows=json.loads((out/case["case_id"]/"metrics.json").read_text())
            histories.extend(dict(row,case_id=case["case_id"],stage=case["stage"],candidate_id=case["candidate_id"],case_status=case["status"]) for row in rows)
    with (out/"histories.csv").open("x",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(histories[0]))
        writer.writeheader();writer.writerows(histories)
    summary={"config_sha256":args.config_sha256,"audit_sha256":sha256(out/"audit.json"),
             "selection_sha256":sha256(out/"selection.json"),"audit_accepted":audit["accepted"],
             "all_cases":workflow["cases"],"selection":selection,"confirmation_summaries":{},"contrasts":[],
             "official_test_evaluation":False,"allocation_gpu_hours":"record separately from terminal Slurm accounting",
             "scope":"Fixed BN48k anchor; fresh paired training seeds conditional on reused validation identities; finite matched nine-candidate search, not a tuned-convergence comparison"}
    lines=["# BN forward-decorrelation and measured-work comparison","",
           f"Complete-cohort audit accepted: **{audit['accepted']}**. Confirmation cases: {audit['complete_confirmation']}/32.","",
           "Every method receives momentum SGD, the same nine-candidate/two-seed search budget, and 30 seconds of measured synchronized update work. The clock excludes evaluation/export; its final overshoot is charged and reported. Data loading and full child duration are separate costs.","",
           "| Method | Selected peak LR | A damping / FD learning rate | Boundary dimensions | Development mean CE |","|---|---:|---:|---|---:|"]
    for method,row in selection["selected"].items():
        lines.append(f"| {method} | {row['peak_lr']} | {row.get('relative_damping',row.get('decorrelation_lr','—'))} | {row['boundary_dimensions']} | {row['mean_ce']:.6f} |")
    lines += ["","All candidate scores and failures are in summary.json. A boundary selection is retained without an automatic search extension. This fixed family does not establish tuned convergence or optimizer-independent superiority."]
    if audit["accepted"]:
        index={(c["method"],c["seed"]):c for c in workflow["cases"] if c["stage"]=="confirmation"}
        lines += ["","| Method | Accuracy (%) mean ± SD | CE mean ± SD | Updates mean | Update seconds mean | Child wall seconds mean |","|---|---:|---:|---:|---:|---:|"]
        for method in config["methods"]:
            cases=[index[method,seed] for seed in config["confirmation_seeds"]]
            values={key:moments([case["endpoint"][field]*scale for case in cases]) for key,field,scale in
                    [("accuracy_pp","validation_accuracy",100),("cross_entropy","validation_loss",1),("updates","step",1),("update_seconds","training_seconds",1)]}
            values["child_wall_seconds"]=moments([case["invocation_wall_seconds"] for case in cases])
            values["peak_cuda_allocated_bytes"]=moments([case["endpoint"]["peak_cuda_allocated_bytes"] for case in cases])
            values["decorrelation_learned_state_count"]=moments([case["endpoint"]["decorrelation_learned_state_count"] for case in cases])
            summary["confirmation_summaries"][method]=values
            lines.append(f"| {method} | {values['accuracy_pp']['mean']:.4f} ± {values['accuracy_pp']['sd']:.4f} | {values['cross_entropy']['mean']:.6f} ± {values['cross_entropy']['sd']:.6f} | {values['updates']['mean']:.1f} | {values['update_seconds']['mean']:.6f} | {values['child_wall_seconds']['mean']:.4f} |")
        for treatment,control in (("ndfa","fd_dfa"),("ndfa","dfa"),("ndfa","bp"),("fd_dfa","dfa")):
            primary = treatment == "ndfa" and control == "fd_dfa"
            contrast={"name":treatment+" minus "+control,"endpoint":"30 measured update seconds","primary":primary}
            for key,field,scale in [("accuracy_pp","validation_accuracy",100),("cross_entropy","validation_loss",1),("update_seconds","training_seconds",1)]:
                contrast[key]=moments([(index[treatment,seed]['endpoint'][field]-index[control,seed]['endpoint'][field])*scale for seed in config['confirmation_seeds']])
            contrast["child_wall_seconds"]=moments([index[treatment,seed]['invocation_wall_seconds']-index[control,seed]['invocation_wall_seconds'] for seed in config['confirmation_seeds']])
            if primary:
                vals=contrast["accuracy_pp"]["values"];observed=abs(sum(vals))
                contrast["supplemental_two_sided_signflip_p"]=sum(abs(sum(s*v for s,v in zip(signs,vals))) >= observed-1e-12 for signs in itertools.product((-1,1),repeat=8))/256
            summary["contrasts"].append(contrast)
            lines += ["",f"{contrast['name']}: accuracy {contrast['accuracy_pp']['mean']:+.4f} pp, paired 95% t interval {contrast['accuracy_pp']['ci95']}; CE {contrast['cross_entropy']['mean']:+.6f}, interval {contrast['cross_entropy']['ci95']}."]
        # Each declared secondary contrast requires all eight pairs to reach 5000.
        fixed={(r['method'],r['seed']):r for r in histories if r['stage']=='confirmation' and r['step']==config['secondary_step_endpoint']}
        summary["same_step_secondary_complete"]=len(fixed)==32
        summary["same_step_secondary_reached"]={m:sum((m,seed) in fixed for seed in config['confirmation_seeds']) for m in config['methods']}
        for treatment,control in (("ndfa","fd_dfa"),("ndfa","dfa"),("ndfa","bp"),("fd_dfa","dfa")):
            if all((m,seed) in fixed for m in (treatment,control) for seed in config['confirmation_seeds']):
                summary["contrasts"].append({"name":treatment+" minus "+control,"endpoint":"secondary step5000","primary":False,
                    **{key:moments([(fixed[treatment,seed][field]-fixed[control,seed][field])*scale for seed in config['confirmation_seeds']])
                       for key,field,scale in [("accuracy_pp","validation_accuracy",100),("cross_entropy","validation_loss",1),("update_seconds","training_seconds",1)]}})
        lines += ["","One primary comparison: A−FD-DFA final accuracy. The exact two-sided sign-flip test is supplemental and assumes exchangeable paired signs. Other metrics and comparisons are descriptive; all are retained. Intervals describe training-seed uncertainty on the same validation split, not uncertainty from sampling a new dataset.","","FD-DFA uses Ahmad's pinned dense forward mechanism before every linear map, retaining BN and direct feedback injected before the outgoing decorator. It is a matched adaptation, not a reproduction of published FA/Adam training. All decorator calculations and numerical guards are charged to update work; persistent decorator matrices/means and validation cost are separately visible in case artifacts."]
    else:
        lines += ["","**No confirmation inference:** the complete prespecified cohort was not accepted. All recorded cases and failures remain in the inventory."]
    lines += ["","All successful cases save final inference/BN/decorator states, validation logits, optimizer/sampler artifacts, and complete declared histories. Step5000 has metrics only. Exact training resume and independent GPU replay are not claimed. No official-test data were loaded."]
    write_json(out/"summary.json",summary)
    (out/"report.md").write_text("\n".join(lines)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(1,2,figsize=(9,3.5),layout="constrained")
    colors={"bp":"#222222","dfa":"#7F7F7F","ndfa":"#0072B2","fd_dfa":"#D55E00"}
    for method in config["methods"]:
        for i,seed in enumerate(config["confirmation_seeds"]):
            curve=[r for r in histories if r['stage']=='confirmation' and r['method']==method and r['seed']==seed]
            for axis,metric,scale in [(axes[0],"validation_accuracy",100),(axes[1],"validation_loss",1)]:
                axis.plot([r['training_seconds'] for r in curve],[r[metric]*scale for r in curve],
                          color=colors[method],alpha=.5,lw=1,label=method if i==0 else None)
    for axis in axes:
        axis.set_xlabel("Synchronized update work (s)");axis.grid(alpha=.2);axis.legend(frameon=False)
    axes[0].set_ylabel("Validation accuracy (%)");axes[1].set_ylabel("Validation cross-entropy")
    figure.suptitle("All eight fresh seed trajectories; fixed BN48k protocol")
    figure.savefig(out/"curves.pdf");figure.savefig(out/"curves.png",dpi=160);plt.close(figure)
    print(json.dumps({"audit_accepted":audit['accepted'],"report_dir":str(out)}))


if __name__=="__main__":
    main()
