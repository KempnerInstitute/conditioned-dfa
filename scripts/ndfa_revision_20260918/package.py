"""Build a fresh compact review export with corrected science and all follow-ups."""
from pathlib import Path
import argparse,hashlib,io,json,re,sys,tempfile,zipfile,pickle
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from build_ndfa_completion_package import transform_details

def sha(data):return hashlib.sha256(data).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--paper-build',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);paper=ROOT/'drafts/Info-DFA'
    build=json.loads((a.paper_build/'build_receipt.json').read_text());assert build['accepted']
    assert json.loads((ROOT/'results/ndfa_revision_followups_audit_20260918/audit.json').read_text())['complete']
    base=ROOT/'results/ndfa_original_restoration_20260916/v5/iclr_supplement.zip'
    replacements={};records={};derived={}
    text_suffix={'.py','.json','.csv','.md','.tex','.sbatch','.txt','.sty','.bst'}
    def source(name,path):
        data=path.read_bytes();changes=[]
        if path.suffix in text_suffix:
            value,changes=transform_details(data.decode(),ROOT,paper=name.startswith('paper/') and path.suffix=='.tex');data=value.encode()
        replacements[name]=data;records[name]={'original_sha256':sha(path.read_bytes()),'export_sha256':sha(data),'bytes':len(data),'transformations':changes}
    def generated(name,data,changes):
        if not isinstance(data,bytes):data=(json.dumps(data,indent=2)+'\n').encode()
        replacements[name]=data;records[name]={'original_sha256':None,'export_sha256':sha(data),'bytes':len(data),'transformations':changes}
    manifest=json.loads((paper/'manuscript_manifest.json').read_text())
    for name in manifest['file_sha256']:
        if name=='conditioned_dfa_arxiv.tex':continue
        source('paper/'+name,paper/name)
    source('paper/conditioned_dfa_iclr.pdf',a.paper_build/'conditioned_dfa_iclr.pdf')
    # Figure-generator input is a serialized figure object produced locally from
    # the archived numerical plotting script, and is hash-bound like other assets.
    cachebytes=(ROOT/'assets/ndfa_revision_20260918/figure_cache.pkl').read_bytes()
    assert not any(t in cachebytes.lower() for t in [b'hsafaai',b'/n/holylabs',b'/n/home'])
    generated('revision/figure_cache.pkl',cachebytes,['Only figure objects retained; unused local input-path metadata removed.'])
    code=list((ROOT/'scripts/ndfa_revision_20260918').glob('*.py'))
    code=[p for p in code if p.name not in ['original_figures.py','package.py','integrate_followups.py','verify_package.py','visual_check.py']]
    code += [ROOT/'scripts/visual_revision_20260915/redesign.py',ROOT/'analysis/plot_ndfa_factor_mechanism.py',ROOT/'analysis/make_error_kndfa_replication_figure.py']
    for name in ['ndfa_revision_math.py','compute_infodfa_seedlevel_stats.py','aggregate_spatialkron_controls.py','aggregate_spatialkron_sweep.py','aggregate_spatial_kron.py','aggregate_actwhiten.py','aggregate_bpwhiten.py','validate_mode_timing.py']:
        code.append(ROOT/'analysis'/name)
    plan=json.loads((ROOT/'configs/ndfa_revision_followups_20260918.json').read_text())
    code += [ROOT/name for name in plan['source_sha256']]
    code += [ROOT/'tests/test_ndfa_revision_math.py',ROOT/'slurm/ndfa_revision_20260918.sbatch']
    for file in set(code):source(str(file.relative_to(ROOT)),file)
    for name in ['results/infodfa_alignment_dynamics_v1/alignment_dynamics.csv',
                 'results/infodfa_feedback_variance_v1/feedback_variance_cells.csv']:
        source(name,ROOT/name)
    source('verify_revision.py',ROOT/'scripts/ndfa_revision_20260918/verify_revision.py')
    for name in ['ndfa_revision_followups_protocol_20260918.md','ndfa_imagenet_provenance_20260918.json']:
        source('docs/research/'+name,ROOT/'docs/research'/name)
    source('configs/ndfa_revision_followups_20260918.json',ROOT/'configs/ndfa_revision_followups_20260918.json')
    for directory in ['ndfa_revision_saved_audit_20260918','ndfa_revision_followups_audit_20260918','infodfa_seedlevel_stats_corrected_20260918','infodfa_mode_timing_corrected_20260918','ndfa_spatial_controls_corrected_20260918','ndfa_spatial_sweep_corrected_20260918']:
        for file in (ROOT/'results'/directory).rglob('*'):
            if file.is_file() and file.suffix in {'.json','.csv','.md'}:source(str(file.relative_to(ROOT)),file)
    torch.set_num_threads(2)
    runroot=ROOT/'results/ndfa_revision_followups_20260918'
    for file in runroot.rglob('*.json'):source(str(file.relative_to(ROOT)),file)
    for file in runroot.rglob('*.pt'):
        if file.name not in ['validation.pt','test.pt']:continue
        data=torch.load(file,map_location='cpu',weights_only=True);z=data['logits'].double();labels=data['labels'];loss=(torch.logsumexp(z,1)-z[torch.arange(len(z)),labels]).numpy()
        buffer=io.BytesIO();np.savez_compressed(buffer,loss=loss.astype('float32'),prediction=z.argmax(1).numpy().astype('uint8'),label=labels.numpy().astype('uint8'))
        name=str(file.relative_to(ROOT).with_name(file.stem+'_compact.npz'))
        generated(name,buffer.getvalue(),['Per-example float64 CE rounded to float32; exact predicted and true classes; logits and model state omitted.'])
        derived[name]={'original_prediction_sha256':sha(file.read_bytes()),'examples':len(loss),'original_mean_ce':float(loss.mean()),'export_mean_ce':float(loss.astype('float32').astype('float64').mean())}
    generated('revision/COMPACT_PREDICTIONS.json',derived,['Derived prediction provenance'])
    generated('revision/README.md',b'''# September 18 scientific revision

Run `python -B verify_revision.py` after the original `reproduce.py` checks.
This verifies every new endpoint from compact per-example losses/classes, the
corrected whole-seed intervals, the additional diagonal comparison and the
complete-risk counterexample. The original-logit audit was performed before
export; this compact archive does not rerun that logit calculation or inference.
All 202 planned cases are retained, including the 32 validation-only cases.
Checkpoint and optimizer states are omitted; their hashes remain in receipts.

Training source/configuration pins in historical receipts describe ORIGINAL
bytes. Anonymous path substitutions change export bytes. Reproducing training
requires a relocated configuration, appropriate local dataset path, and newly
recorded source pins; do not silently rewrite historical receipts.

The revised figure generator is `scripts/ndfa_revision_20260918/redraw.py`.
Set NDFA_FIGURE_CACHE=revision/figure_cache.pkl, NDFA_PAPER_FIGURES and
NDFA_FIGURE_OUTPUT to new directories outside this untouched extraction.
The cache contains locally constructed Matplotlib figures from retained source
measurements. Use the pinned package artifact, not an unrelated pickle.

Orthogonalized-DFA/Muon is discussed as overlapping related work; no new
optimizer benchmark is claimed. The targeted error-side diagonal ablation
answers the selected mechanism question. Unrelated development studies and
redundant historical graphics remain outside the revised manuscript.
''',['Revision-specific reproduction instructions'])
    archive=a.output/'iclr_supplement.zip'
    with zipfile.ZipFile(base) as old,zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as new:
        oldmanifest=json.loads(old.read('review_package/PACKAGE_MANIFEST.json'))
        for name,record in oldmanifest['files'].items():
            if name.startswith('paper/') or name in replacements or name in ['README.md','CURATION.json']:continue
            data=old.read('review_package/'+name);assert sha(data)==record['export_sha256'];replacements[name]=data;records[name]=record
        readme=old.read('review_package/README.md').decode()
        readme='# Current scientific revision — 18 September 2026\n\nThe paper and revised analyses supersede the earlier presentation. Run:\n\n```bash\npython -B reproduce.py --checks integrity,statistics,legacy,focused,paper,smoke\npython -B verify_revision.py\npython -B -m pytest -q -p no:cacheprovider tests/test_ndfa_revision_math.py\n```\n\nSee revision/README.md for new evidence and export scope. Historical claims of unchanged mathematics or figure assets no longer apply.\n\n## Earlier package documentation\n\n'+readme
        generated('README.md',readme.encode(),['Current instructions plus historical provenance'])
        cur=json.loads(old.read('review_package/CURATION.json'));cur['paper_revision']={'scope':'Complete-risk and seed-dependence corrections; all targeted follow-ups; compact revised figures','scientific_data_and_code_unchanged':False}
        generated('CURATION.json',cur,['Updated scope'])
        # Bind only active scientific assets and local source; no identifying entrypoint.
        pubmanifest={'schema_version':3,'release_status':'revised_scientific_review_export','files':records,'base_archive_sha256':sha(base.read_bytes())}
        manifestbytes=(json.dumps(pubmanifest,indent=2)+'\n').encode()
        for name,data in sorted({**replacements,'PACKAGE_MANIFEST.json':manifestbytes}.items()):
            info=zipfile.ZipInfo('review_package/'+name,date_time=(1980,1,1,0,0,0));info.create_system=0;info.compress_type=zipfile.ZIP_DEFLATED;new.writestr(info,data,compresslevel=9)
    receipt={'files':len(records),'zip_bytes':archive.stat().st_size,'zip_sha256':sha(archive.read_bytes()),'manifest_sha256':sha(manifestbytes),'paper_sha256':build['versions']['iclr']['sha256'],'within_100MB':archive.stat().st_size<100_000_000,'validation':'pending'}
    (a.output/'build_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');assert receipt['within_100MB'],receipt
    print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
