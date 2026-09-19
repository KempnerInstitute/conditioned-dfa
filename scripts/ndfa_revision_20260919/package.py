"""Update the verified compact evidence archive with the curated manuscript."""
from pathlib import Path
import argparse,hashlib,json,sys,zipfile
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from build_ndfa_completion_package import transform_details


def sha(b):return hashlib.sha256(b).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--paper-build',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--base',type=Path,default=ROOT/'results/ndfa_revision_release_20260918/review_04/iclr_supplement.zip')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    paper=ROOT/'drafts/Info-DFA';build=json.loads((args.paper_build/'build_receipt.json').read_text());assert build['accepted']
    manifest=json.loads((paper/'manuscript_manifest.json').read_text());payload={};records={}
    with zipfile.ZipFile(args.base) as old:
        old_manifest=json.loads(old.read('review_package/PACKAGE_MANIFEST.json'))
        for name,record in old_manifest['files'].items():
            if name.startswith('paper/'):continue
            data=old.read('review_package/'+name);assert sha(data)==record['export_sha256']
            payload[name]=data;records[name]=record
    def add(name,path):
        data=path.read_bytes();changes=[]
        if path.suffix in {'.py','.tex','.sty','.bst','.json','.csv','.md'}:
            value,changes=transform_details(data.decode(),ROOT,paper=name.startswith('paper/') and path.suffix=='.tex')
            data=value.encode()
        payload[name]=data;records[name]={'original_sha256':sha(path.read_bytes()),'export_sha256':sha(data),
                                        'bytes':len(data),'transformations':changes}
    def generated(name,text):
        data=text.encode();payload[name]=data;records[name]={'original_sha256':None,'export_sha256':sha(data),'bytes':len(data),'transformations':['Second-review documentation']}
    for name in manifest['file_sha256']:
        if name!='conditioned_dfa_arxiv.tex':add('paper/'+name,paper/name)
    add('paper/conditioned_dfa_iclr.pdf',args.paper_build/'conditioned_dfa_iclr.pdf')
    for path in (ROOT/'assets/ndfa_revision_20260919').iterdir():
        if path.is_file():add(str(path.relative_to(ROOT)),path)
    add('study_ledger.json',ROOT/'assets/ndfa_revision_20260919/study_ledger.json')
    for name in ['verify.py','redraw.py','linear_simulation.py','prepare_evidence.py']:
        path=ROOT/'scripts/ndfa_revision_20260919'/name;add(str(path.relative_to(ROOT)),path)
    for name in ['aggregate_infodfa_bn_baseline.py','write_infodfa_paper_tables.py']:
        add('analysis/'+name,ROOT/'analysis'/name)
    generated('revision/SECOND_REVIEW.md','''# Second-review correction and supplementary curation

The active PDF contains the evidence supporting the main claims. Exhaustive
development and secondary comparison records remain in this archive. Removed
projected-step/rank, vision-curve and ImageNet-routing trajectory figures are
historical presentations; their old error bars are not current estimates.

Run `python -B scripts/ndfa_revision_20260919/verify.py` for the remaining
global-seed uncertainty corrections, common nuisance-ratio analysis, follow-up
intervals, and two mathematical qualifications. The seed-level SEMs condition
on the designed grid and retained feedback draws. They do not quantify new
dataset or hyperparameter-selection uncertainty.

The current figure entry point is `scripts/ndfa_revision_20260919/redraw.py`.
It calls the preceding generator for unchanged panels. The archived cache is
only an input; current captions and replication definitions are in the PDF.
Set NDFA_FIGURE_CACHE=revision/figure_cache.pkl and point NDFA_FIGURE_OUTPUT
and NDFA_PAPER_FIGURES to fresh directories outside this extraction.
`linear_simulation.py` regenerates the small population illustration; it is
not a training benchmark. `prepare_evidence.py` requires original raw archives
and exports the compact summaries included here; offline checks use those
summaries without retraining or accessing the test datasets again.

Study IDs S1--S8 resolve in `study_ledger.json`. Its historical source and
configuration hashes describe original bytes, before anonymous path exports.
They must not be silently rewritten to describe exported configurations.
''')
    generated('README.md','''# Conditioned DFA: curated review evidence

The active manuscript includes the second-review scientific corrections and
the reorganized supplement. Start with `paper/conditioned_dfa_iclr.pdf`.

```bash
python -B reproduce.py --checks integrity,statistics,legacy,focused,paper,smoke
python -B verify_revision.py
python -B scripts/ndfa_revision_20260919/verify.py
python -B -m pytest -q -p no:cacheprovider tests/test_ndfa_revision_math.py
```

See `revision/SECOND_REVIEW.md` for corrected uncertainty units, figure
reproduction and the cohort ledger. `revision/README.md` describes the
preceding experimental follow-ups. Compact records verify saved endpoints;
they do not repeat model training, inference or hardware timing.
''')
    cur=json.loads(payload['CURATION.json']);cur['second_review']={
        'scope':'Uncentered-moment interpretation, conditional seed-level uncertainty, joint-limit and clock qualifications, study-based supplement.',
        'active_figures':manifest['figures'],'removed_presentations':['projected-step/rank panels','exploratory vision curves','routing trajectories','duplicate endpoint reference','exhaustive development grids','secondary pairwise contrasts'],
        'all_experimental_records_preserved':True}
    generated('CURATION.json',json.dumps(cur,indent=2)+'\n')
    pub={'schema_version':4,'release_status':'second_review_and_supplement_curation','files':records,'base_archive_sha256':sha(args.base.read_bytes())}
    mbytes=(json.dumps(pub,indent=2)+'\n').encode()
    archive=args.output/'iclr_supplement.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,data in sorted({**payload,'PACKAGE_MANIFEST.json':mbytes}.items()):
            info=zipfile.ZipInfo('review_package/'+name,date_time=(1980,1,1,0,0,0));info.create_system=0
            z.writestr(info,data,compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
    receipt={'files':len(records),'zip_bytes':archive.stat().st_size,'zip_sha256':sha(archive.read_bytes()),
             'paper_sha256':build['versions']['iclr']['sha256'],'within_100MB':archive.stat().st_size<100_000_000,'validation':'pending'}
    (args.output/'build_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');assert receipt['within_100MB']
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
