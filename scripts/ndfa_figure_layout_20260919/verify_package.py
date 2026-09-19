"""Validate the figure-only revision from a fresh anonymous extraction."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,tempfile,zipfile
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from build_ndfa_anonymous_package import scan_package

def main():
    p=argparse.ArgumentParser();p.add_argument('--package-dir',type=Path,required=True);a=p.parse_args();out=a.package_dir/'fresh_validation';out.mkdir(exist_ok=False)
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    results={}
    with tempfile.TemporaryDirectory(prefix='ndfa_revised_review_') as temp:
        with zipfile.ZipFile(a.package_dir/'iclr_supplement.zip') as z:z.extractall(temp)
        work=Path(temp)/'review_package'
        env.update(NDFA_FIGURE_CACHE=str(work/'revision/figure_cache.pkl'),
                   NDFA_FIGURE_OUTPUT=str(Path(temp)/'figure_previews'),
                   NDFA_PAPER_FIGURES=str(Path(temp)/'figure_pdfs'))
        commands={
            'integrity_and_paper':[sys.executable,'-B','reproduce.py','--checks','integrity,paper'],
            'compact_figures':[sys.executable,'-B','scripts/ndfa_figure_layout_20260919/redraw.py',
                '--output',env['NDFA_FIGURE_OUTPUT'],'--paper-figures',env['NDFA_PAPER_FIGURES']]}
        for name,command in commands.items():
            q=subprocess.run(command,cwd=work,env=env,capture_output=True,text=True,timeout=900)
            (out/(name+'.log')).write_text(q.stdout+q.stderr);results[name]={'exit_code':q.returncode}
            print(name,q.returncode,flush=True)
            if q.returncode:raise RuntimeError(name+' failed; see '+str(out/(name+'.log')))
        # Rebuilt figure pixels must match the active paper assets, not merely
        # complete without error. The unchanged mode-timing figure is verified
        # by the package integrity and full-paper rebuild above.
        import fitz
        plotted=json.loads((Path(temp)/'figure_previews/manifest.json').read_text())['figures']
        for name in plotted:
            actual=fitz.open(Path(temp)/'figure_pdfs'/(name+'.pdf'))
            expected=fitz.open(work/'paper/figures'/(name+'.pdf'))
            assert actual[0].get_pixmap().samples==expected[0].get_pixmap().samples, name
        results['compact_figures']['pixel_identical_assets']=len(plotted)
        scan=scan_package(work)
        approved=[];unresolved=[]
        for finding in scan['findings']:
            if finding['kind']=='required_identifying_copyright_notice' and (work/finding['path']).read_text().startswith('MIT License') and 'Copyright (c) 2026 the authors of the accompanying anonymous submission' in (work/finding['path']).read_text():
                approved.append({**finding,'resolution':'Previously authorized anonymous first-party notice; original canonical attribution unchanged.'})
            elif finding['path']=='paper/natbib.sty' and finding['kind']=='identifying_institution_token' and finding['token']=='Harvard' and hashlib.sha256((work/finding['path']).read_bytes()).hexdigest()==hashlib.sha256((ROOT/'drafts/Info-DFA/natbib.sty').read_bytes()).hexdigest():
                approved.append({**finding,'resolution':'Unmodified third-party natbib commands named harvarditem/harvardand; citation-style compatibility, not author affiliation.'})
            else:unresolved.append(finding)
        scan.update(findings=unresolved,approved_context=approved,ready_for_anonymous_distribution=not unresolved)
        (out/'anonymity_scan.json').write_text(json.dumps(scan,indent=2)+'\n')
        results['anonymity']=scan['ready_for_anonymous_distribution']
        assert results['anonymity'],str(unresolved[:5])
    results.update(accepted=True,fresh_extraction_outside_workspace=True,model_training=False,new_test_evaluation=False,scope='Three revised figure assets and fresh manuscript build; original experiment records unchanged; previous full scientific validation retained.')
    (out/'receipt.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))
if __name__=='__main__':main()
