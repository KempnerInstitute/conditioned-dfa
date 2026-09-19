"""Refresh the anonymous evidence archive after the horizontal figure revision."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from build_ndfa_completion_package import transform_details


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--paper-build", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    paper = ROOT / "drafts/Info-DFA"
    build = json.loads((a.paper_build / "build_receipt.json").read_text())
    assert build["accepted"]
    with zipfile.ZipFile(a.base) as z:
        payload = {name.removeprefix("review_package/"): z.read(name) for name in z.namelist()}
    manifest = json.loads(payload.pop("PACKAGE_MANIFEST.json"))
    original = payload.copy()
    records = manifest["files"]
    for name, record in records.items():
        assert sha(payload[name]) == record["export_sha256"]

    def add(name, path):
        data = path.read_bytes()
        changes = []
        if path.suffix in {".py", ".tex", ".sty", ".bst", ".json", ".md"}:
            text, changes = transform_details(data.decode(), ROOT,
                                             paper=name.startswith("paper/") and path.suffix == ".tex")
            data = text.encode()
        payload[name] = data
        records[name] = {"original_sha256": sha(path.read_bytes()), "export_sha256": sha(data),
                         "bytes": len(data), "transformations": changes}

    source = json.loads((paper / "manuscript_manifest.json").read_text())
    for name in source["file_sha256"]:
        if name != "conditioned_dfa_arxiv.tex":
            add("paper/" + name, paper / name)
    add("paper/conditioned_dfa_iclr.pdf", a.paper_build / "conditioned_dfa_iclr.pdf")
    path = ROOT / "scripts/ndfa_figure_layout_20260919/redraw.py"
    add(str(path.relative_to(ROOT)), path)
    note = """# Compact main figures and follow-up evidence

Figures 1, 2 and 4 use a four-panel row at the final manuscript width.
Figure 1 restores the analytic input-conditioning ratio; this is not a
simulated convergence-rate result. Figure 2 and Figure 4 retain every curve,
point, bar and uncertainty value from the corrected preceding revision.
The dashed direction in Figure 1A has both task and nuisance components.
Figure 3 retains all seed points and intervals in a shorter layout. Figure 5
retains all nine original contrasts and adds twelve existing work/width
follow-up contrasts in two panels. All paired means and individual 95% t
intervals are checked against their source summaries. No cohorts are pooled.
The supplementary follow-up figure retains the normalization intervention;
its work and width panels have moved to the main figure, avoiding duplication.
No measurements, training, test evaluation or inferential claims changed.

To reproduce all revised assets, first run the September 19 generator as
described in SECOND_REVIEW.md. Then run the compact layout generator:

```bash
NDFA_FIGURE_CACHE=revision/figure_cache.pkl python -B scripts/ndfa_figure_layout_20260919/redraw.py --output /tmp/ndfa_compact_figures --paper-figures /tmp/ndfa_all_figures
```

Point the preceding generator's NDFA_PAPER_FIGURES to the same output
directory, /tmp/ndfa_all_figures. Use a fresh output location. The unchanged
mode-timing PDF is retained in paper/figures. The compact generator compares
all plotted numerical values with the corrected preceding generator before
exporting Figures 2, 3 and 4, and checks the spectral weights and simulation
means in Figure 1. Its manifest records the six exported figure hashes and
all 21 plotted Figure 5 contrasts. It also checks all normalization points
and means against the preceding supplementary panel.
"""
    for name, text in [("revision/HORIZONTAL_FIGURES.md", note),
                       ("README.md", payload["README.md"].decode()+
                        "\nThe latest revision compacts Figure 3 and adds existing work/width evidence to Figure 5.\n"
                        "See `revision/HORIZONTAL_FIGURES.md` for the current figure reproduction step.\n")]:
        data = text.encode()
        payload[name] = data
        records[name] = {"original_sha256": None, "export_sha256": sha(data),
                         "bytes": len(data), "transformations": ["Figure layout documentation"]}
    changed = sorted(name for name in payload if payload[name] != original.get(name))
    assert all(name.startswith("paper/") or name in ["README.md", "revision/HORIZONTAL_FIGURES.md",
               "scripts/ndfa_figure_layout_20260919/redraw.py"] for name in changed)
    manifest.update(release_status="expanded_main_figures", base_archive_sha256=sha(a.base.read_bytes()))
    payload["PACKAGE_MANIFEST.json"] = (json.dumps(manifest, indent=2)+"\n").encode()
    archive = a.output / "iclr_supplement.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, data in sorted(payload.items()):
            info = zipfile.ZipInfo("review_package/"+name, date_time=(1980,1,1,0,0,0))
            info.create_system = 0
            z.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    receipt = {"files":len(records), "zip_bytes":archive.stat().st_size,
               "zip_sha256":sha(archive.read_bytes()), "changed_files":changed,
               "all_experimental_records_byte_identical":True,
               "paper_sha256":build["versions"]["iclr"]["sha256"],
               "within_100MB":archive.stat().st_size < 100_000_000,
               "validation":"pending"}
    assert receipt["within_100MB"]
    (a.output/"build_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt,indent=2))


if __name__ == "__main__":
    main()
