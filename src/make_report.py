"""Build a compact HTML report from audited, multi-seed PlantShiftEval summaries."""
from __future__ import annotations
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load_json(path: str, default):
    file_path = ROOT / path
    if not file_path.exists():
        return default
    with file_path.open(encoding="utf-8") as handle:
        return json.load(handle)

def num(value, digits=4):
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "—"

def main():
    rows = load_json("report/summary.json", [])
    cross_rows = load_json("report/crossdomain.json", {}).get("results", {})
    body = []
    for row in rows:
        model = row["model"]
        cd = cross_rows.get(model, {})
        body.append(
            "<tr>"
            f"<td>{html.escape(model)}</td><td>{row.get('n_seeds', 0)}</td>"
            f"<td>{num(row.get('macro_f1'))} ± {num(row.get('macro_f1_sd'))}</td>"
            f"<td>{row.get('robustness_n_seeds', 0)}</td>"
            f"<td>{num(row.get('clean'))}</td><td>{num(row.get('robust_acc'))}</td>"
            f"<td>{num(cd.get('acc15')) if cd else '—'}</td></tr>"
        )
    page = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PlantShiftEval — Results summary</title>
<style>
body{font-family:Arial,sans-serif;max-width:1100px;margin:36px auto;padding:0 20px;color:#202833;line-height:1.55}
h1,h2{line-height:1.2}table{border-collapse:collapse;width:100%;font-size:14px}
th,td{border-bottom:1px solid #d9e0e7;padding:9px;text-align:left}th{background:#f3f7fa}
.note{background:#f7f9fb;padding:12px 16px;border-left:3px solid #4d7897}
</style></head><body><h1>PlantShiftEval</h1>
<p><strong>Testing texture-representation hypotheses in plant disease recognition.</strong>
MTD-Net is treated as a candidate hypothesis, not assumed to be universally superior.</p>
<h2>Recorded results</h2><table><thead><tr><th>Model</th><th>Training seeds</th><th>PlantVillage macro-F1</th>
<th>Corruption seeds</th><th>Clean accuracy</th><th>Corruption accuracy</th><th>PlantDoc 15-way accuracy</th></tr></thead><tbody>""" + "".join(body) + """</tbody></table>
<h2>Interpretation limits</h2><div class="note"><ul>
<li>The six-epoch scratch comparison has only three seeds; the mean macro-F1 gap is smaller than MTD-Net's across-seed SD.</li>
<li>The 12-epoch comparison also changes peak learning rate, so it does not isolate epoch count.</li>
<li>Synthetic corruptions do not substitute for independent field-domain validation.</li>
<li>The current PlantDoc adaptation protocol pools mapped target splits for cross-validation; those results are exploratory.</li>
<li>Raw data and model weights are not included; dataset provenance and software versions still need to be pinned.</li>
</ul></div><p>Values are read from tracked aggregate JSON files; this script does not train models.</p></body></html>"""
    output = ROOT / "report" / "PlantShiftEval_summary.html"
    output.write_text(page, encoding="utf-8")
    print(f"Wrote {output}")

if __name__ == "__main__":
    main()
