"""Export actual saved lesion summaries for offline-capable website controls."""
import csv,json
from pathlib import Path
root=Path(__file__).resolve().parents[2]
folder=root/'outputs/rogers2004/tuning/adam'
summary=json.loads((folder/'summary.json').read_text())
# Normalize keys to match JavaScript String(number), including zero.
summary={format(float(k),'g'):v for k,v in summary.items()}
rows=list(csv.DictReader((folder/'lesion_trials.csv').open()))
audit={}
for level in summary:
    subset=[r for r in rows if float(r['lesion'])==float(level)]
    audit[level]=dict(trials=len(subset),unsettled=sum(int(r['nonconvergent_inputs'])>0 for r in subset))
data=dict(levels=sorted(map(float,summary)),summary=summary,audit=audit)
(root/'website/rogers_results.js').write_text('// Generated from tuning/adam; run notebooks/rogers2004/export_website.py to refresh.\nwindow.rogersResults = '+json.dumps(data,separators=(',',':'))+';\n',encoding='utf8')

