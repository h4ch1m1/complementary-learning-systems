"""Export teaching examples from the saved optimized model; no model fitting."""
from pathlib import Path
import json,hashlib
import numpy as np
import torch
import rogers_model as m
torch.set_num_threads(1)
root=m.ROOT
model,env=m.load(root/'outputs/rogers2004/tuning/adam')
items=[3,11,19,24,35,43]
ticks=list(range(29))+[32,40,56,80,120,200,400]
target=torch.tensor(env['targets'])
cue=target[items].repeat(3,1)
mask=torch.zeros_like(cue,dtype=torch.bool)
mask[:6,:40]=True;mask[6:12,40:152]=True;mask[12:,152:]=True
with torch.no_grad():
    history=m.dynamics(model.weight,cue,mask,400)
traces=[]
for modality in range(3):
    traces.append([history[ticks,modality*6+i,:].numpy().astype('float64').round(5).tolist() for i in range(6)])
weights=model.weight.detach().clone()
damage=[]
for trial in range(3):
    levels=[]
    for level in [0,.1,.2,.3,.4,.5]:
        # Independent masks for each level, matching the aggregate protocol.
        generator=torch.Generator().manual_seed(2707+trial*100+round(level*100))
        keep=torch.rand(weights.shape,generator=generator)>=level
        with torch.no_grad():
            model.weight.copy_(weights*keep)
            outputs,residual,end_tick=m.settled_probe(model,env)
        records=[]
        for modality in [0,2]:
            records.append([])
            for i in range(48):
                a=outputs[modality,i]
                winner=int(a[:40].argmax())
                confidence=float(a[winner])
                bits=(a[152:216]>.5).numpy().astype(int)
                records[-1].append(dict(visual=bits.tolist(),name=winner if confidence>.5 else -1,
                    confidence=round(confidence,5),residual=float(residual[modality,i])))
        levels.append(dict(outputs=records,ticks=end_tick,removed=float(((~keep)*model.allowed.bool()).sum()/model.allowed.sum())))
        print('export',trial,level,flush=True)
    damage.append(levels)
with torch.no_grad():model.weight.copy_(weights)
data=dict(labels=list(map(str,env['item_labels'])),names=list(map(str,env['names'])),
    categories=list(map(int,env['category'])),name_ids=list(map(int,env['name_ids'])),
    targets=env['targets'].astype(int).tolist(),items=items,ticks=ticks,traces=traces,
    levels=[0,.1,.2,.3,.4,.5],damage=damage,
    provenance=dict(checkpoint_sha256=hashlib.sha256((root/'outputs/rogers2004/tuning/adam/model.pt').read_bytes()).hexdigest(),
                    lesion_seed_rule='2707 + trial*100 + percentage; independent masks per level',
                    precision='Activities rounded to 5 decimals; classifications computed before rounding',
                    source='Optimized extension, one checkpoint; illustrative masks are separate from the aggregate evaluation'))
(root/'website/rogers_explorer_data.js').write_text('window.rogersExplorerData = '+json.dumps(data,separators=(',',':'))+';\n',encoding='utf8')
print('Exported teaching data',flush=True)
