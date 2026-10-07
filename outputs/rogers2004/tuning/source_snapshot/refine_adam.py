"""Optimization extension: full-batch Adam and expected general-name targets."""
import json,time
import numpy as np
import torch
import torch.nn.functional as F
import rogers_model as m
from tune_model import audit

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    base=m.ROOT/'outputs/rogers2004/tuning'
    model,env=m.load(base/'stable');out=base/'adam';out.mkdir(exist_ok=True)
    target=np.tile(env['targets'],(3,1));cue=target.copy()
    mask=np.zeros_like(cue,dtype=bool);mask[:48,:40]=True;mask[48:96,40:152]=True;mask[96:,152:]=True
    for i,name in enumerate(env['name_ids']):
        if name in m.GENERAL:
            eligible=np.isin(env['category'],m.GENERAL[name])
            target[i,40:]=env['targets'][eligible,40:].mean(0)
    cue,mask,target=map(torch.from_numpy,(cue,mask,target))
    optimizer=torch.optim.Adam([model.weight],lr=.001)
    config=dict(source='stable',optimizer='Adam',lr=.001,max_updates=1000,ticks=56,
                objective='mean BCE ticks21..56 + 1e-6 mean squared allowed weights',
                general_names='expected feature targets instead of sampled exemplars; names unchanged',
                stopping='intact max error < .05 and all inputs stable; no lesion selection')
    (out/'config.json').write_text(json.dumps(config,indent=2));np.savez_compressed(out/'environment.npz',**env)
    rows=[];start=time.perf_counter()
    for step in range(1001):
        if step:
            optimizer.zero_grad()
            h=m.dynamics(model.weight,cue,mask,56)[21:57,:,:216]
            loss=F.binary_cross_entropy(h,target.expand_as(h))+1e-6*model.weight.square().sum()/model.allowed.sum()
            loss.backward();model.weight.grad.mul_(model.allowed);optimizer.step()
            with torch.no_grad():model.weight.mul_(model.allowed)
        if step%50==0:
            with torch.no_grad():r=dict(additional_epoch=step,seconds=time.perf_counter()-start,**audit(model,env))
            rows.append(r);print(json.dumps(r),flush=True)
            (out/'training.json').write_text(json.dumps(rows,indent=2))
            torch.save(dict(state_dict=model.state_dict(),epoch=1000,optimizer_updates=step,seed=12345,data_seed=2004),out/'model.pt')
            if r['max_error']<.05 and r['unsettled']==0:break

if __name__=='__main__':main()
