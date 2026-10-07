"""Explicit extension: supervise 56 ticks, including the late trajectory."""
import json, time, argparse
import numpy as np
import torch
import rogers_model as m
from tune_model import audit

@torch.jit.script
def late_bptt(weight: torch.Tensor, cue: torch.Tensor, clamp: torch.Tensor, target: torch.Tensor):
    h=m.dynamics(weight,cue.unsqueeze(0),clamp.unsqueeze(0),56).squeeze(1)
    delta=torch.zeros_like(h)
    a=h[21:57,:216]
    # Binary cross entropy against hard targets; .01 is only the dead zone.
    active=((a-target).abs()>.01).to(a.dtype)
    delta[21:57,:216]=(.25*8./36.)*(a-target)*active
    mask=torch.cat((clamp,torch.zeros(64,dtype=torch.bool)))
    for t in range(55,0,-1):
        delta[t]+=.75*delta[t+1]+.25*h[t]*(1.-h[t])*(delta[t+1]@weight)
        if t<12: delta[t].masked_fill_(mask,0.)
    return .25*delta[1:].t()@h[:-1]

@torch.jit.script
def late_epoch(w: torch.Tensor, allowed: torch.Tensor, cues: torch.Tensor, masks: torch.Tensor, targets: torch.Tensor, lr: float = .002):
    for i in range(cues.size(0)):
        grad=late_bptt(w,cues[i],masks[i],targets[i])
        w.mul_(1.-.001/144.).add_(grad*allowed,alpha=-lr)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--source',default='data'); ap.add_argument('--epochs',type=int,default=200)
    ap.add_argument('--output',default='stable'); ap.add_argument('--lr',type=float,default=.002)
    a=ap.parse_args(); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    base=m.ROOT/'outputs/rogers2004/tuning'; model,env=m.load(base/a.source)
    out=base/a.output; out.mkdir(exist_ok=True)
    np.savez_compressed(out/'environment.npz',**env)
    (out/'config.json').write_text(json.dumps(dict(source=a.source,epochs=a.epochs,lr=a.lr,ticks=56,
        loss='hard-target BCE, .01 dead zone, mean over ticks21..56 scaled to original total',
        sampling_seed=64321,selection='intact audit only'),indent=2))
    rows=[]; rng=np.random.default_rng(64321); start=time.perf_counter()
    for epoch in range(a.epochs+1):
        if epoch:
            with torch.no_grad(): late_epoch(model.weight,model.allowed,*m.epoch_patterns(env,rng),a.lr)
        if epoch%50==0 or epoch==a.epochs:
            with torch.no_grad(): row=dict(additional_epoch=epoch,seconds=time.perf_counter()-start,**audit(model,env))
            rows.append(row); print(json.dumps(row),flush=True)
            (out/'training.json').write_text(json.dumps(rows,indent=2))
            source_epoch=torch.load(base/a.source/'model.pt',weights_only=True)['epoch']
            torch.save(dict(state_dict=model.state_dict(),epoch=source_epoch+epoch,seed=12345,data_seed=2004),out/'model.pt')

if __name__=='__main__': main()
