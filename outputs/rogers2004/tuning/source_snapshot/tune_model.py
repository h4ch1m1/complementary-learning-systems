"""Controlled continuation experiments; original paper_settings stays untouched."""
import argparse, json, time, hashlib
from pathlib import Path
import numpy as np
import torch
import rogers_model as m

@torch.jit.script
def epoch_step(w: torch.Tensor, allowed: torch.Tensor, cues: torch.Tensor,
               masks: torch.Tensor, targets: torch.Tensor, lr: float, margin: float):
    loss = 0.
    for i in range(cues.size(0)):
        grad, err = m.bptt(w, cues[i], masks[i], targets[i], margin)
        w.mul_(1.-.001/144.).add_(grad*allowed, alpha=-lr)
        loss += float(err)
    return loss/cues.size(0)

def audit(model, env):
    out, residual, ticks = m.settled_probe(model, env)
    checked = torch.ones(3,48,dtype=torch.bool)
    checked[0] = torch.from_numpy(env['name_ids'] >= 4)
    err = (out[:,:,:216]-torch.from_numpy(env['targets']))[checked].abs()
    names = out[2,:,:40]
    correct = (names.argmax(1).numpy()==env['name_ids']) & (names.max(1).values.numpy()>.5)
    return dict(mse=float(err.square().mean()), within_005=float((err<.05).float().mean()),
                bit_accuracy=float((err<.5).float().mean()), naming=float(correct.mean()),
                max_error=float(err.max()), unsettled=int((residual>=1e-5).sum()), ticks=ticks)

def revised_environment():
    env = m.make_environment(2004)
    # Keep exact predicates in their documented slots. Restore four fruit-shared
    # encyclopedic attributes from Fig. 3 within the eight non-label slots.
    # Other non-label encyclopedic attributes are idiosyncratic, not redundant
    # noisy animal/artifact predicates. This remains a reconstruction, not raw data.
    p = env['verbal_p'].copy()
    p[:,96:104] = .2
    p[5,100:104] = .8
    rng = np.random.default_rng(2004)
    rng.random((48,64))
    draws = rng.random((48,112))
    env['targets'][:,40:152] = (draws < p[env['category']]).astype('float32')
    env['verbal_p'] = p
    return env

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--variant', choices=['continue','tight','data'], required=True)
    ap.add_argument('--epochs',type=int,default=400)
    ap.add_argument('--evaluate',action='store_true')
    a=ap.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root=m.ROOT/'outputs/rogers2004/tuning'/a.variant
    root.mkdir(parents=True,exist_ok=True)
    model,env=m.load(m.ROOT/'outputs/rogers2004/paper_settings')
    if a.variant=='data': env=revised_environment()
    margin=.05 if a.variant=='continue' else .01
    rng=np.random.default_rng(54321)
    rows=[]; start=time.perf_counter()
    config=dict(variant=a.variant,initial_checkpoint='paper_settings/model.pt',
                additional_epochs=a.epochs,margin=margin,lr_schedule='0.005 first half, 0.002 second half',
                sampling_seed=54321,selection='intact performance only; no lesion scores',
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root/'config.json').write_text(json.dumps(config,indent=2))
    np.savez_compressed(root/'environment.npz',**env)
    for epoch in range(a.epochs+1):
        if epoch:
            cues,masks,targets=m.epoch_patterns(env,rng)
            with torch.no_grad():
                loss=epoch_step(model.weight,model.allowed,cues,masks,targets,
                                .005 if epoch<=a.epochs//2 else .002,margin)
        if epoch%100==0 or epoch==a.epochs:
            with torch.no_grad(): result=audit(model,env)
            row=dict(additional_epoch=epoch,seconds=time.perf_counter()-start,**result)
            rows.append(row); print(json.dumps(row),flush=True)
            (root/'training.json').write_text(json.dumps(rows,indent=2))
            torch.save(dict(state_dict=model.state_dict(),epoch=400+epoch,seed=12345,data_seed=2004),root/'model.pt')
    if a.evaluate: m.evaluate(model,env,root,50,lesion_seed=1707)

if __name__=='__main__': main()
