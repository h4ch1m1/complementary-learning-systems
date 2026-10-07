import json,hashlib,shutil
import torch
import rogers_model as m

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    root=m.ROOT/'outputs/rogers2004/tuning'
    candidates={}
    for name in ['continue','tight','data','stable','adam']:
        rows=json.loads((root/name/'training.json').read_text())
        candidates[name]=rows[-1]
    eligible={k:r for k,r in candidates.items() if r['naming']==1 and r['bit_accuracy']==1 and r['unsettled']==0}
    chosen=max(eligible,key=lambda k:eligible[k]['within_005'])
    record=dict(variant=chosen,rule='Among final intact checkpoints with perfect naming and binary features and no unsettled inputs, maximize within-.05 fraction.',
                candidates=candidates,lesion_seed=1707,lesion_trials_per_nonzero_level=50,
                limitation='Different environments are not a like-for-like clinical fit; inputs chosen from Figure 3, not lesion curves.')
    (root/'selection.json').write_text(json.dumps(record,indent=2))
    snapshots=root/'source_snapshot';snapshots.mkdir(exist_ok=True)
    manifest={}
    for name in ['rogers_model.py','tune_model.py','refine_stability.py','refine_adam.py','validate_tuning.py','report_tuning.py']:
        source=m.ROOT/'notebooks/rogers2004'/name
        shutil.copy2(source,snapshots/name);manifest[name]=hashlib.sha256(source.read_bytes()).hexdigest()
    (snapshots/'sha256.json').write_text(json.dumps(manifest,indent=2))
    model,env=m.load(root/chosen)
    print('Selected using intact data only: '+chosen,flush=True)
    m.evaluate(model,env,root/chosen,trials=50,lesion_seed=1707)

if __name__=='__main__':main()
