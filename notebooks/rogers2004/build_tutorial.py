"""Build the Colab teaching notebook from the paper page and executable model."""
from pathlib import Path
import re,base64
import nbformat as nb
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
page=(ROOT/'website/semantic_memory.html').read_text(encoding='utf8')
n=nb.v4.new_notebook();n.metadata.update(kernelspec=dict(display_name='Python 3',language='python',name='python3'),colab=dict(name='Rogers2004_semantic_memory.ipynb',toc_visible=True))
cells=[]
def md(s):cells.append(nb.v4.new_markdown_cell(s.strip()))
def code(s,hidden=False):
 c=nb.v4.new_code_cell(s.strip())
 if hidden:c.metadata.update(cellView='form',jupyter={'source_hidden':True})
 cells.append(c)
def image_file(path,alt):
 p=ROOT/path;c=nb.v4.new_markdown_cell(f'![{alt}](attachment:{p.name})');c.attachments={p.name:{'image/png':base64.b64encode(p.read_bytes()).decode()}};cells.append(c)
def section(number):
 match=re.search(r'<h2(?: id="[^"]+")?>'+str(number)+r'\. (.*?)</h2>(.*?)(?=<h2|</article>)',page,re.S)
 return match[1],match[2]
intro=re.search(r'<article class="book-section" id="introduction">(.*?)<div class="chapter-links">',page,re.S)[1]
intro=re.sub(r'<h1>(.*?)</h1>',r'# \1',intro)
md(intro)
md('''## Tutorial objectives

This tutorial connects three questions: how information about an object is represented, how a partial cue evokes the rest of that information, and how deleting connections changes the response.

The notebook contains the article, executable PyTorch examples, and interactive comparisons. The teaching sequence follows the pattern of [Neuromatch tutorials](https://compneuro.neuromatch.io/tutorials/W1D5_DeepLearning/student/W1D5_Tutorial1.html): an explanation, a short calculation, and an experiment. It is an independent paper reproduction, not an official Neuromatch tutorial.

The demonstrations use the saved optimized reconstruction shown on the website. The initial paper-settings reconstruction remains separately available. Training differences and data provenance are recorded in the linked reproduction notes.''')
md('''## Setup

The setup cell locates a local checkout or downloads the public repository in Colab. Dependencies are installed only when missing. All default cells run on a CPU; a full retraining run is optional.''')
code('''# @title Environment and repository
import importlib.util, subprocess, sys
from pathlib import Path
packages = {'numpy':'numpy', 'torch':'torch', 'matplotlib':'matplotlib',
            'pandas':'pandas', 'ipywidgets':'ipywidgets', 'PIL':'pillow'}
missing = [package for module, package in packages.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *missing])
ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p/'notebooks/rogers2004/rogers_model.py').is_file()), None)
if ROOT is None:
    ROOT = Path.cwd()/'papers'
    if not (ROOT/'notebooks/rogers2004/rogers_model.py').is_file():
        if ROOT.exists():
            raise RuntimeError('The papers directory exists but is not this repository.')
        subprocess.check_call(['git', 'clone', '--depth', '1',
                              'https://github.com/h4ch1m1/complementary-learning-systems.git', str(ROOT)])
sys.path.insert(0, str(ROOT/'notebooks/rogers2004'))
import json, copy
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, HTML, SVG, Image, Markdown, clear_output
from PIL import Image as PILImage
import rogers_model as m
from tune_model import audit
try:
    from google.colab import output
    output.enable_custom_widget_manager()
except ImportError:
    pass
torch.set_num_threads(1)
torch.manual_seed(12345)
plt.rcParams.update({'figure.dpi':110, 'axes.spines.top':False, 'axes.spines.right':False})
model, env = m.load(ROOT/'outputs/rogers2004/tuning/adam')
model.eval()
labels = list(map(str, env['item_labels']))
photo_sources = json.loads((ROOT/'assets/rogers_objects/sources.json').read_text(encoding='utf8'))
targets = torch.from_numpy(env['targets'])
POOLS = {'Name':slice(0,40), 'Verbal description':slice(40,152), 'Visual features':slice(152,216)}
print(f'{len(labels)} objects; {targets.shape[1]} visible units; {m.HIDDEN} semantic units.')''',True)
code('''# @title Plotting helpers
import re

def reference_image(ax, item):
    key = re.sub(r'_\\d+$', '', labels[item])
    src = photo_sources[key]['src']
    with PILImage.open(ROOT/'website'/src) as im:
        ax.imshow(im.copy())
    ax.axis('off')
    ax.set_title(labels[item])

def feature_grid(ax, values, title, vmax=1):
    values = np.asarray(values)
    columns = 16 if values.size == 112 else 8
    ax.imshow(values.reshape(-1, columns), cmap='Blues', vmin=0, vmax=vmax, interpolation='nearest')
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])

def cue_for(item, modality):
    cue = targets[item:item+1].clone()
    mask = torch.zeros_like(cue, dtype=torch.bool)
    mask[:, POOLS[modality]] = True
    return cue, mask

OBJECTS = [(label,i) for i,label in enumerate(labels)]
PHOTO_CREDITS = '\\n'.join(f'- {v["title"]}: [source and attribution]({v["source"]})' for v in photo_sources.values())''',True)
title,body=section(1);md('## 1. '+title+'\n\n'+re.search(r'<p>.*?</p>',body,re.S)[0])
md('''The following comparison uses the stored feature vectors. A shared active feature contributes to the intersection; a feature active in either object contributes to the union. Their ratio summarizes overlap.''')
code('''def compare_objects(first=3, second=4, pool='Visual features'):
    av = env['targets'][first, POOLS[pool]]
    bv = env['targets'][second, POOLS[pool]]
    shared = int(np.logical_and(av,bv).sum())
    union = int(np.logical_or(av,bv).sum())
    fig, ax = plt.subplots(2,2,figsize=(8,5), gridspec_kw={'width_ratios':[1,2]})
    for row,item,values in [(0,first,av),(1,second,bv)]:
        reference_image(ax[row,0],item)
        feature_grid(ax[row,1],values,pool)
    plt.tight_layout(); plt.show()
    print(f'{shared} shared active features / {union} active in either object: {shared/max(union,1):.1%}')

widgets.interact(compare_objects,
    first=widgets.Dropdown(options=OBJECTS,value=3,description='First'),
    second=widgets.Dropdown(options=OBJECTS,value=4,description='Second'),
    pool=widgets.Dropdown(options=['Visual features','Verbal description'],description='Features'));''')
svg=re.search(r'<svg viewBox="0 0 700 270".*?</svg>',body,re.S)[0]
code('display(SVG('+repr(svg)+'))',True)
md('The model connects all three forms of information to the same layer of 64 hidden units. Each arrow direction has independently learned weights.')
code('''allowed = m.connection_mask()
assert allowed.shape == (280,280)
assert torch.count_nonzero(allowed[:216,:216]) == 0
print('Permitted connections:', int(allowed.sum()))
fig,ax = plt.subplots(figsize=(5,5))
ax.imshow(allowed, cmap='Greys', interpolation='nearest')
ax.set(xlabel='Sending unit',ylabel='Receiving unit',title='Connections allowed by the architecture')
ax.axvline(215.5,color='#3675a8');ax.axhline(215.5,color='#3675a8');plt.show()''')
md('''**Checkpoint.** A semantic unit has no preset object label. An object is represented by the joint activity of all 64 units. The mask above permits visible–hidden and hidden–hidden connections, while excluding direct visible–visible connections.''')
title,body=section(2);before=body.split('<section class="teaching-demo"')[0]
md('## 2. '+title+'\n\n'+before)
md('''### One recurrent update

The matrix uses rows for receiving units and columns for sending units. The calculation below shows the contribution from incoming activity, the retained net input, and the resulting activity for one hidden unit.''')
code('''item = labels.index('chicken')
cue, mask = cue_for(item,'Name')
full_cue = torch.cat((cue,torch.zeros(1,64)),dim=1)
full_mask = torch.cat((mask,torch.zeros(1,64,dtype=torch.bool)),dim=1)
u0 = torch.where(full_mask,(full_cue*2-1)*15.9357739741644,torch.full_like(full_cue,-2.))
a0 = torch.where(full_mask,full_cue,torch.sigmoid(u0))
with torch.no_grad():
    incoming = a0 @ model.weight.T - 2
    u1 = .75*u0 + .25*incoming
    a1 = torch.sigmoid(u1)
    actual = m.dynamics(model.weight,cue,mask,1)[1]
assert torch.allclose(a1[:,216:],actual[:,216:])
unit = 216
pd.DataFrame({'quantity':['previous net input','weighted input minus bias','new net input','new activity'],
              'value':[u0[0,unit].item(),incoming[0,unit].item(),u1[0,unit].item(),a1[0,unit].item()]})''')
md('''### Activity following a cue

We present a name, a verbal description, or a visual pattern and follow the network through successive updates. The input is fixed in states 0–11 and released at state 12. The animation uses the actual recurrent computation. Later displayed frames are farther apart in simulation time.''')
code('''ticks = list(range(29))+[32,40,56,80,120,200,400]
trace_cache = {}
def retrieval(item=3, modality='Name', frame=0):
    key = (item,modality)
    if key not in trace_cache:
        cue,mask = cue_for(item,modality)
        with torch.no_grad():
            trace_cache[key] = m.dynamics(model.weight,cue,mask,400)[:,0].numpy()
    activity = trace_cache[key][ticks[frame]]
    fig = plt.figure(figsize=(10,5))
    gs = fig.add_gridspec(2,3,width_ratios=[.7,1,1])
    reference_image(fig.add_subplot(gs[:,0]),item)
    for j,(name,sl) in enumerate([('Names',slice(0,40)),('Verbal descriptors',slice(40,152)),
                                  ('Visual features',slice(152,216)),('Semantic units',slice(216,280))]):
        ax=fig.add_subplot(gs[j//2,1+j%2]);feature_grid(ax,activity[sl],name)
    plt.tight_layout();plt.show()
    winner = int(activity[:40].argmax())
    print('Name response:', str(env['names'][winner]) if activity[winner]>.5 else 'No response')

trace_item=widgets.Dropdown(options=OBJECTS,value=3,description='Object')
trace_cue=widgets.Dropdown(options=list(POOLS),description='Cue')
frame=widgets.IntSlider(min=0,max=len(ticks)-1,value=0,description='Progress',continuous_update=False)
play=widgets.Play(min=0,max=len(ticks)-1,interval=260)
widgets.jslink((play,'value'),(frame,'value'))
trace_output=widgets.interactive_output(retrieval,{'item':trace_item,'modality':trace_cue,'frame':frame})
display(widgets.VBox([widgets.HBox([trace_item,trace_cue]),widgets.HBox([play,frame]),trace_output]))''')
md('''### Learning changes the connections

A short optimization experiment starts from a copy of the saved weights. The loss is calculated from the visible outputs at updates 21–56. Backpropagation through time assigns the error to connections that influenced those outputs. Adam then adjusts the permitted weights. This is the optimization extension, rather than the original 400-epoch training procedure.''')
code('''practice_model = copy.deepcopy(model)
training_targets = targets.repeat(3,1)
training_cues = training_targets.clone()
training_masks = torch.zeros_like(training_cues,dtype=torch.bool)
training_masks[:48,:40]=True
training_masks[48:96,40:152]=True
training_masks[96:,152:]=True
for i,name_id in enumerate(env['name_ids']):
    if name_id in m.GENERAL:
        eligible = np.isin(env['category'],m.GENERAL[name_id])
        training_targets[i,40:] = targets[eligible,40:].mean(0)
optimizer = torch.optim.Adam([practice_model.weight],lr=.001)
losses=[]
for step in range(20):
    optimizer.zero_grad()
    prediction=m.dynamics(practice_model.weight,training_cues,training_masks,56)[21:57,:,:216]
    loss=torch.nn.functional.binary_cross_entropy(prediction,training_targets.expand_as(prediction))
    loss=loss+1e-6*practice_model.weight.square().sum()/practice_model.allowed.sum()
    loss.backward()
    practice_model.weight.grad.mul_(practice_model.allowed)
    optimizer.step()
    with torch.no_grad():practice_model.weight.mul_(practice_model.allowed)
    losses.append(loss.item())
assert np.isfinite(losses).all()
plt.plot(range(1,21),losses);plt.xlabel('Additional Adam update');plt.ylabel('Training objective');plt.show()''')
md('''### Optional original-schedule training

The default run uses saved weights so the whole lesson remains short. The optional cell below trains a fresh reconstruction for 400 epochs and writes its results to a separate directory. This run uses the original reconstructed environment, so it does not reproduce the optimized checkpoint above.''')
code('''# @title Optional full training
RUN_FULL_TRAINING = False # @param {type:"boolean"}
if RUN_FULL_TRAINING:
    retrained_model,retrained_env=m.train(seed=12345,data_seed=2004,epochs=400,lr=.005,
                                         gradient_scale=1.,output=ROOT/'outputs/rogers2004/colab_training')
else:
    print('The lesson continues with the saved checkpoint.')''')
title,body=section(3);md('## 3. '+title+'\n\n'+body.strip())
md('''**Checkpoint.** Releasing the cue removes the external constraint, not the recurrent connections. The next activity pattern still depends on the preceding activity. Convergence to a stable pattern is a property to check, rather than an automatic consequence of using recurrence.''')
title,body=section(4);table=re.search(r'<table>.*?</table>',body,re.S)[0];md('## 4. '+title+'\n\n'+table)
md('''### Feature loss in an individual damaged network

We compare intact and damaged visual responses to the same cue. A name cue corresponds to drawing from a name; a visual cue followed by its removal corresponds to delayed copying. Each damage sample removes a different set of connections. No retraining or rescaling follows damage.''')
code('''with torch.no_grad():
    intact_outputs,intact_residual,intact_ticks=m.settled_probe(model,env)
damage_cache={}
def damaged_response(item=3, task='Drawing from a name', damage=.2, sample=1):
    modality='Name' if task=='Drawing from a name' else 'Visual features'
    key=(item,modality,damage,sample)
    if key not in damage_cache:
        generator=torch.Generator().manual_seed(2707+(sample-1)*100+round(damage*100))
        keep=torch.rand(model.weight.shape,generator=generator)>=damage
        cue,mask=cue_for(item,modality)
        with torch.no_grad():
            state,residual,end=m.settle(model.weight*keep,cue,mask)
        damage_cache[key]=(state[0].numpy(),float(residual[0]),end)
    state,residual,end=damage_cache[key]
    before=intact_outputs[0 if modality=='Name' else 2,item,152:216].numpy()>.5
    after=state[152:216]>.5
    changes=np.where(before & after,1,np.where(before & ~after,2,np.where(~before & after,3,0)))
    from matplotlib.colors import ListedColormap
    fig,ax=plt.subplots(1,3,figsize=(10,3),gridspec_kw={'width_ratios':[.8,1,1]})
    reference_image(ax[0],item);feature_grid(ax[1],before,'Intact response')
    ax[2].imshow(changes.reshape(8,8),vmin=0,vmax=3,cmap=ListedColormap(['#eef1f3','#3675a8','#d18538','#9470ad']))
    ax[2].set(title='Damaged: retained / omitted / added',xticks=[],yticks=[])
    plt.tight_layout();plt.show()
    winner=int(state[:40].argmax())
    print('Name:',str(env['names'][winner]) if state[winner]>.5 else 'No response',
          '| Omitted:',int((before & ~after).sum()),'| Added:',int((~before & after).sum()))
    if residual>=1e-5:print('Response unsettled at the update limit.')

widgets.interact(damaged_response,
 item=widgets.Dropdown(options=OBJECTS,value=3,description='Object'),
 task=widgets.Dropdown(options=['Drawing from a name','Delayed copying'],description='Task'),
 damage=widgets.SelectionSlider(options=[0.,.1,.2,.3,.4,.5],value=.2,description='Damage',continuous_update=False),
 sample=widgets.Dropdown(options=[1,2,3],description='Sample'));''')
md('''We compare performance at ten damage levels to examine how each task changes as more connections are removed. Nonzero levels average 50 masks applied to one trained network.''')
code('''summary=json.loads((ROOT/'outputs/rogers2004/tuning/adam/summary.json').read_text())
metrics={'Naming':'naming/all/correct','Matching: close':'matching/close',
         'Matching: unrelated':'matching/unrelated',
         'Broad sorting':'sorting/picture/animal_artifact/general',
         'Specific sorting':'sorting/picture/animal_artifact/specific'}
levels=sorted(map(float,summary))
fig,ax=plt.subplots(figsize=(8,4))
for label,key in metrics.items():
    ax.plot(levels,[summary[str(level)][key]['mean'] for level in levels],marker='o',label=label)
ax.set(xlabel='Fraction of connections removed',ylabel='Accuracy',ylim=(0,1.05));ax.legend();plt.show()''')
md('''The saved aggregate results can also be recalculated with the evaluator. The optional cell uses fresh masks from the same documented seed and leaves the supplied results intact.''')
code('''# @title Optional full lesion evaluation
RUN_FULL_EVALUATION=False # @param {type:"boolean"}
if RUN_FULL_EVALUATION:
    recomputed=m.evaluate(model,env,ROOT/'outputs/rogers2004/colab_evaluation',trials=50,lesion_seed=1707)
else:
    print('The aggregate figure uses the saved 50-mask evaluation.')''')
title,body=section(5);md('## 5. '+title)
image_file(Path('outputs/rogers2004/tuning/lesion_validation.png'),'Task performance across damage levels')
for p in re.findall(r'<p>.*?</p>',body,re.S):md(p)
md(re.search(r'<table>.*?</table>',body,re.S)[0])
code('''# @title Recalculate the intact-network audit
with torch.no_grad():
    intact_check=audit(model,env)
pd.DataFrame([intact_check]).T.rename(columns={0:'Measured value'})''')
md('''## Resources

- [Original paper](https://doi.org/10.1037/0033-295X.111.1.205)
- [Original reconstruction notebook](https://github.com/h4ch1m1/complementary-learning-systems/blob/main/notebooks/rogers2004/semantic_deterioration.ipynb)
- [Experiment report](https://github.com/h4ch1m1/complementary-learning-systems/blob/main/outputs/rogers2004/tuning/REPORT.md)
- [Settings and reproduction notes](https://github.com/h4ch1m1/complementary-learning-systems/blob/main/notes/ROGERS2004_NOTES.md)
- [Raw lesion results](https://github.com/h4ch1m1/complementary-learning-systems/blob/main/outputs/rogers2004/tuning/adam/lesion_trials.csv)
- [Image credits](https://h4ch1m1.github.io/complementary-learning-systems/website/rogers_image_sources.html)''')
code('''# @title Reference image sources
display(Markdown(PHOTO_CREDITS))''',True)
n.cells=cells;nb.validate(n);nb.write(n,HERE/'tutorial.ipynb');print('Created',len(cells),'cells')
