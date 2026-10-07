(() => {
'use strict';
const d=window.rogersExplorerData;
if(!d)return;
const $=id=>document.getElementById(id);
const categoryNames=['bird','mammal','vehicle','household object','tool','fruit'];
function options(select,indices){indices.forEach(i=>{const o=document.createElement('option');o.value=i;o.textContent=d.labels[i]+' · '+categoryNames[d.categories[i]];select.append(o);});}
function grid(host,values,prefix,targets=null,colors=null){
 host.replaceChildren();host.classList.add('unit-grid');
 host.setAttribute('role','img');host.setAttribute('aria-label',prefix+': '+values.length+' units. '+values.filter(v=>v>.5).length+' above 0.5.');
 values.forEach((v,i)=>{const cell=document.createElement('span');cell.className='unit';
 cell.style.background=colors?colors[i]:'rgb('+Math.round(224-175*v)+','+Math.round(231-114*v)+','+Math.round(237-69*v)+')';
 if(targets?.[i])cell.classList.add('target-on');
 cell.title=prefix+' '+(i+1)+': '+Number(v).toFixed(3)+(targets?' · target '+targets[i]:'');
 host.append(cell);});
}
const indices=d.labels.map((_,i)=>i);
options($('feature-a'),indices);options($('feature-b'),indices);
$('feature-a').value='3';$('feature-b').value='4';
function referenceImage(side,index,prefix='feature'){
 const suffix=side?'-'+side:'';
 const key=d.labels[index].replace(/_\d+$/,'');
 const photo=window.rogersObjectImages?.[key];
 const img=$(prefix+'-image'+suffix),link=$(prefix+'-image-link'+suffix);
 if(photo?.src){
  if(img.getAttribute('src')!==photo.src)img.src=photo.src;
  img.alt=photo.title+' reference image';
  img.hidden=false;link.href='rogers_image_sources.html#'+key;
  link.title=photo.title+' — illustrative reference; image credits';
 }else{img.hidden=true;img.removeAttribute('src');link.removeAttribute('href');}
}
function compare(){
 const a=+ $('feature-a').value,b=+ $('feature-b').value,pool=$('feature-pool').value;
 const lo=pool==='visual'?152:40, hi=pool==='visual'?216:152;
 const av=d.targets[a].slice(lo,hi),bv=d.targets[b].slice(lo,hi);
 const colors=av.map((v,i)=>v&&bv[i]?'#3675a8':v?'#d18538':bv[i]?'#5b9b83':'#eef1f3');
 grid($('feature-grid-a'),av,pool,null,colors.map((c,i)=>av[i]?c:'#eef1f3'));
 grid($('feature-grid-b'),bv,pool,null,colors.map((c,i)=>bv[i]?c:'#eef1f3'));
 $('feature-label-a').textContent=d.labels[a];$('feature-label-b').textContent=d.labels[b];
 referenceImage('a',a);referenceImage('b',b);
 const shared=av.filter((v,i)=>v&&bv[i]).length,union=av.filter((v,i)=>v||bv[i]).length;
 $('feature-overlap').textContent=shared+' active features in common out of '+union+' active in either object (overlap '+(union?100*shared/union:100).toFixed(1)+'%).';
}
['feature-a','feature-b','feature-pool'].forEach(id=>['input','change'].forEach(event=>$(id).addEventListener(event,compare)));compare();

options($('trace-item'),d.items);$('trace-item').value=String(d.items[0]);
$('trace-step').max=String(d.ticks.length-1);
let timer=null;
function stop(){if(timer)clearInterval(timer);timer=null;$('trace-play').setAttribute('aria-label','Play retrieval');$('trace-play').title='Play';$('trace-play').setAttribute('aria-pressed','false');}
function trace(){
 const item=+ $('trace-item').value,mi=+ $('trace-cue').value,k=+ $('trace-step').value,t=d.ticks[k];
 referenceImage('',item,'trace');
 const a=d.traces[mi][d.items.indexOf(item)][k],target=d.targets[item];
 $('trace-step').setAttribute('aria-valuetext','Update '+t);
 const pools=[['names',0,40],['verbal',40,152],['visual',152,216],['semantic',216,280]];
 pools.forEach(([name,lo,hi],i)=>{
  grid($('trace-'+name),a.slice(lo,hi),name,i<3?target.slice(lo,hi):null);
  $('trace-'+name).parentElement.classList.toggle('clamped',t<12&&i===mi);
 });
 const ranked=a.slice(0,40).map((v,i)=>({v,i})).sort((x,y)=>y.v-x.v).slice(0,3);
 $('trace-readout').textContent='Most active names: '+ranked.map(r=>d.names[r.i]+' '+r.v.toFixed(2)).join(' · ')+'. '+(ranked[0].v>.5?'Name response: '+d.names[ranked[0].i]+'.':'No name exceeds the 0.5 response threshold.');
 const mae=a.slice(0,216).reduce((s,v,i)=>s+Math.abs(v-target[i]),0)/216;
 $('trace-error').textContent='Mean absolute difference from this object’s visible target: '+mae.toFixed(3)+'.';
}
$('trace-step').addEventListener('input',()=>{stop();trace();});
['trace-item','trace-cue'].forEach(id=>['input','change'].forEach(event=>$(id).addEventListener(event,()=>{stop();$('trace-step').value='0';trace();})));
$('trace-play').addEventListener('click',()=>{
 if(timer){stop();return;}
 if(+$('trace-step').value===d.ticks.length-1)$('trace-step').value='0';
 $('trace-play').setAttribute('aria-label','Pause retrieval');$('trace-play').title='Pause';$('trace-play').setAttribute('aria-pressed','true');
 timer=setInterval(()=>{let k=+$('trace-step').value;if(k>=d.ticks.length-1){stop();return;}$('trace-step').value=String(k+1);trace();},260);
});
window.addEventListener('hashchange',stop);
document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
trace();

options($('lesion-item'),indices);$('lesion-item').value='3';
function lesion(){
 const item=+ $('lesion-item').value,mi=+ $('lesion-cue').value,trial=+ $('lesion-trial').value,k=+ $('lesion-step').value;
 referenceImage('',item,'lesion');
 const run=d.damage[trial][k],base=d.damage[trial][0].outputs[mi][item],out=run.outputs[mi][item];
 const colors=out.visual.map((v,i)=>v&&base.visual[i]?'#3675a8':v?'#9470ad':base.visual[i]?'#d18538':'#eef1f3');
 grid($('lesion-before'),base.visual,'intact visual');
 grid($('lesion-after'),out.visual,'damaged visual',null,colors);
 const omit=base.visual.filter((v,i)=>v&&!out.visual[i]).length,add=base.visual.filter((v,i)=>!v&&out.visual[i]).length;
 const name=out.name<0?'No response':d.names[out.name],before=base.name<0?'No response':d.names[base.name];
 $('lesion-level').textContent=Math.round(d.levels[k]*100)+'%';
 $('lesion-step').setAttribute('aria-valuetext',Math.round(d.levels[k]*100)+' percent damage');
 $('lesion-readout').textContent='Name: '+before+' → '+name+' (strongest name activity '+out.confidence.toFixed(2)+'). Visual features omitted: '+omit+'; added: '+add+'.';
 if(out.residual>=1e-5)$('lesion-readout').textContent+=' Response unsettled.';
 if(mi===0&&d.name_ids[item]<4)$('lesion-readout').textContent+=' General-name cue.';

}
['lesion-item','lesion-cue','lesion-trial'].forEach(id=>['input','change'].forEach(event=>$(id).addEventListener(event,lesion)));
$('lesion-step').addEventListener('input',lesion);lesion();
window.addEventListener('pageshow',()=>{compare();trace();lesion();});
})();
