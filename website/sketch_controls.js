// Add pencil borders only to controls that already have a visible border.
(() => {
 const styleURL=new URL('sketch_controls.css',document.currentScript.src).href;
 function enhance(doc){
  if(doc.documentElement.dataset.sketchEnhanced)return;doc.documentElement.dataset.sketchEnhanced="true";
  if(!doc.querySelector('link[data-sketch-controls]')){const link=doc.createElement('link');link.rel='stylesheet';link.href=styleURL;link.dataset.sketchControls='';doc.head.append(link);}
  function update(){
   doc.querySelectorAll('select,button,input[type=number],input[type=text],output,[id^=number]').forEach(el=>{
    if(el.classList.contains('sketch-control')||el.closest('.page-toc,aside')||el.id==='trace-play')return;
    const s=doc.defaultView.getComputedStyle(el);
    if(parseFloat(s.borderTopWidth)>0&&s.borderTopStyle!=='none'&&s.borderTopColor!=='transparent'&&s.borderTopColor!=='rgba(0, 0, 0, 0)')el.classList.add('sketch-control');
   });
   doc.querySelectorAll('iframe').forEach(frame=>{
    if(frame.dataset.sketchWatched)return;frame.dataset.sketchWatched='';
    const visit=()=>{try{if(frame.contentDocument?.head)enhance(frame.contentDocument);}catch{}};
    frame.addEventListener('load',visit);visit();
   });
  }
  update();let pending=false;new MutationObserver(()=>{if(pending)return;pending=true;requestAnimationFrame(()=>{pending=false;update();});}).observe(doc.body,{childList:true,subtree:true});
 }
 enhance(document);
})();
