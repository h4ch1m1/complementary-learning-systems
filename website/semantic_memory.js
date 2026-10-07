(() => {
  const articles = [...document.querySelectorAll('.book-section')];
  const list = document.getElementById('page-headings');
  let observer;
  function updateChapter() {
    const requested = location.hash.slice(1);
    const target = document.getElementById(requested);
    const article = articles.find(a => a.id === requested) || target?.closest('.book-section') || articles[0];
    articles.forEach(a => { a.hidden = a !== article; });
    document.querySelectorAll('[data-chapter]').forEach(a => {
      if (a.dataset.chapter === article.id) a.setAttribute('aria-current','page');
      else a.removeAttribute('aria-current');
    });
    observer?.disconnect(); list.replaceChildren();
    const headings = [...article.querySelectorAll('h1,h2')];
    const mark = index => [...list.children].forEach((b,i) => {
      if (i === index) b.setAttribute('aria-current','location'); else b.removeAttribute('aria-current');
    });
    headings.forEach((h,i) => {
      const b = document.createElement('button'); b.type='button'; b.textContent=h.textContent;
      b.addEventListener('click', () => {h.scrollIntoView({block:'start'}); mark(i);}); list.append(b);
    });
    observer = new IntersectionObserver(entries => {
      const visible=entries.find(e=>e.isIntersecting); if(visible) mark(headings.indexOf(visible.target));
    },{rootMargin:'0px 0px -65% 0px'});
    headings.forEach(h=>observer.observe(h)); mark(0);
    if (target && target !== article) target.scrollIntoView({block:'start'});
    else window.scrollTo(0,0);
  }
  window.addEventListener('hashchange',updateChapter); updateChapter();
  const slider=document.getElementById('damage'), bars=document.getElementById('result-bars');
  const metrics=[['Naming','naming/all/correct'],['Matching: close foil','matching/close'],['Matching: distant foil','matching/distant'],['Matching: unrelated foil','matching/unrelated'],['Broad picture sorting','sorting/picture/animal_artifact/general'],['Specific picture sorting','sorting/picture/animal_artifact/specific']];
  function updateResults() {
    const data=window.rogersResults;
    if(!data){bars.textContent='Results could not be loaded. The measured figure is available below.';return;}
    const level=data.levels[Number(slider.value)], result=data.summary[String(level)];
    document.getElementById('damage-label').textContent=Math.round(level*100)+'%';slider.setAttribute('aria-valuetext',Math.round(level*100)+' percent of connections removed');bars.replaceChildren();
    metrics.forEach(([label,key])=>{
      const row=document.createElement('div');row.className='result-row';
      const name=document.createElement('span');name.textContent=label;
      const track=document.createElement('div');track.className='result-track';track.setAttribute('aria-hidden','true');
      const fill=document.createElement('div');fill.className='result-fill';fill.style.width=(result[key].mean*100)+'%';track.append(fill);
      const value=document.createElement('span');value.className='result-value';value.textContent=(result[key].mean*100).toFixed(1)+'%';
      row.append(name,track,value);bars.append(row);
    });
  }
  slider.addEventListener('input',updateResults);updateResults();
})();

