// Keep the chapter index synchronized with the visible reading position.
(() => {
  const theme = document.querySelector('link[href="book_theme.css"]');
  if (theme) document.head.appendChild(theme);
  let observer;
  function refresh() {
    document.querySelectorAll('.chapter-links a').forEach(a => {
      if (/open interactive|直接打开交互/i.test(a.textContent)) a.remove();
    });
    const child = document.getElementById('rat-extension-link');
    if (child) child.textContent = 'Extension: image associations';
    const article = document.querySelector('.book-section:not([hidden])');
    const headings = [...article.querySelectorAll('h1,h2,h3')];
    const buttons = [...document.querySelectorAll('#page-headings button')];
    const mark = index => buttons.forEach((button, i) => {
      if (i === index) button.setAttribute('aria-current', 'location');
      else button.removeAttribute('aria-current');
    });
    observer?.disconnect();
    observer = new IntersectionObserver(entries => {
      const visible = entries.filter(e => e.isIntersecting);
      if (visible.length) mark(headings.indexOf(visible[0].target));
    }, {rootMargin: '0px 0px -70% 0px'});
    headings.forEach((heading, i) => {
      observer.observe(heading);
      buttons[i]?.addEventListener('click', () => mark(i));
    });
    const notebooks = {semantic:'semantic/semantic_knowledge',rat:'rat/rat_consolidation',monkey:'monkey/monkey_discrimination',equations:'two_system/two_system_model'};
    const download = document.querySelector('.book-toolbar a[download]');
    if (download) download.href = '../notebooks/' + (notebooks[article.id] || 'semantic/semantic_knowledge') + '.ipynb';
    mark(0);
  }
  window.addEventListener('hashchange', refresh);
  refresh();
})();
