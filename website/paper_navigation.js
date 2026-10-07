// Add future papers here; each page keeps its own chapter navigation.
(() => {
  const papers = [
    {
      id: 'cls-1995',
      file: 'semantic_knowledge.html',
      title: 'Why There Are Complementary Learning Systems in the Hippocampus and Neocortex: Insights From the Successes and Failures of Connectionist Models of Learning and Memory',
      citation: 'McClelland, McNaughton & O’Reilly · 1995',
    },
    {
      id: 'rogers-2004',
      file: 'semantic_memory.html',
      title: 'Structure and Deterioration of Semantic Memory: A Neuropsychological and Computational Investigation',
      citation: 'Rogers et al. · 2004',

    },
  ];
  const host = document.querySelector('body > aside .book-title');
  const current = papers.find(paper => paper.id === document.body.dataset.paper);
  if (!host || !current) return;

  const picker = document.createElement('details');
  picker.className = 'paper-picker';
  const summary = document.createElement('summary');
  summary.setAttribute('aria-label', `Choose paper. Current paper: ${current.title}`);
  const title = document.createElement('span');
  title.className = 'paper-picker-title';
  title.textContent = current.title;
  summary.append(title);

  const links = document.createElement('nav');
  links.className = 'paper-options';
  links.setAttribute('aria-label', 'Papers');
  papers.forEach(paper => {
    const link = document.createElement('a');
    link.href = paper.file;
    const name = document.createElement('span');
    name.textContent = paper.title;
    link.append(name);
    if (paper.id === current.id) {
      link.setAttribute('aria-current', 'page');
      link.addEventListener('click', event => {
        event.preventDefault();
        picker.open = false;
        summary.focus();
      });
    }
    links.append(link);
  });
  picker.append(summary, links);
  host.replaceWith(picker);
  document.addEventListener('click', event => {
    if (!picker.contains(event.target)) picker.open = false;
  });
  picker.addEventListener('keydown', event => {
    if (event.key === 'Escape') {
      picker.open = false;
      summary.focus();
    }
  });
})();

