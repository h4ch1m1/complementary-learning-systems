// Papers are chapters; each paper's existing section links stay together.
(() => {
  const siteTitle = 'Neural Network Models of Learning and Memory';
  const papers = [
    {
      id: 'cls-1995', file: 'semantic_knowledge.html', label: 'Complementary learning systems',
      title: 'Why There Are Complementary Learning Systems in the Hippocampus and Neocortex: Insights From the Successes and Failures of Connectionist Models of Learning and Memory',
      citation: 'McClelland, McNaughton & O’Reilly (1995)', last: 'equations',
    },
    {
      id: 'rogers-2004', file: 'semantic_memory.html', label: 'Semantic memory',
      title: 'Structure and Deterioration of Semantic Memory: A Neuropsychological and Computational Investigation',
      citation: 'Rogers et al. (2004)', last: 'model',
    },
    {
      id: 'tcm-2002', file: 'temporal_context.html', label: 'Temporal context model',
      title: 'A Distributed Representation of Temporal Context',
      citation: 'Howard & Kahana (2002)', last: 'model',
    },
  ];
  const aside = document.querySelector('body > aside');
  if (!aside) return;
  const currentIndex = papers.findIndex(paper => paper.id === document.body.dataset.paper);
  const current = papers[currentIndex];
  const overview = document.body.dataset.paper === 'overview';
  if (!current && !overview) return;
  const home = overview ? 'index.html' : '../index.html';
  const paperURL = paper => (overview ? 'website/' : '') + paper.file;
  const element = (tag, className, text) => {
    const node = document.createElement(tag);
    node.className = className;
    if (text) node.textContent = text;
    return node;
  };
  const link = (href, className, text) => {
    const node = element('a', className, text); node.href = href; return node;
  };
  const sections = aside.querySelector('nav');
  const sectionLinks = sections ? [...sections.querySelectorAll('[data-chapter]')] : [];
  const title = link(home, 'book-title', siteTitle);
  const toggle = element('button', 'navigation-toggle', 'Chapters');
  toggle.type = 'button'; toggle.setAttribute('aria-expanded', 'false');
  toggle.setAttribute('aria-controls', 'book-navigation');
  toggle.addEventListener('click', () => toggle.setAttribute('aria-expanded', String(toggle.getAttribute('aria-expanded') !== 'true')));
  const navigation = element('nav', 'book-navigation');
  navigation.id = 'book-navigation'; navigation.setAttribute('aria-label', 'Tutorial chapters');
  const overviewLink = link(home, 'overview-link', 'Overview');
  if (overview) overviewLink.setAttribute('aria-current', 'page');
  navigation.append(overviewLink);
  const chapters = element('ol', 'paper-chapters');
  papers.forEach((paper, i) => {
    const chapter = element('li', 'paper-chapter');
    const heading = link(paperURL(paper) + '#introduction', 'paper-link');
    heading.append(element('span', 'chapter-name', `${i + 1}. ${paper.label}`), element('span', 'chapter-citation', paper.citation));
    chapter.append(heading);
    if (paper === current) {
      heading.setAttribute('aria-current', 'true');
      sections.className = 'paper-sections';
      sections.setAttribute('aria-label', 'Sections in this chapter');
      sections.querySelectorAll('.nav-caption').forEach(node => node.remove());
      chapter.append(sections);
    }
    chapters.append(chapter);
  });
  navigation.append(chapters);
  aside.replaceChildren(title, toggle, navigation);

  if (current) {
    const introduction = document.getElementById('introduction');
    const heading = element('header', 'paper-heading');
    heading.append(element('p', 'chapter-kicker', `Chapter ${currentIndex + 1} · ${current.label}`), element('p', 'paper-name', current.title), element('p', 'paper-citation', current.citation));
    introduction.prepend(heading);
    // Replace scattered back/next links, retaining downloads and source links.
    document.querySelectorAll('.chapter-links a, .page-footer a').forEach(node => {
      if (sectionLinks.some(section => section.getAttribute('href') === node.getAttribute('href'))) node.remove();
    });
    document.querySelectorAll('.chapter-links, .page-footer').forEach(node => { if (!node.children.length) node.remove(); });
    sectionLinks.forEach((section, i) => {
      const footer = element('nav', 'chapter-pagination');
      footer.setAttribute('aria-label', 'Reading order');
      const previousPaper = papers[currentIndex - 1], nextPaper = papers[currentIndex + 1];
      const previous = i > 0
        ? [sectionLinks[i - 1].getAttribute('href'), `← ${sectionLinks[i - 1].textContent}`]
        : previousPaper ? [paperURL(previousPaper) + '#' + previousPaper.last, `← Chapter ${currentIndex}: ${previousPaper.label}`] : [home, '← Overview'];
      footer.append(link(previous[0], 'previous-section', previous[1]));
      const next = sectionLinks[i + 1];
      const following = next ? [next.getAttribute('href'), next.textContent + ' →']
        : nextPaper ? [paperURL(nextPaper) + '#introduction', `Chapter ${currentIndex + 2}: ${nextPaper.label} →`] : [home, 'Return to Overview →'];
      footer.append(link(following[0], 'next-section', following[1]));
      document.getElementById(section.dataset.chapter).append(footer);
    });
  }
  const updateTitle = () => { document.title = current ? `${siteTitle} · ${current.citation}` : siteTitle; };
  updateTitle();
  window.addEventListener('hashchange', updateTitle);
})();
