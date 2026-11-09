(() => {
'use strict';
const $ = id => document.getElementById(id);
const BLUE = '#3675a8', ORANGE = '#d18538', GREY = '#8a8f94', GREEN = '#5b9b83', INK = '#65716d';
const PAPER = {beta: .402, gamma: 1, tau: .247, d: 7.24};
const N = 12;
const WORDS = ['apple', 'river', 'key', 'tiger', 'umbrella', 'train', 'candle', 'forest', 'cup', 'moon', 'shoe', 'piano'];
const CONDITIONS = ['immediate', 'delayed', 'continuous'];
const TITLES = {immediate: 'Immediate recall', delayed: 'Delayed recall', continuous: 'Continuous distractor'};
const SEQUENCES = {
  immediate: Array(N).fill('f'),
  delayed: [...Array(N).fill('f'), 'D'],
  continuous: [...Array(N).fill(['f', 'D']).flat(), 'D'],
};
const PLACEMENTS = {
  paper: SEQUENCES.continuous,
  'last-retention': [...Array(N - 1).fill(['f', 'D']).flat(), 'f', 'D'],
  'between-only': [...Array(N - 1).fill(['f', 'D']).flat(), 'f'],
};
const PAPER_POINTS = {
  immediate: {pfr: [.003, .003, .003, .008, .010, .013, .020, .042, .062, .120, .240, .455]},
  delayed: {pfr: [.022, .025, .025, .028, .035, .037, .048, .062, .105, .140, .185, .265],
            crp: {'-5': .041, '-4': .052, '-3': .072, '-2': .096, '-1': .145, 1: .272, 2: .198, 3: .173, 4: .145, 5: .152}},
  continuous: {pfr: [.040, .048, .042, .037, .045, .042, .040, .047, .050, .070, .123, .397],
               crp: {'-5': .044, '-4': .046, '-3': .060, '-2': .090, '-1': .200, 1: .285, 2: .143, 3: .101, 4: .097, 5: .096}},
};

const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
const basis = (dim, k) => { const v = new Array(dim).fill(0); v[k] = 1; return v; };

function createModel(n, dim, p) {
  return {n, p, ctx: basis(dim, n), MFT: [...Array(n)].map((_, i) => basis(dim, i)),
          MTF: [...Array(n)].map(() => new Array(dim).fill(0)), next: n + 1};
}
function cloneModel(m) {
  return {n: m.n, p: m.p, ctx: m.ctx.slice(), MFT: m.MFT.map(r => r.slice()), MTF: m.MTF.map(r => r.slice()), next: m.next};
}
function present(m, item) {
  const {beta, gamma} = m.p;
  const incoming = m.MFT[item].slice();
  const c = dot(m.ctx, incoming);
  const rho = Math.sqrt(1 + beta * beta * (c * c - 1)) - beta * c;
  m.ctx = m.ctx.map((v, i) => rho * v + beta * incoming[i]);
  const B = 1 / Math.sqrt(gamma * gamma + 1 + 2 * gamma * dot(incoming, m.ctx));
  m.MFT[item] = incoming.map((v, i) => gamma * B * v + B * m.ctx[i]);
  m.MTF[item] = m.MTF[item].map((v, i) => v + m.ctx[i]);
  return {rho, context: m.ctx.slice()};
}
function distract(m) {
  const r = Math.pow(1 - m.p.beta * m.p.beta, m.p.d / 2);
  m.ctx = m.ctx.map(v => r * v);
  m.ctx[m.next] += Math.sqrt(1 - r * r);
  m.next += 1;
}
function study(tokens, p) {
  const n = tokens.filter(t => t === 'f').length, nd = tokens.filter(t => t === 'D').length;
  const m = createModel(n, n + 1 + Math.max(nd, 1), p);
  const history = [];
  let item = 0;
  tokens.forEach(t => { if (t === 'f') history.push(present(m, item++).context); else distract(m); });
  return {model: m, history};
}
function activations(m) { return m.MTF.map(row => dot(row, m.ctx)); }
function softmax(logits, excluded = []) {
  const masked = logits.map((v, i) => excluded.includes(i) ? -Infinity : v);
  const top = Math.max(...masked);
  const e = masked.map(v => v === -Infinity ? 0 : Math.exp(v - top));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map(v => v / s);
}
function probabilities(m, excluded = []) { return softmax(activations(m).map(a => 2 * a / m.p.tau), excluded); }
function transitions(m, exclude = true) {
  const first = probabilities(m);
  const second = first.map((_, i) => { const b = cloneModel(m); present(b, i); return probabilities(b, exclude ? [i] : []); });
  return {first, second};
}
function lagCrp(first, second) {
  const n = first.length, out = {};
  for (let lag = 1 - n; lag < n; lag++) {
    if (!lag) continue;
    let num = 0, den = 0;
    for (let i = 0; i < n; i++) {
      if (i + lag < 0 || i + lag >= n) continue;
      num += first[i] * second[i][i + lag];
      den += first[i];
    }
    out[lag] = num / den;
  }
  return out;
}
function predict(tokens, p) {
  const {model} = study(tokens, p);
  const {first, second} = transitions(model);
  return {pfr: first, crp: lagCrp(first, second), model};
}
window.tcm = {WORDS, present, distract, study, probabilities, transitions, lagCrp, predict, SEQUENCES, PLACEMENTS, PAPER, PAPER_POINTS};
if (typeof document === 'undefined') return;

const SVGNS = 'http://www.w3.org/2000/svg';
function el(name, attrs = {}, parent) {
  const node = document.createElementNS(SVGNS, name);
  Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, v));
  if (parent) parent.append(node);
  return node;
}
function fmt(v, digits = 3) { return Number(v).toFixed(digits); }

function chart(host, opts) {
  const W = opts.W || 330, H = opts.H || 240, L = 46, R = 12, T = 30, B = 44;
  const [x0, x1] = opts.xRange, yMax = opts.yMax;
  const sx = x => L + (x - x0) / (x1 - x0) * (W - L - R);
  const sy = y => H - B - y / yMax * (H - T - B);
  host.replaceChildren();
  const svg = el('svg', {viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': opts.label || opts.title || ''}, host);
  if (opts.title) el('text', {x: (L + W - R) / 2, y: 18, 'text-anchor': 'middle', 'font-size': 14, fill: '#323232'}, svg).textContent = opts.title;
  el('path', {d: `M${L} ${T - 4} Q${L - 1.5} ${(T + H - B) / 2} ${L} ${H - B} Q${(L + W - R) / 2} ${H - B + 1.5} ${W - R} ${H - B}`,
              fill: 'none', stroke: INK, 'stroke-width': 1.3, 'stroke-linecap': 'round'}, svg);
  opts.yTicks.forEach(y => {
    el('line', {x1: L - 4, x2: L, y1: sy(y), y2: sy(y), stroke: INK}, svg);
    el('text', {x: L - 7, y: sy(y) + 4, 'text-anchor': 'end', 'font-size': 11, fill: '#555'}, svg).textContent = y;
  });
  opts.xTicks.forEach(x => {
    el('line', {x1: sx(x), x2: sx(x), y1: H - B, y2: H - B + 4, stroke: INK}, svg);
    el('text', {x: sx(x), y: H - B + 17, 'text-anchor': 'middle', 'font-size': 11, fill: '#555'}, svg).textContent = x;
  });
  if (opts.xLabel) el('text', {x: (L + W - R) / 2, y: H - 6, 'text-anchor': 'middle', 'font-size': 12, fill: '#444'}, svg).textContent = opts.xLabel;
  if (opts.yLabel) el('text', {x: 12, y: (T + H - B) / 2, 'text-anchor': 'middle', 'font-size': 12, fill: '#444', transform: `rotate(-90 12 ${(T + H - B) / 2})`}, svg).textContent = opts.yLabel;
  opts.series.forEach(s => {
    s.segments.forEach(points => {
      const pts = points.filter(([, y]) => Number.isFinite(y));
      if (s.line && pts.length > 1) el('polyline', {points: pts.map(([x, y]) => `${sx(x)},${sy(Math.min(y, yMax))}`).join(' '), fill: 'none', stroke: s.color, 'stroke-width': s.width || 2, 'stroke-linejoin': 'round', 'stroke-dasharray': s.dash || ''}, svg);
      pts.forEach(([x, y]) => {
        const cx = sx(x), cy = sy(Math.min(y, yMax));
        if (s.marker === 'square') el('rect', {x: cx - 4.5, y: cy - 4.5, width: 9, height: 9, fill: 'none', stroke: s.color, 'stroke-width': 1.6}, svg);
        else if (s.marker === 'circle') el('circle', {cx, cy, r: 3.4, fill: s.color}, svg);
      });
    });
  });
}
const lagSplit = (map, lim = 5) => [[...Array(lim)].map((_, k) => [k - lim, map[k - lim]]), [...Array(lim)].map((_, k) => [k + 1, map[k + 1]])];
function pfrChart(host, condition, pfr) {
  const series = [];
  series.push({segments: [pfr.map((y, i) => [i + 1, y])], color: BLUE, line: true, marker: 'circle'});
  chart(host, {W: 270, H: 220, title: TITLES[condition], xRange: [.5, 12.5], xTicks: [1, 3, 6, 9, 12], yMax: .85, yTicks: [0, .2, .4, .6, .8],
               xLabel: 'Serial position', yLabel: 'Probability of first recall', series});
}
function crpChart(host, condition, crp) {
  chart(host, {W: 270, H: 220, title: TITLES[condition], xRange: [-5.6, 5.6], xTicks: [-5, -3, -1, 1, 3, 5], yMax: .45, yTicks: [0, .1, .2, .3, .4],
               xLabel: 'Lag', yLabel: 'Conditional response probability',
               series: [{segments: lagSplit(crp), color: BLUE, line: true, marker: 'circle'}]});
}

const chapters = [...document.querySelectorAll('.book-section')];
const toc = $('page-headings');
let observer;
function updateChapter() {
  const requested = location.hash.slice(1);
  const target = document.getElementById(requested);
  const article = chapters.find(a => a.id === requested) || target?.closest('.book-section') || chapters[0];
  chapters.forEach(a => { a.hidden = a !== article; });
  document.querySelectorAll('[data-chapter]').forEach(a => {
    if (a.dataset.chapter === article.id) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
  });
  observer?.disconnect(); toc.replaceChildren();
  const headings = [...article.querySelectorAll('h1,h2')];
  const mark = index => [...toc.children].forEach((b, i) => { if (i === index) b.setAttribute('aria-current', 'location'); else b.removeAttribute('aria-current'); });
  headings.forEach((h, i) => {
    const b = document.createElement('button'); b.type = 'button'; b.textContent = h.textContent;
    b.addEventListener('click', () => { h.scrollIntoView({block: 'start'}); mark(i); }); toc.append(b);
  });
  observer = new IntersectionObserver(entries => {
    const visible = entries.find(e => e.isIntersecting); if (visible) mark(headings.indexOf(visible.target));
  }, {rootMargin: '0px 0px -65% 0px'});
  headings.forEach(h => observer.observe(h)); mark(0);
  if (target && target !== article) target.scrollIntoView({block: 'start'}); else window.scrollTo(0, 0);
}
window.addEventListener('hashchange', updateChapter); updateChapter();


const PLAY_SVG = '<svg viewBox="0 0 48 48" width="38" height="38" aria-hidden="true" focusable="false"><g class="play-mark" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M15 9 Q27 16 37 24 Q27 32 15 39 Q13 25 15 9 Z"/><path d="M17 11 Q26 18 34 24 Q27 30 17 36" opacity=".35"/></g><g class="pause-mark" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M17 11 Q16 24 17 37 M30 11 Q32 24 30 37"/></g></svg>';
document.querySelectorAll('.play-button').forEach(b => { b.innerHTML = PLAY_SVG; });
const shade = v => `rgb(${Math.round(238 - 184 * v)},${Math.round(241 - 124 * v)},${Math.round(243 - 75 * v)})`;
const signed = lag => (lag > 0 ? '+' : '−') + Math.abs(lag);

function contextStrip(host, ctx, highlight = -1) {
  host.replaceChildren();
  const cells = [['Initial', ctx[N]], ...WORDS.map((w, k) => [w, ctx[k]])];
  const rest = ctx.slice(N + 1).reduce((s, v) => s + v * v, 0);
  if (rest > 1e-12) cells.push(['Math', Math.sqrt(rest)]);
  cells.forEach(([label, v], k) => {
    const cell = document.createElement('div');
    cell.className = 'context-cell' + (k - 1 === highlight ? ' current' : '');
    const swatch = document.createElement('span');
    swatch.style.background = shade(Math.max(0, Math.min(1, v)));
    const name = document.createElement('small');
    name.textContent = label;
    cell.title = `${label}: ${fmt(v)}`;
    cell.append(swatch, name);
    host.append(cell);
  });
}
function wordBars(host, values, opts) {
  host.replaceChildren();
  values.forEach((v, k) => {
    const col = document.createElement('div');
    col.className = 'word-bar' + (opts.dim?.(k) ? ' dim' : '') + (opts.mark?.(k) ? ' marked' : '');
    const track = document.createElement('div'); track.className = 'word-bar-track';
    const fill = document.createElement('div'); fill.className = 'word-bar-fill';
    fill.style.height = Math.max(0, Math.min(1, v / opts.max)) * 100 + '%';
    fill.style.background = opts.color || BLUE;
    track.append(fill);
    const value = document.createElement('span'); value.className = 'word-bar-value';
    value.textContent = opts.dim?.(k) ? '' : (v >= .995 ? '1.00' : v.toFixed(2).replace(/^0/, ''));
    col.append(value, track);
    if (opts.labels !== false) {
      const label = document.createElement(opts.onClick ? 'button' : 'span');
      label.className = 'word-bar-label';
      if (opts.onClick) { label.type = 'button'; label.disabled = !!opts.dim?.(k); label.addEventListener('click', () => opts.onClick(k)); }
      label.textContent = WORDS[k];
      col.append(label);
    }
    if (opts.badge?.(k)) { const b = document.createElement('span'); b.className = 'word-bar-badge'; b.textContent = opts.badge(k); col.prepend(b); }
    host.append(col);
  });
}

(() => {
  let frames = [], timer = null;
  const step = $('study-step'), play = $('study-play');
  function build() {
    const beta = +$('study-beta').value;
    $('study-beta-value').textContent = fmt(beta, 2);
    const m = createModel(N, N + 1, {...PAPER, beta});
    frames = [{ctx: m.ctx.slice(), contexts: []}];
    const contexts = [];
    for (let i = 0; i < N; i++) { contexts.push(present(m, i).context); frames.push({ctx: m.ctx.slice(), contexts: contexts.slice()}); }
  }
  function draw() {
    const k = +step.value, f = frames[k];
    $('study-word').textContent = k ? WORDS[k - 1] : 'Ready';
    $('study-count').textContent = k ? `Word ${k} of ${N}` : 'No words presented';
    contextStrip($('study-context'), f.ctx, k - 1);
    const sims = WORDS.map((_, j) => j < f.contexts.length ? dot(f.contexts[j], f.ctx) : 0);
    wordBars($('study-similarity'), sims, {max: 1, color: GREY, dim: j => j >= f.contexts.length});
    if (k >= 2) $('study-readout').textContent = `After “${WORDS[k - 1]}”: the “${WORDS[0]}” component is ${fmt(f.ctx[0], 2)}, and similarity to its study context is ${fmt(sims[0], 2)}.`;
    else $('study-readout').textContent = k ? 'The first word enters context as the initial context fades.' : 'Play or move the slider to follow context during study.';
  }
  function stop() { clearInterval(timer); timer = null; play.setAttribute('aria-pressed', 'false'); play.title = 'Play'; }
  play.addEventListener('click', () => {
    if (timer) { stop(); return; }
    if (+step.value === N) step.value = '0';
    play.setAttribute('aria-pressed', 'true'); play.title = 'Pause';
    draw();
    timer = setInterval(() => { if (+step.value >= N) { stop(); return; } step.value = String(+step.value + 1); draw(); }, 900);
  });
  step.addEventListener('input', () => { stop(); draw(); });
  $('study-beta').addEventListener('input', () => { build(); draw(); });
  build(); step.value = String(N); draw();
})();

(() => {
  function draw() {
    const tau = +$('choice-tau').value;
    $('choice-tau-value').textContent = fmt(tau, 2);
    const {model} = study(SEQUENCES.immediate, {...PAPER, tau});
    const a = activations(model), pr = probabilities(model);
    wordBars($('choice-activation'), a, {max: 1, color: GREY, labels: false});
    wordBars($('choice-probability'), pr, {max: 1, color: BLUE});
  }
  $('choice-tau').addEventListener('input', draw); draw();
})();

(() => {
  let model, recalled;
  function reset() {
    const condition = $('recall-condition').value, gamma = +$('recall-gamma').value;
    $('recall-gamma-value').textContent = fmt(gamma, 1);
    model = study(SEQUENCES[condition], {...PAPER, gamma}).model;
    recalled = [];
    draw();
  }
  function choose(k) {
    if (recalled.includes(k) || recalled.length >= N) return;
    present(model, k); recalled.push(k); draw();
  }
  function sample() {
    const pr = probabilities(model, recalled);
    let u = Math.random(), k = 0;
    while (k < N - 1 && (u -= pr[k]) > 0) k++;
    while (recalled.includes(k)) k = (k + 1) % N;
    choose(k);
  }
  function draw() {
    const done = recalled.length >= N;
    const pr = done ? new Array(N).fill(0) : probabilities(model, recalled);
    wordBars($('recall-board'), pr, {max: Math.max(.5, ...pr), color: BLUE, dim: k => recalled.includes(k), onClick: choose,
                                       badge: k => recalled.includes(k) ? String(recalled.indexOf(k) + 1) : ''});
    contextStrip($('recall-context'), model.ctx, recalled.length ? recalled[recalled.length - 1] : -1);
    const seq = recalled.map((k, i) => i ? `${WORDS[k]} (${signed(k - recalled[i - 1])})` : WORDS[k]);
    $('recall-sequence').textContent = recalled.length ? 'Recall order: ' + seq.join(' → ') : 'No words recalled yet.';
    $('recall-sample').disabled = done;
  }
  $('recall-sample').addEventListener('click', sample);
  $('recall-reset').addEventListener('click', reset);
  ['recall-condition', 'recall-gamma'].forEach(id => $(id).addEventListener('input', reset));
  reset();
})();

(() => {
  function draw() {
    const condition = $('time-condition').value;
    const p = {beta: +$('time-beta').value, gamma: 1, tau: +$('time-tau').value, d: +$('time-d').value};
    $('time-beta-value').textContent = fmt(p.beta, 3);
    $('time-tau-value').textContent = fmt(p.tau, 3);
    $('time-d-value').textContent = fmt(p.d, 2);
    const {pfr, crp} = predict(SEQUENCES[condition], p);
    pfrChart($('time-pfr'), condition, pfr);
    crpChart($('time-crp'), condition, crp);
  }
  ['time-condition', 'time-beta', 'time-tau', 'time-d'].forEach(id => $(id).addEventListener('input', draw));
  $('time-reset').addEventListener('click', () => { $('time-beta').value = PAPER.beta; $('time-tau').value = PAPER.tau; $('time-d').value = PAPER.d; draw(); });
  draw();
})();

})();
