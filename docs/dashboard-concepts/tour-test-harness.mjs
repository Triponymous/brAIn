import { createContext, runInContext } from 'node:vm';
import { readFileSync } from 'node:fs';

export const read = name => readFileSync(new URL(name, import.meta.url), 'utf8');

class Element {
  constructor(tag = 'div') {
    this.tag = tag; this.attrs = new Map(); this.dataset = {}; this.events = new Map();
    this.style = { setProperty() {}, removeProperty() {} };
    const classes = new Set();
    this.classList = { add: x => classes.add(x), remove: x => classes.delete(x), contains: x => classes.has(x) };
    this.children = []; this.hidden = false; this.open = false; this.isConnected = true;
    this.rect = { left: 40, right: 600, top: 80, bottom: 280, width: 560, height: 200 };
  }
  addEventListener(type, handler) { this.events.set(type, [...(this.events.get(type) || []), handler]); }
  emit(type, data = {}) {
    const event = { target: this, type, preventDefault() {}, stopPropagation() {}, ...data };
    for (const handler of this.events.get(type) || []) handler(event);
  }
  setAttribute(name, value) { this.attrs.set(name, String(value)); }
  getAttribute(name) { return this.attrs.get(name) ?? null; }
  hasAttribute(name) { return this.attrs.has(name); }
  removeAttribute(name) { this.attrs.delete(name); }
  append(child) { this.children.push(child); child.parent = this; }
  getBoundingClientRect() { return this.rect; }
  getClientRects() { return this.hidden ? [] : [this.rect]; }
  get offsetHeight() { return this.rect.height; }
  focus() { this.doc.activeElement = this; }
  showModal() { this.open = true; }
  close() { if (this.open) { this.open = false; this.emit('close'); } }
  matches(selector) {
    return selector.split(',').some(part => {
      const s = part.trim();
      const attribute = s.match(/^\[([^=\]]+)(?:="([^"]*)")?\]$/);
      if (attribute) return this.hasAttribute(attribute[1]) && (attribute[2] === undefined || this.getAttribute(attribute[1]) === attribute[2]);
      return s === this.tag || s === this.tag + ':not(:disabled)' && !this.disabled ||
        s === 'a[href]' && this.tag === 'a' && this.hasAttribute('href');
    });
  }
  closest(selector) { return this.matches(selector) ? this : this.parent?.closest(selector) ?? null; }
  contains(element) { return element === this || this.children.some(child => child.contains(element)); }
  querySelector(selector) { return this.children.find(child => child.matches(selector)) ?? this.children.map(child => child.querySelector(selector)).find(Boolean) ?? null; }
}

export function harness({ visited = false, blockedStorage = false, page = 'live', language = null } = {}) {
  const context = createContext({});
  runInContext(read('onboarding-data.js') + read('onboarding-en.js') + read('onboarding-i18n.js') + read('onboarding-layout.js'), context);
  const data = runInContext('ObservatoryTourData', context);
  const ids = new Map(), targets = new Map(), routes = [], frames = new Map(), writes = [];
  const doc = new Element('document');
  const make = (tag = 'div') => { const node = new Element(tag); node.doc = doc; return node; };
  doc.body = make('body'); doc.body.dataset.page = page; doc.activeElement = doc.body;
  doc.documentElement = { clientWidth: 1280 };
  for (const match of read('observatory.html').matchAll(/\bid="([^"]+)"/g)) ids.set(match[1], make());
  const welcome = ids.get('tour-welcome'), card = ids.get('tour-card');
  welcome.tag = 'dialog'; card.hidden = true; ids.get('tour-spotlight').hidden = true;
  card.rect = { left: 0, right: 376, top: 0, bottom: 440, width: 376, height: 440 };
  const content = make(); content.setAttribute('class', 'tour-card-content');
  const baseQuery = card.querySelector.bind(card);
  card.querySelector = selector => selector === '.tour-card-content' ? content : baseQuery(selector);
  const buttons = new Map();
  for (const action of ['open', 'dismiss', 'start', 'back', 'next', 'exit', 'focus']) {
    const button = make('button'); button.setAttribute('data-tour-' + action, '');
    buttons.set(action, button);
    if (['back', 'next', 'exit', 'focus'].includes(action)) card.append(button);
  }
  for (const step of data.steps) {
    if (!targets.has(step.target)) targets.set(step.target, make());
  }
  doc.getElementById = id => ids.get(id);
  doc.createElement = tag => make(tag);
  doc.querySelector = selector => selector === 'dialog[open]' ? (welcome.open ? welcome : null) :
    selector === '.tour-launch' ? buttons.get('open') : targets.get(selector) ?? null;
  const bindings = [];
  for (const match of read('observatory.html').matchAll(/<([\w-]+)\b([^>]*\bdata-tour-(?:text|label|language|open)\b[^>]*)>/g)) {
    const id = match[2].match(/\bid="([^"]+)"/)?.[1];
    const node = id ? ids.get(id) : match[2].includes('data-tour-open') ? buttons.get('open') : make(match[1]);
    for (const attr of match[2].matchAll(/(data-tour-[\w-]+)(?:="([^"]*)")?/g)) {
      node.setAttribute(attr[1], attr[2] || '');
      node.dataset[attr[1].slice(5).replace(/-([a-z])/g, (_, c) => c.toUpperCase())] = attr[2] || '';
    }
    bindings.push(node);
  }
  doc.querySelectorAll = selector => bindings.filter(node => node.matches(selector));
  const location = new URL('http://127.0.0.1:4178/observatory.html#' + page);
  const win = new Element('window'); win.innerHeight = 900; win.scrollY = 120;
  win.scrollTo = ({ top }) => { win.scrollY = top; };
  let raf = 0;
  Object.assign(context, { document: doc, window: win, location, URL,
    localStorage: {
      getItem(key) { if (blockedStorage) throw Error('blocked'); return key === data.storageKey ? (visited ? 'seen' : null) : language; },
      setItem(key, value) { if (blockedStorage) throw Error('blocked'); if (key === data.storageKey) visited = true; else language = value; writes.push([key, value]); }
    },
    history: { replaceState(_state, _title, url) { location.href = String(url); } },
    setPage() { doc.body.dataset.page = location.hash.slice(1); routes.push(doc.body.dataset.page); },
    stop() {}, requestAnimationFrame(fn) { frames.set(++raf, fn); return raf; },
    cancelAnimationFrame(id) { frames.delete(id); },
    ResizeObserver: class { observe() {} },
    fetch() { throw Error('The tour must not make network requests'); },
    CaptureControls: new Proxy({}, { get() { throw Error('The tour must not control capture'); } }),
    LiveWorkspace: new Proxy({}, { get() { throw Error('The tour must not connect or export'); } })
  });
  runInContext(read('onboarding.js'), context);
  const flush = () => { const pending = [...frames.values()]; frames.clear(); pending.forEach(fn => fn()); };
  const click = action => { doc.emit('click', { target: buttons.get(action) }); flush(); };
  const chapter = page => { const node = ids.get('tour-chapter'); node.value = page; node.emit('change'); flush(); };
  const changeLanguage = (value, id = 'tour-card-language') => { const node = ids.get(id); node.value = value; node.emit('change'); flush(); };
  return { context, doc, win, data, ids, targets, buttons, routes, writes, click, chapter, changeLanguage, bindings, flush, make };
}
