import { Script, runInContext } from 'node:vm';
import assert from 'node:assert/strict';
import test from 'node:test';
import { harness, read } from './tour-test-harness.mjs';

test('tour scripts parse; every chapter and target binds to the actual markup', () => {
  const html = read('observatory.html');
  for (const name of ['onboarding-data.js', 'onboarding-layout.js', 'onboarding.js']) new Script(read(name));
  const { data } = harness();
  assert.equal(data.steps.length, 18);
  assert.equal(new Set(data.steps.map(step => step.id)).size, data.steps.length);
  const pages = [...new Set(data.steps.map(step => step.page))];
  assert.deepEqual(pages, Array.from(data.chapters, chapter => chapter[0]));
  for (const step of data.steps) {
    assert(step.description && step.caution && step.source);
    if (step.target.startsWith('#')) assert(html.includes(`id="${step.target.slice(1)}"`), step.id);
    else assert(new RegExp('class="[^"]*\\b' + step.target.slice(1) + '\\b').test(html), step.id);
    if (step.task) {
      assert(['change', 'click', 'input'].includes(step.event));
      assert(!/capture|export|teach|confirm|connect/.test(step.action));
      assert(step.success);
    }
  }
  const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map(match => match[1]);
  assert.equal(new Set(ids).size, ids.length);
  assert(html.indexOf('src="onboarding-data.js"') < html.indexOf('src="onboarding.js"'));
});

test('first visit offers the guide without navigation; only a seen flag is persisted', () => {
  const h = harness({ page: 'journal' });
  assert(h.ids.get('tour-welcome').open);
  assert(h.ids.get('tour-card').hidden);
  assert.equal(h.doc.body.dataset.page, 'journal');
  assert.equal(h.routes.length, 0);
  assert.deepEqual(h.writes, [[h.data.storageKey, 'seen']]);
  h.click('dismiss');
  assert(!h.ids.get('tour-welcome').open);
  assert.equal(h.doc.activeElement, h.buttons.get('open'));
});

test('returning visitors are not interrupted, but can reopen the guide', () => {
  const h = harness({ visited: true });
  assert(!h.ids.get('tour-welcome').open);
  h.click('open');
  assert(h.ids.get('tour-welcome').open);
  h.click('start');
  assert.equal(h.ids.get('tour-card').dataset.step, 'connection');
});

test('blocked storage does not prevent opening, navigation or exit', () => {
  const h = harness({ blockedStorage: true });
  h.click('start'); h.click('next'); h.click('exit');
  assert(h.ids.get('tour-card').hidden);
  assert.equal(h.doc.body.dataset.page, 'live');
  assert.equal(h.writes.length, 0);
});

test('the entire tour advances without connecting, capturing, annotating or exporting', () => {
  const h = harness({ page: 'journal' });
  h.click('start');
  for (const [index, step] of h.data.steps.entries()) {
    assert.equal(h.ids.get('tour-card').dataset.step, step.id);
    assert.equal(h.doc.body.dataset.page, step.page);
    assert.equal(h.ids.get('tour-progress').value, index + 1);
    assert(!h.ids.get('tour-card').hidden);
    h.click('next');
  }
  assert(h.ids.get('tour-card').hidden);
  assert(h.ids.get('tour-spotlight').hidden);
  assert.equal(h.doc.body.dataset.page, 'journal');
  assert.equal(h.win.scrollY, 120);
  assert(!h.doc.body.classList.contains('tour-active'));
  assert.equal(h.writes.length, 1);
  for (const target of h.targets.values()) {
    assert(!target.hasAttribute('data-tour-highlight'));
    assert(!target.hasAttribute('aria-describedby'));
    assert(!target.hasAttribute('tabindex'));
  }
});

test('back, arbitrary chapter selection and missing target all keep an exit path', () => {
  const h = harness(); h.click('start'); h.click('back');
  assert.equal(h.ids.get('tour-card').dataset.step, 'connection');
  h.chapter('vocabulary');
  assert.equal(h.ids.get('tour-card').dataset.step, 'vocabulary');
  h.click('back'); assert.equal(h.ids.get('tour-card').dataset.step, 'replay');
  h.targets.delete('.methods-lead'); h.chapter('methods');
  assert(h.ids.get('tour-spotlight').hidden);
  assert(h.buttons.get('focus').disabled);
  h.click('next'); h.click('exit');
  assert(h.ids.get('tour-card').hidden);
});

test('optional exercises acknowledge only the expected event and never auto-advance', () => {
  const h = harness(); h.click('start'); h.chapter('overview'); h.click('next');
  const control = h.make('select'); control.closest = selector => selector === '#moment-select' ? control : null;
  h.doc.emit('click', { target: control });
  assert.equal(h.ids.get('tour-result').textContent, '');
  h.doc.emit('change', { target: control }); h.flush();
  assert(h.ids.get('tour-result').textContent.startsWith('✓'));
  assert.equal(h.ids.get('tour-card').dataset.step, 'contexts');
  h.click('next'); assert.equal(h.ids.get('tour-result').textContent, '');
});

test('Escape restores the origin, while ordinary navigation respects the chosen page', () => {
  const h = harness({ page: 'vocabulary' }); h.click('start');
  h.doc.emit('keydown', { key: 'Escape', target: h.buttons.get('next') });
  assert(h.ids.get('tour-card').hidden);
  assert.equal(h.doc.body.dataset.page, 'vocabulary');
  h.click('open'); h.click('start');
  h.context.location.hash = 'methods'; h.doc.body.dataset.page = 'methods';
  h.win.emit('hashchange');
  assert(h.ids.get('tour-card').hidden);
  assert.equal(h.doc.body.dataset.page, 'methods');
});

test('cleanup preserves pre-existing descriptions and tabindex; a modal gets Escape first', () => {
  const h = harness(); const target = h.targets.get('.live-connection');
  target.setAttribute('aria-describedby', 'existing-note'); target.setAttribute('tabindex', '0');
  h.click('start'); assert.equal(target.getAttribute('aria-describedby'), 'existing-note tour-description');
  h.ids.get('tour-welcome').open = true;
  h.doc.emit('keydown', { key: 'Escape', target: h.buttons.get('next') });
  assert(!h.ids.get('tour-card').hidden);
  h.ids.get('tour-welcome').open = false; h.click('exit');
  assert.equal(target.getAttribute('aria-describedby'), 'existing-note');
  assert.equal(target.getAttribute('tabindex'), '0');
});

test('positioning stays on screen and keeps the visible spotlight clear of the card', () => {
  const h = harness(); const compute = runInContext('ObservatoryTourLayout.compute', h.context);
  for (const [width, height] of [[320,568], [390,844], [677,987], [1440,900], [844,390]]) {
    for (const left of [0, 80, width * .55]) for (const top of [-100, 24, height * .4, height + 10]) {
      const rect = { left, top, right: Math.min(width - 16, left + 320), bottom: top + 250 };
      const result = compute(rect, { width, height }, { width: Math.min(376, width - 32), height: height * .52 });
      assert(result.x >= 0 && result.x + result.width <= width);
      assert(result.y >= 0 && result.y < height);
      if (result.visible) {
        const b = result.box;
        assert(b.left >= 0 && b.right <= width && b.top >= 0 && b.bottom <= height);
        assert(b.bottom <= result.y || b.right <= result.x || b.left >= result.x + result.width);
      }
    }
  }
});

test('tour animation is opt-in to motion; live controls are not restyled or auto-clicked', () => {
  const css = read('onboarding.css'), js = read('onboarding.js'), html = read('observatory.html');
  assert(css.includes('@media (prefers-reduced-motion: no-preference)'));
  assert(!/transition:\s*(all|top|left|width|height)/.test(css));
  assert(html.includes('aria-modal="false" aria-labelledby="tour-title"'));
  assert(html.includes('lang="de"'));
  assert(!/\.click\(|dispatchEvent\(|fetch\(|\.check\(/.test(js));
});
