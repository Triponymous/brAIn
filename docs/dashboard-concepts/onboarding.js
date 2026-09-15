(() => {
  const { steps, chapters, storageKey } = ObservatoryTourData;
  const el = id => document.getElementById(id);
  const welcome = el('tour-welcome'), card = el('tour-card'), spotlight = el('tour-spotlight');
  let index = -1, entry = null, target = null, addedTabIndex = false, scheduled = 0;
  const active = () => index >= 0;
  const viewport = () => ({ width: document.documentElement.clientWidth, height: window.innerHeight });

  function seen() {
    try { return localStorage.getItem(storageKey) === 'seen'; } catch { return false; }
  }
  function remember() {
    try { localStorage.setItem(storageKey, 'seen'); } catch { /* Tours also work without browser storage. */ }
  }
  function captureEntry() {
    entry = { page: document.body.dataset.page, scroll: window.scrollY, focus: document.activeElement };
  }
  function visit(page) {
    if (document.body.dataset.page === page) return;
    const url = new URL(location.href);
    url.hash = page;
    history.replaceState(null, '', url);
    setPage();
  }
  function restoreFocus() {
    const previous = entry?.focus;
    const focus = previous?.isConnected && previous.getClientRects().length && previous !== document.body ?
      previous : document.querySelector('.tour-launch');
    focus?.focus({ preventScroll: true });
  }
  function openWelcome() {
    if (welcome.open || document.querySelector('dialog[open]')) return;
    if (!active()) captureEntry();
    stop();
    welcome.showModal();
    el('tour-welcome-title').focus();
    remember();
  }
  function clearTarget() {
    if (!target) return;
    target.removeAttribute('data-tour-highlight');
    const description = (target.getAttribute('aria-describedby') || '').split(' ')
      .filter(id => id && id !== 'tour-description');
    if (description.length) target.setAttribute('aria-describedby', description.join(' '));
    else target.removeAttribute('aria-describedby');
    if (addedTabIndex && target.getAttribute('tabindex') === '-1') target.removeAttribute('tabindex');
    target = null;
    addedTabIndex = false;
  }
  function markTarget() {
    clearTarget();
    target = document.querySelector(steps[index].target);
    if (!target) return;
    target.setAttribute('data-tour-highlight', steps[index].id);
    const description = target.getAttribute('aria-describedby');
    target.setAttribute('aria-describedby', [description, 'tour-description'].filter(Boolean).join(' '));
    if (!target.hasAttribute('tabindex') && !target.matches('button,input,select,a[href]')) {
      target.setAttribute('tabindex', '-1');
      addedTabIndex = true;
    }
  }
  function layout(scroll = false) {
    if (!active()) return;
    const valid = target?.isConnected && target.getClientRects().length;
    card.querySelector('[data-tour-focus]').disabled = !valid;
    const bounds = valid ? target.getBoundingClientRect() : { left: 0, right: 0, top: 0, bottom: 0 };
    let result = ObservatoryTourLayout.compute(bounds, viewport(), card.getBoundingClientRect());
    document.body.style.setProperty('--tour-clearance', card.offsetHeight + 48 + 'px');
    if (scroll && valid) {
      const available = Math.max(50, result.limit - 32);
      const top = 24 + Math.max(0, (available - Math.min(bounds.height, available)) / 2);
      window.scrollTo({ top: Math.max(0, window.scrollY + bounds.top - top), behavior: 'instant' });
      result = ObservatoryTourLayout.compute(target.getBoundingClientRect(), viewport(), card.getBoundingClientRect());
    }
    card.style.left = result.x + 'px';
    card.style.top = result.y + 'px';
    spotlight.hidden = !valid || !result.visible;
    if (spotlight.hidden) return;
    const { box } = result;
    Object.assign(spotlight.style, { left: box.left + 'px', top: box.top + 'px',
      width: box.right - box.left + 'px', height: box.bottom - box.top + 'px' });
  }
  function scheduleLayout() {
    if (!active() || scheduled) return;
    scheduled = requestAnimationFrame(() => { scheduled = 0; layout(); });
  }
  function go(next) {
    if (!Number.isInteger(next) || next < 0 || next >= steps.length) return;
    if (!entry) captureEntry();
    index = next;
    welcome.close();
    const step = steps[index];
    visit(step.page);
    card.hidden = false;
    card.dataset.step = step.id;
    document.body.classList.add('tour-active');
    for (const [id, value] of Object.entries({ 'tour-title': step.title, 'tour-source': step.source,
      'tour-description': step.description, 'tour-caution': step.caution,
      'tour-task': step.task || '', 'tour-result': '', 'tour-marker': String(index + 1).padStart(2, '0'),
      'tour-count': `${String(index + 1).padStart(2, '0')} / ${steps.length}` })) el(id).textContent = value;
    el('tour-exercise').hidden = !step.task;
    el('tour-progress').max = steps.length;
    el('tour-progress').value = index + 1;
    el('tour-chapter').value = step.page;
    card.querySelector('[data-tour-back]').disabled = index === 0;
    card.querySelector('[data-tour-next]').textContent = index === steps.length - 1 ? 'Fertig ✓' : 'Weiter →';
    markTarget();
    card.querySelector('.tour-card-content').scrollTop = 0;
    el('tour-title').focus({ preventScroll: true });
    layout(true);
    // The existing router moves the shared graph on its next frame.
    requestAnimationFrame(() => { if (active() && steps[index] === step) layout(true); });
  }
  function finish(restore = true) {
    index = -1;
    clearTarget();
    cancelAnimationFrame(scheduled);
    scheduled = 0;
    card.hidden = spotlight.hidden = true;
    document.body.classList.remove('tour-active');
    document.body.style.removeProperty('--tour-clearance');
    if (restore && entry) {
      visit(entry.page);
      window.scrollTo({ top: entry.scroll, behavior: 'instant' });
      restoreFocus();
    }
    entry = null;
  }
  function focusTarget() {
    layout(true);
    if (!target) return;
    const focusable = 'button:not(:disabled),select:not(:disabled),input:not(:disabled),a[href],[tabindex="0"]';
    const focus = target.matches(focusable) ? target : target.querySelector(focusable) || target;
    focus.focus({ preventScroll: true });
  }
  function exercise(event) {
    if (!active()) return;
    const step = steps[index];
    if (step.event !== event.type || !event.target.closest(step.action)) return;
    el('tour-result').textContent = '✓ ' + step.success;
    scheduleLayout();
  }

  for (const [page, label] of chapters) {
    const option = document.createElement('option');
    option.value = page; option.textContent = label; el('tour-chapter').append(option);
  }
  document.addEventListener('click', event => {
    const button = event.target.closest('[data-tour-open],[data-tour-dismiss],[data-tour-start],'
      + '[data-tour-chapter],[data-tour-back],[data-tour-next],[data-tour-exit],[data-tour-focus]');
    if (!button) return;
    if (button.hasAttribute('data-tour-open')) openWelcome();
    else if (button.hasAttribute('data-tour-dismiss')) welcome.close();
    else if (button.hasAttribute('data-tour-start')) go(0);
    else if (button.hasAttribute('data-tour-chapter')) go(steps.findIndex(step => step.page === button.dataset.tourChapter));
    else if (button.hasAttribute('data-tour-back')) go(index - 1);
    else if (button.hasAttribute('data-tour-next')) index === steps.length - 1 ? finish() : go(index + 1);
    else if (button.hasAttribute('data-tour-exit')) finish();
    else if (button.hasAttribute('data-tour-focus')) focusTarget();
  });
  el('tour-chapter').addEventListener('change', event => go(steps.findIndex(step => step.page === event.target.value)));
  for (const type of ['click', 'change', 'input']) document.addEventListener(type, exercise);
  welcome.addEventListener('close', () => { if (!active()) { restoreFocus(); entry = null; } });
  document.addEventListener('keydown', event => {
    if (!active() || document.querySelector('dialog[open]')) return;
    if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); finish(); }
    if (!card.contains(event.target) || event.target.matches('select,input,textarea')) return;
    if (event.key === 'ArrowRight' && index < steps.length - 1) { event.preventDefault(); go(index + 1); }
    if (event.key === 'ArrowLeft' && index > 0) { event.preventDefault(); go(index - 1); }
  }, true);
  window.addEventListener('hashchange', () => {
    if (active() && location.hash !== '#' + steps[index].page) finish(false);
  });
  window.addEventListener('resize', scheduleLayout);
  window.addEventListener('scroll', scheduleLayout, { passive: true });
  new ResizeObserver(scheduleLayout).observe(card);
  if (!seen()) openWelcome();
})();
