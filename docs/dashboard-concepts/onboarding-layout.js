const ObservatoryTourLayout = (() => {
  const clamp = (value, min, max) => Math.max(min, Math.min(value, Math.max(min, max)));

  function compute(rect, viewport, card) {
    const gap = 18, margin = 16;
    const width = Math.min(card.width, viewport.width - margin * 2);
    const height = Math.min(card.height, viewport.height - margin * 2);
    const right = viewport.width - rect.right - gap - margin;
    const left = rect.left - gap - margin;
    const beside = viewport.width >= 1000 && Math.max(left, right) >= width;
    const x = beside && left > right ? rect.left - width - gap :
      beside ? rect.right + gap : viewport.width - width - margin;
    const y = beside ? clamp(rect.top, margin, viewport.height - height - margin) :
      viewport.height - height - margin;
    const limit = beside ? viewport.height - margin : y - gap;
    const box = { left: clamp(rect.left - 6, 8, viewport.width - 8),
      top: Math.max(margin, rect.top - 6),
      right: clamp(rect.right + 6, 8, viewport.width - 8),
      bottom: Math.min(rect.bottom + 6, limit) };
    return { x, y, width, beside, limit, box,
      visible: box.bottom - box.top > 12 && box.right - box.left > 12 };
  }

  return { compute };
})();
