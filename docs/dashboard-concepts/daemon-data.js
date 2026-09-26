/* The persistent daemon's contracts. Only confirmed server state may describe a source as shared. */
const DaemonData = (() => {
  const keys = ['keystroke_rate', 'mouse_rate', 'idle', 'active_app', 'mic'];
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  function consent(data) {
    if (!data || data.schema !== 'brain.consent.v1' || !Number.isSafeInteger(data.revision) || data.revision < 0 ||
        typeof data.paused !== 'boolean' || Object.keys(data.sources ?? {}).length !== keys.length) throw Error('Unsupported source controls');
    for (const key of keys) {
      const source = data.sources[key];
      if (!source || typeof source.enabled !== 'boolean' || !['disabled', 'waiting', 'available'].includes(source.status) ||
          (source.changed_at !== null && !finite(source.changed_at)) ||
          source.enabled === (source.status === 'disabled')) throw Error('Invalid source state');
    }
    if (data.paused !== !keys.some(key => data.sources[key].enabled)) throw Error('Pause contradicts the sources');
    return data;
  }
  function change(data, enabled) {
    consent(data);
    const fields = Object.keys(enabled);
    if (!fields.length || fields.some(key => !keys.includes(key) || typeof enabled[key] !== 'boolean'))
      throw Error('Invalid source selection');
    return { revision: data.revision, enabled: { ...enabled } };
  }
  function feel(data) {
    if (!data || typeof data.paused !== 'boolean' || !Array.isArray(data.known_labels) || data.known_labels.length > 500 ||
        !data.known_labels.every(label => typeof label === 'string' && label.length && label.length <= 64) ||
        (data.recognized !== null && !data.known_labels.includes(data.recognized)) ||
        !finite(data.confidence) || data.confidence < 0 || data.confidence > 1 ||
        (data.pending_ask !== null && !finite(data.pending_ask?.at)) ||
        (data.paused && data.recognized !== null)) throw Error('Unsupported felt-state reply');
    return data;
  }
  function daemon(data) {
    if (!data || typeof data.running !== 'boolean' ||
        (data.running ? !Number.isSafeInteger(data.pid) || data.pid < 1 : data.pid !== null && data.pid !== undefined))
      throw Error('Unsupported daemon status');
    return data;
  }
  return { keys, consent, change, feel, daemon };
})();
