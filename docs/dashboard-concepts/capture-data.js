/* Only confirmed server policy may describe capture as on or off. */
const CaptureData = (() => {
  const keys = ['keystroke_rate', 'mouse_rate', 'idle', 'active_app'];
  function validate(data) {
    if (!data || data.schema !== 'brain.capture.v1' || data.scope !== 'observation_runner' ||
        data.persistence !== 'session_only' || typeof data.session_id !== 'string' ||
        !data.session_id.length || data.session_id.length > 64 ||
        !Number.isSafeInteger(data.revision) || data.revision < 0 ||
        typeof data.armed !== 'boolean' || typeof data.failed !== 'boolean' ||
        Object.keys(data.inputs ?? {}).length !== keys.length) throw Error('Unsupported capture controls');
    for (const key of keys) {
      const input = data.inputs[key];
      if (!input || typeof input.enabled !== 'boolean' ||
          !['disabled', 'waiting', 'available', 'permission_required', 'unavailable', 'error'].includes(input.status) ||
          (!data.armed && input.enabled) || (!input.enabled && input.status !== 'disabled') ||
          (input.enabled && input.status === 'disabled') || (input.status === 'error' && !data.failed))
        throw Error('Invalid capture state');
    }
    return data;
  }
  function change(data, enabled) {
    validate(data);
    const fields = Object.keys(enabled);
    if (!fields.length || fields.some(key => !keys.includes(key) || typeof enabled[key] !== 'boolean'))
      throw Error('Invalid input selection');
    return { session_id: data.session_id, revision: data.revision, enabled: { ...enabled } };
  }
  return { keys, validate, change };
})();
