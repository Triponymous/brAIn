/* Transport boundary for recorded model output. Never substitutes demo data. */
const LiveData = (() => {
  const keys = ['s', 'e', 'c', 'w'];
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  function validate(data) {
    if (!data || data.schema !== 'brain.telemetry.v1' || typeof data.session_id !== 'string' ||
        data.session_id.length > 64 || !data.session_id.length) throw Error('Unsupported telemetry contract');
    if (!['waiting', 'streaming', 'stale', 'error', 'paused'].includes(data.status) ||
        !['desktop_metadata', 'desktop_sensors', 'disabled', 'test_fixture'].includes(data.source?.input_kind)) throw Error('Unknown data provenance');
    if (data.age_s !== null && (!finite(data.age_s) || data.age_s < 0)) throw Error('Invalid sample age');
    if (data.capture) {
      CaptureData.validate(data.capture);
      if (data.capture.session_id !== data.session_id) throw Error('Capture session mismatch');
    }
    for (const key of keys) {
      const count = data.architecture?.[key];
      if (!Number.isInteger(count) || count < 1 || count > 4096) throw Error('Invalid architecture');
    }
    if (!Array.isArray(data.frames) || data.frames.length > 200) throw Error('Invalid batch');
    if (data.source.input_kind === 'disabled' && data.frames.length) throw Error('Disabled inputs cannot have recorded frames');
    let tick = 0, elapsed = -1;
    for (const frame of data.frames) {
      if (!Number.isSafeInteger(frame.tick) || frame.tick <= tick || !finite(frame.elapsed_s) ||
          frame.elapsed_s < 0 || frame.elapsed_s < elapsed || !Number.isFinite(Date.parse(frame.captured_at))) throw Error('Invalid sample order');
      tick = frame.tick; elapsed = frame.elapsed_s;
      if (frame.capture_revision != null && (!Number.isSafeInteger(frame.capture_revision) || frame.capture_revision < 0 ||
          (data.capture && frame.capture_revision > data.capture.revision))) throw Error('Invalid capture revision');
      for (const key of keys) {
        const region = frame.regions?.[key], count = data.architecture[key];
        if (!Array.isArray(region?.output) || region.output.length !== count ||
            !region.output.every(value => value === 0 || value === 1)) throw Error('Invalid output vector');
        if (key === 'e' ? region.membrane !== null : !Array.isArray(region.membrane) ||
            region.membrane.length !== count || !region.membrane.every(finite)) throw Error('Invalid membrane vector');
      }
      for (const key of ['DA', 'NE', 'ACh', '5HT']) {
        if (!finite(frame.modulators?.[key])) throw Error('Invalid model modulator');
      }
      for (const key of ['keystroke_rate', 'mouse_rate', 'idle', 'active_app', 'microphone', 'wearable']) {
        const sensor = frame.sensors?.[key];
        if (!sensor || !['available', 'unavailable', 'disabled', 'permission_required', 'not_connected'].includes(sensor.status))
          throw Error('Missing sensor provenance');
        if (sensor.status === 'available' && (!finite(sensor.observed_at) ||
            (key === 'active_app' ? !['development','browser','communication','writing','media','design','other'].includes(sensor.value) :
              !finite(sensor.value) || sensor.value < 0))) throw Error('Invalid sensor measurement');
      }
    }
    return data;
  }

  function merge(previous, data) {
    validate(data);
    if (previous?.meta.session_id === data.session_id &&
        (keys.some(key => previous.meta.architecture[key] !== data.architecture[key]) ||
          previous.meta.source.input_kind !== data.source.input_kind)) throw Error('Source changed without a new session');
    const reset = !previous || previous.meta.session_id !== data.session_id || data.reset || data.gap;
    const frames = reset ? [] : [...previous.frames];
    for (const frame of data.frames) {
      const last = frames.at(-1);
      if (last && frame.tick <= last.tick) continue;
      if (last && frame.elapsed_s < last.elapsed_s) throw Error('Clock moved backwards');
      frames.push(frame);
    }
    return { meta: { ...data, frames: undefined }, frames: frames.slice(-200),
      gaps: (reset ? 0 : previous.gaps) + Number(!!data.gap) };
  }

  function activity(frame) {
    return keys.flatMap(key => frame.regions[key].output);
  }

  function exportWindow(store, cursor, unit) {
    return structuredClone({ schema: 'brain.observation.export.v1', exported_at: new Date().toISOString(),
      provenance: store.meta, frames: store.frames, selected_tick: cursor?.tick ?? null, unit,
      limitations: ['Actual computational output, not a biological brain measurement',
        'Post-step membrane, not the pre-reset peak', 'No measured synaptic edges or anatomical localization',
        'No learned emotion labels, wearable inputs, or verified product benefit',
        'Rolling window only; not a complete session or model checkpoint'] });
  }
  return { validate, merge, activity, exportWindow };
})();
