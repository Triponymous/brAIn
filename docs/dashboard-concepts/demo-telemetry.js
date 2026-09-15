// An inspectable fixture generator, deliberately not presented as a snnTorch run.
const DemoTelemetry = (() => {
  const meta = Object.freeze({
    id: 'obs-demo-03', generator: 'procedural-lif-v1', seed: 4271,
    frames: 200, sampleHz: 100, durationSeconds: 2,
    edgeSeed: 92712, layoutSeed: 441, synthetic: true
  });
  const cache = new Map();

  function generate(scenario) {
    const scenarioIndex = Object.keys(scenarios).indexOf(scenario);
    const heat = scenarios[scenario].heat;
    const result = new Map();
    nodes.forEach((node, order) => {
      const rand = random(meta.seed + scenarioIndex * 100003 + order * 917);
      const voltage = new Float32Array(meta.frames);
      const output = new Uint8Array(meta.frames);
      const cumulative = new Uint16Array(meta.frames);
      let v = rand() * .5, count = 0;
      for (let frame = 0; frame < meta.frames; frame++) {
        const drive = heat[node.g] * (.035 + rand() * .055) *
          (1 + .32 * Math.sin(frame * .11 + node.index));
        if (node.key === 'e') {
          output[frame] = rand() < heat[node.g] * .3 ? 1 : 0;
          voltage[frame] = output[frame];
        } else {
          v = .975 * v + drive;
          output[frame] = v >= 1 ? 1 : 0;
          voltage[frame] = Math.min(v, 1);
          if (output[frame]) v = 0;
        }
        count += output[frame];
        cumulative[frame] = count;
      }
      result.set(node.id, { voltage, output, cumulative });
    });
    cache.set(scenario, result);
    return result;
  }

  function series(scenario, id) {
    return (cache.get(scenario) || generate(scenario)).get(id);
  }

  function sample(scenario, id, frame) {
    const trace = series(scenario, id);
    let glow = 0;
    for (let back = 0; back <= 4 && back <= frame; back++) {
      if (trace.output[frame - back]) { glow = 1 - back * .18; break; }
    }
    return { voltage: trace.voltage[frame], output: trace.output[frame], glow,
      count: trace.cumulative[frame], total: trace.cumulative[meta.frames - 1] };
  }

  function activity(scenario, frame) {
    return Float32Array.from(nodes, node => sample(scenario, node.id, frame).glow);
  }

  function snapshotSeries(scenario, id) {
    const trace = series(scenario, id);
    return { unit: id, voltage_au: Array.from(trace.voltage), output: Array.from(trace.output) };
  }

  return Object.freeze({ meta, series, sample, activity, snapshotSeries });
})();
