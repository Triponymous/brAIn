/**
 * brAIntest Pet Face — Eye Animation
 *
 * Connects to the braind WebSocket and animates two eyes based on
 * the brain's neuromodulator levels.
 *
 * Eye states (driven by real brain state, not scripted):
 *   idle:     normal size, slow blink ~5s
 *   curious:  wider eyes, faster blinks
 *   alarmed:  very wide, small pupils, brief freeze
 *   sleepy:   half-closed lids, very slow blink
 *   asleep:   eyes closed, gentle breathing
 */

// --- Push-to-talk via global hotkey (Option+Space) ---
let voiceState = 'idle'; // idle | listening | speaking

if (window.__TAURI__) {
  window.__TAURI__.event.listen('push-to-talk', async () => {
    if (voiceState !== 'idle') return;
    voiceState = 'listening';

    try {
      const resp = await fetch('http://localhost:8000/api/voice-chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ duration: 5 }),
      });

      const blob = await resp.blob();
      if (blob.size > 0) {
        voiceState = 'speaking';
        const audio = new Audio(URL.createObjectURL(blob));
        audio.onended = () => { voiceState = 'idle'; };
        audio.play();
      } else {
        voiceState = 'idle';
      }
    } catch (err) {
      console.error('Voice chat failed:', err);
      voiceState = 'idle';
    }
  });
}

// --- WebSocket connection ---
let brainState = null;
let wsReconnectTimer = null;

function connectWS() {
  const ws = new WebSocket('ws://localhost:8000/ws');
  ws.onmessage = (ev) => {
    try { brainState = JSON.parse(ev.data); } catch {}
  };
  ws.onclose = () => {
    wsReconnectTimer = setTimeout(connectWS, 2000);
  };
  ws.onerror = () => ws.close();
}
connectWS();

// --- Canvas setup ---
const canvas = document.getElementById('eyes');
const ctx = canvas.getContext('2d');
let W, H;

function resize() {
  const dpr = devicePixelRatio || 1;
  W = window.innerWidth;
  H = window.innerHeight;
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
resize();
window.addEventListener('resize', resize);

// --- Eye state machine ---
const STATES = { idle: 0, curious: 1, alarmed: 2, sleepy: 3, asleep: 4 };

function getTargetState(mods) {
  // Voice states override modulator-driven states
  if (voiceState === 'listening') return STATES.curious; // wide eyes, attentive
  if (voiceState === 'speaking') return STATES.idle;     // calm while speaking

  if (!mods) return STATES.idle;
  const da = mods.DA || 0;
  const ne = mods.NE || 0;
  const ach = mods.ACh || 0;
  const sht = mods['5HT'] || 0;

  if (ne > 0.5) return STATES.alarmed;
  if (da > 0.3 && ach > 0.3) return STATES.curious;
  if (da < 0.05 && ne < 0.05 && ach < 0.05 && sht < 0.05) return STATES.asleep;
  if (da < 0.1 && ne < 0.1) return STATES.sleepy;
  return STATES.idle;
}

// Smoothly interpolated animation params
let eyeOpenness = 1.0;      // 0 = closed, 1 = normal, 1.3 = wide
let pupilSize = 1.0;        // 0.5 = tiny, 1 = normal, 1.2 = dilated
let breathOffset = 0;       // gentle up/down "breathing"
let blinkTimer = 0;
let blinkPhase = 0;         // 0 = open, >0 = in blink animation
let blinkInterval = 4000 + Math.random() * 3000;
let tremor = 0;             // for alarmed shake

// State targets
const stateParams = {
  [STATES.idle]:    { openness: 1.0, pupil: 1.0, blinkRate: 5000 },
  [STATES.curious]: { openness: 1.25, pupil: 1.15, blinkRate: 2500 },
  [STATES.alarmed]: { openness: 1.4, pupil: 0.5, blinkRate: 10000 },
  [STATES.sleepy]:  { openness: 0.5, pupil: 0.8, blinkRate: 8000 },
  [STATES.asleep]:  { openness: 0.05, pupil: 0.6, blinkRate: 15000 },
};

let currentState = STATES.idle;
let targetOpenness = 1.0;
let targetPupil = 1.0;

function lerp(a, b, t) { return a + (b - a) * t; }

// --- Main animation loop ---
let lastTime = performance.now();

function animate(now) {
  const dt = (now - lastTime) / 1000;
  lastTime = now;

  // Update state from brain
  const mods = brainState ? brainState.modulators : null;
  const newState = getTargetState(mods);
  if (newState !== currentState) {
    currentState = newState;
    const p = stateParams[currentState];
    targetOpenness = p.openness;
    targetPupil = p.pupil;
    blinkInterval = p.blinkRate;
  }

  // Smooth interpolation
  eyeOpenness = lerp(eyeOpenness, targetOpenness, dt * 3);
  pupilSize = lerp(pupilSize, targetPupil, dt * 3);

  // Breathing
  breathOffset = Math.sin(now / 2000) * 2;

  // Blink logic
  blinkTimer += dt * 1000;
  if (blinkPhase > 0) {
    blinkPhase -= dt * 8; // blink speed
    if (blinkPhase < 0) blinkPhase = 0;
  } else if (blinkTimer > blinkInterval) {
    blinkTimer = 0;
    blinkPhase = 1.0; // start blink
    blinkInterval = stateParams[currentState].blinkRate * (0.7 + Math.random() * 0.6);
  }

  // Tremor for alarmed state
  tremor = currentState === STATES.alarmed
    ? (Math.random() - 0.5) * 2
    : lerp(tremor, 0, dt * 5);

  // --- Draw ---
  ctx.clearRect(0, 0, W, H);

  // Background: very subtle radial glow when active
  if (brainState) {
    const activity = (mods?.DA || 0) * 0.3 + (mods?.NE || 0) * 0.2;
    if (activity > 0.02) {
      const grad = ctx.createRadialGradient(W/2, H/2, 0, W/2, H/2, W * 0.6);
      grad.addColorStop(0, `rgba(52, 211, 153, ${activity * 0.15})`);
      grad.addColorStop(1, 'rgba(0, 0, 0, 0)');
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, W, H);
    }
  }

  const centerY = H * 0.45 + breathOffset;

  // Draw each eye
  drawEye(W * 0.33 + tremor, centerY, false);
  drawEye(W * 0.67 + tremor, centerY, true);

  // Connection indicator
  ctx.fillStyle = brainState ? '#34d399' : '#ef4444';
  ctx.beginPath();
  ctx.arc(W - 8, 8, 3, 0, Math.PI * 2);
  ctx.fill();

  requestAnimationFrame(animate);
}

function drawEye(x, y, isRight) {
  const baseRadius = W * 0.12;
  const eyeH = baseRadius * eyeOpenness;

  // Blink: reduce height
  const blinkFactor = blinkPhase > 0
    ? 1 - Math.sin(blinkPhase * Math.PI) * 0.95
    : 1;
  const finalH = eyeH * blinkFactor;

  if (finalH < 1) return; // fully closed

  // Eye white (actually dark green-gray)
  ctx.save();
  ctx.beginPath();
  ctx.ellipse(x, y, baseRadius, finalH, 0, 0, Math.PI * 2);
  ctx.clip();

  // Eye fill — subtle gradient
  const eyeGrad = ctx.createRadialGradient(x, y, 0, x, y, baseRadius);
  eyeGrad.addColorStop(0, 'rgba(52, 211, 153, 0.25)');
  eyeGrad.addColorStop(0.7, 'rgba(52, 211, 153, 0.12)');
  eyeGrad.addColorStop(1, 'rgba(52, 211, 153, 0.05)');
  ctx.fillStyle = eyeGrad;
  ctx.fillRect(x - baseRadius, y - finalH, baseRadius * 2, finalH * 2);

  // Pupil
  const pRadius = baseRadius * 0.35 * pupilSize;
  ctx.beginPath();
  ctx.arc(x, y, pRadius, 0, Math.PI * 2);
  ctx.fillStyle = '#34d399';
  ctx.fill();

  // Highlight dot
  ctx.beginPath();
  ctx.arc(x + pRadius * 0.3, y - pRadius * 0.3, pRadius * 0.2, 0, Math.PI * 2);
  ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
  ctx.fill();

  ctx.restore();

  // Eye outline
  ctx.beginPath();
  ctx.ellipse(x, y, baseRadius, finalH, 0, 0, Math.PI * 2);
  ctx.strokeStyle = 'rgba(52, 211, 153, 0.3)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // Eyelid (top, for sleepy/blink state)
  if (blinkFactor < 0.95 || eyeOpenness < 0.8) {
    const lidY = y - finalH;
    const lidH = eyeH * (1 - blinkFactor) + (1 - Math.min(1, eyeOpenness)) * eyeH * 0.5;
    ctx.fillStyle = '#0a0a0f';
    ctx.fillRect(x - baseRadius - 2, lidY - 5, baseRadius * 2 + 4, lidH + 5);
  }
}

requestAnimationFrame(animate);
