/**
 * brAIntest Pet Face — Full Avatar (Tabbie.me / Cozmo style)
 *
 * White face on black AMOLED background. Two rounded-rectangle eyes
 * with pupils + one bezier mouth. All parameters driven CONTINUOUSLY
 * by neuromodulators (DA, NE, ACh, 5HT) — no discrete emotion states.
 *
 * Spec: PET_FACE_SPEC.md
 * Display: 466×466 round AMOLED (or any canvas, scales proportionally)
 */

// ─── Config ───
const DESIGN_SIZE = 466; // spec reference size; we scale to actual canvas

// Layout (spec Section 2 + 9)
const LAYOUT = {
  leftEye:   { cx: 155, cy: 200 },
  rightEye:  { cx: 311, cy: 200 },
  mouth:     { cx: 233, cy: 310 },
};

// Default eye (spec Section 3)
const EYE_DEFAULT = {
  width: 80,
  height: 90,
  radius: 30,
  tilt: 0,          // degrees, positive = clockwise
};

// Pupil (spec Section 3)
const PUPIL_DEFAULT = {
  radius: 12,
  color: '#000000',
};

// Mouth (spec Section 4)
const MOUTH_DEFAULT = {
  width: 50,
  strokeWidth: 4,
  curveY: -5, // negative = smile (control point above line)
};

// ─── Push-to-talk ───
let voiceState = 'idle';

if (window.__TAURI__) {
  window.__TAURI__.event.listen('push-to-talk', async () => {
    if (voiceState !== 'idle') return;
    voiceState = 'listening';
    try {
      const port = window.__BRAIND_PORT || 8000;
      const resp = await fetch(`http://localhost:${port}/api/voice-chat`, {
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

// ─── WebSocket ───
let brainState = null;
let sleepMode = false;

function connectWS() {
  const port = window.__BRAIND_PORT || 8000;
  const ws = new WebSocket(`ws://localhost:${port}/ws`);
  ws.onmessage = (ev) => {
    try {
      const data = JSON.parse(ev.data);
      brainState = data;
      sleepMode = !!data.sleep_mode;
    } catch {}
  };
  ws.onclose = () => setTimeout(connectWS, 2000);
  ws.onerror = () => ws.close();
}
connectWS();

// ─── Canvas ───
const canvas = document.getElementById('eyes');
const ctx = canvas.getContext('2d');
let W, H, S; // width, height, scale factor

function resize() {
  const dpr = devicePixelRatio || 1;
  W = window.innerWidth;
  H = window.innerHeight;
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  S = Math.min(W, H) / DESIGN_SIZE;
}
resize();
window.addEventListener('resize', resize);

// ─── Smoothed face parameters (all floats, interpolated per frame) ───
const face = {
  // Eye shape
  eyeWidthMul: 1.0,    // multiplier on EYE_DEFAULT.width
  eyeHeightMul: 1.0,   // multiplier on EYE_DEFAULT.height
  eyeRadiusMul: 1.0,   // corner radius multiplier
  eyeTiltL: 0,          // left eye tilt in degrees
  eyeTiltR: 0,          // right eye tilt in degrees

  // Pupil
  pupilRadius: PUPIL_DEFAULT.radius,
  pupilOffX: 0,         // gaze offset from eye center
  pupilOffY: 0,

  // Mouth
  mouthWidth: MOUTH_DEFAULT.width,
  mouthCurve: MOUTH_DEFAULT.curveY, // negative=smile, positive=frown
  mouthOpacity: 1.0,

  // Animation
  blinkPhase: 0,        // 0=open, 1=peak of blink
  breathPhase: 0,       // breathing animation phase
  animSpeed: 1.0,       // overall animation speed multiplier

  // Gaze target (for idle drift)
  gazeTargetX: 0,
  gazeTargetY: 0,
  gazeTimer: 0,
};

// Targets (computed from modulators, face lerps toward these)
const target = { ...face };

// ─── Modulator → Face Parameter Mapping (spec Section 7) ───
function updateTargetsFromModulators(mods) {
  const da  = mods?.DA  || 0;
  const ne  = mods?.NE  || 0;
  const ach = mods?.ACh || 0;
  const sht = mods['5HT'] || 0;

  // Normalize modulators to 0-1 range.
  // Real measured ranges: calm=0.01-0.05, active=0.05-0.15, spike=0.15-0.30
  // Scale so 0.15 (active) → 0.5, and 0.30 (spike) → 1.0
  const daN  = Math.min(1, da  * 3.3);
  const neN  = Math.min(1, ne  * 3.3);
  const achN = Math.min(1, ach * 3.3);
  const shtN = Math.min(1, sht * 3.3);

  // --- DA (Dopamine): excitement, reward ---
  // Eye height: low DA=0.7x, high DA=1.2x
  const daHeight = 0.7 + daN * 0.5;
  // Pupil size: low DA=8px, high DA=15px
  const daPupil = 8 + daN * 7;
  // Mouth curve: low DA=slight frown (+3), high DA=smile (-10)
  const daMouth = 3 - daN * 13;
  // Blink: high DA = relaxed (less blinking handled in blink logic)

  // --- NE (Noradrenaline): alertness, stress ---
  // Animation speed: low NE=slow, high NE=snappy
  const neSpeed = 0.7 + neN * 0.8;
  // Pupil movement range increases with NE
  const neGazeRange = 10 + neN * 15;
  // Eye tilt: NE>0.7 → alert/angry inner tilt
  const neTilt = neN > 0.7 ? (neN - 0.7) * 40 : 0; // up to 12 degrees

  // --- ACh (Acetylcholine): focus, attention ---
  // Gaze stability: high ACh = locked forward
  const achGazeStability = achN; // 0=drifty, 1=locked
  // Pupil shrink when focused: ACh>0.7 → smaller pupils
  const achPupilMod = achN > 0.7 ? -4 * (achN - 0.7) / 0.3 : 0;
  // Mouth fades when focused
  const achMouthOpacity = achN > 0.5 ? 1.0 - (achN - 0.5) * 1.4 : 1.0;

  // --- 5HT (Serotonin): calm, contentment ---
  // Overall speed: high 5HT = slower, calmer
  const shtSpeed = 1.5 - shtN * 0.8; // 1.5x when low, 0.7x when high
  // Eye height slightly shorter when content (relaxed squint)
  const shtHeight = shtN > 0.7 ? 1.0 - (shtN - 0.7) * 0.3 : 1.0;
  // Corner radius: rounder when calm
  const shtRadius = 1.0 + shtN * 0.3;

  // --- Blend all influences ---
  target.eyeHeightMul = daHeight * shtHeight;
  target.eyeWidthMul = 1.0 + neN * 0.1; // slightly wider when alert
  target.eyeRadiusMul = shtRadius;
  target.eyeTiltL = neTilt;   // inner edge rises on left
  target.eyeTiltR = -neTilt;  // mirrored on right

  target.pupilRadius = Math.max(6, Math.min(16, daPupil + achPupilMod));
  target.mouthCurve = daMouth;
  target.mouthWidth = MOUTH_DEFAULT.width + daN * 20; // wider smile when happy
  target.mouthOpacity = Math.max(0.1, achMouthOpacity);
  target.animSpeed = (neSpeed + shtSpeed) / 2;

  // Gaze: if ACh is high, lock forward. Otherwise drift.
  if (achGazeStability > 0.7) {
    target.gazeTargetX = 0;
    target.gazeTargetY = 0;
  }
  // NE affects gaze range (stored for idle gaze logic)
  target._gazeRange = neGazeRange;
  target._gazeStability = achGazeStability;

  // --- Sleep override ---
  if (sleepMode) {
    target.eyeHeightMul = 0.08; // nearly closed
    target.pupilRadius = 6;
    target.mouthCurve = 0; // neutral
    target.mouthOpacity = 0.3;
    target.animSpeed = 0.5;
  }
}

// ─── Blink logic ───
let blinkTimer = 0;
let nextBlinkIn = 3000 + Math.random() * 3000;

function updateBlink(dtMs) {
  if (face.blinkPhase > 0) {
    face.blinkPhase -= dtMs / 150 * face.animSpeed; // 150ms total blink
    if (face.blinkPhase < 0) face.blinkPhase = 0;
  } else {
    blinkTimer += dtMs;
    // Blink interval: base 3-6s, modified by DA (relaxed=longer) and NE (alert=shorter)
    const mods = brainState?.modulators;
    const da = (mods?.DA || 0) * 10;
    const ne = (mods?.NE || 0) * 10;
    const ach = (mods?.ACh || 0) * 10;
    let interval = nextBlinkIn;
    // High ACh (focus): suppress blinks (8-10s)
    if (ach > 0.7) interval = 8000 + Math.random() * 2000;
    // High NE (stress): more blinks (1.5-2.5s)
    else if (ne > 0.7) interval = 1500 + Math.random() * 1000;

    if (blinkTimer > interval) {
      blinkTimer = 0;
      face.blinkPhase = 1.0;
      nextBlinkIn = 3000 + Math.random() * 3000;
    }
  }
}

// ─── Idle gaze drift ───
function updateGaze(dtMs) {
  face.gazeTimer -= dtMs;
  if (face.gazeTimer <= 0) {
    const range = target._gazeRange || 15;
    const stability = target._gazeStability || 0;
    // High stability = small random movement
    const factor = 1.0 - stability * 0.8;
    target.gazeTargetX = (Math.random() - 0.5) * 2 * range * factor;
    target.gazeTargetY = (Math.random() - 0.5) * 2 * (range * 0.7) * factor;
    face.gazeTimer = 1000 + Math.random() * 2000;
  }
}

// ─── Interpolation ───
function lerp(a, b, t) { return a + (b - a) * Math.min(1, t); }

function updateFace(dtMs) {
  const t = dtMs / 1000 * 3 * face.animSpeed; // ~3 per second base rate
  face.eyeWidthMul   = lerp(face.eyeWidthMul,   target.eyeWidthMul,   t);
  face.eyeHeightMul  = lerp(face.eyeHeightMul,  target.eyeHeightMul,  t);
  face.eyeRadiusMul  = lerp(face.eyeRadiusMul,  target.eyeRadiusMul,  t);
  face.eyeTiltL      = lerp(face.eyeTiltL,       target.eyeTiltL,      t);
  face.eyeTiltR      = lerp(face.eyeTiltR,       target.eyeTiltR,      t);
  face.pupilRadius   = lerp(face.pupilRadius,    target.pupilRadius,   t);
  face.pupilOffX     = lerp(face.pupilOffX,      target.gazeTargetX,   t * 0.5);
  face.pupilOffY     = lerp(face.pupilOffY,      target.gazeTargetY,   t * 0.5);
  face.mouthWidth    = lerp(face.mouthWidth,     target.mouthWidth,    t);
  face.mouthCurve    = lerp(face.mouthCurve,     target.mouthCurve,    t);
  face.mouthOpacity  = lerp(face.mouthOpacity,   target.mouthOpacity,  t);
  face.animSpeed     = lerp(face.animSpeed,      target.animSpeed,     t * 0.3);

  // Breathing: slow sine wave
  face.breathPhase += dtMs / 1000 * 0.5 * face.animSpeed;
}

// ─── Drawing ───
function drawRoundedRect(cx, cy, w, h, r, tiltDeg) {
  ctx.save();
  ctx.translate(cx * S, cy * S);
  if (tiltDeg) ctx.rotate(tiltDeg * Math.PI / 180);
  const hw = w * S / 2;
  const hh = h * S / 2;
  const cr = Math.min(r * S, hw, hh);

  ctx.beginPath();
  ctx.moveTo(-hw + cr, -hh);
  ctx.lineTo(hw - cr, -hh);
  ctx.quadraticCurveTo(hw, -hh, hw, -hh + cr);
  ctx.lineTo(hw, hh - cr);
  ctx.quadraticCurveTo(hw, hh, hw - cr, hh);
  ctx.lineTo(-hw + cr, hh);
  ctx.quadraticCurveTo(-hw, hh, -hw, hh - cr);
  ctx.lineTo(-hw, -hh + cr);
  ctx.quadraticCurveTo(-hw, -hh, -hw + cr, -hh);
  ctx.closePath();
  ctx.restore();
}

function drawEye(cx, cy, tiltDeg, isRight) {
  const w = EYE_DEFAULT.width * face.eyeWidthMul;
  let h = EYE_DEFAULT.height * face.eyeHeightMul;
  const r = EYE_DEFAULT.radius * face.eyeRadiusMul;

  // Blink: reduce height
  if (face.blinkPhase > 0) {
    const blinkAmount = Math.sin(face.blinkPhase * Math.PI);
    h *= (1 - blinkAmount * 0.95);
  }

  if (h < 2) {
    // Fully closed: draw thin line
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = 2 * S;
    ctx.beginPath();
    ctx.moveTo((cx - w / 2) * S, cy * S);
    ctx.lineTo((cx + w / 2) * S, cy * S);
    ctx.stroke();
    return;
  }

  // Eye white
  drawRoundedRect(cx, cy, w, h, r, tiltDeg);
  ctx.fillStyle = '#FFFFFF';
  ctx.fill();

  // Pupil
  const px = cx + face.pupilOffX;
  const py = cy + face.pupilOffY;
  // Clamp pupil within eye bounds
  const maxOffX = (w / 2 - face.pupilRadius - 4);
  const maxOffY = (h / 2 - face.pupilRadius - 4);
  const clampedX = cx + Math.max(-maxOffX, Math.min(maxOffX, face.pupilOffX));
  const clampedY = cy + Math.max(-maxOffY, Math.min(maxOffY, face.pupilOffY));

  ctx.beginPath();
  ctx.arc(clampedX * S, clampedY * S, face.pupilRadius * S, 0, Math.PI * 2);
  ctx.fillStyle = PUPIL_DEFAULT.color;
  ctx.fill();

  // Highlight (spec: adds life)
  const hlR = face.pupilRadius * 0.25;
  ctx.beginPath();
  ctx.arc((clampedX + face.pupilRadius * 0.3) * S, (clampedY - face.pupilRadius * 0.3) * S, hlR * S, 0, Math.PI * 2);
  ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
  ctx.fill();
}

function drawMouth(cx, cy) {
  if (face.mouthOpacity < 0.05) return;

  const halfW = face.mouthWidth / 2;
  const x1 = cx - halfW;
  const x2 = cx + halfW;
  const cpY = cy + face.mouthCurve; // negative curve = control point above = smile

  ctx.save();
  ctx.globalAlpha = face.mouthOpacity;
  ctx.strokeStyle = '#FFFFFF';
  ctx.lineWidth = MOUTH_DEFAULT.strokeWidth * S;
  ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(x1 * S, cy * S);
  ctx.quadraticCurveTo(cx * S, cpY * S, x2 * S, cy * S);
  ctx.stroke();
  ctx.restore();
}

// ─── Main loop ───
let lastTime = performance.now();

function animate(now) {
  const dtMs = Math.min(now - lastTime, 50); // cap at 50ms to avoid jumps
  lastTime = now;

  // Update targets from brain
  const mods = brainState?.modulators;
  if (mods) {
    updateTargetsFromModulators(mods);
  } else {
    // No connection: gentle idle
    target.eyeHeightMul = 1.0;
    target.eyeWidthMul = 1.0;
    target.pupilRadius = 12;
    target.mouthCurve = -3;
    target.mouthOpacity = 0.8;
    target.animSpeed = 0.8;
  }

  // Update animations
  updateBlink(dtMs);
  updateGaze(dtMs);
  updateFace(dtMs);

  // Breathing offset
  const breathY = Math.sin(face.breathPhase * Math.PI * 2) * 3;

  // ─── Draw ───
  ctx.clearRect(0, 0, W, H);

  // Background: pure black (AMOLED)
  ctx.fillStyle = '#000000';
  ctx.fillRect(0, 0, W, H);

  // Circular viewport mask (for round display)
  ctx.save();
  ctx.beginPath();
  ctx.arc(W / 2, H / 2, Math.min(W, H) / 2, 0, Math.PI * 2);
  ctx.clip();

  // Subtle glow when active
  if (mods) {
    const da = (mods.DA || 0) * 10;
    if (da > 0.2) {
      const grad = ctx.createRadialGradient(W / 2, H / 2, 0, W / 2, H / 2, W * 0.4);
      grad.addColorStop(0, `rgba(255, 255, 255, ${Math.min(0.03, da * 0.02)})`);
      grad.addColorStop(1, 'rgba(0, 0, 0, 0)');
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, W, H);
    }
  }

  // Eyes
  const eyeY = LAYOUT.leftEye.cy + breathY;
  drawEye(LAYOUT.leftEye.cx, eyeY, face.eyeTiltL, false);
  drawEye(LAYOUT.rightEye.cx, eyeY, face.eyeTiltR, true);

  // Mouth
  drawMouth(LAYOUT.mouth.cx, LAYOUT.mouth.cy + breathY * 0.5);

  // Blush when happy (DA high)
  if (mods) {
    const da = (mods.DA || 0) * 10;
    if (da > 0.5) {
      const blushAlpha = Math.min(0.15, (da - 0.5) * 0.3);
      ctx.fillStyle = `rgba(255, 107, 157, ${blushAlpha})`; // #FF6B9D
      ctx.beginPath();
      ctx.ellipse(130 * S, 260 * S, 15 * S, 8 * S, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.beginPath();
      ctx.ellipse(336 * S, 260 * S, 15 * S, 8 * S, 0, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // Sleep Zzz
  if (sleepMode && face.eyeHeightMul < 0.15) {
    const zzzPhase = (now / 1000) % 3;
    ctx.fillStyle = `rgba(255, 255, 255, ${0.3 + Math.sin(zzzPhase * Math.PI) * 0.2})`;
    ctx.font = `${14 * S}px monospace`;
    ctx.fillText('z', (300 + zzzPhase * 10) * S, (160 - zzzPhase * 20) * S);
    ctx.font = `${18 * S}px monospace`;
    ctx.fillText('z', (320 + zzzPhase * 8) * S, (140 - zzzPhase * 25) * S);
    ctx.font = `${22 * S}px monospace`;
    ctx.fillText('Z', (340 + zzzPhase * 6) * S, (120 - zzzPhase * 30) * S);
  }

  ctx.restore();

  // Connection indicator (outside clip)
  ctx.fillStyle = brainState ? '#34d399' : '#ef4444';
  ctx.beginPath();
  ctx.arc(W - 8, 8, 3, 0, Math.PI * 2);
  ctx.fill();

  // Voice state indicator
  if (voiceState === 'listening') {
    ctx.strokeStyle = '#00C8C8'; // ADYZEN cyan
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(W / 2, H / 2, Math.min(W, H) / 2 - 5, 0, Math.PI * 2);
    ctx.stroke();
  }

  requestAnimationFrame(animate);
}

requestAnimationFrame(animate);

// ─── Click-through + Alt-drag to reposition ───
// Default: click-through (mouse events pass through to desktop).
// Hold Alt/Option: click-through off → drag to reposition.
// Release Alt: click-through back on.
(function() {
  if (!window.__TAURI_INTERNALS__) return;
  const invoke = window.__TAURI_INTERNALS__.invoke;

  // Alt down → make window interactive
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Alt') invoke('set_ignore_cursor', { ignore: false });
  });

  // Alt up → back to click-through
  window.addEventListener('keyup', (e) => {
    if (e.key === 'Alt') invoke('set_ignore_cursor', { ignore: true });
  });

  // Mouse down while Alt held → start window drag
  document.addEventListener('mousedown', (e) => {
    if (e.altKey) invoke('start_drag');
  });

  // Lost focus → ensure click-through is re-enabled
  window.addEventListener('blur', () => invoke('set_ignore_cursor', { ignore: true }));
})();
