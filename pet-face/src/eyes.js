// Placeholder — Task 5 will replace with full eye animation
const canvas = document.getElementById('eyes');
const ctx = canvas.getContext('2d');

function resize() {
  canvas.width = window.innerWidth * devicePixelRatio;
  canvas.height = window.innerHeight * devicePixelRatio;
  ctx.scale(devicePixelRatio, devicePixelRatio);
}
resize();
window.addEventListener('resize', resize);

function draw() {
  const w = window.innerWidth;
  const h = window.innerHeight;

  ctx.clearRect(0, 0, w, h);

  // Two simple eyes
  ctx.fillStyle = '#34d399';
  ctx.beginPath();
  ctx.arc(w * 0.35, h * 0.45, 15, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.arc(w * 0.65, h * 0.45, 15, 0, Math.PI * 2);
  ctx.fill();

  requestAnimationFrame(draw);
}
draw();
