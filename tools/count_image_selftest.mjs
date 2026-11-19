// Regression check for tools/count_image.html without a browser or lab data.
//   node tools/count_image_selftest.mjs
//
// The detector is pure (typed arrays in, particles out), so this builds a
// synthetic polydisperse bed in memory and checks the count, the two radius
// classes and the centre accuracy.  The TIFF reader is not exercised here; it
// was verified byte-for-byte against PIL on a real Basler frame and on
// hand-written 8/16-bit, little/big-endian, 1-row and 64-row strips.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const html = fs.readFileSync(path.join(here, 'count_image.html'), 'utf8');
const a = html.indexOf('// ===== PURE LOGIC');
const b = html.indexOf('// ===== END PURE LOGIC');
if (a < 0 || b < 0) {
  console.error('FAIL: PURE LOGIC markers not found in count_image.html');
  process.exit(1);
}
const api = new Function(html.slice(html.indexOf('\n', a) + 1, b) +
  '\nreturn {detectParticles};')();

const W = 1200, H = 900;
const discs = [];
let seed = 12345;
const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
while (discs.length < 60) {
  const r = discs.length % 3 === 0 ? 26 : 18;
  const x = r + 10 + rnd() * (W - 2 * r - 20);
  const y = r + 10 + rnd() * (H - 2 * r - 20);
  if (discs.some((d) => Math.hypot(d.x - x, d.y - y) < d.r + r + 4)) continue;
  discs.push({ x, y, r });
}
const g = new Float32Array(W * H).fill(0.16);
for (const d of discs) {
  const x0 = Math.max(0, Math.floor(d.x - d.r)), x1 = Math.min(W, Math.ceil(d.x + d.r));
  const y0 = Math.max(0, Math.floor(d.y - d.r)), y1 = Math.min(H, Math.ceil(d.y + d.r));
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const dx = x - d.x, dy = y - d.y;
      if (dx * dx + dy * dy <= d.r * d.r) g[y * W + x] = 0.74;
    }
  }
}
for (let i = 0; i < g.length; i++) g[i] += (rnd() - 0.5) * 0.02;

const res = api.detectParticles(g, W, H, { rMin: 10, rMax: 40, sensitivity: 0.6, maxSide: 1600 });
let fails = 0;
function check(name, ok, detail) {
  console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail ? '  ' + detail : ''));
  if (!ok) fails++;
}

check('count', Math.abs(res.particles.length - discs.length) <= 2,
      `${res.particles.length} vs ${discs.length}`);
let worst = 0;
for (const d of discs) {
  const near = Math.min(...res.particles.map((p) => Math.hypot(p.x - d.x, p.y - d.y)));
  if (near > worst) worst = near;
}
check('every disc found', worst < 3, `worst centre distance ${worst.toFixed(2)} px`);
const small = res.particles.filter((p) => p.r < 22).length;
check('two radius classes', Math.abs(small - 40) <= 2, `${small} small, ${res.particles.length - small} large`);

process.exit(fails ? 1 : 0);
