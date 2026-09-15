// Local link integrity only; external availability and factual review are separate.
import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const files = execFileSync('git', ['ls-files', '--cached', '--others', '--exclude-standard', '-z'], { cwd: root })
  .toString().split('\0').filter(file => file.endsWith('.md'));
const failures = [];
let links = 0;
for (const file of new Set(files)) {
  const source = readFileSync(resolve(root, file), 'utf8').replace(/```[\s\S]*?```/g, '');
  for (const match of source.matchAll(/!?\[[^\]\n]*\]\(([^)\n]+)\)/g)) {
    const target = match[1].trim().replace(/^<|>$/g, '');
    if (/^(https?:|mailto:|#)/.test(target)) continue;
    const path = decodeURIComponent(target.split('#')[0]);
    if (!path) continue;
    links++;
    if (path.startsWith('/') || !existsSync(resolve(root, dirname(file), path))) {
      failures.push(`${file}: ${target}`);
    }
  }
}
if (failures.length) {
  console.error(failures.join('\n'));
  process.exitCode = 1;
} else console.log(`PASS: ${new Set(files).size} Markdown files; ${links} local links resolve.`);
