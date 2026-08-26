#!/usr/bin/env node
// Rewrites local img/ references in post bodies to the deployed site's absolute
// image URLs. Front matter is intentionally left untouched because the theme
// expects header-img to be a site-relative path.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = process.cwd();
const POSTS_DIR = path.join(ROOT, '_posts');
const CONFIG_FILE = path.join(ROOT, '_config.yml');
const DRY_RUN = process.argv.includes('--dry-run');

function siteUrl() {
  if (process.env.SITE_URL?.trim()) {
    return process.env.SITE_URL.trim().replace(/\/$/, '');
  }

  const config = fs.readFileSync(CONFIG_FILE, 'utf8');
  const match = config.match(/^url:\s*["']?([^\s"'#]+)["']?\s*(?:#.*)?$/m);
  if (!match) throw new Error('SITE_URL is unset and _config.yml has no url setting');
  return match[1].replace(/\/$/, '');
}

function splitFrontMatter(content) {
  const match = content.match(/^---\r?\n[\s\S]*?\r?\n---\r?\n/);
  if (!match) return ['', content];
  return [match[0], content.slice(match[0].length)];
}

export function rewritePost(content, imageBaseUrl) {
  const [frontMatter, body] = splitFrontMatter(content);
  const absoluteImgRoot = `${imageBaseUrl}/img/`;
  let inFence = false;

  const rewrittenBody = body
    .split(/(?<=\n)/)
    .map((line) => {
      if (/^\s*(```|~~~)/.test(line)) {
        inFence = !inFence;
        return line;
      }
      if (inFence) return line;

      return line
        // Markdown images: ![alt](/img/a.png), ![alt](img/a.png), etc.
        .replace(
          /(!\[[^\]]*\]\(\s*<?)(?:(?:\.\.?\/)|\/)*img\//g,
          `$1${absoluteImgRoot}`,
        )
        // HTML images with quoted src attributes.
        .replace(
          /(<img\b[^>]*?\bsrc\s*=\s*["'])(?:(?:\.\.?\/)|\/)*img\//gi,
          `$1${absoluteImgRoot}`,
        );
    })
    .join('');

  return frontMatter + rewrittenBody;
}

function main() {
  const baseUrl = siteUrl();
  const posts = fs
    .readdirSync(POSTS_DIR)
    .filter((name) => name.endsWith('.md'))
    .sort();

  let changedFiles = 0;
  for (const name of posts) {
    const file = path.join(POSTS_DIR, name);
    const content = fs.readFileSync(file, 'utf8');
    const rewritten = rewritePost(content, baseUrl);
    if (rewritten === content) continue;

    changedFiles += 1;
    if (!DRY_RUN) fs.writeFileSync(file, rewritten);
    console.log(`  ${DRY_RUN ? '• would rewrite' : '✓ rewrote'} ${name}`);
  }

  console.log(
    `${DRY_RUN ? 'Would rewrite' : 'Rewrote'} ${changedFiles} post(s) with image root ${baseUrl}/img/`,
  );
}

if (process.argv[1] && fileURLToPath(import.meta.url) === path.resolve(process.argv[1])) {
  main();
}
