#!/usr/bin/env node
// Generates YAML front matter for Jekyll posts that lack it.
//
// Deterministic fields (no LLM needed):
//   - title      : first "# heading" in the body, else filename slug
//   - date       : from the "YYYY-MM-DD-" filename prefix
//   - header-img : first image in the body (local or https), else a stable
//                  random pick from IMAGE_POOL
//
// LLM fields (DeepSeek API, cached in .llm-cache/cache.json):
//   - tags, subtitle, tldr
//
// Usage:
//   node .github/scripts/generate-frontmatter.mjs          # inject front matter
//   node .github/scripts/generate-frontmatter.mjs --strip  # remove generated front matter

import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';

const ROOT = process.cwd();
const POSTS_DIR = path.join(ROOT, '_posts');
const CACHE_DIR = path.join(ROOT, '.llm-cache');
const CACHE_FILE = path.join(CACHE_DIR, 'cache.json');

// Image files under img/ that should NOT be used as auto cover images
// (site chrome / avatars). Sub-directories like in-posts/ and posts-img/ are
// also excluded because those images belong to specific posts.
const EXCLUDE_COVERS = new Set([
  'favicon.ico',
  'avatar.jpg',
  'avatar-about.jpg',
  '404-bg.jpg',
  'img-archive.jpg',
]);

const COVER_EXT = ['.jpg', '.jpeg', '.png', '.webp'];

const DEEPSEEK_URL = 'https://api.deepseek.com/chat/completions';
const MODEL = 'deepseek-v4-flash';
const MAX_INPUT_CHARS = 6000;
const GENERATED_MARKER = '# @generated';

function sha256(s) {
  return crypto.createHash('sha256').update(s).digest('hex');
}

function loadApiKey() {
  if (process.env.DEEPSEEK_API_KEY && process.env.DEEPSEEK_API_KEY.trim()) {
    return process.env.DEEPSEEK_API_KEY.trim();
  }
  try {
    const env = fs.readFileSync(path.join(ROOT, '.env'), 'utf8');
    for (const line of env.split(/\r?\n/)) {
      const m = line.match(/^\s*DEEPSEEK_API_KEY\s*=\s*(.+?)\s*$/);
      if (m) return m[1].replace(/^["']|["']$/g, '');
    }
  } catch {}
  return null;
}

function yaml(s) {
  return JSON.stringify(String(s));
}

function listPosts() {
  if (!fs.existsSync(POSTS_DIR)) return [];
  return fs.readdirSync(POSTS_DIR).filter((f) => f.endsWith('.md')).sort();
}

function slugToTitle(slug) {
  return slug
    .replace(/[-_]+/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase())
    .trim();
}

function extractTitle(content) {
  const m = content.match(/^#\s+(.+?)\s*$/m);
  return m ? m[1].trim() : null;
}

function extractDate(filename) {
  const m = filename.match(/^(\d{4})-(\d{2})-(\d{2})-/);
  return m ? `${m[1]}-${m[2]}-${m[3]} 12:00:00` : null;
}

function extractFirstImage(content) {
  const m = content.match(/!\[[^\]]*\]\(\s*([^)\s]+)/);
  if (!m) return null;
  const url = m[1].trim();
  if (!url || url.startsWith('#')) return null;
  return url;
}

function listCoverImages() {
  let pool = [];
  try {
    pool = fs
      .readdirSync(path.join(ROOT, 'img'))
      .filter((f) => COVER_EXT.includes(path.extname(f).toLowerCase()) && !EXCLUDE_COVERS.has(f))
      .sort();
  } catch {}
  return pool;
}

function pickImage(filename) {
  const pool = listCoverImages();
  if (pool.length === 0) return 'img/home-bg.jpg';
  const h = crypto.createHash('sha256').update(filename).digest();
  const idx = h.readUInt32BE(0) % pool.length;
  return `img/${pool[idx]}`;
}

function readCache() {
  try {
    return JSON.parse(fs.readFileSync(CACHE_FILE, 'utf8'));
  } catch {
    return {};
  }
}

function writeCache(cache) {
  fs.mkdirSync(CACHE_DIR, { recursive: true });
  fs.writeFileSync(CACHE_FILE, JSON.stringify(cache, null, 2));
}

async function callDeepSeek(content, apiKey) {
  const res = await fetch(DEEPSEEK_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: MODEL,
      messages: [
        {
          role: 'system',
          content:
            '你是一名中文技术博客的编辑助手。请阅读用户提供的文章正文,输出一个 JSON 对象,包含以下字段:\n' +
            '- tags: 3~5 个技术主题标签(字符串数组)\n' +
            '- subtitle: 一句话副标题,概括文章核心内容\n' +
            '- tldr: 一段中文总结(不超过 300 字),概括文章的背景、方法、要点和结论,可以写得详细一些,不必局限于几句话\n' +
            '只输出 JSON,不要输出任何其他内容。',
        },
        { role: 'user', content: content.slice(0, MAX_INPUT_CHARS) },
      ],
      response_format: { type: 'json_object' },
      temperature: 0.3,
      max_tokens: 1200,
    }),
  });
  if (!res.ok) {
    throw new Error(`DeepSeek HTTP ${res.status}: ${(await res.text()).slice(0, 200)}`);
  }
  const data = await res.json();
  const text = data.choices?.[0]?.message?.content;
  if (!text) throw new Error('DeepSeek returned empty content');
  return JSON.parse(text);
}

function buildFrontMatter({ title, date, headerImg, tags, subtitle, tldr }) {
  const lines = ['---', GENERATED_MARKER];
  lines.push(`title: ${yaml(title)}`);
  if (date) lines.push(`date: ${date}`);
  if (headerImg) lines.push(`header-img: ${yaml(headerImg)}`);
  if (subtitle) lines.push(`subtitle: ${yaml(subtitle)}`);
  if (tldr) lines.push(`tldr: ${yaml(tldr)}`);
  if (tags && tags.length) {
    lines.push('tags:');
    for (const t of tags) lines.push(`    - ${yaml(t)}`);
  }
  lines.push('---', '');
  return lines.join('\n');
}

function stripFrontMatter(content) {
  const m = content.match(/^---\r?\n([\s\S]*?\n)---\r?\n/);
  if (!m) return content;
  if (!m[1].includes(GENERATED_MARKER)) return content;
  return content.slice(m[0].length);
}

async function generate() {
  const apiKey = loadApiKey();
  const cache = readCache();
  let cacheDirty = false;

  for (const f of listPosts()) {
    const file = path.join(POSTS_DIR, f);
    let content = fs.readFileSync(file, 'utf8');
    if (content.startsWith('---')) continue; // already has front matter

    const slug = f.replace(/^\d{4}-\d{2}-\d{2}-/, '').replace(/\.md$/, '');
    const title = extractTitle(content) || slugToTitle(slug);
    const date = extractDate(f);
    const headerImg = extractFirstImage(content) || pickImage(f);

    let tags = [];
    let subtitle = '';
    let tldr = '';
    const hash = sha256(content);
    const cached = cache[f];
    if (cached && cached.hash === hash) {
      tags = cached.tags || [];
      subtitle = cached.subtitle || '';
      tldr = cached.tldr || '';
    } else if (apiKey) {
      try {
        const r = await callDeepSeek(content, apiKey);
        tags = Array.isArray(r.tags) ? r.tags.filter((t) => typeof t === 'string').slice(0, 5) : [];
        subtitle = typeof r.subtitle === 'string' ? r.subtitle : '';
        tldr = typeof r.tldr === 'string' ? r.tldr : '';
        cache[f] = { hash, tags, subtitle, tldr };
        cacheDirty = true;
        console.log(`  ✨ LLM fields generated for ${f}`);
      } catch (e) {
        console.error(`  ⚠️ DeepSeek failed for ${f}: ${e.message}`);
      }
    } else {
      console.warn(`  ⚠️ no DEEPSEEK_API_KEY; skipping LLM fields for ${f}`);
    }

    const fm = buildFrontMatter({ title, date, headerImg, tags, subtitle, tldr });
    fs.writeFileSync(file, fm + content);
    console.log(`  ✓ ${f} (title="${title}")`);
  }

  if (cacheDirty) writeCache(cache);
}

function strip() {
  for (const f of listPosts()) {
    const file = path.join(POSTS_DIR, f);
    const content = fs.readFileSync(file, 'utf8');
    const stripped = stripFrontMatter(content);
    if (stripped !== content) {
      fs.writeFileSync(file, stripped);
      console.log(`  ↺ stripped ${f}`);
    }
  }
}

if (process.argv[2] === '--strip') {
  strip();
} else {
  generate().catch((e) => {
    console.error(e);
    process.exit(1);
  });
}
