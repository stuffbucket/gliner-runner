// Checks that need no model: accessibility (axe), load time and size, phone-width layout, and links that fail.
// The same visit saves each key page's text for the comprehension stage.
import { writeFileSync } from "node:fs";
import { join } from "node:path";

import { open, readPage, shortUrl } from "./page.mjs";

const AXE_TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"];
const PHONE = { width: 390, height: 844 };

/** A file name for a page, for example `how-it-works` for `/how-it-works/` and `home` for `/`. */
export const pageSlug = (href, siteRoot) => shortUrl(href, siteRoot).replace(/^\/|\/$/g, "").replace(/[^a-z0-9]+/gi, "-") || "home";

async function measure(page) {
  return page.evaluate(() => {
    const [nav] = performance.getEntriesByType("navigation");
    const resources = performance.getEntriesByType("resource");
    const paint = performance.getEntriesByName("first-contentful-paint")[0];
    return {
      loadMs: Math.round(nav?.loadEventEnd ?? 0),
      firstContentfulPaintMs: Math.round(paint?.startTime ?? 0),
      requests: resources.length + 1,
      transferKb: Math.round((resources.reduce((sum, r) => sum + (r.transferSize || 0), 0) + (nav?.transferSize ?? 0)) / 1024),
    };
  });
}

/** Elements wider than the phone screen, which make the page scroll sideways. */
async function overflow(page) {
  return page.evaluate(() => {
    const width = document.documentElement.clientWidth;
    if (document.documentElement.scrollWidth <= width + 1) return [];
    return [...document.querySelectorAll("body *")]
      .filter((el) => {
        const box = el.getBoundingClientRect();
        if (box.right <= width + 1 || box.width === 0) return false;
        // Inside a scrolling box (a code block or a table wrapper) the overflow is intended.
        for (let p = el.parentElement; p; p = p.parentElement) if (/(auto|scroll)/.test(getComputedStyle(p).overflowX)) return false;
        return true;
      })
      .slice(0, 5)
      .map((el) => `${el.tagName.toLowerCase()}${el.className && typeof el.className === "string" ? `.${el.className.trim().split(/\s+/).join(".")}` : ""} (${Math.round(el.getBoundingClientRect().right)}px)`);
  });
}

export async function checkPages({ browser, AxeBuilder, siteRoot, urls, pagesDir, shotsDir, log }) {
  const desktop = await browser.newContext({ viewport: { width: 1280, height: 800 } });
  const phone = await browser.newContext({ viewport: PHONE, isMobile: true, hasTouch: true, deviceScaleFactor: 2 });
  const pages = [];
  const links = new Map();
  try {
    for (const url of urls) {
      const page = await desktop.newPage();
      const status = await open(page, url);
      const seen = await readPage(page, siteRoot);
      const slug = pageSlug(url, siteRoot);
      writeFileSync(join(pagesDir, `${slug}.txt`), `${seen.title}\n\n${seen.text}`);
      const axe = await new AxeBuilder({ page }).withTags(AXE_TAGS).analyze();
      const timing = await measure(page);
      for (const link of seen.links) if (!links.has(link.href)) links.set(link.href, { ...link, from: shortUrl(url, siteRoot) });
      await page.close();

      const small = await phone.newPage();
      await open(small, url);
      const wide = await overflow(small);
      await small.screenshot({ type: "jpeg", quality: 60, path: join(shotsDir, `phone-${slug}.jpg`) });
      await small.close();

      pages.push({
        url, path: shortUrl(url, siteRoot), slug, status, title: seen.title, timing, phoneOverflow: wide,
        axe: axe.violations.map((v) => ({
          id: v.id, impact: v.impact, help: v.help, helpUrl: v.helpUrl,
          nodes: v.nodes.length, targets: v.nodes.slice(0, 3).map((n) => n.target.join(" ")),
          summary: v.nodes[0]?.failureSummary?.split("\n").slice(0, 3).join(" ") ?? "",
        })),
      });
      log(`checked ${shortUrl(url, siteRoot)}: ${axe.violations.length} axe rules, ${timing.loadMs} ms, ${wide.length ? "overflows on a phone" : "fits a phone"}`);
    }
  } finally {
    await desktop.close();
    await phone.close();
  }
  return { pages, links: [...links.values()] };
}

/** Requests every internal link once and returns the ones that fail. External links are checked only if asked. */
export async function checkLinks(links, { external = false } = {}) {
  const broken = [];
  for (const link of links.filter((l) => l.internal || external)) {
    const status = await fetch(link.href, { method: link.internal ? "GET" : "HEAD", redirect: "follow", signal: AbortSignal.timeout(15_000) })
      .then((r) => r.status, (e) => e.message);
    if (status !== 200) broken.push({ ...link, status });
  }
  return broken;
}
