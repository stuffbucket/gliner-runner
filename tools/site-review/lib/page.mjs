// What a reader sees on one page: the text, headings and links, read from the live DOM.

/** The visible text, headings and links of the page. Links are absolute; `internal` means under the site root. */
export async function readPage(page, siteRoot) {
  const seen = await page.evaluate(() => {
    const visible = (el) => el.getClientRects().length > 0 && getComputedStyle(el).visibility !== "hidden";
    const main = document.querySelector("main") ?? document.body;
    return {
      title: document.title,
      text: main.innerText,
      allText: document.body.innerText,
      headings: [...document.querySelectorAll("h1, h2, h3")].filter(visible).map((h) => `${h.tagName.toLowerCase()} ${h.innerText.trim()}`),
      links: [...document.querySelectorAll("a[href]")]
        .filter(visible)
        .map((a) => ({ text: (a.innerText || a.getAttribute("aria-label") || a.title || "").trim().replace(/\s+/g, " "), href: a.href })),
    };
  });
  const here = page.url().split("#")[0];
  const unique = new Map();
  for (const link of seen.links) {
    const href = link.href.split("#")[0] === here ? null : link.href;
    if (!href || !/^https?:/.test(href) || unique.has(href)) continue;
    unique.set(href, { text: link.text.slice(0, 80) || "(no text)", href, internal: href.startsWith(siteRoot) });
  }
  return { url: page.url(), ...seen, links: [...unique.values()] };
}

/** Lower case, plain quotes, single spaces: so a quote the model copied can be found in the page text. */
export const normalize = (text) =>
  text.toLowerCase().replace(/[‘’]/g, "'").replace(/[“”]/g, '"').replace(/[…]/g, "...").replace(/\s+/g, " ").trim();

/** Whether `quote` is on the page. Models trim or re-punctuate quotes, so a long enough prefix also counts. */
export function quoteOnPage(quote, pageText) {
  const q = normalize(quote ?? "").replace(/^["'.\s]+|["'.\s]+$/g, "");
  if (q.length === 0) return false;
  const text = normalize(pageText);
  return text.includes(q) || (q.length > 40 && text.includes(q.slice(0, 40)));
}

/** A path under the site root for prompts and the report, for example `start/` for `<root>start/`. */
export const shortUrl = (href, siteRoot) => (href.startsWith(siteRoot) ? `/${href.slice(siteRoot.length)}` : href);

/** Opens `url` and waits for client-rendered parts (the playground is React, rendered in the browser only). */
export async function open(page, url) {
  const response = await page.goto(url, { waitUntil: "load", timeout: 60_000 });
  await page.waitForLoadState("networkidle", { timeout: 10_000 }).catch(() => {});
  if (url.includes("/playground")) await page.waitForSelector(".pg-app", { timeout: 15_000 }).catch(() => {});
  return response?.status() ?? 0;
}
