// Generates hard/data/feedback600.json: 600 short product feedback texts, 300 positive and 300 negative.
// Deterministic (no randomness): items x phrases.
import { writeFileSync } from "node:fs";
const items = ["blender", "headset", "backpack", "kettle", "lamp", "office chair", "phone case", "rice cooker", "keyboard", "water bottle",
  "vacuum", "tent", "jacket", "speaker", "coffee grinder", "toaster", "monitor", "mattress", "drill", "printer", "umbrella",
  "router", "frying pan", "desk", "camera", "bicycle helmet", "hair dryer", "suitcase", "mouse", "fan"];
const pos = [
  "I love this {i}, it works perfectly.",
  "The {i} is excellent and well made.",
  "Really happy with my new {i}.",
  "This {i} exceeded my expectations.",
  "Fantastic {i}, I would buy it again.",
  "The {i} is great value and does exactly what I need.",
  "Five stars, the {i} is wonderful.",
  "My {i} arrived quickly and works beautifully.",
  "Best {i} I have ever owned.",
  "The {i} is sturdy, quiet and easy to use.",
];
const neg = [
  "The {i} broke after two days.",
  "Terrible {i}, a complete waste of money.",
  "I am very disappointed with this {i}.",
  "The {i} stopped working within a week.",
  "Do not buy this {i}, it is awful.",
  "The {i} feels cheap and flimsy.",
  "One star, the {i} is useless.",
  "My {i} arrived broken and support never answered.",
  "Worst {i} I have ever bought.",
  "The {i} is loud, slow and hard to use.",
];
// Neutral context so each text is a few sentences long, as real comments are.
const context = [
  "I ordered it in the spring after reading a few reviews online, and I have used it most days since then.",
  "It was a birthday present for my partner, who had been asking for one for a long time.",
  "We bought it for our new apartment when we moved to the city last year.",
  "I picked it up in the store on my way home from work because my old one had stopped working.",
  "My sister recommended the brand, so I chose the same model she has at her house.",
];
const out = [];
for (const [n, i] of items.entries()) {
  for (let k = 0; k < 10; k++) {
    const c = context[(n + k) % context.length];
    out.push({ text: `${c} ${pos[k].replace("{i}", i)}`, answer: "positive" });
    out.push({ text: `${c} ${neg[k].replace("{i}", i)}`, answer: "negative" });
  }
}
const rows = out.map((r, k) => ({ id: `F${String(k + 1).padStart(3, "0")}`, ...r }));
writeFileSync(new URL("./data/feedback600.json", import.meta.url), JSON.stringify(rows, null, 1) + "\n");
console.log(rows.length);
