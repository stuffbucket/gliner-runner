// The deterministic checks every candidate must pass before any model reads it: length, sentence length,
// reading grade, stall words and meta asides. Each slot sets its own limits in slots.json.

/** Words as a reader counts them: runs of letters or digits, with inner apostrophes and hyphens kept. */
export function words(text) {
  return text.match(/[\p{L}\p{N}]+(?:['’-][\p{L}\p{N}]+)*/gu) ?? [];
}

/** Sentences, split after ., ! or ? (and runs of them). A trailing fragment with no full stop is a sentence too. */
export function sentences(text) {
  return text
    .replace(/\s+/g, " ")
    .split(/(?<=[.!?…]["”’)]?)\s+/u)
    .map((s) => s.trim())
    .filter((s) => words(s).length > 0);
}

/**
 * Syllables in one English word, by the usual heuristic: drop endings that add no syllable (a silent final "e",
 * "-es" and "-ed" after most consonants; "-le" is kept, since it adds one), drop a leading "y", then count runs of
 * one or two vowels. Right for most short everyday words, which is what a grade-5 limit needs. A number counts
 * as two syllables.
 */
export function syllables(word) {
  const w = word.toLowerCase().replace(/[^a-z]/g, "");
  if (w.length === 0) return /\d/.test(word) ? 2 : 0;
  if (w.length <= 3) return 1;
  const core = w.replace(/(?:[^laeiouy]es|[^laeiouytd]ed|[^laeiouy]e)$/, "").replace(/^y/, "");
  return Math.max(1, core.match(/[aeiouy]{1,2}/g)?.length ?? 0);
}

/** Flesch-Kincaid grade level: 0.39 (words / sentences) + 11.8 (syllables / words) - 15.59. */
export function grade(text) {
  const ws = words(text);
  const n = Math.max(1, sentences(text).length);
  if (ws.length === 0) return 0;
  const syl = ws.reduce((sum, w) => sum + syllables(w), 0);
  return 0.39 * (ws.length / n) + 11.8 * (syl / ws.length) - 15.59;
}

const escape = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/** Every check for one text against one slot's limits. `reasons` is empty when the text passes. */
export function check(text, limits) {
  const ws = words(text);
  const ss = sentences(text);
  const longest = Math.max(0, ...ss.map((s) => words(s).length));
  const g = grade(text);
  const reasons = [];
  if (limits.maxWords && ws.length > limits.maxWords) reasons.push(`${ws.length} words (limit ${limits.maxWords})`);
  if (limits.maxSentences && ss.length > limits.maxSentences) reasons.push(`${ss.length} sentences (limit ${limits.maxSentences})`);
  if (limits.maxSentenceWords && longest > limits.maxSentenceWords)
    reasons.push(`a sentence of ${longest} words (limit ${limits.maxSentenceWords})`);
  if (limits.maxGrade != null && g > limits.maxGrade) reasons.push(`grade ${g.toFixed(1)} (limit ${limits.maxGrade})`);
  // Stall words match whole words, with or without a plural "s": "label" also catches "labels".
  const stalls = (limits.stallWords ?? []).filter((w) => new RegExp(`(?<![\\p{L}\\p{N}])${escape(w)}s?(?![\\p{L}\\p{N}])`, "iu").test(text));
  if (stalls.length > 0) reasons.push(`stall word: ${stalls.join(", ")}`);
  const meta = (limits.meta ?? []).filter((m) => new RegExp(`(?<![\\p{L}])${escape(m)}(?![\\p{L}])`, "iu").test(text));
  if (meta.length > 0) reasons.push(`meta: ${meta.join(", ")}`);
  return { words: ws.length, sentences: ss.length, longestSentence: longest, grade: Number(g.toFixed(2)), reasons };
}
