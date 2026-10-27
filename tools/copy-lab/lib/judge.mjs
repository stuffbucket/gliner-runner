// Pairwise judging: every pair of entrants, in both orders (to cancel position bias), by each judge model. Each
// judgement answers two questions (understood faster and remembered; sounds like a person); each is worth half a
// win. The ranking is a Bradley-Terry fit over all games, with the plain win rate alongside.
import { pool } from "./model.mjs";
import { judgeMessages } from "./prompts.mjs";

/**
 * Bradley-Terry strengths by the MM algorithm (Hunter 2004). `games` are { a, b, score } with score the share of
 * the game `a` won (0, 0.5 or 1). One virtual half-win and half-loss against an average player keeps a player who
 * won or lost everything finite.
 */
export function bradleyTerry(ids, games, iterations = 500) {
  let p = Object.fromEntries(ids.map((id) => [id, 1]));
  for (let it = 0; it < iterations; it += 1) {
    const next = {};
    for (const i of ids) {
      let wins = 0.5;
      let denom = 1 / (p[i] + 1);
      for (const g of games) {
        if (g.a !== i && g.b !== i) continue;
        const other = g.a === i ? g.b : g.a;
        wins += g.a === i ? g.score : 1 - g.score;
        denom += 1 / (p[i] + p[other]);
      }
      next[i] = wins / denom;
    }
    // Normalise so the geometric mean is 1.
    const scale = Math.exp(ids.reduce((s, id) => s + Math.log(next[id]), 0) / ids.length);
    for (const id of ids) next[id] /= scale;
    p = next;
  }
  return p;
}

export async function runJudging({ slot, entrants, client, concurrency, log }) {
  const jobs = [];
  for (let i = 0; i < entrants.length; i += 1) {
    for (let j = 0; j < entrants.length; j += 1) {
      if (i === j) continue;
      for (const judge of slot.judges) jobs.push({ judge, a: entrants[i], b: entrants[j] });
    }
  }
  log(`judge: ${entrants.length} entrants, ${jobs.length} judgements by ${slot.judges.join(", ")}`);
  const games = (
    await pool(jobs, concurrency, async ({ judge, a, b }) => {
      try {
        const reply = await client.json({ stage: "judge", model: judge, messages: judgeMessages(a.text, b.text), temperature: 0, maxTokens: 3000 });
        const pick = (v) => (/^\s*A/i.test(String(v)) ? 1 : /^\s*B/i.test(String(v)) ? 0 : 0.5);
        const faster = pick(reply.faster);
        const person = pick(reply.person);
        return { judge, a: a.id, b: b.id, faster, person, score: (faster + person) / 2, why: reply.why };
      } catch (failure) {
        log(`judge ${judge} failed on ${a.id} vs ${b.id}: ${failure.message}`);
        return null;
      }
    })
  ).filter(Boolean);
  const ids = entrants.map((e) => e.id);
  const strength = bradleyTerry(ids, games);
  for (const e of entrants) {
    const mine = games.filter((g) => g.a === e.id || g.b === e.id);
    const share = (field) => mine.reduce((s, g) => s + (g.a === e.id ? g[field] : 1 - g[field]), 0) / Math.max(1, mine.length);
    e.judge = {
      winRate: Number(share("score").toFixed(3)),
      faster: Number(share("faster").toFixed(3)),
      person: Number(share("person").toFixed(3)),
      strength: Number(strength[e.id].toFixed(3)),
      games: mine.length,
    };
  }
  // The share of games the text shown first won: 0.5 means no position bias (both orders are played anyway).
  const firstWins = games.reduce((s, g) => s + g.score, 0) / Math.max(1, games.length);
  return { games, failures: jobs.length - games.length, positionBias: Number(firstWins.toFixed(3)) };
}
