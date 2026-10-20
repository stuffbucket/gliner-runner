// "20 questions" for a classifier: the player describes something without naming it, and the game asks "Is it a
// zebra?" until it is right. Every question put to the player counts toward the 20: a guess, and any follow-up
// question the game might ask. Working things out is free: to choose what to ask, the game classifies the clues
// against the options at each level of the tree in site/src/data/twenty-questions.json ("Is it an animal, a
// vegetable or a mineral?", then a group, then the things in that group), as often as it needs to, without asking
// the player anything.
//
// The player is on the game's side, so this is a search, not a contest. How likely each thing is follows from the
// clues alone: the product of the model's shares on the way down (P(animal) x P(zoo animal | animal) x
// P(zebra | zoo animal)). A "no" only takes that thing away, leaving the others in the same order, and guessing the
// most likely thing left each time both needs the fewest guesses on average and wins most often within 20.
// So the game:
//   - finds the most likely thing best-first: a group's chance is more than that of anything inside it, so once
//     the likeliest thing found beats every group not yet looked inside, it is the likeliest of all, usually
//     after only a few of the tree's levels;
//   - keeps a classification only while the clues are the ones it was made for: a new clue makes every level
//     worth asking again, so what the game guesses depends on the clues as they stand, not the order they came in;
//   - rules out a wrong guess by name (a kiwi is a bird and a fruit, and is guessed once).
// The functions are pure, so the playground and the simulator (site/scripts/twenty-questions/simulate.mjs) play
// exactly the same game.

export const MAX_QUESTIONS = 20;

/**
 * Search settings, tuned with site/scripts/twenty-questions/tune.mjs:
 * - `temperature`: each classification's probabilities are raised to 1/temperature and renormalised (above 1
 *   trusts the classifier's first choice less).
 */
export const TUNING = { temperature: 1.4 };

/** Things that read better without "a" ("Is it gold?"). */
const MASS = new Set([
  "gold", "silver", "iron", "copper", "sand", "rice", "wheat", "coffee", "tea", "chocolate", "popcorn", "lettuce",
  "broccoli", "spinach", "garlic", "ginger", "cinnamon", "pepper", "corn", "bread", "pasta", "sugar", "cotton",
  "paper", "wood", "grass", "moss", "seaweed", "glass", "plastic", "water", "ice", "snow", "rain", "lightning",
  "fire", "oil", "rubber", "salt", "chalk", "clay", "steel", "honey", "cereal", "oatmeal", "flour", "mint", "basil",
  "lava", "fog", "thunder", "wind", "money", "jewelry", "lego", "bamboo", "celery", "cabbage", "cauliflower",
  "asparagus", "parsley", "oregano", "thyme", "rosemary", "cilantro", "dill", "nutmeg", "vanilla", "saffron",
  "paprika", "turmeric", "maple syrup", "soy sauce", "ketchup", "peanut butter", "jam", "tofu", "oatmeal", "quinoa",
  "barley", "oats", "lumber", "cork", "linen", "hemp", "cardboard", "rope", "orange juice", "lemonade", "cider",
  "wine", "beer", "cocoa", "hay", "straw", "ivy", "clover", "lavender", "aluminum", "marble", "granite", "concrete",
]);
const PLURAL = /(fries|scissors|jeans|glasses|pants|headphones|shorts|sunglasses|binoculars|tongs|oats|beans|peas|lentils|chips|noodles|grapes|leaves)$/;
const article = (label) => (MASS.has(label) || PLURAL.test(label) ? "" : /^[aeiou]/.test(label) ? "an " : "a ");

/** "a zebra", "an owl", "gold", "french fries". */
export const phrase = (label) => `${article(label)}${label}`;

const SEPARATOR = " › ";
export const key = (path) => path.join(SEPARATOR);

export function nodeAt(tree, path) {
  return path.reduce((node, label) => node?.children?.find((child) => child.label === label), tree);
}

export const isGuess = (node) => node.children === undefined;

/** What a classification of `node`'s children sends: their labels, and their descriptions if any have one. */
export function levelOf(node) {
  const described = node.children.filter((child) => child.description);
  return {
    labels: node.children.map((child) => child.label),
    descriptions: described.length > 0 ? Object.fromEntries(described.map((child) => [child.label, child.description])) : undefined,
  };
}

/** "Is it an animal, a vegetable or a mineral?" */
export function question(options) {
  const phrases = options.map((option) => phrase(option.label));
  return `Is it ${phrases.length > 1 ? `${phrases.slice(0, -1).join(", ")} or ${phrases.at(-1)}` : phrases[0]}?`;
}

/** The question a classification answers: the options themselves at the top, then "What kind of animal is it?" and
 * "Which zoo animal is it?" (the options are on the card already). */
export function levelQuestion(path, options) {
  if (path.length === 0) return question(options);
  return path.length === 1 ? `What kind of ${path[0]} is it?` : `Which ${path.at(-1)} is it?`;
}

/** "Is it a zebra?" */
export const guessQuestion = (label) => `Is it ${phrase(label)}?`;

/**
 * A new game. `levels` holds the latest classification of each level (by nodeKey, "" is the root) and the clues
 * it was made for; only those made for the current clues count (the trace in twentyQuestionsTrace.js reuses them
 * too). `ruledOut` holds the names of wrong guesses, and `asked` counts the questions put to the player.
 */
export const newGame = (tree, clues) => ({
  tree,
  clues,
  levels: new Map(),
  ruledOut: new Set(),
  asked: 0,
  phase: "asking", // asking | guessing | won | lost
  guess: undefined,
  note: "",
});

const copy = (game) => ({
  ...game,
  ruledOut: new Set(game.ruledOut),
  levels: new Map(game.levels),
});

/**
 * What to do next, best-first from the root: `{ kind: "guess", label, nodeKey, p, number, text }` when the most
 * likely thing is known, or `{ kind: "classify", path, nodeKey, options, labels, descriptions, clues, again, text }`
 * for the most likely group not yet classified for the current clues (`again` if it was, for older clues). Nothing
 * once every thing is ruled out.
 */
export function nextStep(game, tuning = TUNING) {
  const frontier = [{ path: [], node: game.tree, p: 1 }];
  while (frontier.length > 0) {
    const best = frontier.reduce((top, entry, index) => (entry.p > frontier[top].p ? index : top), 0);
    const { path, node, p } = frontier.splice(best, 1)[0];
    const nodeKey = key(path);
    if (isGuess(node)) {
      if (game.ruledOut.has(node.label)) continue;
      const number = game.asked + 1;
      return { kind: "guess", nodeKey, label: node.label, p, number, text: `Question ${number} of ${MAX_QUESTIONS} · ${guessQuestion(node.label)}` };
    }
    const level = game.levels.get(nodeKey);
    if (level?.clues !== game.clues) {
      const again = level !== undefined;
      const question = levelQuestion(path, node.children);
      return {
        kind: "classify",
        path,
        nodeKey,
        options: node.children,
        ...levelOf(node),
        clues: game.clues,
        again,
        text: again ? `With your new clue · ${question}` : question,
      };
    }
    const { labels } = levelOf(node);
    const shares = temper(labels, level.probabilities, tuning);
    node.children.forEach((child, index) => frontier.push({ path: [...path, child.label], node: child, p: p * shares[index] }));
  }
  return undefined;
}

/** Each label's share of one classification, raised to 1/temperature and renormalised (see TUNING). */
export function temper(labels, probabilities, tuning = TUNING) {
  const tempered = labels.map((label) => (probabilities[label] ?? 0) ** (1 / tuning.temperature));
  const total = tempered.reduce((sum, value) => sum + value, 0) || 1;
  return tempered.map((value) => value / total);
}

/**
 * The game after a classification of `step.clues` (it costs no question). If the clues changed while the model
 * was thinking, the level is out of date at once and the next step classifies it again.
 */
export function afterClassification(game, step, probabilities) {
  const state = copy(game);
  state.levels.set(step.nodeKey, { clues: step.clues, probabilities });
  const top = [...step.options].sort((a, b) => (probabilities[b.label] ?? 0) - (probabilities[a.label] ?? 0))[0];
  state.note = top === undefined ? "" : isGuess(top) ? "" : `Most likely ${phrase(top.label)}.`;
  return state;
}

/** The game while it asks the guess `step` (the guess counts as a question). */
export function startGuess(game, step) {
  const state = copy(game);
  state.asked = step.number;
  state.phase = "guessing";
  state.guess = step.label;
  return state;
}

/** The game after the player answered the current guess. */
export function afterReply(game, yes) {
  const state = copy(game);
  if (yes) {
    state.phase = "won";
    return state;
  }
  state.ruledOut.add(game.guess);
  state.note = `Not ${phrase(game.guess)}.`;
  state.guess = undefined;
  state.phase = state.asked >= MAX_QUESTIONS || nextStep(state, TUNING) === undefined ? "lost" : "asking";
  return state;
}

export function verdict(game) {
  return game.phase === "won"
    ? `Got it in ${game.asked} question${game.asked === 1 ? "" : "s"}: it's ${phrase(game.guess)}!`
    : `You win: ${MAX_QUESTIONS} questions and it still doesn't know. What was it?`;
}
