// "20 questions" for a classifier: the player describes something without naming it. The game asks two kinds of
// question, each counting toward the 20:
//   - a classification of the clues against the options at one level of the tree in
//     site/src/data/twenty-questions.json ("Is it an animal, a vegetable or a mineral?", then a group, then the
//     things in that group), which tells it how likely each option is;
//   - a guess ("Is it a zebra?"), answered yes or no by the player.
// It plays best-first: every option it has scored carries the probability of the whole path to it
// (P(animal) x P(mammal | animal) x P(zebra | mammal)), and each turn it either guesses the most likely thing or
// opens up the most likely group it has not looked inside yet. A wrong guess only removes that one thing, so the
// game moves on to another group as soon as that group is more likely than what is left of the current one.
// The functions are pure, so the playground and the simulator (site/scripts/twenty-questions/simulate.mjs) play
// exactly the same game.

export const MAX_QUESTIONS = 20;

/**
 * Search settings, tuned with site/scripts/twenty-questions/tune.mjs:
 * - `discount`: an unexplored group is worth its probability times this when compared with a guess, because
 *   opening it costs a question and still leaves a guess to make;
 * - `temperature`: each classification's probabilities are raised to 1/temperature and renormalised (above 1
 *   trusts the classifier's first choice less).
 */
export const TUNING = { discount: 0.45, temperature: 1 };

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
const pathOf = (nodeKey) => (nodeKey === "" ? [] : nodeKey.split(SEPARATOR));

export function nodeAt(tree, path) {
  return path.reduce((node, label) => node?.children?.find((child) => child.label === label), tree);
}

export const isGuess = (node) => node.children === undefined;

/** "Is it an animal, a vegetable or a mineral?" */
export function question(options) {
  const phrases = options.map((option) => phrase(option.label));
  return `Is it ${phrases.length > 1 ? `${phrases.slice(0, -1).join(", ")} or ${phrases.at(-1)}` : phrases[0]}?`;
}

/** "Is it a zebra?" */
export const guessQuestion = (label) => `Is it ${phrase(label)}?`;

/**
 * A new game. `scores` holds, for every node scored so far, the probability of the path to it ("" is the root);
 * `opened` the nodes whose children have been classified; `ruledOut` the wrong guesses.
 */
export const newGame = (tree, clues) => ({
  tree,
  clues,
  scores: new Map([["", 1]]),
  opened: new Set(),
  ruledOut: new Set(),
  asked: 0,
  phase: "asking", // asking | guessing | won | lost
  guess: undefined,
  note: "",
});

const copy = (game) => ({
  ...game,
  scores: new Map(game.scores),
  opened: new Set(game.opened),
  ruledOut: new Set(game.ruledOut),
});

/**
 * What to do next: `{ kind: "guess", label, nodeKey, text }` for the most likely thing, or
 * `{ kind: "classify", path, options, labels, descriptions, text }` to open the most likely unexplored group.
 */
export function nextStep(game, tuning = TUNING) {
  let best;
  for (const [nodeKey, score] of game.scores) {
    if (game.opened.has(nodeKey) || game.ruledOut.has(nodeKey)) continue;
    const value = isGuess(nodeAt(game.tree, pathOf(nodeKey))) ? score : score * tuning.discount;
    if (best === undefined || value > best.value) best = { nodeKey, value };
  }
  if (best === undefined) return undefined;
  const path = pathOf(best.nodeKey);
  const node = nodeAt(game.tree, path);
  const number = game.asked + 1;
  if (isGuess(node)) {
    return { kind: "guess", nodeKey: best.nodeKey, label: node.label, number, text: `Question ${number} of ${MAX_QUESTIONS} · ${guessQuestion(node.label)}` };
  }
  const options = node.children;
  const described = options.filter((option) => option.description);
  return {
    kind: "classify",
    path,
    nodeKey: best.nodeKey,
    options,
    number,
    labels: options.map((option) => option.label),
    descriptions: described.length > 0 ? Object.fromEntries(described.map((o) => [o.label, o.description])) : undefined,
    text: `Question ${number} of ${MAX_QUESTIONS} · ${question(options)}`,
  };
}

/** The game after a classification: each option gets the probability of the path to it. */
export function afterClassification(game, step, probabilities, tuning = TUNING) {
  const state = copy(game);
  state.asked = step.number;
  state.opened.add(step.nodeKey);
  const parent = game.scores.get(step.nodeKey) ?? 0;
  const tempered = step.options.map((option) => (probabilities[option.label] ?? 0) ** (1 / tuning.temperature));
  const total = tempered.reduce((sum, value) => sum + value, 0) || 1;
  step.options.forEach((option, index) => {
    state.scores.set(key([...step.path, option.label]), (parent * (tempered[index] ?? 0)) / total);
  });
  const top = [...step.options].sort((a, b) => (probabilities[b.label] ?? 0) - (probabilities[a.label] ?? 0))[0];
  state.note = top === undefined ? "" : isGuess(top) ? "" : `Most likely ${phrase(top.label)}.`;
  if (state.asked >= MAX_QUESTIONS) state.phase = "lost";
  return state;
}

/** The game while it asks the guess `step` (the guess counts as a question). */
export function startGuess(game, step) {
  const state = copy(game);
  state.asked = step.number;
  state.phase = "guessing";
  state.guess = step.label;
  state.guessKey = step.nodeKey;
  return state;
}

/** The game after the player answered the current guess. */
export function afterReply(game, yes) {
  const state = copy(game);
  if (yes) {
    state.phase = "won";
    return state;
  }
  state.ruledOut.add(game.guessKey);
  state.note = `Not ${phrase(game.guess)}.`;
  state.guess = undefined;
  state.guessKey = undefined;
  state.phase = state.asked >= MAX_QUESTIONS || nextStep(state, TUNING) === undefined ? "lost" : "asking";
  return state;
}

export function verdict(game) {
  return game.phase === "won"
    ? `Got it in ${game.asked} question${game.asked === 1 ? "" : "s"}: it's ${phrase(game.guess)}!`
    : `You win: ${MAX_QUESTIONS} questions and it still doesn't know. What was it?`;
}
