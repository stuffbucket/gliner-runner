// "20 questions" for a classifier: the visitor describes something without naming it, and each question is one
// classification of those clues. The labels start with the classic "animal, vegetable or mineral?" and narrow down
// this tree until the last level, where every label is a guess.

export const MAX_QUESTIONS = 20;
/** After this many wrong guesses in one group, the game decides the group itself was wrong and goes back up. */
export const WRONG_GUESSES_PER_GROUP = 2;
export const SAMPLE_CLUES = "It is black and white with stripes, and it lives in herds on the African grassland.";

const group = (label, description, things) => ({ label, description, children: things.map((thing) => ({ label: thing })) });

export const TREE = {
  label: "anything",
  children: [
    {
      label: "animal",
      description: "A living creature",
      children: [
        group("mammal", "A furry animal that feeds its young milk", ["dog", "cat", "horse", "cow", "elephant", "lion", "zebra", "giraffe", "whale", "bat", "rabbit", "mouse", "bear", "monkey"]),
        group("bird", "An animal with feathers and wings", ["eagle", "penguin", "parrot", "owl", "chicken", "duck", "flamingo"]),
        group("fish", "An animal that lives and breathes in water", ["shark", "goldfish", "salmon", "clownfish"]),
        group("reptile or amphibian", "A scaly or slimy animal such as a snake, lizard or frog", ["snake", "turtle", "crocodile", "frog", "lizard"]),
        group("insect or bug", "A small creature with six or eight legs", ["bee", "butterfly", "ant", "spider", "ladybug", "mosquito"]),
      ],
    },
    {
      label: "vegetable",
      description: "A plant, or something that grows on a plant or is made from one",
      children: [
        group("fruit", "A sweet part of a plant that people eat", ["apple", "banana", "orange", "strawberry", "grape", "watermelon", "lemon", "pineapple"]),
        group("vegetable to eat", "A savoury plant food such as a root or a leaf", ["carrot", "potato", "tomato", "onion", "broccoli", "pumpkin", "lettuce"]),
        group("tree or flower", "A plant that grows in the ground", ["oak tree", "rose", "sunflower", "cactus", "palm tree", "tulip"]),
        group("grain, nut or bean", "Seeds that people eat or drink", ["rice", "wheat", "peanut", "coffee", "chocolate", "popcorn"]),
      ],
    },
    {
      label: "mineral",
      description: "Something that is not alive: an object, a material, a machine or a place",
      children: [
        group("vehicle", "Something that carries people or things", ["car", "bicycle", "airplane", "train", "boat", "rocket", "bus"]),
        group("tool or household object", "Something people use at home or at work", ["hammer", "scissors", "chair", "table", "lamp", "key", "umbrella", "clock"]),
        group("electronic device", "A machine that runs on electricity", ["phone", "computer", "television", "camera", "headphones", "robot"]),
        group("musical instrument", "Something people play to make music", ["piano", "guitar", "drum", "violin", "trumpet"]),
        group("clothing", "Something people wear", ["shirt", "shoe", "hat", "sock", "jacket"]),
        group("natural material or place", "Something found in nature that is not alive", ["rock", "diamond", "gold", "sand", "volcano", "mountain", "ocean", "moon"]),
      ],
    },
  ],
};

/** Things that read better without "a" ("Is it gold?"). */
const MASS = new Set(["gold", "sand", "rice", "wheat", "coffee", "chocolate", "popcorn", "lettuce", "broccoli"]);
const article = (label) => (MASS.has(label) ? "" : /^[aeiou]/.test(label) ? "an " : "a ");
/** "a zebra", "an owl", "gold". */
export const phrase = (label) => `${article(label)}${label}`;

export function nodeAt(path) {
  return path.reduce((node, label) => node.children.find((child) => child.label === label), TREE);
}

export const isGuess = (node) => node.children === undefined;
export const key = (path) => path.join(" › ");

/** "Is it an animal, a vegetable or a mineral?" */
export function question(options) {
  const phrases = options.map((option) => phrase(option.label));
  return `Is it ${phrases.length > 1 ? `${phrases.slice(0, -1).join(", ")} or ${phrases.at(-1)}` : phrases[0]}?`;
}

/** "Is it a zebra?" */
export const guessQuestion = (label) => `Is it ${phrase(label)}?`;

/**
 * A new game. `path` is where the next question is asked; `excluded` holds the keys of options ruled out (wrong
 * guesses and abandoned groups); `wrong` counts wrong guesses per group.
 */
export const newGame = (clues = SAMPLE_CLUES) => ({
  clues,
  path: [],
  excluded: new Set(),
  wrong: new Map(),
  asked: 0,
  phase: "asking", // asking | guessing | won | lost
  guess: undefined,
  note: "",
});

/** The options still open at `path`, backing up past levels whose options are all ruled out. */
export function openOptions(game) {
  let path = game.path;
  for (;;) {
    const options = nodeAt(path).children.filter((child) => !game.excluded.has(key([...path, child.label])));
    if (options.length > 0 || path.length === 0) {
      return { path, options };
    }
    game.excluded.add(key(path));
    path = path.slice(0, -1);
  }
}
