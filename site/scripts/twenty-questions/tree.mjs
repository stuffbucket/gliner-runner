// Builds the playground's "20 questions" tree (site/src/data/twenty-questions.json) from answers.json, which also
// holds the clues the simulator plays with. Only labels and descriptions go to the page.
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

export const ANSWERS = join(dirname(fileURLToPath(import.meta.url)), "answers.json");

export function loadAnswers(path = ANSWERS) {
  return JSON.parse(readFileSync(path, "utf8"));
}

export function buildTree(answers) {
  return {
    sample: answers.sample,
    tree: {
      label: "anything",
      children: answers.categories.map((category) => ({
        label: category.label,
        description: category.description,
        children: category.groups.map((group) => ({
          label: group.label,
          description: group.description,
          children: category.items
            .filter((item) => item.group === group.label)
            .map((item) => ({ label: item.name, ...(item.description ? { description: item.description } : {}) })),
        })),
      })),
    },
  };
}
