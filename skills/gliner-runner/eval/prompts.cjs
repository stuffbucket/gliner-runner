// Prompt variants for the gliner-runner skill eval.
// A: no skill, only a one-line description. B: the skill as the system prompt.
// SKILL_FILE overrides which SKILL.md variant B reads (used to compare skill versions).
"use strict";
const { readFileSync } = require("node:fs");
const { join } = require("node:path");

const ONE_LINE =
  "gliner-runner is a local zero-shot text classifier with a CLI and an HTTP server (https://github.com/stuffbucket/gliner-runner).";
const AGENT = "You are a coding agent helping a developer. Answer the request directly and concretely.";

function skillText() {
  const file = process.env.SKILL_FILE || join(__dirname, "..", "SKILL.md");
  return readFileSync(file, "utf8");
}

const messages = (system, task) => [
  { role: "system", content: system },
  { role: "user", content: task },
];

module.exports.noSkill = ({ vars }) => messages(`${AGENT}\n\n${ONE_LINE}`, vars.task);
module.exports.withSkill = ({ vars }) =>
  messages(`${AGENT}\n\n${ONE_LINE}\n\nYou have this skill loaded:\n\n${skillText()}`, vars.task);
