// Prompt variants for the hard tier: the same A/B system prompts as ../prompts.cjs, with the run rules
// appended to every task. SKILL_FILE still selects which skill text B reads.
"use strict";
const base = require("../prompts.cjs");

const RULES = `Rules for the script:
- Write one complete script: Node.js 24 using only built-in modules (global fetch is available), or a bash script using curl and jq.
- gliner-runner serve is already running at http://127.0.0.1:8099 with the models decide-340m, decide-multilingual and decide-1b installed.
- The script runs in its own directory, which holds the input file. It has no network access except that server and no packages to install. It must finish within 2 minutes.
- Print only the result to stdout, nothing else.
- Reply with the script in a single fenced code block.`;

const withRules = (vars) => ({ vars: { ...vars, task: `${vars.task}\n\n${RULES}` } });

module.exports.noSkill = ({ vars }) => base.noSkill(withRules(vars));
module.exports.withSkill = ({ vars }) => base.withSkill(withRules(vars));
