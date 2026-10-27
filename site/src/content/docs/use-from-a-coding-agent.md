---
title: Use it from a coding agent
description: A skill that teaches a coding agent such as Claude Code to use GLiNER Runner, how to install it, and the Markdown pages an agent can read.
---

Ask a coding agent, such as Claude Code, to sort your support tickets, and it can write the code
that calls GLiNER Runner. Install the skill below and the agent knows how. An agent without the
skill can still read this site, in Markdown.

## What the skill is

A skill is a folder of instructions that a coding agent reads when a task needs them. The GLiNER
Runner skill is in the repository, in
[`skills/gliner-runner/`](https://github.com/stuffbucket/gliner-runner/tree/main/skills/gliner-runner).
The instructions are in one file, `SKILL.md`. It holds the same facts as this site, written short,
for an agent.

With the skill installed, ask the agent for a job such as "send each support ticket to billing,
support or shipping". The agent uses the skill to write the request, and the code that reads the
answer.

## Install it in Claude Code

Claude Code reads skills from two folders:

- `~/.claude/skills/` in your home folder, for all your projects.
- `.claude/skills/` in a project, for that project only. Commit it, and everyone who works on the
  project gets the skill too.

Copy the skill folder into one of them. For all your projects:

```sh
git clone --depth 1 https://github.com/stuffbucket/gliner-runner.git
mkdir -p ~/.claude/skills
cp -R gliner-runner/skills/gliner-runner ~/.claude/skills/
```

Claude Code picks up the new skill in a session that is already running. If `~/.claude/skills/`
did not exist when the session started, type `/reload-skills`. The agent uses the skill when your
request matches it. You can also start it yourself with `/gliner-runner`.

Other coding agents that read `SKILL.md` skills keep them in their own folder. See the agent's
documentation.

The skill is instructions, not the program. GLiNER Runner, a model and a running server must be on
your computer. The skill tells the agent how to install them. To do it yourself, see
[Get started](/gliner-runner/start/).

## What it teaches

- **When to use it, and when not.** Choosing labels from a list, or deciding which tags apply.
  Not writing, summarizing, or pulling a name or a number out of a text. See
  [How it works](/gliner-runner/how-it-works/#what-it-can-and-cant-do).
- **How to install and start it,** and to ask you before it downloads a model. See
  [Get started](/gliner-runner/start/).
- **The request.** Every field, and that the task name and the labels are read by the model. See
  [Command line and HTTP](/gliner-runner/command-line-and-http/#send-a-request-with-curl).
- **One label or several,** and what `confidence` and `probabilities` mean in each. See
  [What the numbers mean](/gliner-runner/how-it-works/#what-the-numbers-mean).
- **Thresholds.** Keep the cutoffs in your code, tune them on your own texts, and over HTTP set
  both thresholds to go below 0.5. See
  [Lower the threshold](/gliner-runner/command-line-and-http/#lower-the-threshold).
- **Many texts, long texts and web pages.** One server, `/v1/batch`, the queue limit, splitting
  long documents, and browser permissions. See
  [Many texts](/gliner-runner/command-line-and-http/#many-texts) and
  [Use it in a React app](/gliner-runner/use-in-react/).
- **Errors.** What each error code means and what to do. See
  [Errors](/gliner-runner/command-line-and-http/#errors).
- **When answers look wrong.** Overlapping labels, a missing "none of these" label, a long text
  decided by one sentence, or the wrong model for the language.

## Pages for agents

An agent without the skill can read this site in Markdown. The list of pages is at
[`llms.txt`](https://stuffbucket.github.io/gliner-runner/llms.txt). Every documentation page has a
Markdown copy at the same address, with `.md` in place of the last `/`. For example, the Markdown
copy of
`https://stuffbucket.github.io/gliner-runner/how-it-works/` is
[`https://stuffbucket.github.io/gliner-runner/how-it-works.md`](https://stuffbucket.github.io/gliner-runner/how-it-works.md).

The running server describes its requests and replies at `http://127.0.0.1:8090/openapi.json`.

## Next

[Choose a model](/gliner-runner/models/): the three models, their sizes and languages, and how they
were tested.
