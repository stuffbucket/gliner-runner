// The prompts: drafting (one per angle), the disinterested friend who retells a candidate, the grader's rubric and
// the pairwise judge. The brief is the product's facts and the user's own words; the voice is the landing page's.
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const repo = join(here, "../../..");

/** Bump when a prompt changes in a way that should invalidate cached retells and judgements. */
export const PROMPT_VERSION = 2;

const headlines = () => {
  try {
    return Object.values(JSON.parse(readFileSync(join(repo, "site/src/data/headlines.json"), "utf8")).headlines);
  } catch {
    return ["Tossing the junk mail before you reach the front door.", "Knowing which call to take during dinner."];
  }
};

const BRIEF = `What the thing is: GLiNER Runner, a small free program. You give it a short piece of text (a message, an email,
a review) and a few options you wrote yourself, such as billing, shipping or support; urgent or can wait; happy or
angry. It picks the option that fits (or, if you ask, every option that fits, or none). It never writes anything and
does not chat. It cannot answer with an option you did not give it. It makes the kind of fast call a person makes at a
glance without stopping to think: tossing junk mail, knowing which call to take during dinner, knowing which desk to
take a problem to. It runs on your own computer or in your browser, and your text never leaves your device.

The reader: has used ChatGPT and assumes any "AI model" is a chatbot that writes. Has no interest in how it works
inside. Needs a picture they could pass on to a friend in 15 seconds.`;

const USERS_WORDS = [
  "Most people will think of a model as meaning an LLM... help them understand at a very high level what the model is doing so that they have a mechanical analog that they could explain to someone in 15 seconds who has no interest in understanding the guts.",
  "This model is a thinking fast model. Something that people do all the time in their head that they understand... Not too abstract or a completely different tack.",
  "Nobody wakes up in the morning wanting to sort things into labels.",
  "Showing people how something works is far stronger than telling them.",
];

const STYLE = `Style: plain and unmannered, like the landing page. Short words. Concrete everyday things. Second person is fine.
No eyebrows or set-ups ("Think of it as", "Here's the thing", "Simply put", "Meet"). No flourishes, no stacked
metaphors, no "magic", no rhetorical questions, no exclamation marks, no em dashes, no colons that set up a reveal.
No meta asides about the page or the reader ("this page", "in short", "below").`;

export const ANGLES = {
  glance: "Start from an everyday glance decision the reader already makes (sorting the mail, picking which call to answer, which counter to queue at), then say the program makes that same kind of quick call about a piece of text.",
  "multiple-choice": "Use the multiple-choice picture: you write the choices, it reads the text and picks one (or a few). Keep it at the level of a school quiz, not a test of facts.",
  "not-chat": "Contrast it with a chatbot: it doesn't write, it picks. The reader assumes AI means chat; correct that in plain words, then say what it does instead.",
  "users-words": `Build from the user's own words below. Keep their plain register and their idea (a fast, in-your-head call people make all day), but say it to the reader, not about the reader.\n${USERS_WORDS.map((w) => `- "${w}"`).join("\n")}`,
  "tiny-example": "Show, don't tell: one tiny concrete example in words (a message, two or three choices you gave it, and the one it picks), plus at most one sentence that says what that means.",
  "show-lead-in": "Write the one sentence that sits right above a hands-on demo. In the demo, a kid's line (\"Please please please can I stay up late?\") is shown with sliders for the ways a kid might ask: begging, bargaining, flattery, a polite ask, and so on. The program has set each slider to how well that way fits the line, and the reader can drag one to see a different line. The sentence must still make sense read alone, and should say what the program does, not describe the demo's controls.",
};

export function draftingMessages({ slot, angle, count }) {
  const limits = [
    `at most ${slot.maxWords} words in all`,
    slot.maxSentences ? `at most ${slot.maxSentences} sentence${slot.maxSentences > 1 ? "s" : ""}` : null,
    `no sentence longer than ${slot.maxSentenceWords} words`,
    `Flesch-Kincaid grade ${slot.maxGrade} or lower (short words, short sentences)`,
    `never use these words: ${slot.stallWords.join(", ")}`,
  ].filter(Boolean);
  return [
    {
      role: "system",
      content: `You write copy for a small open-source project's website.\n\n${BRIEF}\n\nThe voice to match is the landing page's. Its headlines:\n${headlines().map((h) => `- ${h}`).join("\n")}\n\n${STYLE}`,
    },
    {
      role: "user",
      content: `The slot: ${slot.where}. ${slot.purpose}\n\nHard limits (a script checks them and throws out anything that breaks one):\n${limits.map((l) => `- ${l}`).join("\n")}\n\nThe angle: ${ANGLES[angle]}\n\nWrite ${count} different candidates for this slot. Make them genuinely different from each other, not small rewordings. Reply with JSON only: {"candidates": ["...", "..."]}`,
    },
  ];
}

export function personaMessages(text) {
  return [
    {
      role: "system",
      content:
        "You are a friend of someone who works on websites. You have no interest in technology and none in how software works. You have used ChatGPT a few times. Your friend shows you a few lines from a website and asks you about them. Answer only from what the lines say, in your own plain words. Do not use anything you know from elsewhere. Reply with JSON only.",
    },
    {
      role: "user",
      content: `The lines:\n"""\n${text}\n"""\n\n(a) In one sentence, what does it do?\n(b) Is it like ChatGPT? Why or why not?\n(c) Would you keep reading? yes or no. These lines start a longer page, and you skip most pages like it: say yes only if they were quick to get and made you a little curious, and no if they were slow, vague or dull.\n\nReply as {"what": "...", "like_chatgpt": "...", "keep_reading": "yes" or "no"}`,
    },
  ];
}

export function rubric(facts) {
  const required = facts.filter((f) => !f.optional);
  const optional = facts.filter((f) => f.optional);
  const list = (fs) => fs.map((f) => `- ${f.id}: ${f.fact}`).join("\n");
  return [
    "The output is a reader's retelling of a short text they just read (what it does, and whether it is like ChatGPT).",
    "For each fact below, decide whether the retelling conveys it correctly. A paraphrase counts. A vague answer, a missing fact, or an answer that contradicts the fact (for example, saying it writes replies) does not.",
    `Required facts:\n${list(required)}`,
    optional.length > 0 ? `Optional facts (report them, but they do not count toward the score):\n${list(optional)}` : "",
    `The score is the number of required facts conveyed divided by ${required.length}. Pass only if every required fact is conveyed.`,
    `Begin the reason with every fact id and yes or no, exactly like: ${facts.map((f) => `${f.id}=yes`).join(" ")}. Then give one short sentence of explanation.`,
  ].filter(Boolean).join("\n\n");
}

export function judgeMessages(a, b) {
  return [
    {
      role: "system",
      content: "You compare two short texts written for a website. The reader is a busy person who does not care about technology. Judge only the texts. Reply with JSON only.",
    },
    {
      role: "user",
      content: `Text A:\n"""\n${a}\n"""\n\nText B:\n"""\n${b}\n"""\n\n1. Which would a busy person who does not care about tech understand faster and remember?\n2. Which sounds like a person, not a brochure?\n\nReply as {"faster": "A" or "B", "person": "A" or "B", "why": "one short sentence"}`,
    },
  ];
}
