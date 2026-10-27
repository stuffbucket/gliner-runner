// The retell prompt: a disinterested friend reads only the candidate and answers three questions.
import { personaMessages } from "../lib/prompts.mjs";

export default function prompt({ vars }) {
  return personaMessages(vars.text);
}
