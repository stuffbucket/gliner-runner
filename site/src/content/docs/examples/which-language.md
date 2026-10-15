---
title: Tell which language a message is in
description: Find out whether a message is in English, Spanish, French, German, or Italian, so you can send it to the right person.
---

## The situation

Messages arrive in several languages. Before anyone reads them, you want to know which language
each one is in, so you can send it to someone who reads that language.

## The labels

Each message gets one label. The labels are the names of the languages you expect.

| Label | Meaning |
| --- | --- |
| `English` | The message is written in English |
| `Spanish` | The message is written in Spanish |
| `French` | The message is written in French |
| `German` | The message is written in German |
| `Italian` | The message is written in Italian |

## The command

This uses the default model, `decide-340m`, with no descriptions:

```sh
while IFS= read -r line; do
  printf '%s\n  ' "$line"
  gliner-runner classify --labels "English,Spanish,French,German,Italian" "$line"
done <<'EOF'
Where is the nearest train station?
¿Dónde está la estación de tren más cercana?
Où se trouve la gare la plus proche ?
Wo ist der nächste Bahnhof?
Dov'è la stazione ferroviaria più vicina?
Gracias por todo, nos vemos mañana.
Merci beaucoup pour votre aide.
EOF
```

## What we got

```text
Where is the nearest train station?
  English (confidence 0.568)
¿Dónde está la estación de tren más cercana?
  Spanish (confidence 0.986)
Où se trouve la gare la plus proche ?
  French (confidence 0.983)
Wo ist der nächste Bahnhof?
  German (confidence 0.999)
Dov'è la stazione ferroviaria più vicina?
  Italian (confidence 1.000)
Gracias por todo, nos vemos mañana.
  Spanish (confidence 0.970)
Merci beaucoup pour votre aide.
  French (confidence 0.989)
```

All seven are right. The English sentence has the lowest confidence, 0.568.

## Where it goes wrong

- **A language that is not in your list.** We sent a Portuguese sentence with the same labels. The
  program chose the closest one:

  ```sh
  gliner-runner classify --labels "English,Spanish,French,German,Italian" "Onde fica a estação de comboios mais próxima?"
  ```

  ```text
  Spanish (confidence 0.331)
  ```

  The confidence is low. Add `Portuguese` and any other language you expect to the labels.
- **Very short text.** The single word `ok` was labelled `English` with a confidence of 0.414. One
  word often does not show which language it is.
- **Mixed languages.** `Hola, how are you? Todo bien por aquí.` was labelled `Spanish` (0.988). A
  message in two languages gets one label.
- **The multilingual model was worse at this.** We ran the same seven messages with
  `--model decide-multilingual`. It got the English sentence wrong (`French`, confidence 0.251)
  and gave `Merci beaucoup pour votre aide.` only 0.350. For naming a language, use the default
  model.

## What to try next

- **Send each message to the right team.** Once you know the language, use the
  [other languages example](/gliner-runner/examples/other-languages/) to sort it by topic.
- **Send low confidence to a person.** Anything under 0.5 is worth a look.
