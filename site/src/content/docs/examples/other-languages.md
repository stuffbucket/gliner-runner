---
title: Route messages written in other languages
description: Route Spanish, French, and German messages with the multilingual model.
---

## The situation

Your support inbox gets messages in Spanish, French, and German. You want them sorted into the
same teams as your English messages, without translating them first.

## Get the multilingual model

The default English model is not meant for other languages. Use `decide-multilingual`, which is
about 1.2 GB:

```sh
gliner-runner models install decide-multilingual
```

## The labels

The labels stay in English. They are the names of your teams, and the model reads the message in
its own language. Each message gets one label.

| Label | Description we give it |
| --- | --- |
| `billing` | Charges, invoices, refunds, or payment problems |
| `technical` | Something does not work or shows an error |
| `shipping` | Delivery, tracking, or a missing package |
| `account` | Logging in, passwords, or account details |

## The command

Add `--model decide-multilingual` to the command:

```sh
while IFS= read -r line; do
  printf '%s\n  ' "$line"
  gliner-runner classify --model decide-multilingual \
    --labels "billing,technical,shipping,account" \
    --describe "billing=Charges, invoices, refunds, or payment problems" \
    --describe "technical=Something does not work or shows an error" \
    --describe "shipping=Delivery, tracking, or a missing package" \
    --describe "account=Logging in, passwords, or account details" \
    "$line"
done <<'EOF'
Me cobraron dos veces el mismo pedido y quiero que me devuelvan el dinero.
Mon colis n'est toujours pas arrivé après deux semaines.
Ich kann mich nicht mehr einloggen, das Passwort wird nicht akzeptiert.
Die App stürzt ab, sobald ich die Einstellungen öffne.
La facture de ce mois est plus élevée que prévu.
EOF
```

## What we got

```text
Me cobraron dos veces el mismo pedido y quiero que me devuelvan el dinero.
  billing (confidence 0.724)
Mon colis n'est toujours pas arrivé après deux semaines.
  shipping (confidence 0.699)
Ich kann mich nicht mehr einloggen, das Passwort wird nicht akzeptiert.
  account (confidence 0.492)
Die App stürzt ab, sobald ich die Einstellungen öffne.
  technical (confidence 0.783)
La facture de ce mois est plus élevée que prévu.
  billing (confidence 0.786)
```

The messages are Spanish, French, German, German, and French. All five went to the right team.
The German login message has the lowest confidence, 0.492.

One miss: when we changed the fourth message to `Die App stürzt ab, sobald ich auf "Bezahlen" tippe.`
(the app crashes when I tap "Pay"), the program chose `billing (confidence 0.476)`. The
message is about a crash, but it mentions paying. When a message touches two topics, the answer can
go either way, and a low confidence is the sign.

## What to try next

- **Find out the language first.** See
  [Tell which language a message is in](/gliner-runner/examples/which-language/).
- **Mixed languages.** Put messages in any of the supported languages in the same list. You do not
  have to say which language each one is in.
- **Check the confidence.** Send anything under 0.5 to a person who reads that language.
- **Other languages.** Try your own. We tested Spanish, French, and German.
- **English messages** also work with `decide-multilingual`. The `demo` command shows examples:
  `gliner-runner demo --model decide-multilingual`.
