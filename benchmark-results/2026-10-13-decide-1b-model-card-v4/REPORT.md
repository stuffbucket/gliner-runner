# Pinned model-card example characterization

This run parsed and executed every `model.classify_text` example in the model
card shipped with `fastino/GLiNER2.5-Decide-1B` at `67af949e17bebd20f05fe020a971079889e7a462`.
The card calls its displayed results “potential outputs,” so disagreement is
reported as characterization rather than a test failure.

## Environment

- Hardware: Apple M5 Max
- OS: Darwin 26.7.1
- Python 3.12.12; PyTorch 2.14.0;
  GLiNER2 2.1.0
- Model-card SHA-256: `7a6b15d834f8125d7a798e4f2fa5518149e63eb0085df929e5fb63931cc6834f`
- Discovered `model.classify_text` calls: 21
- Parsed executable examples: 21
- Every discovered example parsed: True

## Execution

| Profile | Load (ms) | Executed | Potential-output matches |
| --- | ---: | ---: | ---: |
| CPU / FP32 | 44708.8 | 21/21 | 11/21 |

## Device output parity

CPU/FP32 is the baseline.

| Candidate | Exact output matches |
| --- | ---: |


## CPU/FP32 outputs

Input text and schemas remain in the pinned upstream card and are not duplicated
in this artifact.

| Example | Matches potential output | Actual output |
| --- | --- | --- |
| Customer support intent | yes | `{"intent": "refund_request"}` |
| Banking request | yes | `{"intent": "transfer_cancel"}` |
| Travel request | no | `{"request": "seat_change"}` |
| Clinic request | yes | `{"request": "book_appointment"}` |
| Review sentiment | no | `{"sentiment": "positive"}` |
| Product aspects | yes | `{"aspects": ["battery", "keyboard", "screen"]}` |
| News topic | yes | `{"topic": "business"}` |
| Document type | yes | `{"document_type": "invoice"}` |
| Email triage | no | `{"intent": "request", "route": "legal", "urgency": "critical"}` |
| Ticket routing | no | `{"queue": "payroll"}` |
| Handoff to a person | yes | `{"handoff": "yes"}` |
| Did the agent finish? | yes | `{"finished": "no"}` |
| Moderation | yes | `{"policy": "personal_data"}` |
| Incident severity | no | `{"severity": "info"}` |
| Urgency score | no | `{"urgency": "4"}` |
| Spam or not | yes | `{"label": "spam"}` |
| Several decisions at once | no | `{"intent": "checkout", "needs_human": "no", "priority": "low", "topics": ["hvac"]}` |
| Question over a passage | no | `{"answer": "yes"}` |
| Book | no | `{"genre": "history"}` |
| Labels with a description | yes | `{"intent": "card_pin_change"}` |
| Ordinal score | no | `{"rating": "5"}` |
