# Pinned model-card example characterization

This run parsed and executed every `model.classify_text` example in the model
card shipped with `fastino/GLiNER2.5-Decide` at `5a7adf72a23b4d311abae6ce050d7f0012bb3416`.
The card calls its displayed results “potential outputs,” so disagreement is
reported as characterization rather than a test failure.

## Environment

- Hardware: Apple M5 Max
- OS: Darwin 26.7
- Python 3.12.12; PyTorch 2.14.0;
  GLiNER2 2.0.0
- Model-card SHA-256: `c92f9ada7e7a5af1620d59fa29b69920c7d1d3851f61e547ddc9d8c2ce5a1b95`
- Parsed examples: 21

## Execution

| Profile | Load (ms) | Executed | Potential-output matches |
| --- | ---: | ---: | ---: |
| CPU / FP32 | 4289.3 | 21/21 | 12/21 |
| MPS / FP16 | 4373.2 | 21/21 | 12/21 |
| MPS / FP32 | 4496.9 | 21/21 | 12/21 |

## Device output parity

CPU/FP32 is the baseline.

| Candidate | Exact output matches |
| --- | ---: |
| MPS/FP16 | 21/21 |
| MPS/FP32 | 21/21 |

## CPU/FP32 outputs

Input text and schemas remain in the pinned upstream card and are not duplicated
in this artifact.

| Example | Matches potential output | Actual output |
| --- | --- | --- |
| Customer support intent | yes | `{"intent": "refund_request"}` |
| Banking request | no | `{"intent": "beneficiary_add"}` |
| Travel request | no | `{"request": "seat_change"}` |
| Clinic request | yes | `{"request": "book_appointment"}` |
| Review sentiment | no | `{"sentiment": "positive"}` |
| Product aspects | yes | `{"aspects": ["battery", "keyboard", "screen"]}` |
| News topic | yes | `{"topic": "business"}` |
| Document type | yes | `{"document_type": "invoice"}` |
| Email triage | no | `{"intent": "request", "route": "legal", "urgency": "critical"}` |
| Ticket routing | yes | `{"queue": "benefits"}` |
| Handoff to a person | no | `{"handoff": "no"}` |
| Did the agent finish? | yes | `{"finished": "no"}` |
| Moderation | yes | `{"policy": "personal_data"}` |
| Incident severity | no | `{"severity": "info"}` |
| Urgency score | yes | `{"urgency": "5"}` |
| Spam or not | yes | `{"label": "spam"}` |
| Several decisions at once | no | `{"intent": "room_change", "needs_human": "yes", "priority": "urgent", "topics": ["hvac"]}` |
| Question over a passage | no | `{"answer": "yes"}` |
| Book | no | `{"genre": "history"}` |
| Labels with a description | yes | `{"intent": "card_pin_change"}` |
| Ordinal score | yes | `{"rating": "7"}` |
