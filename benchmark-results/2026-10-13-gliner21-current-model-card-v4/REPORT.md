# Pinned model-card example characterization

This run parsed and executed every `model.classify_text` example in the model
card shipped with `fastino/GLiNER2.5-Decide` at `9d1bfb848cd16d93ee56ccbc59cd92bf0f284d90`.
The card calls its displayed results “potential outputs,” so disagreement is
reported as characterization rather than a test failure.

## Environment

- Hardware: Apple M5 Max
- OS: Darwin 26.7.1
- Python 3.12.12; PyTorch 2.14.0;
  GLiNER2 2.1.0
- Model-card SHA-256: `09a9705ee8c30392bb5a82ac503c01b889ff7e957a4c96bf7249f05acde709a3`
- Discovered `model.classify_text` calls: 21
- Parsed executable examples: 21
- Every discovered example parsed: True

## Execution

| Profile | Load (ms) | Executed | Potential-output matches |
| --- | ---: | ---: | ---: |
| CPU / FP32 | 4207.6 | 21/21 | 12/21 |

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
