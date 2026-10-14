# Pinned model-card example characterization

This run parsed and executed every `model.classify_text` example in the model
card shipped with `fastino/GLiNER2.5-multi-Decide` at `e173bca1f0c4217d7a55b3b0c705d5ece8a0218d`.
The card calls its displayed results “potential outputs,” so disagreement is
reported as characterization rather than a test failure.

## Environment

- Hardware: Apple M5 Max
- OS: Darwin 26.7.1
- Python 3.12.12; PyTorch 2.14.0;
  GLiNER2 2.1.0
- Model-card SHA-256: `55531b81804b6c8160d41de57ff9559ca20b2f3779ab6e36e471dcd51127d92b`
- Discovered `model.classify_text` calls: 21
- Parsed executable examples: 21
- Every discovered example parsed: True

## Execution

| Profile | Load (ms) | Executed | Potential-output matches |
| --- | ---: | ---: | ---: |
| CPU / FP32 | 3915.2 | 21/21 | 9/21 |

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
| Travel request | yes | `{"request": "change"}` |
| Clinic request | no | `{"request": "refill_prescription"}` |
| Review sentiment | no | `{"sentiment": "positive"}` |
| Product aspects | no | `{"aspects": ["keyboard", "screen"]}` |
| News topic | yes | `{"topic": "business"}` |
| Document type | yes | `{"document_type": "invoice"}` |
| Email triage | yes | `{"intent": "request", "route": "legal", "urgency": "high"}` |
| Ticket routing | no | `{"queue": "payroll"}` |
| Handoff to a person | no | `{"handoff": "no"}` |
| Did the agent finish? | no | `{"finished": "yes"}` |
| Moderation | yes | `{"policy": "personal_data"}` |
| Incident severity | yes | `{"severity": "low"}` |
| Urgency score | no | `{"urgency": "4"}` |
| Spam or not | yes | `{"label": "spam"}` |
| Several decisions at once | no | `{"intent": "checkout", "needs_human": "yes", "priority": "high", "topics": ["hvac"]}` |
| Question over a passage | no | `{"answer": "yes"}` |
| Book | no | `{"genre": "history"}` |
| Labels with a description | yes | `{"intent": "card_pin_change"}` |
| Ordinal score | no | `{"rating": "10"}` |
