# circle_area — ATAF end-to-end demo

The smallest complete demonstration of the ATAF acquisition loop: a tool
is proposed, deployed, refused while pending, approved by a human, and
then successfully invoked.

This v0.1 example writes the tool code by hand (the LLM-driven agent
client arrives in a later session), but every server interaction — the
HTTP calls, the governance refusal, the approval, the execution — is the
real thing.

## Run it

In one terminal, start the server (defaults to port 9123):

```bash
source .venv/bin/activate
ATAF_PORT=9123 ataf-server
```

In another terminal, run the demo:

```bash
source .venv/bin/activate
python examples/circle_area/demo.py
```

Expected output:

```
[1] catalog has 0 tool(s) (version 0)
[2] deployed circle_area_v1 (status PENDING_REVIEW)
    input_schema: {'type': 'object', 'properties': {'radius': {'type': 'number', 'description': 'The radius of the circle, in any unit.'}}, 'required': ['radius']}
[3] invoke while pending -> HTTP 403: TOOL_NOT_AUTHORIZED
[4] approved circle_area_v1
[5] circle_area(radius=10) = 314.1592653589793
```

## What to look at next

- **Interactive API docs:** open <http://127.0.0.1:9123/docs>
- **Machine-readable spec:** <http://127.0.0.1:9123/openapi.json>
- **The deployed code:** `ataf_data/tools/circle_area_v1.py`
- **The audit trail:** `ataf_data/logs/deployment.jsonl`
- **Admin from the CLI instead of HTTP:**
  ```bash
  ataf-admin list
  ataf-admin approve circle_area_v1
  ```
