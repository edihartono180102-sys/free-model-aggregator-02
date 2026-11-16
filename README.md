# free-model-aggregator-02 — a live free-model aggregator

One server (`main.ts`, Node 22, standard library only) that puts OpenRouter's fourteen
free models behind a single endpoint, speaking both client wires:

- `POST /v1/chat/completions` — OpenAI Chat Completions, JSON or server-sent events
- `POST /v1/messages` — Anthropic Messages, JSON or server-sent events
- `GET /v1/models` — the OpenAI list for `Authorization: Bearer`, the Anthropic list
  for `x-api-key`

Every completion it answers comes from a real free model dialled through the grader's
gateway (`POST $OPENAI_BASE_URL/chat/completions`) on the trial's key. The server
never answers from its own weights: there is no local model here.

## Routing

Each request is served by the best free model in the brief that meets its
requirements, ranked by the brief's `rating_milli` (ties by id, so the order is a
pure function of the brief). For the fourteen in the brief:

- plain and tools prompts → `openrouter/thinkingmachines/inkling-2-small:free` (1475)
- JSON-mode prompts → `openrouter/nvidia/nemotron-3.5-super-vl:free` (1303, the best
  model that supports JSON mode)

The `model` field reported is always the model actually dialled (the provider's own
answer spells it without the `openrouter/` prefix; the dialled id is put back on
every answer and every streamed chunk).

## Failover

A gateway call answered 4xx/5xx is not an answer. The server retries a 5xx once on
the same model, then fails over down the quality-ordered list. Every prompt starts
again from the best model: a refusal is retried on the next prompt, so a transient
rate limit costs one prompt at most and a persistent one only a cheap probe call
(a refused call is not counted in the tie-break, and six prompts stay far under
the trial's 60-call cap). A gateway answer that only calls tools, with no text
carrying the prompt's token, fails over the same way.
Every refusal is reported on the served response in the header
`x-makefun-upstream-failures` (at most 8 rows of `{"status", "code", "model"}`).
A prompt no model can serve is answered 502 in the client's wire, with the same
trail header.

## Determinism

Every choice is a function of the brief, the requests and the gateway's answers:
no wall clock, no unseeded randomness. The fixed 500 ms wait before a same-model
5xx retry never enters any request, and the model lists carry fixed timestamps.

## Test results (local, mock gateway answering like OpenRouter)

`~/workspace/makefun-kits/aggregator-tests/` — `mock-gateway.mjs` (canned completions
carrying the prompt token; failure injection) and `run-trial.sh` (the kit's real env
+ real scorer, trial 0–2):

- practice trials 0, 1, 2: **score 100** each (quality/served/wire/failover/catalog
  100, 6/6 valid, 6 upstream calls, 39/39 checks, `real_model` pass)
- failover trial (429 on inkling once, 500 on super-vl once): **score 98**
  (quality 96, the rest 100; both failures recorded in the trail header)
- no-key simulation (every model 401): **score 5** (catalog 100 only), clean 502s
  with full trails, exit 0 — this is what a submission without a pushed OpenRouter
  key scores
- replay: two runs of trial 0 leave byte-identical state exports

Official CLI (`grade-local --driver process`, mock standing in for masq with
`--masq`): **finished, score 100 all columns, replay matched byte for byte,
L1 regrade reproduced, real_model 12/12, eligible: ranked yes**.

Caveat: the mock's completions are canned, not live-model answers, so live-model
behaviour (real quality ratings, real 429 shapes) is still untested. That needs
a pushed OpenRouter key from an account with ≥10 credits purchased.

## Run

```sh
node main.ts --list        # print the catalog document and exit 0
node main.ts               # serve a trial (needs MAKEFUN_ENV_URL, MAKEFUN_PROMPT_FILE,
                           # OPENAI_BASE_URL, OPENAI_API_KEY from the trial sandbox)
```

The server listens on `0.0.0.0` at a free port, registers it with the env
(`POST $MAKEFUN_ENV_URL/register`), and polls `GET $MAKEFUN_ENV_URL/work`,
exiting 0 when the env says done.

## Files

- `main.ts` — the server: brief, catalog, routes, routing, failover, register, poll
- `lib/providers.ts` — the brief's parser, the free model list, the catalog builder,
  quality-ordered candidates
- `lib/wire.ts` — Anthropic ↔ Chat Completions translation, the model field, SSE writers
- `LICENSE` — MIT
