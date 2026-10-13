# OpenRouter live test — hosted decision models

Run 2026-10-13 (UTC ~11:25–11:30) in a Claude Code cloud container, from `main` at
`e7fc1c9`. No key appears anywhere in this file; none was printed or logged.

## Summary

- **How the key reaches openrouter.ai here:** the environment's egress proxy
  injects it. `OPENROUTER_API_KEY` is **not set** in the container; requests
  sent with **no** `Authorization` header get HTTP 200. The earlier
  "401 — proxy key not reaching openrouter.ai" was not reproduced.
- `src/decide_local.py --dry-run` refuses to start without the env var
  (by design), so step 2 ran the Python equivalent via `guard.ask_http(..., key=None, ...)`.
- Both pinned models answer through the Decisions API, return their pinned
  model (dated build) and provider, and cost fractions of a cent.

| model (pin) | answered as | provider | direct call | tonight's state (run 1 / run 2) |
|---|---|---|---|---|
| perplexity/pplx-decider-v1.1-27b | perplexity/pplx-decider-v1.1-27b-20261006 | Perplexity | 200, 0.49 s, 101 in / 1 out tok, $0.00000202 | 6952 ms / 526 ms |
| typesafe/jev-1.13 | typesafe/jev-1.13-20260917 | TypeSafe | 200, 0.32 s, 288 in / 20 out tok, $0.000012096 | 301 ms / 399 ms |

## Step 1 — auth check

(a) Env var:

```
$ [ -n "${OPENROUTER_API_KEY+x}" ] && echo "set, length ${#OPENROUTER_API_KEY}" || echo unset
OPENROUTER_API_KEY unset
```

So the Bearer-header variant could not be tried (there is no key in the
environment to send).

(b) No `Authorization` header (proxy injection):

```
$ curl -sS -o key.json -w 'HTTP %{http_code} time %{time_total}s\n' https://openrouter.ai/api/v1/key
HTTP 200 time 0.568581s
```

Non-secret fields:

| field | value |
|---|---|
| label | a masked key prefix (`sk-or-v1-…`, as OpenRouter displays it; elided here) |
| limit | 100 |
| limit_remaining | 35.391478489 |
| usage | 64.608521511 |
| is_free_tier | False |

After all tests below: `limit_remaining` 35.386588084, `usage` 64.613411916
(Δ ≈ $0.0049 for everything in this file, Yahoo-state runs included).

## Step 2 — tonight's state through both hosted backends

Installed: `pip install numpy pandas yfinance lxml` (yfinance was missing).

The CLI, as asked:

```
$ QQQ_DECIDER_HOME=/tmp/qqq-decider python src/decide_local.py --dry-run
stopped: set OPENROUTER_API_KEY in this machine's environment (never in a file in the repo), or run --engine local
set OPENROUTER_API_KEY in this machine's environment (never in a file in the repo), or run --engine local
exit 1
```

It exits before any network call (`run()` requires the env var for
`--engine hosted`). Since the key works only through proxy injection, the
equivalent was run instead, twice:

```python
import sys, time, json
sys.path.insert(0, 'src')
import guard, decide_local
today, names, state = guard.tonight()
for b, model in decide_local.HOSTED.items():
    p, who, ms = guard.ask_http(state, names, decide_local.DECISIONS, None, model)
    print(b, model, who, ms, p)
```

(`QQQ_DECIDER_HOME=/tmp/qqq-decider`; nothing written to `reports/`; this
skips `hosted_ask`'s pin check, but every `answered_as` below is a dated
build of its pin, so it would have passed.)

```
=== run 1
tonight(): today=20261013 names=11 (MU, BE, STX, SNDK, NBIS, IREN, LRCX, WDC, AMAT, INTC, CAT) state_bytes=1416 build_s=10.7
[http] pin=perplexity/pplx-decider-v1.1-27b answered_as=perplexity/pplx-decider-v1.1-27b-20261006@Perplexity ms=6952 wall_s=6.95
   probs={"MU": 0.514, "BE": 0.485, "STX": 0.61, "SNDK": 0.587, "NBIS": 0.476, "IREN": 0.587, "LRCX": 0.534, "WDC": 0.51, "AMAT": 0.464, "INTC": 0.586, "CAT": 0.481, "BOOK": 0.558}
[jev] pin=typesafe/jev-1.13 answered_as=typesafe/jev-1.13-20260917@TypeSafe ms=301 wall_s=0.30
   probs={"MU": 0.5, "BE": 0.54, "STX": 0.49, "SNDK": 0.46, "NBIS": 0.64, "IREN": 0.39, "LRCX": 0.74, "WDC": 0.61, "AMAT": 0.69, "INTC": 0.56, "CAT": 0.75, "BOOK": 0.35}
=== run 2
tonight(): today=20261013 names=11 (MU, BE, STX, SNDK, NBIS, IREN, LRCX, WDC, AMAT, INTC, CAT) state_bytes=1416 build_s=9.0
[http] pin=perplexity/pplx-decider-v1.1-27b answered_as=perplexity/pplx-decider-v1.1-27b-20261006@Perplexity ms=526 wall_s=0.53
   probs={"MU": 0.514, "BE": 0.485, "STX": 0.61, "SNDK": 0.587, "NBIS": 0.476, "IREN": 0.587, "LRCX": 0.534, "WDC": 0.51, "AMAT": 0.464, "INTC": 0.586, "CAT": 0.481, "BOOK": 0.558}
[jev] pin=typesafe/jev-1.13 answered_as=typesafe/jev-1.13-20260917@TypeSafe ms=399 wall_s=0.40
   probs={"MU": 0.47, "BE": 0.5, "STX": 0.44, "SNDK": 0.42, "NBIS": 0.62, "IREN": 0.38, "LRCX": 0.68, "WDC": 0.63, "AMAT": 0.69, "INTC": 0.59, "CAT": 0.76, "BOOK": 0.37}
```

Observations: the Perplexity Decider is deterministic across the two runs
(identical probabilities); its first call took ~7 s (likely a cold start),
the second 0.5 s. Jev varies by up to ±0.06 between runs and answers in
0.3–0.4 s. The two disagree substantially (e.g. CAT 0.48 vs 0.75, LRCX 0.53
vs 0.68–0.74, BOOK 0.56 vs 0.35–0.37); the decider's probabilities sit in a
narrow 0.46–0.61 band.

## Step 3 — one direct request per model

```
$ curl -sS -w 'HTTP %{http_code} wall %{time_total}s\n' -H 'Content-Type: application/json' \
    -d '{"model": "<model>", "state": "Overnight test: a stock fell 3% today on no news.", "questions": {"q": {"type": "noul", "instructions": "Should it be held overnight?"}}}' \
    https://openrouter.ai/api/alpha/decisions
```

perplexity/pplx-decider-v1.1-27b:

```
HTTP 200 wall 0.491236s
{"model":"perplexity/pplx-decider-v1.1-27b-20261006","answers":{"q":{"type":"noul","noul":0.37719060594125126}},"usage":{"input_tokens":101,"output_tokens":1,"cost":0.00000202},"id":"gen-dec-1791890984-gJNnD7SN1orzkrASTs05","provider":"Perplexity"}
```

typesafe/jev-1.13:

```
HTTP 200 wall 0.320260s
{"model":"typesafe/jev-1.13-20260917","answers":{"q":{"type":"noul","noul":0.48}},"usage":{"input_tokens":288,"output_tokens":20,"cost":0.000012096},"id":"gen-dec-1791890984-gi7hPf6XuHxpTnifCW22","provider":"TypeSafe"}
```

## Errors

None from OpenRouter. The only failure was the CLI's own pre-flight refusal
quoted in step 2 (no `OPENROUTER_API_KEY` in this container's environment).

## Implication for `decide_local.py`

In this container the key exists only in the proxy, so `decide_local.py`
cannot run the hosted engine as written: `run()` raises before any request
when the env var is unset. On the owner's machine, with the key exported, the
code path is unchanged. To run it in a proxy-injected environment, the env-var
check would need an opt-out (e.g. accept a missing key when an explicit flag
says the proxy authenticates). That change was not made here.
