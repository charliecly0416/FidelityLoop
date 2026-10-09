# Paired trace schema, version 1

## Index and identity

`INDEX.json` contains separate campaign objects. `physical` and `predictions` list record ID, relative gzip path, SHA256, policy/window coordinates and repeat or replay-model labels. `source_sha256` lists opaque hashes of the source components used for that projection, without source paths. `pairs` explicitly joins physical and prediction IDs; a prediction may be reused by multiple physical repeats. A physical execution appears once. `excluded_attempts` preserves status/reason categories and source hashes for unusable attempts; its entries are not complete traces.

`record_id`, campaign, policy, window and replay model are archival labels. They are not performance orderings. E2's archived prediction `ppo_C30_selected` and physical `ppo_C30_raw` denote the same checkpoint/tensors and are normalized to the physical label, as documented in its index notes.

Request aliases are `r` followed by 24 hexadecimal characters derived from the original request ID. Uniqueness and identical request metadata are checked within each paired workload. Always join through the pair index, not through a presumed global request population.

## Record fields

Each gzip expands to one JSON object:

| Field | Meaning |
|---|---|
| `schema` | `fidelityloop-trace-v1` |
| `record_id`, `campaign`, `kind`, `policy`, `window` | Indexed identity; kind is physical or prediction |
| `replay_model`, `repeat` | Model for a prediction, repeat for a physical execution; the other is null |
| `horizon_s` | Observation cutoff, 2100 seconds |
| `prices` | Six numeric scenario prices: GPU second, startup event, shutdown event, API input/output token budget, offline deadline miss |
| `requests` | Full arrival denominator with request alias, online/offline type, arrival and deadline seconds, input/output token budgets, terminal status, completion seconds or null, route |
| `resource_intervals` | Per-slot occupied segments: `gpu`, `generation`, `start_s`, `end_s` |
| `ledger_events` | Chargeable event primitives: relative `time_s`, kind start/stop/api_accept, and request alias for API acceptance |
| `events` | Whitelisted lifecycle, request and control events; optional slot, generation, request alias, lifecycle state and executed target |
| `decisions` | Scheduled decision tick `time_s` and acted-upon capacity `target`; 2100 entries per record |
| `reference` | Accepted source P1+/P1 (when comparable), GPU seconds and total scenario cost for regression; not the recomputation inputs |

All times are seconds relative to observation start; setup may be negative and final release may exceed 2100. Slot IDs 0/1 and generation counters are logical identities, not hardware UUIDs. Routes are gpu0/gpu1/synthetic_api/none. Status is completed/censored (failed/unfinished are reserved). A completed request may still be late. The JSON carries token counts, never token IDs or response text.

## Accounting and feasibility

For each occupied segment `[a,b]`, charge `max(0,min(b,T)-max(a,0))` GPU seconds with `T=2100`. Physical segments follow launch-issued to verified release; prediction segments follow the native occupied lifecycle states. Publication preserves these source semantics and checks the resulting sum against the original ledger. This is resource occupation, not useful GPU compute time or energy.

Start/stop/API event charges use the half-open interval `0 <= t < T`; initial ready instances can occupy resources without a window startup charge. API cost is the accepted request's input count times the input price plus maximum output budget times the output price. Offline penalty is every late or unfinished offline arrival times its declared penalty. Setup/cleanup primitives are retained but excluded from these window totals. No provider price or answer-quality claim follows from this arithmetic.

A request is timely iff it has a completed terminal and `arrival <= completion <= min(deadline,T)`. Equality with a deadline is timely. Post-cutoff or absent completion counts as unfinished for this observation. P1+ requires every online and offline arrival timely. P1 requires every offline arrival timely, at least 99 percent of online arrivals timely, and no unfinished arrival. The older V3 native prediction `feasible` flag is not this P1 definition, so its reference P1 is null; strict outcomes are recomputed from request rows.

## Event and decision projection

The event stream retains relevant observed/simulated events rather than inventing a shared hidden simulator state. Its keys are limited to `time_s`, `kind`, `gpu`, `generation`, `request_id`, `state`, and `target`. Kinds retain documented source names such as launch_issued/launch, ready, lifecycle_state/state, shutdown_issued/shutdown, verified_release, request_release/release, dispatch/local_dispatch, request_terminal/complete, api_accept, policy_tick/decision, and observation boundaries. States are off/starting/active/draining/stopping. Optional fields are absent when a source does not provide them. Arbitrary worker messages and runtime metadata are omitted.

`decisions` uses scheduled ticks for comparable target sequences. Physical policy_tick events retain observed relative execution time where available, so scheduled time is not presented as a precisely observed actuation timestamp. These records permit sequence inspection; they omit neural logits, full policy observations, hidden state and model weights, and are not a turnkey policy-replay/training artifact.

## Scientific scope

Do not pool the campaigns or interpret 372 reused pairs as independent predictions. Technical exclusions and scientifically failed valid runs have different roles. The corpus measures these existing deployments; it is neither an unseen-workload benchmark nor a claim that arbitrary hardware, engines, output quality or real API performance are covered.
