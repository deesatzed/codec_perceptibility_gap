# CodecGuard local LLM prompt templates

Use these after the simulation bundle runs. They are optional; the core repo does not require Ollama.

## Prose codec

Answer the task in concise prose. Include the final answer and the key reasoning steps. Do not use JSON.

## Schema codec

Return valid JSON only with this shape:

```json
{
  "answer": "...",
  "claims": [
    {"subject":"...", "predicate":"...", "object":"...", "value": null, "unit": null, "polarity":"affirm", "confidence": 0.0}
  ]
}
```

## Executable-code codec

Return a single Python function named `solve()` that takes no arguments and returns:

```python
{"answer": ..., "claims": [...]}
```

The function must be deterministic and must not import network, filesystem, subprocess, or OS modules.

## Causal-graph codec

Return valid JSON only:

```json
{
  "nodes": [{"id":"n1", "label":"..."}],
  "edges": [{"source":"n1", "target":"n2", "relation":"causes|computes|supports|overrides"}],
  "claims": [
    {"subject":"...", "predicate":"...", "object":"...", "value": null, "unit": null, "polarity":"affirm", "confidence": 0.0}
  ]
}
```

## Scoring note

Parse all codecs into canonical claim atoms, then diff: numeric disagreement, unit mismatch, polarity flip, missing required claim, unsupported extra claim, non-executable code, invalid graph edge, or graph/prose causal mismatch.
