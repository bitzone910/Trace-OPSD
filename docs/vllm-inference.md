# vLLM adapter

This directory is Trace-OPSD's narrow integration boundary for vLLM. It is
not a vendored copy of the vLLM engine or Search-R1's legacy veRL wrapper.

- `client.py`: asynchronous HTTP client for `POST /v1/completions`.
- `server.py`: converts the validated rollout YAML into server launch args.
- `health.py`: checks `/health`, `/v1/models`, and the configured model name.

The rollout environment pins `vllm==0.11.0`. The engine runs as a separate
local process and only listens on `127.0.0.1`; rollout code communicates with
it through the OpenAI-compatible HTTP API. Consequently this package has no
dependency on vLLM's private Python classes, Ray, veRL, FSDP, or DataProto.

Public imports:

```python
from trace_opsd.inference.vllm import VLLMClient, build_server_spec, check_vllm_health
```

The implementation is shared with `trace_opsd.rollout.clients`. Tool-service
HTTP clients remain in `trace_opsd/rollout/clients.py` because they are not
vLLM adapters.
