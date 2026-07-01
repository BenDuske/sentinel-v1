# Third-Party Notices — Sentinel v1

Sentinel v1 is MIT-licensed (see `LICENSE`). It depends on the following third-party
components, all under permissive licenses compatible with MIT. Record any new dependency
here with its license before release.

| Component | Purpose | License |
|-----------|---------|---------|
| FastAPI | web/API framework | MIT |
| Starlette | ASGI toolkit (FastAPI dep) | BSD-3-Clause |
| Pydantic | data validation (FastAPI dep) | MIT |
| Uvicorn | ASGI server | BSD-3-Clause |
| python-multipart | file uploads | Apache-2.0 |
| reportlab | PDF report export | BSD-3-Clause |
| Pillow | imaging (reportlab dependency) | HPND / MIT-CMU (PIL license) |
| charset-normalizer | text-encoding detection (reportlab dependency) | MIT |

## Bundled license texts
Full license texts for the PDF-export dependencies are included in [`licenses/`](licenses/):
`reportlab-LICENSE.txt` (BSD), `Pillow-LICENSE.txt` (HPND/PIL), `charset-normalizer-LICENSE.txt` (MIT).
All are permissive and MIT-compatible; attribution is preserved per each license.

## Models / external
- The LLM runs locally via **Ollama** (MIT) using an open **Qwen** model (Apache-2.0 for the
  open Qwen series) — or any OpenAI-compatible endpoint you configure. You supply the model;
  its license governs its use. No model weights are bundled in this repository.
