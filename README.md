## divisor

> Hands-on procedural generation.

> Divisor is a comprehensive framework for flexible, controllable media generation using state‑of‑the‑art diffusion models such as [Flux](https://github.com/black-forest-labs/flux2) and [MMaDA](https://github.com/Gen-Verse/MMaDA). It gives developers, researchers, and artists fine‑grained, programmatic control over the denoising process that underlies modern generative synthesis, enabling both experimental research and production‑grade workflows approaching real-time speed on consumer hardware.

#### Features:

> - Manual Timestep Control – Step through diffusion timesteps, enabling acute control via dynamic prompt changes, layer‑wise manipulations, and on‑the‑fly parameter tuning.
> - Multimodal - Synthesize a wide variety of diffusion content such as text and images
> - Model‑Agnostic Architecture – Unified utilities abstract inner workings, allowing interchangeable components such as custom LoRA and autoencoders.
> - Extensible Prompt Engineering – Dedicated prompt modules support multi‑modal inputs, system messages, and automatic parsing for LLM‑driven results.
> - Robust State Management & Serialization – Serialize and restore the full generation state (seeds, dropout masks, VAE offsets) for reproducibility and pause‑resume workflows.
> - Fine‑Grained Noise & Variation Controls – Deterministic and stochastic variation mechanisms (linear, cosine, etc.) to blend latents or create consistent variations.
> - Integration with External Resources – Fetches model weights, adapters, and [MIR](https://github.com/darkshapes/mir) specs, ensuring rapid and reproducible setups.
> - User‑Facing Interfaces – Entry points provide CLI/script interfaces ready for use or to integrate into apps that assemble pipelines, adjust parameters, and render results.

#### Requires:<hr>

> **Windows/MacOS/Linux device**
>
> **Nvidia graphics card or M-series chip with 8GB+ VRAM**. AMD support untested.
>
> [UV](https://docs.astral.sh/uv/#installation)
>
> [Git (Windows 10/11)](https://github.com/darkshapes/sdbx/wiki/_Setup-:-Git-%E2%80%90Windows-only%E2%80%90)<br>

#### Install:<hr>

```bash
git clone https://github.com/darkshapes/divisor
cd divisor
uv sync --dev
```

> Linux/Macos

```bash
source .venv/bin/activate
```

> Windows:

```Powershell
Set-ExecutionPolicy Bypass -Scope Process -Force; .venv\Scripts\Activate.ps1
```

#### Run:<hr>

```
dvzr
```

```
usage: divisor --model-type dev --quantization <args>

divisor - low-level diffusion prototyping

options:
  -h, --help            show this help message and exit
  --quantization        Enable quantization (fp8, e5m2, e4m3fn) for the model
  -m, --model-type {dev,schnell,dev2,mini,llm}
                        Model type to use: 'dev' (flux1-dev), 'schnell' (flux1-schnell), or 'dev2' (flux2-dev), 'mini' (flux1-mini). Default:
                        dev

Valid arguments : --ae_id, --width, --height, --guidance, --seed, --prompt, --tiny, --device, --num_steps, --loop, --offload, --compile,
--verbose
```

[![dvzr pytest](https://github.com/darkshapes/divisor/actions/workflows/divisor.yml/badge.svg)](https://github.com/darkshapes/divisor/actions/workflows/divisor.yml)<br>
[<img src="https://img.shields.io/badge/me-__?logo=kofi&logoColor=white&logoSize=auto&label=feed&labelColor=maroon&color=grey&link=https%3A%2F%2Fko-fi.com%2Fdarkshapes">](https://ko-fi.com/darkshapes)<br>
