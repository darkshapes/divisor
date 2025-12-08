## divisor

> Hands-on procedural generation.

> Divisor is a framework enabling flexible media creation using advanced diffusion models like [Flux](https://github.com/black-forest-labs/flux2) and [MMaDA](https://github.com/Gen-Verse/MMaDA). With only off-the-shelf computers, developers, researchers, and artists get fine-grained control over low-level generative processing, making it faster and easier than ever to experiment and develop intuition for the workings of neural networks.

#### Features:<hr>

> - Multimodal Creation - Actively sculpt any content such as text and images.
> - Robust Versioning – Pause, resume, save, or restore states with exacting reproducibility and reversibility.
> - Private & Safe - Compartmentalized and local first, so data never leaves your workspace.
> - Fine‑Grained Noise & Variation Controls – Branch variations to create diverse and consistent inspiration.
> - Integration with External Resources – Start quickly with batteries included: models, adapters, and [MIR](https://github.com/darkshapes/mir) specs.

#### Tech Specs:

> - Manual Timestep Control – Step-by-step processing of dynamic prompts, layer‑wise manipulations, and on‑the‑fly parameter changes.
> - Extensible Prompt Engineering – Dedicated multimodal prompting, system messages, and automatic parsing for LLM‑driven results.
> - Model‑Agnostic Architecture – Unified API abstraction allows interchangeable custom LoRA and autoencoders.
> - User‑Facing Interfaces – CLI and Gradio interfaces ready to use or attach to other apps.
> - Sensible Python Engineering - Uses modern tooling with minimal dependences

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
                        Model type to use: 'dev' (flux1-dev), 'schnell' (flux1-schnell), 'dev2' (flux2-dev), 'mini' (flux1-mini), 'llm' (MMaDA) Default:
                        dev

Valid arguments : --ae_id, --width, --height, --guidance, --seed, --prompt, --tiny, --device, --num_steps, --loop, --offload, --compile,
--verbose
```

[![dvzr pytest](https://github.com/darkshapes/divisor/actions/workflows/divisor.yml/badge.svg)](https://github.com/darkshapes/divisor/actions/workflows/divisor.yml)<br>
[<img src="https://img.shields.io/badge/me-__?logo=kofi&logoColor=white&logoSize=auto&label=feed&labelColor=maroon&color=grey&link=https%3A%2F%2Fko-fi.com%2Fdarkshapes">](https://ko-fi.com/darkshapes)<br>
