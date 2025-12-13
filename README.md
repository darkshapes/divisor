<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/darkshapes/entity-statement/refs/heads/main/png/divisor/divisor_300.png">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/darkshapes/entity-statement/refs/heads/main/png/divisor/divisor_300_light.png">
  <img alt="Cryptic square shaped letters forming the word DIVISOR at a 45 degree slant" src="https://raw.githubusercontent.com/darkshapes/entity-statement/refs/heads/main/png/divisor/divisor_300.png">
</picture><br><br>

## Divisor <br>

Divisor is a framework for hands-on procedural generation. It offers precise control over advanced diffusion model processes, enabling faster and easier experimentation on neural networks with off-the-shelf computers. Requires minimum 8GB+ Video RAM on Nvidia or M-series GPU.

<sub>MacOS, Linux, Windows PC [[ UV](https://docs.astral.sh/uv/#installation) ] [ [Git on Windows](https://github.com/darkshapes/sdbx/wiki/_Setup-:-Git-%E2%80%90Windows-only%E2%80%90) ]</sub>

```bash
git clone https://github.com/darkshapes/divisor
cd divisor
uv sync --dev
```

<sub>MacOS/Linux</sub>

```bash
source .venv/bin/activate
```

<sub>Windows</sub>

```powershell
Set-ExecutionPolicy Bypass -Scope Process -Force; .venv\Scripts\Activate.ps1
```

<hr>

```powershell
dvzr
usage: divisor --model-type dev --quantization <args>

Divisor Multimodal CLI

options:
  -h, --help            show this help message and exit
  --quantization        Enable quantization (fp8, e5m2, e4m3fn) for the model
  -m, --model-type {flux1-dev,mini,flux1-schnell,flux2-dev,mmada,mixcot}
                        Model type to use: ['flux1-dev', 'mini', 'flux1-schnell', 'flux2-dev', 'mmada', 'mixcot'], Default:
                        flux1-dev

Valid arguments : --ae_id, --width, --height, --guidance, --seed, --prompt, --tiny, --device, --num_steps, --loop, --offload, --compile, --verbose
```

[![dvzr pytest](https://github.com/darkshapes/divisor/actions/workflows/divisor.yml/badge.svg)](https://github.com/darkshapes/divisor/actions/workflows/divisor.yml)<br>
[<img src="https://img.shields.io/badge/me-__?logo=kofi&logoColor=white&logoSize=auto&label=feed&labelColor=maroon&color=grey&link=https%3A%2F%2Fko-fi.com%2Fdarkshapes">](https://ko-fi.com/darkshapes)<br>
