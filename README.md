[uv](https://docs.astral.sh/uv/#installation)
[git (Windows 10/11)](https://github.com/darkshapes/sdbx/wiki/_Setup-:-Git-%E2%80%90Windows-only%E2%80%90)

```
git clone https://github.com/darkshapes/divisor
cd divisor
uv sync --dev
dvzr
```

```
usage: divisor --model-type dev --quantization <args>

Divisor CLI - Flux image generation and multimodal understanding

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

```

```
