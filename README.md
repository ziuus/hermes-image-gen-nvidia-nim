# hermes-image-gen-nvidia-nim

NVIDIA NIM image generation backend for [Hermes Agent](https://github.com/NousResearch/hermes-agent).

Adds NVIDIA NIM as an image generation provider in `hermes`, using NVIDIA's
hosted OpenAI-compatible API (`integrate.api.nvidia.com`) or a self-hosted NIM
container. Includes FLUX, Qwen-Image, and Stable Diffusion 3.5 models, all
usable with NVIDIA's **free API credits** (no credit card required).

- Free to start — get a key at [build.nvidia.com](https://build.nvidia.com)
- Text-to-image **and** image-to-image editing
- Hosted or self-hosted base URL
- Integrates with `hermes tools` setup UI (`badge: free credits`)

## Install

The plugin ships as a standalone repo per Hermes policy (third-party product
integrations are not merged into the Hermes tree). Two install options:

### Option A — pip entry point (recommended)

```bash
pip install hermes-image-gen-nvidia-nim
```

Hermes discovers the plugin via the `hermes_agent.plugins` entry-point group
and shows it under `hermes plugins list`.

### Option B — copy into `~/.hermes/plugins`

```bash
mkdir -p ~/.hermes/plugins/image_gen
cp -r hermes_image_gen_nvidia_nim/plugins/image_gen/nvidia_nim \
      ~/.hermes/plugins/image_gen/nvidia_nim
```

## Configure

Set your NVIDIA API key in `~/.hermes/.env` (or run `hermes tools` → Image
Generation → NVIDIA NIM):

```bash
NVIDIA_API_KEY=nvapi-...
```

Optional:

```bash
# Self-hosted NIM container (defaults to https://integrate.api.nvidia.com/v1)
NVIDIA_BASE_URL=http://localhost:8000/v1
```

### Model selection

Priority: `NVIDIA_IMAGE_MODEL` env var → `image_gen.nvidia_nim.model` in
`config.yaml` → `image_gen.model` in `config.yaml` → default
(`black-forest-labs/flux.1-schnell`).

## Models

| Model | Speed | Notes |
| --- | --- | --- |
| `black-forest-labs/flux.1-schnell` | ~4 steps | default; fast, excellent quality |
| `black-forest-labs/flux.1-dev` | ~20-50 steps | maximum quality |
| `black-forest-labs/flux.2-klein-4b` | ~4 steps | lightweight, image-to-image |
| `stabilityai/stable-diffusion-3.5-large` | ~50 steps | high quality, editing |
| `qwen/qwen-image` | ~20-50 steps | multilingual text rendering |
| `qwen/qwen-image-2512` | ~20-50 steps | updated Qwen-Image |
| `qwen/qwen-image-edit` | ~20-50 steps | image-to-image editing |

## Usage

Once enabled, ask Hermes to generate an image. For example:

```bash
hermes -z "generate an image of a cat on a skateboard"
```

Editing models also accept an input image (`image_url` / references).

## Development

Tests require a hermes-agent checkout on `sys.path` because the provider
implements Hermes' `ImageGenProvider` ABC:

```bash
pip install -e ".[dev]"
HERMES_AGENT_REPO=/path/to/hermes-agent pytest
```

## License

Apache-2.0
