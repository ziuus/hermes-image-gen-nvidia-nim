"""NVIDIA NIM image generation backend for Hermes Agent.

Provides access to NVIDIA-hosted and self-hosted NIM visual generative AI models.

Two transport paths:

* **Hosted** (default, no ``NVIDIA_BASE_URL``) — calls the NVIDIA-hosted
  ``genai`` API at ``https://ai.api.nvidia.com/v1/genai/<publisher>/<model>``
  with the model-family-specific request schema. This is what build.nvidia.com
  free credits / hosted inference expose.
* **Self-hosted** (``NVIDIA_BASE_URL`` set) — talks to a local NIM container's
  OpenAI-compatible ``/v1/images/generations`` / ``/v1/images/edits`` via the
  ``openai`` SDK.

Models (hosted genai slugs):
    black-forest-labs/flux.1-schnell     ~4 steps   fast, excellent quality
    black-forest-labs/flux.1-dev         ~20-50    maximum quality
    black-forest-labs/flux.2-klein-4b    ~4 steps   fast, image-to-image
    stabilityai/stable-diffusion-3.5-large  ~50     high quality
    qwen/qwen-image                      ~20-50    multilingual text rendering
    qwen/qwen-image-2512                 ~20-50    updated Qwen-Image
    qwen/qwen-image-edit                 editing   image-to-image editing

All models return base64-encoded images. Images are cached locally under
$HERMES_HOME/cache/images/.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from agent.secret_scope import get_secret
from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    normalize_reference_images,
    resolve_aspect_ratio,
    save_b64_image,
    save_url_image,
    success_response,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model catalog
# ---------------------------------------------------------------------------
#
# The "api_model" is the slug sent to the API (hosted genai path or the
# self-hosted OpenAI-compatible /v1/images/generations endpoint).

_MODELS: Dict[str, Dict[str, Any]] = {
    "black-forest-labs/flux.1-schnell": {
        "display": "FLUX.1-schnell",
        "speed": "~4 steps (8-15s)",
        "strengths": "Fast, excellent quality — default for iteration",
        "price": "free credits",
        "api_model": "black-forest-labs/flux.1-schnell",
        "supports_editing": False,
    },
    "black-forest-labs/flux.1-dev": {
        "display": "FLUX.1-dev",
        "speed": "~20-50 steps (30-60s)",
        "strengths": "Maximum quality, best prompt adherence",
        "price": "free credits",
        "api_model": "black-forest-labs/flux.1-dev",
        "supports_editing": False,
    },
    "black-forest-labs/flux.2-klein-4b": {
        "display": "FLUX.2-klein-4B",
        "speed": "~4 steps (8-15s)",
        "strengths": "Fast, 4B params — lightweight, image-to-image",
        "price": "free credits",
        "api_model": "black-forest-labs/flux.2-klein-4b",
        "supports_editing": True,  # hosted mode="Image Editing"
    },
    "stabilityai/stable-diffusion-3.5-large": {
        "display": "Stable Diffusion 3.5 Large",
        "speed": "~50 steps (45-90s)",
        "strengths": "High quality, strong text rendering",
        "price": "free credits",
        "api_model": "stabilityai/stable-diffusion-3.5-large",
        # Editing is available on the self-hosted NIM path; the hosted genai
        # path gates editing separately via _HOSTED_EDIT_MODELS below.
        "supports_editing": True,
    },
    "qwen/qwen-image": {
        "display": "Qwen-Image",
        "speed": "~20-50 steps (30-60s)",
        "strengths": "Multilingual text rendering (EN/CN), precise editing",
        "price": "free credits",
        "api_model": "qwen/qwen-image",
        "supports_editing": False,
    },
    "qwen/qwen-image-2512": {
        "display": "Qwen-Image-2512",
        "speed": "~20-50 steps (30-60s)",
        "strengths": "Updated Qwen-Image, better quality",
        "price": "free credits",
        "api_model": "qwen/qwen-image-2512",
        "supports_editing": False,
    },
    "qwen/qwen-image-edit": {
        "display": "Qwen-Image-Edit",
        "speed": "~20-50 steps (30-60s)",
        "strengths": "Image-to-image editing, style transfer, object add/remove",
        "price": "free credits",
        "api_model": "qwen/qwen-image-edit",
        "supports_editing": True,
    },
}

DEFAULT_MODEL = "black-forest-labs/flux.1-schnell"

# OpenAI API size format for the self-hosted NIM container API.
_SIZES = {
    "landscape": "1536x1024",
    "square": "1024x1024",
    "portrait": "1024x1536",
}

# Hosted genai API has constrained width/height enums (max 1344). Map aspect
# ratios onto the allowed values (landscape/portrait = 16:9, square = 1:1).
_HOSTED_SIZES = {
    "landscape": (1344, 768),
    "square": (1024, 1024),
    "portrait": (768, 1344),
}

_HOSTED_GENAI_BASE = "https://ai.api.nvidia.com/v1/genai"

# Models whose hosted genai endpoint supports image-to-image editing.
_HOSTED_EDIT_MODELS = {
    "black-forest-labs/flux.2-klein-4b",
    "qwen/qwen-image-edit",
}


def _load_nvidia_config() -> Dict[str, Any]:
    """Read ``image_gen`` from config.yaml (returns {} on any failure)."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception as exc:
        logger.debug("Could not load image_gen config: %s", exc)
        return {}


def _resolve_model() -> Tuple[str, Dict[str, Any]]:
    """Decide which model to use and return ``(model_id, meta)``."""
    env_override = os.environ.get("NVIDIA_IMAGE_MODEL")
    if env_override and env_override in _MODELS:
        return env_override, _MODELS[env_override]

    cfg = _load_nvidia_config()
    nvidia_cfg = (
        cfg.get("nvidia_nim") if isinstance(cfg.get("nvidia_nim"), dict) else {}
    )
    candidate: Optional[str] = None
    if isinstance(nvidia_cfg, dict):
        value = nvidia_cfg.get("model")
        if isinstance(value, str) and value in _MODELS:
            candidate = value
    if candidate is None:
        top = cfg.get("model")
        if isinstance(top, str) and top in _MODELS:
            candidate = top

    if candidate is not None:
        return candidate, _MODELS[candidate]

    return DEFAULT_MODEL, _MODELS[DEFAULT_MODEL]


# ---------------------------------------------------------------------------
# Source-image loading (for image-to-image / edit)
# ---------------------------------------------------------------------------


def _load_image_bytes(ref: str) -> Tuple[bytes, str]:
    """Load image bytes from a URL or local file path.

    Returns ``(data, filename)``. Raises on any network / IO error so the
    caller can surface a clean error_response.
    """
    ref = ref.strip()
    lower = ref.lower()
    if lower.startswith(("http://", "https://")):
        import requests

        resp = requests.get(ref, timeout=60)
        resp.raise_for_status()
        name = ref.split("?", 1)[0].rsplit("/", 1)[-1] or "image.png"
        return resp.content, name
    if lower.startswith("data:"):
        import base64

        header, _, b64 = ref.partition(",")
        ext = "png"
        if "image/" in header:
            ext = header.split("image/", 1)[1].split(";", 1)[0] or "png"
        return base64.b64decode(b64), f"image.{ext}"
    # Local file path — enforce the shared credential-read guard before reading.
    from agent.file_safety import raise_if_read_blocked

    raise_if_read_blocked(ref)
    with open(ref, "rb") as fh:
        data = fh.read()
    name = os.path.basename(ref) or "image.png"
    return data, name


def _to_data_uri(ref: str) -> str:
    """Load a reference and return it as a ``data:image/*;base64,...`` URI."""
    import base64

    data, fname = _load_image_bytes(ref)
    ext = os.path.splitext(fname)[1].lstrip(".").lower() or "png"
    if ext == "jpg":
        ext = "jpeg"
    if ext not in ("png", "jpeg", "gif", "webp"):
        ext = "png"
    return f"data:image/{ext};base64,{base64.b64encode(data).decode()}"


# ---------------------------------------------------------------------------
# Hosted genai transport
# ---------------------------------------------------------------------------


def _build_hosted_payload(
    model_id: str,
    prompt: str,
    aspect: str,
    *,
    sources: Optional[List[str]] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Build the model-family-specific request body for the hosted genai API."""
    width, height = _HOSTED_SIZES.get(aspect, _HOSTED_SIZES["square"])
    is_edit = bool(sources)
    payload: Dict[str, Any] = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "seed": seed if seed is not None else 0,
    }

    if model_id.startswith("black-forest-labs/flux.2-klein"):
        payload["mode"] = "Image Editing" if is_edit else "Image Generation"
        payload["steps"] = 4
        if is_edit:
            payload["image"] = [sources[0]]
    elif model_id.startswith("black-forest-labs/"):
        # FLUX.1 schnell/dev — hosted only supports text-to-image (mode "base").
        payload["steps"] = 4
    elif model_id.startswith("stabilityai/"):
        payload["steps"] = 50
        payload["cfg_scale"] = 5.0
    else:
        # qwen/qwen-image, qwen-image-2512, qwen-image-edit
        if is_edit:
            payload["image"] = sources[0]

    return payload


def _extract_hosted_image(response_body: Any) -> Tuple[Optional[str], Optional[str]]:
    """Pull ``(b64, url)`` from a hosted genai / OpenAI-compatible response.

    Handles the shapes returned by NIM hosted genai (``artifacts[].base64``,
    single ``image`` b64 string) and OpenAI-compatible paths
    (``data[].b64_json`` / ``data[].url``).
    """
    if not isinstance(response_body, dict):
        return None, None

    artifacts = response_body.get("artifacts")
    if isinstance(artifacts, list) and artifacts:
        first = artifacts[0]
        if isinstance(first, dict):
            b64 = first.get("base64")
            if b64:
                return b64, None

    image = response_body.get("image")
    if isinstance(image, str) and image:
        return image, None

    data = response_body.get("data")
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            b64 = first.get("b64_json")
            url = first.get("url")
            return (b64 or None), (url or None)

    return None, None


def _hosted_generate(
    api_key: str,
    model_id: str,
    prompt: str,
    aspect: str,
    sources: List[str],
    seed: Optional[int],
) -> Dict[str, Any]:
    """Call the hosted genai API directly (requests-based)."""
    import requests

    data_uris: List[str] = []
    if sources:
        try:
            data_uris = [_to_data_uri(ref) for ref in sources]
        except Exception as exc:
            return error_response(
                error=f"Could not load source image for editing: {exc}",
                error_type="io_error",
                provider="nvidia-nim",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

    url = f"{_HOSTED_GENAI_BASE}/{model_id}"
    payload = _build_hosted_payload(
        model_id, prompt, aspect, sources=data_uris, seed=seed
    )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=300)
    except Exception as exc:
        logger.debug("NVIDIA NIM hosted image generation failed", exc_info=True)
        return error_response(
            error=f"NVIDIA NIM hosted image generation failed: {exc}",
            error_type="api_error",
            provider="nvidia-nim",
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
        )

    if resp.status_code != 200:
        detail = resp.text[:500]
        if resp.status_code in (401, 403):
            err_type = "auth_error"
        elif resp.status_code == 404:
            err_type = "model_not_found"
        elif resp.status_code == 429:
            err_type = "rate_limited"
        else:
            err_type = "api_error"
        logger.debug(
            "NVIDIA NIM hosted image generation failed: %s %s",
            resp.status_code,
            detail,
        )
        return error_response(
            error=(
                f"NVIDIA NIM hosted API returned {resp.status_code} for "
                f"{model_id}: {detail}"
            ),
            error_type=err_type,
            provider="nvidia-nim",
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
        )

    try:
        body = resp.json()
    except Exception as exc:
        return error_response(
            error=f"NVIDIA NIM hosted API returned invalid JSON: {exc}",
            error_type="api_error",
            provider="nvidia-nim",
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
        )

    b64, url = _extract_hosted_image(body)
    if b64:
        try:
            saved_path = save_b64_image(
                b64, prefix=f"nvidia_{model_id.replace('/', '_')}"
            )
        except Exception as exc:
            return error_response(
                error=f"Could not save image to cache: {exc}",
                error_type="io_error",
                provider="nvidia-nim",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        return success_response(
            image=str(saved_path),
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
            provider="nvidia-nim",
            modality="image" if sources else "text",
            extra={
                "size": f"{_HOSTED_SIZES.get(aspect, (1024, 1024))[0]}x{_HOSTED_SIZES.get(aspect, (1024, 1024))[1]}"
            },
        )
    if url:
        try:
            saved_path = save_url_image(
                url, prefix=f"nvidia_{model_id.replace('/', '_')}"
            )
        except Exception as exc:
            logger.warning(
                "NVIDIA NIM image URL %s could not be cached (%s); falling back to bare URL.",
                url,
                exc,
            )
            image_ref = url
        else:
            image_ref = str(saved_path)
        return success_response(
            image=image_ref,
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
            provider="nvidia-nim",
            modality="image" if sources else "text",
            extra={},
        )

    return error_response(
        error="NVIDIA NIM response contained neither base64 nor URL image data",
        error_type="empty_response",
        provider="nvidia-nim",
        model=model_id,
        prompt=prompt,
        aspect_ratio=aspect,
    )


# ---------------------------------------------------------------------------
# Self-hosted (OpenAI-compatible) transport
# ---------------------------------------------------------------------------


def _openai_generate(
    api_key: str,
    model_id: str,
    prompt: str,
    aspect: str,
    sources: List[str],
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Call a self-hosted NIM container via the OpenAI-compatible API."""
    try:
        import openai
    except ImportError:
        return error_response(
            error="openai Python package not installed (pip install openai)",
            error_type="missing_dependency",
            provider="nvidia-nim",
            aspect_ratio=aspect,
        )

    meta = _MODELS[model_id]
    api_model = meta["api_model"]
    size = _SIZES.get(aspect, _SIZES["square"])
    supports_editing = meta.get("supports_editing", False)

    if sources and not supports_editing:
        return error_response(
            error=(
                f"Model {model_id} does not support image editing on the "
                f"self-hosted NIM path. Use a model with editing support: "
                f"black-forest-labs/flux.2-klein-4b, qwen/qwen-image-edit"
            ),
            error_type="model_mismatch",
            provider="nvidia-nim",
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
        )

    base_url = (
        os.environ.get("NVIDIA_BASE_URL") or "https://integrate.api.nvidia.com/v1"
    )
    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    if sources:
        import io

        try:
            files = []
            for ref in sources:
                data, fname = _load_image_bytes(ref)
                bio = io.BytesIO(data)
                bio.name = fname
                files.append(bio)
        except Exception as exc:
            return error_response(
                error=f"Could not load source image for editing: {exc}",
                error_type="io_error",
                provider="nvidia-nim",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            response = client.images.edit(
                model=api_model,
                image=files if len(files) > 1 else files[0],
                prompt=prompt,
                size=size,
                n=1,
                response_format="b64_json",
            )
        except Exception as exc:
            logger.debug("NVIDIA NIM image edit failed", exc_info=True)
            return error_response(
                error=f"NVIDIA NIM image editing failed: {exc}",
                error_type="api_error",
                provider="nvidia-nim",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )
    else:
        payload: Dict[str, Any] = {
            "model": api_model,
            "prompt": prompt,
            "size": size,
            "n": 1,
            "response_format": "b64_json",
        }
        for opt in ("seed", "steps", "quality"):
            if opt in kwargs and kwargs[opt] is not None:
                payload[opt] = kwargs[opt]

        try:
            response = client.images.generate(**payload)
        except Exception as exc:
            logger.debug("NVIDIA NIM image generation failed", exc_info=True)
            return error_response(
                error=f"NVIDIA NIM image generation failed: {exc}",
                error_type="api_error",
                provider="nvidia-nim",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

    data = getattr(response, "data", None) or []
    if not data:
        return error_response(
            error="NVIDIA NIM returned no image data",
            error_type="empty_response",
            provider="nvidia-nim",
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
        )

    first = data[0]
    b64 = getattr(first, "b64_json", None)
    url = getattr(first, "url", None)
    revised_prompt = getattr(first, "revised_prompt", None)

    if b64:
        try:
            saved_path = save_b64_image(
                b64, prefix=f"nvidia_{model_id.replace('/', '_')}"
            )
        except Exception as exc:
            return error_response(
                error=f"Could not save image to cache: {exc}",
                error_type="io_error",
                provider="nvidia-nim",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        image_ref = str(saved_path)
    elif url:
        try:
            saved_path = save_url_image(
                url, prefix=f"nvidia_{model_id.replace('/', '_')}"
            )
        except Exception as exc:
            logger.warning(
                "NVIDIA NIM image URL %s could not be cached (%s); falling back to bare URL.",
                url,
                exc,
            )
            image_ref = url
        else:
            image_ref = str(saved_path)
    else:
        return error_response(
            error="NVIDIA NIM response contained neither b64_json nor URL",
            error_type="empty_response",
            provider="nvidia-nim",
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
        )

    extra: Dict[str, Any] = {"size": size}
    if revised_prompt:
        extra["revised_prompt"] = revised_prompt

    return success_response(
        image=image_ref,
        model=model_id,
        prompt=prompt,
        aspect_ratio=aspect,
        provider="nvidia-nim",
        modality="image" if sources else "text",
        extra=extra,
    )


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class NvidiaNimImageGenProvider(ImageGenProvider):
    """NVIDIA NIM image generation backend.

    Uses the hosted genai API (``https://ai.api.nvidia.com/v1/genai``) by
    default, or the OpenAI-compatible API of a self-hosted NIM container when
    ``NVIDIA_BASE_URL`` is set. Requires ``NVIDIA_API_KEY`` (``nvapi-...``
    from build.nvidia.com for hosted; any string for self-hosted).
    """

    @property
    def name(self) -> str:
        return "nvidia-nim"

    @property
    def display_name(self) -> str:
        return "NVIDIA NIM"

    def is_available(self) -> bool:
        if not get_secret("NVIDIA_API_KEY"):
            return False
        return True

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": model_id,
                "display": meta["display"],
                "speed": meta["speed"],
                "strengths": meta["strengths"],
                "price": meta["price"],
            }
            for model_id, meta in _MODELS.items()
        ]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "NVIDIA NIM",
            "badge": "free credits",
            "tag": "FLUX, Qwen-Image, SD3.5 via NVIDIA NIM (hosted or self-hosted)",
            "env_vars": [
                {
                    "key": "NVIDIA_API_KEY",
                    "prompt": "NVIDIA API key (nvapi-... from build.nvidia.com, or any for self-hosted)",
                    "url": "https://build.nvidia.com",
                },
                {
                    "key": "NVIDIA_BASE_URL",
                    "prompt": "Self-hosted NIM base URL (default: hosted ai.api.nvidia.com/v1/genai)",
                    "url": "https://docs.nvidia.com/nim/visual-genai/latest/getting-started.html",
                    "optional": True,
                },
            ],
        }

    def capabilities(self) -> Dict[str, Any]:
        return {"modalities": ["text", "image"], "max_reference_images": 16}

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        *,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)

        if not prompt:
            return error_response(
                error="Prompt is required and must be a non-empty string",
                error_type="invalid_argument",
                provider="nvidia-nim",
                aspect_ratio=aspect,
            )

        api_key = get_secret("NVIDIA_API_KEY")
        if not api_key:
            return error_response(
                error=(
                    "NVIDIA_API_KEY not set. Run `hermes tools` → Image "
                    "Generation → NVIDIA NIM to configure, or `hermes setup` "
                    "to add the key. Get a free key at https://build.nvidia.com"
                ),
                error_type="auth_required",
                provider="nvidia-nim",
                aspect_ratio=aspect,
            )

        model_id, _meta = _resolve_model()

        # Collect source images (primary + references) for image-to-image.
        sources: List[str] = []
        if isinstance(image_url, str) and image_url.strip():
            sources.append(image_url.strip())
        for ref in normalize_reference_images(reference_image_urls) or []:
            sources.append(ref)
        sources = sources[:16]  # cap at 16 like gpt-image-2

        # Gate editing on the selected transport.
        if sources:
            supports = _meta.get("supports_editing", False)
            hosted = not os.environ.get("NVIDIA_BASE_URL")
            if hosted and model_id not in _HOSTED_EDIT_MODELS:
                supports = False
            if not supports:
                return error_response(
                    error=(
                        f"Model {model_id} does not support image editing. "
                        f"Use a model with editing support: "
                        f"black-forest-labs/flux.2-klein-4b, "
                        f"qwen/qwen-image-edit"
                    ),
                    error_type="model_mismatch",
                    provider="nvidia-nim",
                    model=model_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )

        if os.environ.get("NVIDIA_BASE_URL"):
            return _openai_generate(api_key, model_id, prompt, aspect, sources, kwargs)
        return _hosted_generate(
            api_key,
            model_id,
            prompt,
            aspect,
            sources,
            seed=kwargs.get("seed"),
        )


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Plugin entry point — wire ``NvidiaNimImageGenProvider`` into the registry."""
    ctx.register_image_gen_provider(NvidiaNimImageGenProvider())
