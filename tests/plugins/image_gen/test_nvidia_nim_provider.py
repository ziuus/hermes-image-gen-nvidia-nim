"""Tests for the NVIDIA NIM image_gen plugin (standalone package).

These tests assert behavior and catalog invariants, not frozen snapshots of
model names or counts. Adding or removing a model must not break the suite.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

nvidia_nim_plugin = importlib.import_module(
    "hermes_image_gen_nvidia_nim.plugins.image_gen.nvidia_nim"
)


_PNG_HEX = (
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000d49444154789c6300010000000500010d0a2db40000000049454e44"
    "ae426082"
)


def _b64_png() -> str:
    import base64

    return base64.b64encode(bytes.fromhex(_PNG_HEX)).decode()


def _fake_response(*, b64=None, url=None, revised_prompt=None):
    item = SimpleNamespace(b64_json=b64, url=url, revised_prompt=revised_prompt)
    return SimpleNamespace(data=[item])


@pytest.fixture(autouse=True)
def _tmp_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield tmp_path


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    return nvidia_nim_plugin.NvidiaNimImageGenProvider()


def _catalog_ids():
    return [
        m["id"] for m in nvidia_nim_plugin.NvidiaNimImageGenProvider().list_models()
    ]


class TestMetadata:
    def test_name(self, provider):
        assert provider.name == "nvidia-nim"
        assert provider.display_name == "NVIDIA NIM"

    def test_default_model_is_catalog_key(self, provider):
        # Invariant: the default model is always a resolvable catalog entry.
        assert provider.default_model() in _catalog_ids()
        _, meta = nvidia_nim_plugin._resolve_model()
        assert meta in nvidia_nim_plugin._MODELS.values()

    def test_catalog_nonempty(self, provider):
        # Behavior: the catalog plumbing produces at least one model.
        assert len(provider.list_models()) >= 1

    def test_catalog_entries_have_fields(self, provider):
        for entry in provider.list_models():
            assert entry["display"]
            assert entry["speed"]
            assert entry["strengths"]
            assert entry["price"] == "free credits"

    def test_catalog_key_matches_api_model(self, provider):
        # Invariant: every advertised id resolves to the API model sent to NIM.
        for model_id, meta in nvidia_nim_plugin._MODELS.items():
            assert meta["api_model"] == model_id

    def test_mixed_editing_support(self, provider):
        # Invariant: at least one model supports image-to-image editing and
        # at least one does not (both code paths stay exercised).
        supports = [
            model_id
            for model_id, meta in nvidia_nim_plugin._MODELS.items()
            if meta.get("supports_editing")
        ]
        does_not = [
            model_id
            for model_id, meta in nvidia_nim_plugin._MODELS.items()
            if not meta.get("supports_editing")
        ]
        assert supports
        assert does_not


class TestAvailability:
    def test_no_api_key_unavailable(self, monkeypatch):
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        assert nvidia_nim_plugin.NvidiaNimImageGenProvider().is_available() is False

    def test_api_key_set_available(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
        assert nvidia_nim_plugin.NvidiaNimImageGenProvider().is_available() is True


class TestModelResolution:
    def test_env_var_override(self, monkeypatch):
        target = _catalog_ids()[0]
        monkeypatch.setenv("NVIDIA_IMAGE_MODEL", target)
        model_id, meta = nvidia_nim_plugin._resolve_model()
        assert model_id == target
        assert meta["api_model"] == target

    def test_config_model(self, tmp_path):
        import yaml

        target = _catalog_ids()[1]
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"image_gen": {"nvidia_nim": {"model": target}}})
        )
        model_id, _ = nvidia_nim_plugin._resolve_model()
        assert model_id == target

    def test_fallback_is_default(self, provider):
        model_id, _ = nvidia_nim_plugin._resolve_model()
        assert model_id == provider.default_model()


class TestGenerate:
    def test_requires_prompt(self, provider):
        res = provider.generate("")
        assert res["success"] is False
        assert res["error_type"] == "invalid_argument"

    def test_requires_key(self, monkeypatch):
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        provider = nvidia_nim_plugin.NvidiaNimImageGenProvider()
        res = provider.generate("a cat")
        assert res["success"] is False
        assert res["error_type"] == "auth_required"

    # Self-hosted path (NVIDIA_BASE_URL set → OpenAI-compatible SDK)
    @patch.object(nvidia_nim_plugin, "save_b64_image")
    def test_generate_success_self_hosted(self, mock_save, provider, monkeypatch):
        monkeypatch.setenv("NVIDIA_BASE_URL", "http://localhost:8000/v1")
        mock_client = MagicMock()
        mock_client.images.generate.return_value = _fake_response(b64=_b64_png())
        mock_save.return_value = "/tmp/cache/img.png"

        fake_openai = MagicMock()
        fake_openai.OpenAI.return_value = mock_client

        with patch.dict("sys.modules", {"openai": fake_openai}):
            res = provider.generate("a red apple")
            assert res["success"] is True
            assert res["provider"] == "nvidia-nim"
            assert res["modality"] == "text"
            assert res["image"] == "/tmp/cache/img.png"

    # Hosted genai path (no NVIDIA_BASE_URL → requests to ai.api.nvidia.com)
    @patch.object(nvidia_nim_plugin, "save_b64_image")
    def test_generate_success_hosted(self, mock_save, provider):
        mock_save.return_value = "/tmp/cache/img.png"
        resp = SimpleNamespace(
            status_code=200,
            text="",
            json=lambda: {
                "artifacts": [{"base64": _b64_png(), "finishReason": "SUCCESS"}]
            },
        )
        with patch("requests.post", return_value=resp) as mock_post:
            res = provider.generate("a red apple")
            assert res["success"] is True
            assert res["provider"] == "nvidia-nim"
            assert res["modality"] == "text"
            assert res["image"] == "/tmp/cache/img.png"
            assert mock_post.call_args.args[0].startswith(
                "https://ai.api.nvidia.com/v1/genai/"
            )

    def test_generate_editing_unsupported(self, provider, monkeypatch):
        editing_model = next(
            m
            for m, meta in nvidia_nim_plugin._MODELS.items()
            if not meta["supports_editing"]
        )
        monkeypatch.setenv("NVIDIA_IMAGE_MODEL", editing_model)
        res = provider.generate("edit", image_url="https://example.com/a.png")
        assert res["success"] is False
        assert res["error_type"] == "model_mismatch"

    @patch.object(nvidia_nim_plugin, "save_b64_image")
    def test_generate_editing_supported_self_hosted(
        self, mock_save, provider, monkeypatch
    ):
        editing_model = next(
            m
            for m, meta in nvidia_nim_plugin._MODELS.items()
            if meta["supports_editing"]
        )
        monkeypatch.setenv("NVIDIA_IMAGE_MODEL", editing_model)
        monkeypatch.setenv("NVIDIA_BASE_URL", "http://localhost:8000/v1")

        mock_client = MagicMock()
        mock_client.images.edit.return_value = _fake_response(b64=_b64_png())
        mock_save.return_value = "/tmp/cache/img_edit.png"

        fake_openai = MagicMock()
        fake_openai.OpenAI.return_value = mock_client

        with patch.dict("sys.modules", {"openai": fake_openai}):
            res = provider.generate("edit this", image_url="data:image/png;base64,AA==")
            assert res["success"] is True
            assert res["provider"] == "nvidia-nim"
            assert res["modality"] == "image"
            assert res["model"] == editing_model
            assert res["image"] == "/tmp/cache/img_edit.png"
            mock_client.images.edit.assert_called_once()

    @patch.object(nvidia_nim_plugin, "save_b64_image")
    def test_generate_editing_supported_hosted(self, mock_save, provider, monkeypatch):
        editing_model = next(
            m
            for m, meta in nvidia_nim_plugin._MODELS.items()
            if meta["supports_editing"]
        )
        monkeypatch.setenv("NVIDIA_IMAGE_MODEL", editing_model)

        mock_save.return_value = "/tmp/cache/img_edit.png"
        resp = SimpleNamespace(
            status_code=200,
            text="",
            json=lambda: {"artifacts": [{"base64": _b64_png()}]},
        )
        with patch("requests.post", return_value=resp) as mock_post:
            res = provider.generate("edit this", image_url="data:image/png;base64,AA==")
            assert res["success"] is True
            assert res["provider"] == "nvidia-nim"
            assert res["modality"] == "image"
            assert res["model"] == editing_model
            assert res["image"] == "/tmp/cache/img_edit.png"
            sent = mock_post.call_args.kwargs["json"]
            assert sent.get("mode") == "Image Editing"
            assert sent["image"] == ["data:image/png;base64,AA=="]


class TestHostedPayload:
    def test_flux_text_to_image(self):
        payload = nvidia_nim_plugin._build_hosted_payload(
            "black-forest-labs/flux.1-schnell", "cat", "square"
        )
        assert payload["prompt"] == "cat"
        assert payload["width"] == 1024
        assert payload["height"] == 1024
        assert payload["steps"] == 4
        assert "mode" not in payload
        assert "image" not in payload

    def test_flux2_edit_mode(self):
        payload = nvidia_nim_plugin._build_hosted_payload(
            "black-forest-labs/flux.2-klein-4b",
            "make it red",
            "landscape",
            sources=["data:image/png;base64,AA=="],
        )
        assert payload["mode"] == "Image Editing"
        assert payload["image"] == ["data:image/png;base64,AA=="]
        assert payload["width"] == 1344
        assert payload["height"] == 768

    def test_flux2_generation_mode(self):
        payload = nvidia_nim_plugin._build_hosted_payload(
            "black-forest-labs/flux.2-klein-4b", "cat", "square"
        )
        assert payload["mode"] == "Image Generation"

    def test_sd35_payload(self):
        payload = nvidia_nim_plugin._build_hosted_payload(
            "stabilityai/stable-diffusion-3.5-large", "cat", "portrait"
        )
        assert payload["steps"] == 50
        assert payload["cfg_scale"] == 5.0
        assert payload["width"] == 768
        assert payload["height"] == 1344

    def test_qwen_edit_image_string(self):
        payload = nvidia_nim_plugin._build_hosted_payload(
            "qwen/qwen-image-edit",
            "remove the hat",
            "square",
            sources=["data:image/png;base64,AA=="],
        )
        assert payload["image"] == "data:image/png;base64,AA=="


class TestHostedExtract:
    def test_artifacts_base64(self):
        b64, url = nvidia_nim_plugin._extract_hosted_image(
            {"artifacts": [{"base64": "ABC", "finishReason": "SUCCESS"}]}
        )
        assert b64 == "ABC"
        assert url is None

    def test_image_string(self):
        b64, url = nvidia_nim_plugin._extract_hosted_image({"image": "ABC"})
        assert b64 == "ABC"
        assert url is None

    def test_openai_data_b64_json(self):
        b64, url = nvidia_nim_plugin._extract_hosted_image(
            {"data": [{"b64_json": "ABC"}]}
        )
        assert b64 == "ABC"
        assert url is None

    def test_openai_data_url(self):
        b64, url = nvidia_nim_plugin._extract_hosted_image(
            {"data": [{"url": "https://example.com/i.png"}]}
        )
        assert b64 is None
        assert url == "https://example.com/i.png"

    def test_empty(self):
        b64, url = nvidia_nim_plugin._extract_hosted_image({})
        assert b64 is None
        assert url is None


class TestRegistration:
    def test_register(self):
        ctx = MagicMock()
        nvidia_nim_plugin.register(ctx)
        ctx.register_image_gen_provider.assert_called_once()
        instance = ctx.register_image_gen_provider.call_args[0][0]
        assert isinstance(instance, nvidia_nim_plugin.NvidiaNimImageGenProvider)
