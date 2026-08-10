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

    @patch.object(nvidia_nim_plugin, "save_b64_image")
    def test_generate_success(self, mock_save, provider):
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
    def test_generate_editing_supported(self, mock_save, provider, monkeypatch):
        editing_model = next(
            m
            for m, meta in nvidia_nim_plugin._MODELS.items()
            if meta["supports_editing"]
        )
        monkeypatch.setenv("NVIDIA_IMAGE_MODEL", editing_model)

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


class TestRegistration:
    def test_register(self):
        ctx = MagicMock()
        nvidia_nim_plugin.register(ctx)
        ctx.register_image_gen_provider.assert_called_once()
        instance = ctx.register_image_gen_provider.call_args[0][0]
        assert isinstance(instance, nvidia_nim_plugin.NvidiaNimImageGenProvider)
