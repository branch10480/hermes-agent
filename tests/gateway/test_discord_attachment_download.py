"""Tests for Discord attachment downloads via the authenticated bot session.

Covers the three download paths (image / audio / document) in
``DiscordAdapter._handle_message()`` and the shared ``_cache_discord_*``
helpers. Verifies that:

- ``att.read()`` is preferred over the legacy URL-based downloaders so
  that Discord's CDN auth (and user-environment DNS quirks) can't block
  media caching. (issues #8242 image 403s, #6587 CDN SSRF false-positives)
- Falls back cleanly to the SSRF-gated ``cache_*_from_url`` helpers
  (image/audio) or SSRF-gated aiohttp (documents) when ``att.read()``
  isn't available or fails.
- The document fallback path now runs through the SSRF gate for
  defense-in-depth. (issue #11345)
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig


def _ensure_discord_mock():
    """Install a mock discord module when discord.py isn't available."""
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return

    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.ui = SimpleNamespace(View=object, button=lambda *a, **k: (lambda fn: fn), Button=object)
    discord_mod.ButtonStyle = SimpleNamespace(success=1, primary=2, secondary=2, danger=3, green=1, grey=2, blurple=2, red=3)
    discord_mod.Color = SimpleNamespace(orange=lambda: 1, green=lambda: 2, blue=lambda: 3, red=lambda: 4, purple=lambda: 5)
    discord_mod.Interaction = object
    discord_mod.Embed = MagicMock
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
    )

    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod

    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402
from gateway.platforms.base import MessageType  # noqa: E402


# Minimal valid image / audio / PDF bytes so the cache_*_from_bytes
# validators accept them. cache_image_from_bytes runs _looks_like_image()
# which checks for magic bytes; PNG's magic is sufficient.
_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
_OGG_BYTES = b"OggS" + b"\x00" * 60
_PDF_BYTES = b"%PDF-1.4\n" + b"fake pdf body" + b"\n%%EOF"

# ISO-BMFF ``ftyp`` box carrying the AVIF major brand -- the layout Pillow
# emits (``\x00\x00\x00\x20ftypavif\x00\x00\x00\x00avifmif1``). Discord serves
# these for Chromium screenshots and for uploads it re-encodes.
_AVIF_BYTES = (
    b"\x00\x00\x00\x20ftypavif\x00\x00\x00\x00avifmif1" + b"\x00" * 32
)


def _make_adapter() -> DiscordAdapter:
    return DiscordAdapter(PlatformConfig(enabled=True, token="***"))


def _make_attachment_with_read(payload: bytes) -> SimpleNamespace:
    """Attachment stub that exposes .read() — the happy-path primary."""
    return SimpleNamespace(
        url="https://cdn.discordapp.com/attachments/fake/file.png",
        filename="file.png",
        size=len(payload),
        read=AsyncMock(return_value=payload),
    )


def _make_attachment_without_read() -> SimpleNamespace:
    """Attachment stub that has no .read() — exercises the URL fallback."""
    return SimpleNamespace(
        url="https://cdn.discordapp.com/attachments/fake/file.png",
        filename="file.png",
        size=1024,
    )


# ---------------------------------------------------------------------------
# _read_attachment_bytes
# ---------------------------------------------------------------------------

class TestReadAttachmentBytes:
    """Unit tests for the low-level att.read() wrapper."""


    @pytest.mark.asyncio
    async def test_returns_none_when_read_raises(self):
        """Bot-session fetch failures are swallowed so callers fall back."""
        adapter = _make_adapter()
        att = SimpleNamespace(
            url="https://cdn.discordapp.com/attachments/fake/file.png",
            filename="file.png",
            read=AsyncMock(side_effect=RuntimeError("403 Forbidden")),
        )

        result = await adapter._read_attachment_bytes(att)

        assert result is None


# ---------------------------------------------------------------------------
# _cache_discord_image
# ---------------------------------------------------------------------------

class TestCacheDiscordImage:


    @pytest.mark.asyncio
    async def test_falls_back_to_url_when_bytes_validator_rejects(self):
        """If att.read() returns garbage that cache_image_from_bytes rejects
        (e.g. an HTML error page), fall back to the URL downloader instead
        of surfacing the validation error to the caller."""
        adapter = _make_adapter()
        att = _make_attachment_with_read(b"<html>forbidden</html>")

        with patch(
            "plugins.platforms.discord.adapter.cache_image_from_bytes",
            side_effect=ValueError("not a valid image"),
        ), patch(
            "plugins.platforms.discord.adapter.cache_image_from_url",
            new_callable=AsyncMock,
            return_value="/tmp/fallback.png",
        ) as mock_url:
            result = await adapter._cache_discord_image(att, ".png")

        assert result == "/tmp/fallback.png"
        mock_url.assert_awaited_once()


    @pytest.mark.asyncio
    async def test_avif_is_cached_from_bytes_not_the_cdn_url(
        self, monkeypatch, tmp_path
    ):
        """AVIF reaches disk through the real validator, no URL fallback.

        Deliberately does NOT mock ``cache_image_from_bytes`` -- the point is
        that the real ``_looks_like_image`` gate accepts ISO-BMFF stills.
        While it did not, every AVIF attachment took the ``cache_image_from_url``
        fallback and, when that failed too, left the caller holding a signed
        ``cdn.discordapp.com`` link that expires.
        """
        cache_dir = tmp_path / "images"
        cache_dir.mkdir()
        monkeypatch.setattr(
            "gateway.platforms.base.get_image_cache_dir", lambda: cache_dir
        )
        adapter = _make_adapter()
        att = _make_attachment_with_read(_AVIF_BYTES)

        with patch(
            "plugins.platforms.discord.adapter.cache_image_from_url",
            new_callable=AsyncMock,
        ) as mock_url:
            result = await adapter._cache_discord_image(att, ".avif")

        mock_url.assert_not_called()
        cached = Path(result)
        assert cached.parent == cache_dir
        assert cached.suffix == ".avif"
        assert cached.read_bytes() == _AVIF_BYTES


# ---------------------------------------------------------------------------
# _cache_discord_audio
# ---------------------------------------------------------------------------

class TestCacheDiscordAudio:
    @pytest.mark.asyncio
    async def test_prefers_att_read_over_url(self):
        adapter = _make_adapter()
        att = _make_attachment_with_read(_OGG_BYTES)

        with patch(
            "plugins.platforms.discord.adapter.cache_audio_from_bytes",
            return_value="/tmp/voice.ogg",
        ) as mock_bytes, patch(
            "plugins.platforms.discord.adapter.cache_audio_from_url",
            new_callable=AsyncMock,
        ) as mock_url:
            result = await adapter._cache_discord_audio(att, ".ogg")

        assert result == "/tmp/voice.ogg"
        mock_bytes.assert_called_once_with(_OGG_BYTES, ext=".ogg")
        mock_url.assert_not_called()


# ---------------------------------------------------------------------------
# _cache_discord_document
# ---------------------------------------------------------------------------

class TestCacheDiscordDocument:

    @pytest.mark.asyncio
    async def test_fallback_blocked_by_ssrf_guard(self):
        """Document fallback path now honors is_safe_url — was missing before.

        Regression guard for #11345: the old aiohttp block skipped the
        SSRF check entirely; a non-CDN ``att.url`` could have reached
        internal-looking hosts. The fallback must now refuse unsafe URLs.
        """
        adapter = _make_adapter()
        att = _make_attachment_without_read()  # no .read → forces fallback

        with patch(
            "plugins.platforms.discord.adapter.is_safe_url", return_value=False
        ) as mock_safe, patch("aiohttp.ClientSession") as mock_session:
            with pytest.raises(ValueError, match="SSRF"):
                await adapter._cache_discord_document(att, ".pdf")

        mock_safe.assert_called_once_with(att.url)
        # aiohttp must NOT be contacted when the URL is blocked.
        mock_session.assert_not_called()


# ---------------------------------------------------------------------------
# Integration: end-to-end via _handle_message
# ---------------------------------------------------------------------------

class TestHandleMessageUsesAuthenticatedRead:
    """E2E: verify _handle_message routes image/audio downloads through
    att.read() so cdn.discordapp.com 403s (#8242) and SSRF false-positives
    on mangled DNS (#6587) no longer block media caching.
    """

    @pytest.mark.asyncio
    async def test_image_downloads_via_att_read_not_url(self, monkeypatch):
        """Image attachments with .read() never call cache_image_from_url."""
        adapter = _make_adapter()
        adapter._client = SimpleNamespace(user=SimpleNamespace(id=999))
        adapter.handle_message = AsyncMock()

        with patch(
            "plugins.platforms.discord.adapter.cache_image_from_bytes",
            return_value="/tmp/img_from_read.png",
        ), patch(
            "plugins.platforms.discord.adapter.cache_image_from_url",
            new_callable=AsyncMock,
        ) as mock_url_download:
            att = SimpleNamespace(
                url="https://cdn.discordapp.com/attachments/fake/file.png",
                filename="file.png",
                content_type="image/png",
                size=len(_PNG_BYTES),
                read=AsyncMock(return_value=_PNG_BYTES),
            )
            # Minimal Discord message stub for _handle_message.
            from datetime import datetime, timezone

            class _FakeDMChannel:
                id = 100
                name = "dm"

            # Patch the DMChannel isinstance check so our fake counts as DM.
            monkeypatch.setattr(
                "plugins.platforms.discord.adapter.discord.DMChannel",
                _FakeDMChannel,
            )
            chan = _FakeDMChannel()
            msg = SimpleNamespace(
                id=1, content="", attachments=[att], mentions=[],
                reference=None,
                created_at=datetime.now(timezone.utc),
                channel=chan,
                author=SimpleNamespace(id=42, display_name="U", name="U"),
            )
            await adapter._handle_message(msg)

        mock_url_download.assert_not_called()
        event = adapter.handle_message.call_args[0][0]
        assert event.media_urls == ["/tmp/img_from_read.png"]
        assert event.media_types == ["image/png"]


    @pytest.mark.asyncio
    async def test_avif_attachment_yields_a_local_path_not_a_cdn_url(
        self, monkeypatch, tmp_path
    ):
        """Whole inbound path for an AVIF attachment, real cache validator.

        Two things had to change for this to hold: ``_looks_like_image`` has
        to admit the ISO-BMFF still, and the adapter's extension whitelist has
        to keep ``.avif`` instead of coercing it to ``.jpg`` and leaving a
        cache file whose name contradicts its bytes.

        The assertion that matters downstream is the first one: while caching
        failed, ``media_urls`` carried the signed CDN link, which the vision
        tools cannot re-fetch once Discord's ``?ex=`` deadline passes.
        """
        cache_dir = tmp_path / "images"
        cache_dir.mkdir()
        monkeypatch.setattr(
            "gateway.platforms.base.get_image_cache_dir", lambda: cache_dir
        )
        adapter = _make_adapter()
        adapter._client = SimpleNamespace(user=SimpleNamespace(id=999))
        adapter.handle_message = AsyncMock()

        cdn_url = "https://cdn.discordapp.com/attachments/fake/shot.avif?ex=deadline"
        att = SimpleNamespace(
            url=cdn_url,
            filename="shot.avif",
            content_type="image/avif",
            size=len(_AVIF_BYTES),
            read=AsyncMock(return_value=_AVIF_BYTES),
        )

        from datetime import datetime, timezone

        class _FakeDMChannel:
            id = 100
            name = "dm"

        monkeypatch.setattr(
            "plugins.platforms.discord.adapter.discord.DMChannel",
            _FakeDMChannel,
        )
        msg = SimpleNamespace(
            id=1, content="", attachments=[att], mentions=[],
            reference=None,
            created_at=datetime.now(timezone.utc),
            channel=_FakeDMChannel(),
            author=SimpleNamespace(id=42, display_name="U", name="U"),
        )

        with patch(
            "plugins.platforms.discord.adapter.cache_image_from_url",
            new_callable=AsyncMock,
        ) as mock_url_download:
            await adapter._handle_message(msg)

        mock_url_download.assert_not_called()
        event = adapter.handle_message.call_args[0][0]
        assert event.media_urls != [cdn_url]
        cached = Path(event.media_urls[0])
        assert cached.parent == cache_dir
        assert cached.suffix == ".avif"
        assert cached.read_bytes() == _AVIF_BYTES
        assert event.media_types == ["image/avif"]


