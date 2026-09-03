"""Native image attachment: platform CDN URLs must reach the provider.

Adapters hand the gateway a *local cache path* for an inbound image, but they
fall back to the platform CDN link when caching fails -- the Discord adapter
does this for an AVIF screenshot today, because
``gateway.platforms.base._looks_like_image`` does not recognise AVIF magic
bytes and ``cache_image_from_bytes`` refuses the payload.

``agent.image_routing.build_native_content_parts`` reads ``image_paths`` from
disk, so such a URL used to be reported as an unreadable path and the picture
dropped out of the turn with only a "skipped N unreadable path(s)" warning.
``GatewayRunner._prepare_inbound_message_text`` now materialises remote refs
into the image cache first.
"""

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner, _native_image_cache_suffix
from gateway.session import SessionSource, build_session_key

# Smallest byte prefix that makes ``_sniff_mime_from_bytes`` say "image/png".
# The routing layer sniffs magic bytes, never the file extension, so a stub
# download only has to get the header right.
_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
    b"\x1f\x15\xc4\x89"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)

_DISCORD_CDN_URL = (
    "https://cdn.discordapp.com/attachments/111/222/screenshot.avif"
    "?ex=deadbeef&is=cafe&hm=abc"
)


def _make_runner() -> GatewayRunner:
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake")},
    )
    runner.adapters = {}
    runner._model = "openai/gpt-4.1-mini"
    runner._base_url = None
    runner._decide_image_input_mode = lambda **_: "native"
    return runner


def _source(chat_id: str = "chat-a") -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        chat_type="private",
        user_name=f"user-{chat_id}",
    )


def _image_event(source: SessionSource, path: str, mime: str = "image/avif") -> MessageEvent:
    return MessageEvent(
        text="what is this",
        message_type=MessageType.PHOTO,
        source=source,
        media_urls=[path],
        media_types=[mime],
    )


@pytest.fixture
def image_cache(monkeypatch, tmp_path):
    """Point the shared image cache at a temp dir for the duration of a test."""
    cache_dir = tmp_path / "image_cache"
    cache_dir.mkdir()
    monkeypatch.setattr(
        "gateway.platforms.base.get_image_cache_dir", lambda: cache_dir
    )
    return cache_dir


def _install_download_stub(monkeypatch, *, behaviour):
    """Replace ``vision_tools._download_image`` and record every call."""
    calls = []

    async def _fake_download(url, destination, max_retries=3, *, max_bytes=None):
        calls.append({"url": url, "destination": destination, "max_bytes": max_bytes})
        return behaviour(url, destination)

    monkeypatch.setattr("tools.vision_tools._download_image", _fake_download)
    return calls


@pytest.mark.asyncio
async def test_remote_media_path_is_downloaded_and_becomes_a_data_url(
    monkeypatch, image_cache
):
    """An ``https://`` media path ends up as an inline base64 image part."""
    def _write_png(url, destination):
        destination.write_bytes(_PNG_BYTES)
        return destination

    calls = _install_download_stub(monkeypatch, behaviour=_write_png)

    runner = _make_runner()
    source = _source()
    await runner._prepare_inbound_message_text(
        event=_image_event(source, _DISCORD_CDN_URL),
        source=source,
        history=[],
    )

    paths = runner._consume_pending_native_image_paths(build_session_key(source))
    assert len(paths) == 1
    resolved = paths[0]
    assert not resolved.startswith("http")
    assert resolved.startswith(str(image_cache))
    # The URL's ``?ex=`` query must not leak into the on-disk filename.
    assert resolved.endswith(".avif")
    assert "?" not in resolved

    assert len(calls) == 1
    assert calls[0]["url"] == _DISCORD_CDN_URL
    # Capped at the 20 MiB provider payload ceiling, not the 50 MiB default.
    assert calls[0]["max_bytes"] == 20 * 1024 * 1024

    from agent.image_routing import build_native_content_parts

    parts, skipped = build_native_content_parts("what is this", paths)
    assert skipped == []
    image_parts = [p for p in parts if p.get("type") == "image_url"]
    assert len(image_parts) == 1
    assert image_parts[0]["image_url"]["url"].startswith("data:image/png;base64,")
    # The raw URL is never handed to the provider for a server-side fetch.
    assert _DISCORD_CDN_URL not in image_parts[0]["image_url"]["url"]


@pytest.mark.asyncio
async def test_download_failure_keeps_the_url_and_degrades_to_skipped(
    monkeypatch, image_cache
):
    """A failed download must not invent new behaviour: the image is skipped."""
    def _boom(url, destination):
        raise RuntimeError("HTTP 403 Forbidden")

    calls = _install_download_stub(monkeypatch, behaviour=_boom)

    runner = _make_runner()
    source = _source()
    await runner._prepare_inbound_message_text(
        event=_image_event(source, _DISCORD_CDN_URL),
        source=source,
        history=[],
    )

    paths = runner._consume_pending_native_image_paths(build_session_key(source))
    assert paths == [_DISCORD_CDN_URL]
    assert len(calls) == 1
    # Nothing partial was left behind in the cache.
    assert list(image_cache.iterdir()) == []

    from agent.image_routing import build_native_content_parts

    parts, skipped = build_native_content_parts("what is this", paths)
    assert skipped == [_DISCORD_CDN_URL]
    assert not [p for p in parts if p.get("type") == "image_url"]


@pytest.mark.asyncio
async def test_local_paths_are_passed_through_without_any_download(
    monkeypatch, image_cache
):
    """The all-local case must not touch the network or reorder anything."""
    def _never(url, destination):  # pragma: no cover - must not run
        raise AssertionError(f"unexpected download of {url}")

    calls = _install_download_stub(monkeypatch, behaviour=_never)

    runner = _make_runner()
    source = _source()
    await runner._prepare_inbound_message_text(
        event=_image_event(source, "/tmp/local-a.png", mime="image/png"),
        source=source,
        history=[],
    )

    assert runner._consume_pending_native_image_paths(
        build_session_key(source)
    ) == ["/tmp/local-a.png"]
    assert calls == []


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://cdn.example.com/a/b/shot.avif?ex=1&hm=2", ".avif"),
        ("https://cdn.example.com/a/b/shot.PNG", ".png"),
        ("https://cdn.example.com/a/b/noextension", ".img"),
        ("https://cdn.example.com/a/b/weird.name%2Fwith%2Fslashes", ".img"),
        ("https://cdn.example.com/a/b/x.thisisaverylongextension", ".img"),
        ("https://cdn.example.com/", ".img"),
    ],
)
def test_cache_suffix_is_sanitised(url, expected):
    assert _native_image_cache_suffix(url) == expected
