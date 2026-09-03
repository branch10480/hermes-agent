"""AVIF / HEIC must survive every hop from platform attachment to provider.

Both formats live in the ISO base media file format (ISO-BMFF) container that
MP4/MOV video also uses, so only the major brand at bytes 8..12 separates an
iPhone photo from a video clip. Hermes sniffed that brand in exactly one place
(``agent.image_routing._sniff_mime_from_bytes``) and nowhere else, which broke
the two gates a real attachment has to pass:

1. ``gateway.platforms.base._looks_like_image`` -- the inbound cache gate.
   A miss makes ``cache_image_from_bytes`` raise, the adapter falls back to the
   platform CDN link, and the picture drops out of the turn once that signed
   link expires or refuses an unauthenticated fetch.
2. ``tools.vision_tools._detect_image_mime_type_from_bytes`` -- the resolver
   behind ``vision_analyze`` and the ``image_input_mode: text`` pre-pass.
   A miss makes ``tools.image_source._finalize`` raise ``NotAnImage``, so even
   a successfully cached AVIF could not be looked at.

Both now delegate to ``agent.image_routing.sniff_iso_bmff_image_mime``, so the
brand table cannot drift between them. Video brands must keep returning False /
None at both gates -- they are the reason the container check alone is unsafe.
"""

from io import BytesIO
from pathlib import Path

import pytest

from agent.image_routing import (
    _sniff_mime_from_bytes,
    sniff_iso_bmff_image_mime,
)
from gateway.platforms.base import (
    _looks_like_image,
    cache_image_from_bytes,
    cleanup_image_cache,
)
from tools.vision_tools import (
    _detect_image_mime_type_from_bytes,
    _normalize_to_supported_image,
)


# ---------------------------------------------------------------------------
# Fixtures
#
# Synthetic ISO-BMFF headers rather than encoder output: the brand table is the
# thing under test, and hardcoding the byte layout keeps these cases running on
# installs without a libavif / pillow-heif build. The layout mirrors what
# Pillow actually emits -- a real AVIF starts
# ``\x00\x00\x00\x20ftypavif\x00\x00\x00\x00avifmif1``.
# ---------------------------------------------------------------------------

def _iso_bmff(brand: bytes) -> bytes:
    """Minimal ISO-BMFF ``ftyp`` box carrying *brand* as the major brand."""
    assert len(brand) == 4
    return (
        b"\x00\x00\x00\x20"      # box size
        + b"ftyp"                 # box type
        + brand                   # major brand  (bytes 8..12 -- the decider)
        + b"\x00\x00\x00\x00"    # minor version
        + brand + b"mif1"        # compatible brands
        + b"\x00" * 32           # filler so the box has a body
    )


PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 64
GIF_BYTES = b"GIF89a" + b"\x00" * 64
WEBP_BYTES = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"\x00" * 64
BMP_BYTES = b"BM" + b"\x00" * 64
HTML_BYTES = b"<html><body>403 Forbidden</body></html>"
SVG_BYTES = b'<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>'

# Still-image brands.
AVIF_BYTES = _iso_bmff(b"avif")
AVIS_BYTES = _iso_bmff(b"avis")
HEIC_BYTES = _iso_bmff(b"heic")
MIF1_BYTES = _iso_bmff(b"mif1")

# Video brands sharing the same container -- must NOT be taken for images.
MP4_BYTES = _iso_bmff(b"mp42")
ISOM_BYTES = _iso_bmff(b"isom")
MOV_BYTES = _iso_bmff(b"qt  ")
M4V_BYTES = _iso_bmff(b"M4V ")


def _real_avif_bytes() -> bytes:
    """Encoder-produced AVIF, or skip when this install has no AVIF plugin."""
    PIL_Image = pytest.importorskip("PIL.Image")
    buf = BytesIO()
    try:
        PIL_Image.new("RGB", (8, 8), (200, 30, 40)).save(buf, format="AVIF")
    except Exception as exc:  # pragma: no cover - depends on the Pillow build
        pytest.skip(f"Pillow cannot encode AVIF here: {exc}")
    return buf.getvalue()


def _real_heic_bytes() -> bytes:
    """Encoder-produced HEIC, or skip when ``pillow_heif`` is unavailable."""
    PIL_Image = pytest.importorskip("PIL.Image")
    pillow_heif = pytest.importorskip("pillow_heif")
    pillow_heif.register_heif_opener()
    buf = BytesIO()
    try:
        PIL_Image.new("RGB", (8, 8), (30, 40, 200)).save(buf, format="HEIF")
    except Exception as exc:  # pragma: no cover - depends on the build
        pytest.skip(f"Pillow cannot encode HEIC here: {exc}")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# The shared brand table
# ---------------------------------------------------------------------------

class TestSniffIsoBmffImageMime:
    @pytest.mark.parametrize(
        "blob,expected",
        [
            (AVIF_BYTES, "image/avif"),
            (AVIS_BYTES, "image/avif"),
            (HEIC_BYTES, "image/heic"),
            (MIF1_BYTES, "image/heic"),
        ],
    )
    def test_image_brands(self, blob, expected):
        assert sniff_iso_bmff_image_mime(blob) == expected

    @pytest.mark.parametrize(
        "blob", [MP4_BYTES, ISOM_BYTES, MOV_BYTES, M4V_BYTES]
    )
    def test_video_brands_are_not_images(self, blob):
        assert sniff_iso_bmff_image_mime(blob) is None

    def test_non_iso_bmff_and_short_input(self):
        assert sniff_iso_bmff_image_mime(PNG_BYTES) is None
        assert sniff_iso_bmff_image_mime(HTML_BYTES) is None
        # A truncated header must not index past the end of the buffer.
        assert sniff_iso_bmff_image_mime(b"\x00\x00\x00\x20ftyp") is None
        assert sniff_iso_bmff_image_mime(b"") is None

    def test_image_routing_sniffer_delegates_here(self):
        """The original owner of the table still reports the same mimes."""
        assert _sniff_mime_from_bytes(AVIF_BYTES) == "image/avif"
        assert _sniff_mime_from_bytes(HEIC_BYTES) == "image/heic"
        assert _sniff_mime_from_bytes(MP4_BYTES) is None


# ---------------------------------------------------------------------------
# Gate 1: the inbound image cache
# ---------------------------------------------------------------------------

class TestLooksLikeImage:
    @pytest.mark.parametrize(
        "blob",
        [AVIF_BYTES, AVIS_BYTES, HEIC_BYTES, MIF1_BYTES],
        ids=["avif", "avis", "heic", "mif1"],
    )
    def test_accepts_iso_bmff_stills(self, blob):
        assert _looks_like_image(blob) is True

    @pytest.mark.parametrize(
        "blob",
        [PNG_BYTES, JPEG_BYTES, GIF_BYTES, WEBP_BYTES, BMP_BYTES],
        ids=["png", "jpeg", "gif", "webp", "bmp"],
    )
    def test_pre_existing_formats_still_accepted(self, blob):
        assert _looks_like_image(blob) is True

    @pytest.mark.parametrize(
        "blob",
        [MP4_BYTES, ISOM_BYTES, MOV_BYTES, M4V_BYTES],
        ids=["mp42", "isom", "qt", "m4v"],
    )
    def test_rejects_iso_bmff_video(self, blob):
        """The container is shared with video; only the brand may admit it."""
        assert _looks_like_image(blob) is False

    def test_rejects_html_error_page_and_svg(self):
        """The reason this gate exists at all stays intact."""
        assert _looks_like_image(HTML_BYTES) is False
        assert _looks_like_image(SVG_BYTES) is False
        assert _looks_like_image(b"") is False

    def test_cache_writes_avif_and_janitor_reclaims_it(
        self, monkeypatch, tmp_path
    ):
        """An AVIF now reaches disk, and the hourly janitor still sweeps it.

        ``cleanup_image_cache`` walks the directory rather than globbing known
        extensions, so a new suffix cannot leak files -- pinned here so a
        future glob-based rewrite is caught.
        """
        cache_dir = tmp_path / "images"
        cache_dir.mkdir()
        monkeypatch.setattr(
            "gateway.platforms.base.get_image_cache_dir", lambda: cache_dir
        )

        cached = Path(cache_image_from_bytes(AVIF_BYTES, ext=".avif"))

        assert cached.suffix == ".avif"
        assert cached.read_bytes() == AVIF_BYTES
        assert cached.parent == cache_dir

        # Backdate past the retention window, then sweep.
        import os as _os

        _os.utime(cached, (0, 0))
        assert cleanup_image_cache(max_age_hours=1) == 1
        assert not cached.exists()


# ---------------------------------------------------------------------------
# Gate 2: the vision resolver + the PNG normalisation the providers need
# ---------------------------------------------------------------------------

class TestVisionToolsAcceptsIsoBmff:
    @pytest.mark.parametrize(
        "blob,expected",
        [
            (AVIF_BYTES, "image/avif"),
            (HEIC_BYTES, "image/heic"),
        ],
        ids=["avif", "heic"],
    )
    def test_detect_mime_from_bytes(self, blob, expected):
        assert _detect_image_mime_type_from_bytes(blob) == expected

    @pytest.mark.parametrize(
        "blob", [MP4_BYTES, ISOM_BYTES], ids=["mp42", "isom"]
    )
    def test_video_still_falls_through_to_the_video_sniffers(self, blob):
        assert _detect_image_mime_type_from_bytes(blob) is None

    def test_local_avif_resolves_and_is_transcoded_to_png(self, tmp_path):
        """End of the text-mode / vision_analyze path for a cached AVIF.

        ``resolve_image_source`` used to raise ``NotAnImage`` on these bytes;
        now it types them, and normalisation hands the provider PNG rather
        than an ``image/avif`` media_type Anthropic answers with a 400 --
        which, baked into immutable history, would wedge the whole session.
        """
        import asyncio

        from tools.image_source import ResolveContext, resolve_image_source

        raw = _real_avif_bytes()
        src = tmp_path / "screenshot.avif"
        src.write_bytes(raw)

        resolved = asyncio.run(
            resolve_image_source(str(src), ResolveContext(task_id=None))
        )
        assert resolved.mime == "image/avif"
        assert resolved.data == raw

        out_path, mime, err = _normalize_to_supported_image(src, resolved.mime)

        assert err is None
        assert mime == "image/png"
        assert out_path is not None and out_path != src
        assert out_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    def test_local_heic_is_transcoded_to_png(self, tmp_path):
        """HEIC needs ``pillow_heif`` registered before ``Image.open`` sees it.

        Normalisation reuses ``agent.image_routing._transcode_to_png`` for
        exactly that registration; a bare ``Image.open`` rejects every iPhone
        photo.
        """
        src = tmp_path / "photo.heic"
        src.write_bytes(_real_heic_bytes())

        out_path, mime, err = _normalize_to_supported_image(src, "image/heic")

        assert err is None
        assert mime == "image/png"
        assert out_path is not None
        assert out_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    def test_supported_formats_are_passed_through_untouched(self, tmp_path):
        """No new transcode for formats providers already ingest."""
        src = tmp_path / "already.png"
        src.write_bytes(PNG_BYTES)

        out_path, mime, err = _normalize_to_supported_image(src, "image/png")

        assert (out_path, mime, err) == (src, "image/png", None)
