"""Download logic, tested against a fake HTTP server (no network access)."""

import hashlib
import io
import urllib.request

import pytest

from insurance_cross_sell import data

PAYLOAD = bytes(range(256)) * 40_000  # ~10 MB, spans several range chunks


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


@pytest.fixture
def fake_server(monkeypatch):
    calls = []

    def urlopen(request, timeout=None):
        headers = request.headers if isinstance(request, urllib.request.Request) else {}
        calls.append(headers.get("Range"))
        if "Range" in headers:
            start, end = map(int, headers["Range"].removeprefix("bytes=").split("-"))
            return FakeResponse(PAYLOAD[start : end + 1])
        return FakeResponse(PAYLOAD)

    monkeypatch.setattr(data.urllib.request, "urlopen", urlopen)
    return calls


SHA = hashlib.sha256(PAYLOAD).hexdigest()


def test_parallel_download_reassembles_file(tmp_path, fake_server):
    out = data.download_data(tmp_path / "f.csv", url="http://x", sha256=SHA, size=len(PAYLOAD))

    assert out.read_bytes() == PAYLOAD
    assert len(fake_server) == -(-len(PAYLOAD) // data.RANGE_CHUNK)  # one request per chunk
    assert all(r and r.startswith("bytes=") for r in fake_server)


def test_single_stream_download(tmp_path, fake_server):
    out = data.download_data(tmp_path / "f.csv", url="http://x", sha256=SHA, size=None)

    assert out.read_bytes() == PAYLOAD
    assert fake_server == [None]


def test_checksum_mismatch_removes_partial_file(tmp_path, fake_server):
    with pytest.raises(OSError, match="Checksum mismatch"):
        data.download_data(tmp_path / "f.csv", url="http://x", sha256="0" * 64, size=len(PAYLOAD))

    assert list(tmp_path.iterdir()) == []


def test_existing_file_is_not_downloaded_again(tmp_path, fake_server):
    target = tmp_path / "f.csv"
    target.write_text("cached")

    data.download_data(target, url="http://x")

    assert fake_server == []
    assert target.read_text() == "cached"
