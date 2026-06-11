import os
from unittest.mock import patch

from download_models import add_token, download_url, file_ok, redact_download_secrets

os.environ["CIVITAI_TOKEN"] = "secret-civitai-token"
import download_models as dm

dm.CIVITAI_TOKEN = "secret-civitai-token"

assert "[redacted]" in redact_download_secrets("Bearer secret-civitai-token failed")
assert "[redacted]" in redact_download_secrets("https://civitai.com/x?token=secret-civitai-token")
assert "secret-civitai-token" not in redact_download_secrets(add_token("https://civitai.com/api/download/models/1"))

calls = {"requests": 0}


def fake_aria2_fail(*_args, **_kwargs):
    raise __import__("subprocess").CalledProcessError(22, ["aria2c"])


def fake_requests(url, target, progress=None):
    calls["requests"] += 1
    target.write_bytes(b"x" * (1024 * 1024 + 1))


with patch.object(dm, "download_url_aria2", side_effect=fake_aria2_fail), patch.object(
    dm, "download_url_requests", side_effect=fake_requests
), patch.object(dm, "_have_aria2", return_value=True):
    tmp = dm.Path(os.environ.get("TEMP", "/tmp")) / "muse_download_smoke.bin"
    if tmp.exists():
        tmp.unlink()
    download_url("https://civitai.com/api/download/models/1", tmp)
    assert calls["requests"] == 1
    assert file_ok(tmp)
    tmp.unlink()

print("download_models.smoke: ok")
