import os

import smoke_bootstrap  # noqa: F401

from worker_startup_banner import format_banner, mapped_port, muse_urls, public_host, vast_port_mapping_warning

os.environ["PUBLIC_IPADDR"] = "14.179.88.48"
os.environ["VAST_TCP_PORT_3000"] = "34346"
os.environ["VAST_TCP_PORT_8188"] = "34788"

flask_url, comfy_url = muse_urls()
assert flask_url == "http://14.179.88.48:34346", flask_url
assert comfy_url == "http://14.179.88.48:34788", comfy_url
assert mapped_port(3000) == 34346
assert public_host() == "14.179.88.48"

banner = format_banner(probe_services=False)
assert "Flask  http://14.179.88.48:34346" in banner
assert "Comfy  http://14.179.88.48:34788" in banner
assert "HF_TOKEN" in banner

os.environ.pop("VAST_TCP_PORT_3000", None)
os.environ.pop("VAST_TCP_PORT_8188", None)
warn = vast_port_mapping_warning(3000, 8188)
assert "VAST_TCP_PORT" in warn, warn

print("worker_startup_banner.smoke: ok")
