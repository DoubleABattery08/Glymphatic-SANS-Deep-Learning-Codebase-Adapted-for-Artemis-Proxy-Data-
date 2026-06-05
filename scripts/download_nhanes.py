"""Fetch the NHANES 2017-2018 component files used for the control cohort.

Deterministic: the cycle, component list, and URLs are fixed in config, so a
re-run retrieves the same files. Each file is written to ``data/external/nhanes``
and recorded in a manifest with its source URL, byte size, and SHA-256 so the
provenance can be audited and a reviewer can confirm an identical download.

Run from the repository root::

    python scripts/download_nhanes.py
"""

from __future__ import annotations

import hashlib
import json

import requests

from artemis_proxy import config

_CHUNK_BYTES = 1 << 16
_TIMEOUT_SECONDS = 120
# CDC's WAF rejects requests without a conventional browser user agent.
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; artemis-proxy/0.1)"}


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_component(stem: str, dest_dir) -> dict:
    """Download one ``.XPT`` component, returning its manifest entry.

    An existing non-empty file is left in place (the download is idempotent); its
    checksum is still recomputed so the manifest always reflects the bytes on
    disk.
    """

    url = f"{config.NHANES_BASE_URL}/{stem}.XPT"
    dest = dest_dir / f"{stem}.XPT"
    if not (dest.exists() and dest.stat().st_size > 0):
        response = requests.get(
            url, headers=_HEADERS, timeout=_TIMEOUT_SECONDS, stream=True
        )
        response.raise_for_status()
        with open(dest, "wb") as handle:
            for chunk in response.iter_content(chunk_size=_CHUNK_BYTES):
                handle.write(chunk)
    return {
        "stem": stem,
        "url": url,
        "bytes": dest.stat().st_size,
        "sha256": _sha256(dest),
        "role": config.NHANES_COMPONENTS[stem],
    }


def main() -> None:
    config.NHANES_RAW.mkdir(parents=True, exist_ok=True)
    manifest = {
        "cycle": config.NHANES_CYCLE,
        "base_url": config.NHANES_BASE_URL,
        "components": [
            download_component(stem, config.NHANES_RAW)
            for stem in config.NHANES_COMPONENTS
        ],
    }
    manifest_path = config.NHANES_RAW / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    total_mb = sum(entry["bytes"] for entry in manifest["components"]) / 1e6
    print(
        f"Downloaded {len(manifest['components'])} NHANES {config.NHANES_CYCLE} "
        f"files ({total_mb:.1f} MB) to {config.NHANES_RAW}"
    )


if __name__ == "__main__":
    main()
