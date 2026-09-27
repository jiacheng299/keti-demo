"""Download all three pinned model assets using only the Python standard library."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def load_manifest(root=ROOT):
    data = json.loads((root / "config/model_downloads.json").read_text(encoding="utf-8"))
    if data.get("version") != 1:
        raise ValueError("Unsupported model manifest")
    return data["models"]


def model_path(item, root=ROOT):
    target = (root / item["path"]).resolve()
    if not target.is_relative_to((root / "models").resolve()):
        raise ValueError("Model path must stay in the project's models directory")
    return target


def verified(path, item):
    if not path.is_file() or path.stat().st_size != item["bytes"]:
        return False
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest() == item["sha256"]


def ensure_model(item, root=ROOT, opener=urllib.request.urlopen):
    target = model_path(item, root)
    if verified(target, item):
        return target
    if not item["url"].startswith("https://"):
        raise ValueError("Model downloads require HTTPS")
    target.parent.mkdir(parents=True, exist_ok=True)
    # Do not replace an existing file until the entire replacement is verified.
    for attempt in range(2):
        temporary = None
        try:
            request = urllib.request.Request(item["url"], headers={"User-Agent": "KetiDemo-Setup/1.0"})
            with opener(request, timeout=60) as response, tempfile.NamedTemporaryFile(
                dir=target.parent, suffix=".download", delete=False
            ) as file:
                temporary = Path(file.name)
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > item["bytes"]:
                        raise ValueError("Downloaded model exceeds expected size")
                    file.write(chunk)
            if not verified(temporary, item):
                raise ValueError("Downloaded model checksum mismatch")
            temporary.replace(target)
            return target
        except (OSError, ValueError):
            if attempt == 1:
                raise
        finally:
            if temporary:
                temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify only, without downloading")
    args = parser.parse_args()
    failures = []
    for item in load_manifest():
        try:
            if args.check:
                if not verified(model_path(item), item):
                    raise ValueError("missing or checksum mismatch")
            else:
                print(f"Preparing {item['id']} ...", flush=True)
                ensure_model(item)
            print(f"OK: {item['path']}", flush=True)
        except (OSError, ValueError) as error:
            failures.append(item["id"])
            print(f"FAILED: {item['id']}: {error}", flush=True)
    if failures:
        print("Check network/proxy access to the sources in config/model_downloads.json and rerun.")
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
