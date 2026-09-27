"""Start the shared demo with local credential management hidden."""

import argparse
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8502)
    parser.add_argument("--public", action="store_true", help="Use upload limits suitable for the public tunnel")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    folder = root / "runs" / "share"
    folder.mkdir(parents=True, exist_ok=True)
    os.environ["DEMO_SHARED_INSTANCE"] = "1"
    if args.public:
        os.environ["DEMO_PUBLIC_INSTANCE"] = "1"
    sys.argv = ["streamlit", "run", str(root / "app.py"), "--server.address="+args.host,
                "--server.port="+str(args.port), "--server.headless=true",
                "--browser.gatherUsageStats=false", "--server.maxUploadSize="+str(95 if args.public else 500)]
    from streamlit.web.cli import main as streamlit_main
    streamlit_main()


if __name__ == "__main__":
    main()
