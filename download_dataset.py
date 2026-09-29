import argparse
import io
import os
import time
import zipfile
from pathlib import Path

import requests
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")
DEFAULT_OUT = ROOT / "pat-papercup-seg"

with open(ROOT / "data.yaml") as f:
    rf = yaml.safe_load(f)["roboflow"]

parser = argparse.ArgumentParser()
parser.add_argument("--version", type=int, default=rf["version"])
parser.add_argument("--format", default="yolov8")
parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
args = parser.parse_args()

api_key = os.environ.get("ROBOFLOW_API_KEY")
if not api_key:
    raise SystemExit("Coloca tu API key de Roboflow en el archivo .env (ROBOFLOW_API_KEY=...).")

url = f"https://api.roboflow.com/{rf['workspace']}/{rf['project']}/{args.version}/{args.format}"
print(f"Solicitando export {rf['workspace']}/{rf['project']} v{args.version} ({args.format})...")
resp = requests.get(url, params={"api_key": api_key}, timeout=60)
while resp.status_code == 202:
    print(f"Roboflow está generando el export... {resp.json().get('progress', 0):.0%}")
    time.sleep(5)
    resp = requests.get(url, params={"api_key": api_key}, timeout=60)
if resp.status_code != 200:
    raise SystemExit(f"Error {resp.status_code}: {resp.text}")
link = resp.json()["export"]["link"]

print("Descargando...")
data = requests.get(link, timeout=600).content

args.out.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(data)) as z:
    z.extractall(args.out)

for split in ("train", "valid", "test"):
    images = args.out / split / "images"
    if images.is_dir():
        print(f"{split}: {sum(1 for _ in images.iterdir())} imágenes")
print(f"Dataset en {args.out}")
