"""Download and extract the Oxford-IIIT Pet dataset into data/.

    python scripts/download_data.py            (~800 MB download, ~1 min on a good connection)

Source: https://www.robots.ox.ac.uk/~vgg/data/pets/   (files are hosted on thor.robots.ox.ac.uk)
"""
import argparse
import tarfile
import urllib.request
from pathlib import Path

URLS = {
    "images.tar.gz": "https://thor.robots.ox.ac.uk/pets/images.tar.gz",
    "annotations.tar.gz": "https://thor.robots.ox.ac.uk/pets/annotations.tar.gz",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--keep-archives", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for fname, url in URLS.items():
        target_dir = out / fname.split(".")[0]
        if target_dir.exists():
            print(f"{target_dir} exists, skipping")
            continue
        arc = out / fname
        if not arc.exists():
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, arc)
        print(f"extracting {arc}")
        with tarfile.open(arc) as t:
            t.extractall(out, filter="data")
        if not args.keep_archives:
            arc.unlink()
    n = len(list((out / "images").glob("*.jpg")))
    print(f"done: {n} images in {out / 'images'}")


if __name__ == "__main__":
    main()
