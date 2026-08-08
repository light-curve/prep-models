"""Download Astromer 1 ZTF DR20 g-band weights and ZTF DR17 survey records."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import requests
from prep_models_utils.zenodo import download_zenodo_file, get_zenodo_files

from astromer1_ztfdr20_prep.config import (
    CONF,
    WEIGHTS_DIR,
    ZENODO_DATA_KEY,
    ZENODO_INDEX_KEY,
    ZENODO_RECORD_ID,
    ZTF_RECORDS_DIR,
    ZTF_RECORDS_FILENAME,
)

# ZTF DR17 g-band light curves are fetched from the public SNAD Viewer API at
# coordinates taken from the first well-observed objects in the Zenodo train.csv
# catalog (Nakoneczny et al. 2025), the actual training dataset for this model.
_SNAD_BASE = "https://db.ztf.snad.space/api/v3"
_SNAD_DR = "dr17"
_SEARCH_RADIUS_ARCSEC = 1
_MIN_OBS = 50

TRAIN_CSV_ZENODO_RECORD = "16410988"
TRAIN_CSV_ZENODO_KEY = "train.csv"


def _stream_train_coords(n_coords: int) -> list[tuple[float, float]]:
    """Stream the first rows of train.csv and return coords with many obs."""
    files = get_zenodo_files(TRAIN_CSV_ZENODO_RECORD)
    url = next(f["links"]["self"] for f in files if f["key"] == TRAIN_CSV_ZENODO_KEY)
    coords: list[tuple[float, float]] = []
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        # iter_lines handles Content-Encoding: gzip transparently
        lines = r.iter_lines(decode_unicode=True)
        reader = csv.DictReader(lines)
        for row in reader:
            try:
                n_obs = int(row["n_obs"])
                if n_obs < _MIN_OBS:
                    continue
                coords.append((float(row["ra"]), float(row["dec"])))
                if len(coords) >= n_coords:
                    break
            except ValueError:
                continue
            except KeyError:
                continue
    return coords


def _fetch_gband_lcs(ra: float, dec: float) -> list[dict]:
    """Return all ZTF g-band light curves within _SEARCH_RADIUS_ARCSEC of (ra, dec)."""
    url = f"{_SNAD_BASE}/data/{_SNAD_DR}/circle/full/json"
    r = requests.get(
        url,
        params={"ra": ra, "dec": dec, "radius_arcsec": _SEARCH_RADIUS_ARCSEC},
        timeout=60,
    )
    r.raise_for_status()
    results = []
    for oid, obj in r.json().items():
        if obj["meta"]["filter"] != "zg":
            continue
        lc = obj["lc"]
        if len(lc) < _MIN_OBS:
            continue
        results.append({"oid": oid, "lc": lc})
    return results


def download_ztf_records(
    records_dir: Path, n_curves: int = 20, *, force: bool = False
) -> None:
    """Fetch ZTF DR17 g-band light curves and save as a parquet file."""
    out_path = records_dir / ZTF_RECORDS_FILENAME
    if out_path.exists() and not force:
        print(f"ZTF records already present at {out_path}, skipping.")
        return

    records_dir.mkdir(parents=True, exist_ok=True)
    print(f"Streaming train.csv coordinates from Zenodo {TRAIN_CSV_ZENODO_RECORD} ...")
    # Fetch more coords than needed to account for SNAD misses
    coords = _stream_train_coords(n_coords=n_curves * 5)
    print(f"Got {len(coords)} candidate coordinates.")

    rows: list[dict] = []
    seen_oids: set[str] = set()

    for ra, dec in coords:
        if len(seen_oids) >= n_curves:
            break
        try:
            objects = _fetch_gband_lcs(ra, dec)
        except requests.RequestException as e:
            print(f"  Warning: SNAD query failed for ({ra:.4f}, {dec:.4f}): {e}")
            continue
        for obj in objects:
            oid = obj["oid"]
            if oid in seen_oids:
                continue
            for pt in obj["lc"]:
                rows.append(
                    {
                        "oid": oid,
                        "mjd": pt["mjd"],
                        "mag": pt["mag"],
                        "magerr": pt["magerr"],
                    }
                )
            seen_oids.add(oid)
            if len(seen_oids) >= n_curves:
                break

    if not rows:
        raise RuntimeError("Could not fetch any ZTF g-band light curves from SNAD.")

    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(seen_oids)} ZTF g-band light curves to {out_path}")


def _download_weights(*, force: bool = False) -> None:
    """Download ANN_clf checkpoint from Zenodo and rename to weights.*."""
    weights_index = WEIGHTS_DIR / "weights.index"
    if weights_index.exists() and not force:
        print(f"Weights already present at {WEIGHTS_DIR}, skipping.")
        return

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    for zenodo_key, local_name in [
        (ZENODO_DATA_KEY, "weights.data-00000-of-00001"),
        (ZENODO_INDEX_KEY, "weights.index"),
    ]:
        dest = download_zenodo_file(
            ZENODO_RECORD_ID, zenodo_key, WEIGHTS_DIR, extract_zip=False
        )
        dest.rename(WEIGHTS_DIR / local_name)

    conf_path = WEIGHTS_DIR / "conf.json"
    with open(conf_path, "w") as f:
        json.dump(CONF, f, indent=4)
    print(f"Weights saved to {WEIGHTS_DIR}")


def run_download(*, force: bool = False) -> None:
    _download_weights(force=force)
    download_ztf_records(ZTF_RECORDS_DIR, force=force)
