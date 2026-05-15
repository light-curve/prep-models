"""ZTF DR17 g-band light curve fetcher via the SNAD Viewer API.

Coordinates come from the first well-observed objects in the Zenodo train.csv
catalog (Nakoneczny et al. 2025), which is the actual training dataset for
this model.  Light curves are fetched from the public SNAD ZTF DR17 database
and cached locally as a parquet file.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from prep_models_utils.zenodo import get_zenodo_files

_SNAD_BASE = "https://db.ztf.snad.space/api/v3"
_SNAD_DR = "dr17"
_SEARCH_RADIUS_ARCSEC = 1
_MIN_OBS = 50

TRAIN_CSV_ZENODO_RECORD = "16410988"
TRAIN_CSV_ZENODO_KEY = "train.csv"
ZTF_RECORDS_FILENAME = "ztf_dr17_gband.parquet"

_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ZTF_RECORDS_DIR = _DATA_DIR / "ztf-records"


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
            except ValueError, KeyError:
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


def load_ztf_curves(n_samples: int, records_dir: Path) -> list[dict]:
    """Load ZTF g-band light curves from the cached parquet file."""
    path = records_dir / ZTF_RECORDS_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run 'prep-models astromer1-ztfdr20 download' first."
        )

    df = pd.read_parquet(path)
    curves = []
    for oid, grp in df.groupby("oid"):
        grp = grp.sort_values("mjd")
        curves.append(
            {
                "lcid": str(oid),
                "survey": "ztf_dr17_g",
                "mjd": grp["mjd"].to_numpy(dtype=np.float32),
                "magnitude": grp["mag"].to_numpy(dtype=np.float32),
                "error": grp["magerr"].to_numpy(dtype=np.float32),
            }
        )
        if len(curves) >= n_samples:
            break

    print(f"Loaded {len(curves)} ZTF DR17 g-band light curves")
    return curves
