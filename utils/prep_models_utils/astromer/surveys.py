"""Shared survey record loading for Astromer test data generation."""

from __future__ import annotations

from pathlib import Path

ALCOCK_GDRIVE_ID = "1N4-n3OUAA0J2jFF6fO2MmbykzcxzLEuX"
ATLAS_GDRIVE_ID = "1OnSF3EULdJpFY4MwRZLgmXa6DauziR8W"


def download_records(
    gdrive_id: str,
    records_dir: Path,
    survey_name: str,
    *,
    force: bool = False,
) -> None:
    import zipfile
    import gdown

    if not force and any(records_dir.rglob("*.record")):
        print(f"{survey_name} records already present at {records_dir}, skipping.")
        return

    records_dir.mkdir(parents=True, exist_ok=True)
    tmp_zip = records_dir.parent / f"{survey_name}.zip"
    print(f"Downloading {survey_name} records from Google Drive ...")
    # verify=False works around an SSL EOF issue on Python 3.10 / macOS arm64
    gdown.download(id=gdrive_id, output=str(tmp_zip), quiet=False, verify=False)
    print("Extracting ...")
    with zipfile.ZipFile(tmp_zip) as zf:
        zf.extractall(records_dir)
    tmp_zip.unlink()
    print(f"{survey_name} records saved to {records_dir}")


def _deserialize_record(raw):
    """Parse TFRecord using dim_0/dim_1/dim_2 schema (shared by Alcock and ATLAS)."""
    import tensorflow as tf

    context_features = {
        "label": tf.io.FixedLenFeature([], dtype=tf.int64),
        "length": tf.io.FixedLenFeature([], dtype=tf.int64),
        "id": tf.io.FixedLenFeature([], dtype=tf.string),
    }
    sequence_features = {
        f"dim_{i}": tf.io.VarLenFeature(dtype=tf.float32) for i in range(3)
    }

    context, sequence = tf.io.parse_single_sequence_example(
        serialized=raw,
        context_features=context_features,
        sequence_features=sequence_features,
    )
    cols = [
        tf.cast(tf.sparse.to_dense(sequence[f"dim_{i}"]), tf.float32) for i in range(3)
    ]
    inp = tf.stack(cols, axis=2)[0]  # [n_obs, 3]: dim_0=mjd, dim_1=mag, dim_2=err

    return {
        "input": inp,
        "lcid": context["id"],
        "label": tf.cast(context["label"], tf.int32),
    }


def load_survey_curves(
    n_samples: int,
    records_dir: Path,
    gdrive_id: str,
    survey_name: str,
) -> list[dict]:
    import tensorflow as tf

    if not any(records_dir.rglob("*.record")):
        download_records(gdrive_id, records_dir, survey_name)

    record_files = sorted(records_dir.rglob("*.record"))
    # prefer val records — smaller holdout subset
    val_files = [f for f in record_files if "val" in f.parts]
    if val_files:
        record_files = val_files

    curves = []
    for rec_path in record_files:
        if len(curves) >= n_samples:
            break
        ds = tf.data.TFRecordDataset([str(rec_path)])
        for raw in ds:
            if len(curves) >= n_samples:
                break
            s = _deserialize_record(raw)
            inp = s["input"].numpy()
            curves.append(
                {
                    "lcid": s["lcid"].numpy().decode(),
                    "label": int(s["label"].numpy()),
                    "survey": survey_name,
                    "mjd": inp[:, 0],
                    "magnitude": inp[:, 1],
                    "error": inp[:, 2],
                }
            )

    print(f"Loaded {len(curves)} {survey_name} light curves")
    return curves


def load_mixed_curves(
    n_samples: int,
    alcock_records_dir: Path,
    atlas_records_dir: Path,
) -> list[dict]:
    """Load n_samples curves split evenly between Alcock and ATLAS."""
    n_alcock = n_samples // 2
    n_atlas = n_samples - n_alcock

    alcock = load_survey_curves(
        n_alcock, alcock_records_dir, ALCOCK_GDRIVE_ID, "alcock"
    )
    atlas = load_survey_curves(n_atlas, atlas_records_dir, ATLAS_GDRIVE_ID, "atlas")
    return alcock + atlas
