"""Export AstroNet-TINHO to ONNX via tf2onnx.

TINHO is a compressed (weight-clustered) variant of the T2 time-series
transformer, trained on PLAsTiCC to classify 14 types of astronomical
transients.  The upstream repository ships the trained SavedModels directly,
so no separate weight download is needed.

Two embedding models are exported (GlobalAveragePooling output, embed_dim=32):

  tinho_gr.onnx      – g+r bands only, no redshift
                       input:  flux (B, 100, 2)  float32
                       output: embedding (B, 32)  float32

  tinho_ugrizy.onnx  – all 6 LSST bands + host redshift
                       inputs: flux (B, 100, 6)  float32
                               redshift (B, 2)   float32
                       output: embedding (B, 32)  float32

Embeddings are taken from the GlobalAveragePooling1D layer, before the
final weight-clustered Dense+softmax classifier.
"""

from __future__ import annotations

from pathlib import Path


_CODE_DIR = Path(__file__).resolve().parent.parent / "code"
_MODELS_DIR = _CODE_DIR / "astronet" / "tinho" / "models" / "plasticc"

# Model directory name prefixes; we glob for the exact name
_GR_PREFIX = "model-GR-noZ-28341-"
_UGRIZY_PREFIX = "model-UGRIZY-wZ-31367-"

_VARIANTS = [
    {
        "prefix": _GR_PREFIX,
        "out_name": "tinho_gr.onnx",
        "input_names": ["flux"],
        "output_names": ["embedding"],
    },
    {
        "prefix": _UGRIZY_PREFIX,
        "out_name": "tinho_ugrizy.onnx",
        "input_names": ["flux", "redshift"],
        "output_names": ["embedding"],
    },
]


def _find_model(prefix: str) -> Path:
    matches = [
        p for p in _MODELS_DIR.iterdir() if p.name.startswith(prefix) and p.is_dir()
    ]
    if not matches:
        raise FileNotFoundError(
            f"No SavedModel matching '{prefix}*' in {_MODELS_DIR}. "
            "Ensure the git submodule (models/astronet-tinho/code) is initialised."
        )
    return matches[0]


def _rename_io(model_proto, input_names: list[str], output_names: list[str]):
    """Rename ONNX model inputs/outputs in-place."""
    import onnx

    old_inputs = [i.name for i in model_proto.graph.input]
    old_outputs = [o.name for o in model_proto.graph.output]

    mapping = {}
    for old, new in zip(old_inputs, input_names):
        mapping[old] = new
    for old, new in zip(old_outputs, output_names):
        mapping[old] = new

    for node in model_proto.graph.node:
        node.input[:] = [mapping.get(n, n) for n in node.input]
        node.output[:] = [mapping.get(n, n) for n in node.output]
    for inp in model_proto.graph.input:
        if inp.name in mapping:
            inp.name = mapping[inp.name]
    for out in model_proto.graph.output:
        if out.name in mapping:
            out.name = mapping[out.name]

    onnx.checker.check_model(model_proto)
    return model_proto


def _make_embedding_model(model):
    """Return a new Keras model that outputs the GlobalAveragePooling embedding."""
    import tensorflow as tf

    gap_layer = next(
        layer
        for layer in model.layers
        if isinstance(layer, tf.keras.layers.GlobalAveragePooling1D)
    )
    return tf.keras.Model(inputs=model.inputs, outputs=gap_layer.output)


def run_export(out_dir: Path) -> None:
    import onnx
    import tensorflow as tf
    import tf2onnx

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for variant in _VARIANTS:
        saved_model_path = _find_model(variant["prefix"])
        out_path = out_dir / variant["out_name"]

        print(f"Loading {saved_model_path.name} …")
        full_model = tf.keras.models.load_model(str(saved_model_path), compile=False)
        embed_model = _make_embedding_model(full_model)
        print(f"  Input spec: {embed_model.input_spec}")
        print(f"  Embedding dim: {embed_model.output_shape[-1]}")

        print(f"  Converting → {out_path} …")
        model_proto, _ = tf2onnx.convert.from_keras(embed_model, opset=17)
        _rename_io(model_proto, variant["input_names"], variant["output_names"])
        onnx.save(model_proto, str(out_path))
        print(f"  Written: {out_path}")

    print(f"\nAll ONNX files written to {out_dir}")
