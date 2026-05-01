from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import onnx
import tf2onnx

from .common import AstromerConfig


def build_encoder_fn(encoder, aggregation: str):
    import tensorflow as tf

    window_size = (
        encoder.input_shape[0][1] if isinstance(encoder.input_shape, list) else 200
    )
    sig = [
        tf.TensorSpec(shape=(None, window_size, 1), dtype=tf.float32, name="input"),
        tf.TensorSpec(shape=(None, window_size, 1), dtype=tf.float32, name="times"),
        tf.TensorSpec(shape=(None, window_size, 1), dtype=tf.float32, name="mask_in"),
    ]

    # ONNX interface convention: mask_in=1 for valid observations, 0 for padding.
    # Encoder internal convention is inverted (0=visible, 1=hidden/padding), so we flip.
    if aggregation == "mean":

        @tf.function(input_signature=sig)
        def fn(inp, times, mask_in):
            z = encoder(
                {"input": inp, "times": times, "mask_in": 1.0 - mask_in}, training=False
            )
            return tf.math.divide_no_nan(
                tf.reduce_sum(z * mask_in, axis=1),
                tf.reduce_sum(mask_in, axis=1),
            )

    elif aggregation == "max":

        @tf.function(input_signature=sig)
        def fn(inp, times, mask_in):
            z = encoder(
                {"input": inp, "times": times, "mask_in": 1.0 - mask_in}, training=False
            )
            neg_inf = tf.fill(tf.shape(z), -1e9)
            z_masked = tf.where(mask_in > 0.5, z, neg_inf)
            return tf.reduce_max(z_masked, axis=1)

    elif aggregation == "full":

        @tf.function(input_signature=sig)
        def fn(inp, times, mask_in):
            return encoder(
                {"input": inp, "times": times, "mask_in": 1.0 - mask_in}, training=False
            )

    else:
        raise ValueError(f"Unknown aggregation: {aggregation!r}")

    return fn


def run_export(
    output_dir: Path,
    *,
    config: AstromerConfig,
    load_model: Callable[[], tuple[object, object]],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {config.model_name} model ...")
    model, _ = load_model()
    encoder = model.get_layer("encoder")

    for aggregation in ("mean", "max", "full"):
        filename = f"{config.output_prefix}_{aggregation}.onnx"
        out_path = output_dir / filename
        print(f"Exporting {filename} ({aggregation} pooling) ...")

        fn = build_encoder_fn(encoder, aggregation)
        tf2onnx.convert.from_function(
            fn,
            input_signature=fn.input_signature,
            opset=13,
            output_path=str(out_path),
        )
        print(f"  Saved: {out_path}")
        model_proto = onnx.load(str(out_path))
        for output in model_proto.graph.output:
            dims = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
            print(f"  Output '{output.name}': {dims}")

    print("Export complete.")
