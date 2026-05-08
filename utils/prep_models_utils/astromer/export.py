from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import onnx
import tf2onnx

from .common import AstromerConfig


def build_encoder_fn(encoder):
    """Build a tf.function returning mean, max, and sequence embeddings in one pass."""
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
    @tf.function(input_signature=sig)
    def fn(inp, times, mask_in):
        z = encoder(
            {"input": inp, "times": times, "mask_in": 1.0 - mask_in}, training=False
        )
        mean_out = tf.math.divide_no_nan(
            tf.reduce_sum(z * mask_in, axis=1),
            tf.reduce_sum(mask_in, axis=1),
        )
        neg_inf = tf.fill(tf.shape(z), tf.constant(-1e9, dtype=z.dtype))
        z_masked = tf.where(mask_in > 0.5, z, neg_inf)
        max_out = tf.reduce_max(z_masked, axis=1)
        sequence_out = z
        return mean_out, max_out, sequence_out

    return fn


def _rename_onnx_outputs(proto: onnx.ModelProto, names: list[str]) -> onnx.ModelProto:
    """Rename graph outputs in-place to the given names (by position)."""
    old_names = [o.name for o in proto.graph.output]
    for old, new in zip(old_names, names):
        if old == new:
            continue
        for out in proto.graph.output:
            if out.name == old:
                out.name = new
        for node in proto.graph.node:
            for i, tensor_name in enumerate(node.output):
                if tensor_name == old:
                    node.output[i] = new
            for i, tensor_name in enumerate(node.input):
                if tensor_name == old:
                    node.input[i] = new
        for vi in proto.graph.value_info:
            if vi.name == old:
                vi.name = new
    return proto


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

    filename = f"{config.output_prefix}.onnx"
    out_path = output_dir / filename
    print(f"Exporting {filename} (mean, max, sequence) ...")

    fn = build_encoder_fn(encoder)
    tf2onnx.convert.from_function(
        fn,
        input_signature=fn.input_signature,
        opset=13,
        output_path=str(out_path),
    )

    proto = onnx.load(str(out_path))
    proto = _rename_onnx_outputs(proto, ["mean", "max", "sequence"])
    onnx.save(proto, str(out_path))

    print(f"  Saved: {out_path}")
    proto = onnx.load(str(out_path))
    for output in proto.graph.output:
        dims = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
        print(f"  Output '{output.name}': {dims}")

    print("Export complete.")
