"""ONNX graph utilities shared across model sub-packages."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def convert_bf16_to_f32(src: Path, dst: Path) -> None:
    """Rewrite an ONNX model from bfloat16 weights/casts to float32.

    Three passes over the graph:
    1. Initializers  — bfloat16 weight tensors are cast to float32.
    2. Cast nodes    — any Cast(to=BFLOAT16) becomes Cast(to=FLOAT).
                       The resulting float32→float32 no-op casts are
                       removed automatically by ORT's graph optimizer.
    3. value_info    — intermediate-tensor type annotations are updated
                       so ORT's type-checker does not flag mismatches.
    """
    import onnx
    from onnx import TensorProto, numpy_helper

    proto = onnx.load(str(src))

    for init in proto.graph.initializer:
        if init.data_type == TensorProto.BFLOAT16:
            arr = numpy_helper.to_array(init).astype(np.float32)
            init.CopyFrom(numpy_helper.from_array(arr, name=init.name))

    for node in proto.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and attr.t.data_type == TensorProto.BFLOAT16:
                    arr = numpy_helper.to_array(attr.t).astype(np.float32)
                    attr.t.CopyFrom(numpy_helper.from_array(arr))
        elif node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.BFLOAT16:
                    attr.i = TensorProto.FLOAT

    for vi in proto.graph.value_info:
        if vi.type.tensor_type.elem_type == TensorProto.BFLOAT16:
            vi.type.tensor_type.elem_type = TensorProto.FLOAT

    onnx.save(proto, str(dst))
