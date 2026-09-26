"""Export the Maven light-curve encoder to ONNX."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import torch
from torch import nn

from maven_prep.config import N_MAX_OBS, NBAND, OUTPUT_PREFIX, SEQ_LEN
from maven_prep.runtime import MavenLightCurveEncoder, load_model


class MavenEmbedder(nn.Module):
    """ONNX wrapper exposing pooled, CLIP and per-timestep light-curve features.

    Re-assembles upstream `TransformerWithTimeEmbeddings.forward` from its
    submodules so that the per-timestep transformer output is also available.
    The band embedding is precomputed as a buffer: upstream builds it from
    `x.shape[1] // nband` at runtime, which the tracer would freeze anyway, and
    the sequence length is fixed to the training layout (NBAND x N_MAX_OBS).
    """

    def __init__(self, model: MavenLightCurveEncoder) -> None:
        super().__init__()
        enc = model.lightcurve_encoder
        self.embedding_mag = enc.embedding_mag
        self.embedding_t = enc.embedding_t
        self.band_emb = enc.band_emb
        self.transformer = enc.transformer
        self.encoder_projection = enc.projection
        self.lightcurve_projection = model.lightcurve_projection
        self.register_buffer(
            "band_idx",
            torch.arange(NBAND).repeat_interleave(N_MAX_OBS),
            persistent=False,
        )

    def forward(
        self,
        mag: torch.Tensor,  # (B, SEQ_LEN) float32
        time: torch.Tensor,  # (B, SEQ_LEN) float32
        mask: torch.Tensor,  # (B, SEQ_LEN) float32, 1=valid, 0=pad
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask_bool = mask > 0.5
        x = self.embedding_mag(mag[..., None]) + self.embedding_t(time)
        x = x + self.band_emb(self.band_idx)[None, :, :]
        sequence = self.transformer(x, mask_bool)  # (B, SEQ_LEN, EMB)

        mask_f = mask_bool.to(sequence.dtype)[..., None]
        mean = (sequence * mask_f).sum(1) / mask_f.sum(1).clamp(min=1.0)

        clip = self.lightcurve_projection(self.encoder_projection(mean))
        clip = clip / clip.norm(dim=-1, keepdim=True)
        return mean, clip, sequence


def _check_against_upstream(
    model: MavenLightCurveEncoder, embedder: MavenEmbedder
) -> None:
    """The wrapper must reproduce upstream `lightcurve_embeddings_with_projection`."""
    rng = torch.Generator().manual_seed(0)
    mag = 18.0 + torch.randn(4, SEQ_LEN, generator=rng)
    time = torch.rand(4, SEQ_LEN, generator=rng) * 150.0
    mask = torch.ones(4, SEQ_LEN)
    mask[1, 40:N_MAX_OBS] = 0.0
    mask[2, N_MAX_OBS + 10 :] = 0.0
    mask[3, :N_MAX_OBS] = 0.0
    with torch.no_grad():
        expected = model(mag, time, mask.bool())
        _, clip, _ = embedder(mag, time, mask)
    diff = (expected - clip).abs().max().item()
    print(f"Wrapper vs upstream max abs diff: {diff:.2e}")
    if diff > 1e-5:
        raise RuntimeError("ONNX wrapper does not match the upstream forward pass")


def run_export(output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Maven light-curve encoder ...")
    model = load_model()
    embedder = MavenEmbedder(model)
    embedder.eval()
    _check_against_upstream(model, embedder)

    dummy = (
        torch.full((2, SEQ_LEN), 18.0),
        torch.linspace(0.0, 100.0, SEQ_LEN).repeat(2, 1),
        torch.ones(2, SEQ_LEN),
    )
    input_names = ["mag", "time", "mask"]
    output_names = ["mean", "clip", "sequence"]
    dynamic_axes = {name: {0: "batch"} for name in input_names + output_names}

    out_path = output_dir / f"{OUTPUT_PREFIX}.onnx"
    print(f"Exporting {out_path.name} ({', '.join(output_names)}) ...")
    torch.onnx.export(
        embedder,
        dummy,
        str(out_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=18,
        dynamo=False,
    )

    proto = onnx.load(str(out_path))
    onnx.checker.check_model(proto)
    for out in proto.graph.output:
        dims = [d.dim_param or d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"  Output '{out.name}': {dims}")

    _check_onnx(embedder, out_path)
    print("Export complete.")


def _check_onnx(embedder: MavenEmbedder, path: Path) -> None:
    import onnxruntime as rt

    rng = np.random.default_rng(1)
    mag = (18.0 + rng.normal(size=(3, SEQ_LEN))).astype(np.float32)
    time = (rng.uniform(0.0, 150.0, size=(3, SEQ_LEN))).astype(np.float32)
    mask = np.ones((3, SEQ_LEN), dtype=np.float32)
    mask[0, 70:N_MAX_OBS] = 0.0
    mask[2, N_MAX_OBS + 5 :] = 0.0

    sess = rt.InferenceSession(str(path))
    onnx_out = sess.run(None, {"mag": mag, "time": time, "mask": mask})
    with torch.no_grad():
        torch_out = embedder(
            torch.from_numpy(mag), torch.from_numpy(time), torch.from_numpy(mask)
        )
    for name, o, t in zip(["mean", "clip", "sequence"], onnx_out, torch_out):
        diff = np.abs(o - t.numpy()).max()
        print(f"  ONNX vs torch '{name}' max abs diff: {diff:.2e}")
        if not np.allclose(o, t.numpy(), atol=1e-4):
            raise RuntimeError(f"ONNX output '{name}' does not match torch")
