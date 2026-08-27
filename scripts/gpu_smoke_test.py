"""GPU smoke test: 4 GPUs, bf16, and one cached case round-tripping."""
import os, glob
import numpy as np
import torch


def main():
    print("=" * 60)
    print("GPU_NV_M smoke test")
    print("=" * 60)

    print(f"torch                : {torch.__version__}")
    print(f"cuda available       : {torch.cuda.is_available()}")
    n = torch.cuda.device_count()
    print(f"device count         : {n}  (expected 4 on GPU_NV_M)")
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        print(f"  cuda:{i} {p.name}  {p.total_memory / 1e9:.1f} GB  sm_{p.major}{p.minor}")

    assert torch.cuda.is_available(), "no CUDA device visible"
    if n != 4:
        print(f"WARNING: expected 4 GPUs, saw {n}")

    # bf16 is the project's precision of choice; A10G is Ampere so it is native.
    bf16_ok = torch.cuda.is_bf16_supported()
    print(f"bf16 supported       : {bf16_ok}")
    assert bf16_ok, "bf16 unsupported -- the training recipe assumes it"
    with torch.autocast("cuda", dtype=torch.bfloat16):
        a = torch.randn(512, 512, device="cuda")
        out = (a @ a).float()
    print(f"bf16 matmul          : ok (mean {out.mean().item():.4f})")

    # NCCL presence: multi-GPU DDP depends on it. Linux SPCS has it; Windows
    # would not, which is why all training runs remotely.
    print(f"nccl available       : {torch.distributed.is_nccl_available()}")

    print(f"cpu count            : {os.cpu_count()}  (expected 44)")

    # One cached case: shape, dtype, and the label invariant.
    hits = sorted(glob.glob("/mnt/data/cache/**/*.npz", recursive=True))
    print(f"cached npz visible   : {len(hits)}")
    if hits:
        with np.load(hits[0]) as z:
            print(f"  file        : {hits[0]}")
            print(f"  img shape   : {z['img'].shape}  dtype {z['img'].dtype}")
            print(f"  orig shape  : {z['orig_shape']}")
            if "seg" in z.files:
                vals = np.unique(z["seg"])
                print(f"  label values: {vals.tolist()}")
                assert vals.max() <= 3, (
                    f"label {vals.max()} > 3: this is NOT BraTS 2023 "
                    "(label 4 is the 2021 convention and silently empties ET)"
                )
                print("  label check : OK (no 4s -- ET is label 3)")
    else:
        print("  no cache mounted; stage the cache before training")

    # MONAI import and a forward pass through the real model.
    from monai.networks.nets import SegResNetDS
    print("monai import         : ok")
    net = SegResNetDS(
        spatial_dims=3, init_filters=32, in_channels=4, out_channels=3,
        blocks_down=(1, 2, 2, 4), dsdepth=4,
    ).cuda()
    x = torch.randn(1, 4, 128, 128, 128, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        y = net(x)
    shapes = [tuple(t.shape) for t in (y if isinstance(y, (list, tuple)) else [y])]
    print(f"forward 128^3        : {shapes}")
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"peak VRAM (batch 1)  : {peak:.2f} GB of 24 GB")
    print(f"projected batch 2    : ~{peak * 2:.2f} GB")

    print("\nSMOKE TEST PASSED")
    return {"gpus": n, "bf16": bf16_ok, "peak_gb": peak, "cached": len(hits)}


if __name__ == "__main__":
    __return__ = main()
