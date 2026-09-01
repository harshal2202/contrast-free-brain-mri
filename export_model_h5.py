"""Export a PyTorch generator checkpoint to an HDF5 weights file."""

import argparse

import h5py
import torch


def export_checkpoint(checkpoint_path, output_path):
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    with h5py.File(output_path, "w") as h5_file:
        h5_file.attrs["format"] = "PyTorch generator state_dict"
        h5_file.attrs["source_checkpoint"] = checkpoint_path
        weights = h5_file.create_group("state_dict")

        for name, tensor in state.items():
            weights.create_dataset(name, data=tensor.detach().cpu().numpy())

    print(f"Exported {len(state)} tensors to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a generator checkpoint to HDF5")
    parser.add_argument("--ckpt", required=True, help="Input PyTorch checkpoint")
    parser.add_argument("--out", required=True, help="Output HDF5 file")
    args = parser.parse_args()
    export_checkpoint(args.ckpt, args.out)
