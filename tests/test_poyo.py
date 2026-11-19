import copy
import hashlib
import os
from pathlib import Path

import boto3
import numpy as np
import pytest
import torch
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from tqdm import tqdm

from torch_brain.batching import collate
from torch_brain.data import ArrayDict, Data, IrregularTimeSeries
from torch_brain.models.poyo import POYO, _one_hot_embedding

# The checkpoints all have the same readout dimension
# (since they were trained for hand-velocity decoding)
READOUT_DIM = 2

# Updatable SHA256 checksum of the pretrained POYO model checkpoint
MANIFEST = Path(__file__).parent / "poyo_checkpoint.sha256"
EXPECTED_SHA, NAME = MANIFEST.read_text().split()

BUCKET = "torch-brain"
KEY = "model-zoo/poyo_mp.ckpt"
CACHE_DIR = Path(
    os.environ.get("TORCH_BRAIN_CACHE", Path.home() / ".cache" / "torch_brain")
)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope="session")
def pretrained_checkpoint():
    """Download the pretrained POYO model checkpoint from S3."""
    checkpoint_path = CACHE_DIR / Path(KEY).name

    if checkpoint_path.exists() and _sha256(checkpoint_path) != EXPECTED_SHA:
        checkpoint_path.unlink()

    if not checkpoint_path.exists():
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = checkpoint_path.with_name(f"{checkpoint_path.name}.{os.getpid()}.tmp")
        try:
            s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

            # Get file size for progress bar
            file_info = s3.head_object(Bucket=BUCKET, Key=KEY)
            file_size = file_info["ContentLength"]

            # Download with progress bar
            with tqdm(
                total=file_size,
                unit="B",
                unit_scale=True,
                desc=f"Downloading {checkpoint_path.name}",
            ) as pbar:
                s3.download_file(
                    BUCKET,
                    KEY,
                    str(tmp),
                    Callback=lambda bytes_transferred: pbar.update(bytes_transferred),
                )

            if _sha256(tmp) != EXPECTED_SHA:
                pytest.fail(
                    f"Downloaded POYO checkpoint has incorrect SHA256 checksum: {_sha256(tmp)}"
                )

            tmp.replace(checkpoint_path)

        except (BotoCoreError, ClientError) as e:
            pytest.fail(f"POYO checkpoint not cached and download failed: {e}")

        finally:
            tmp.unlink(missing_ok=True)

    return checkpoint_path


def test_load_pretrained_basic(pretrained_checkpoint):
    """Test loading a pretrained POYO model with default settings"""

    # Load the pretrained model
    model = POYO.load_pretrained(checkpoint_path=pretrained_checkpoint)

    assert isinstance(model, POYO)

    assert hasattr(model, "sequence_length")
    assert hasattr(model, "latent_step")
    assert hasattr(model, "num_latents_per_step")

    assert hasattr(model, "readout")
    assert isinstance(model.readout, torch.nn.Linear)
    assert model.readout.out_features == READOUT_DIM


def test_load_pretrained_forward_pass(pretrained_checkpoint):
    """Test that the loaded model can perform a forward pass"""

    model = POYO.load_pretrained(checkpoint_path=pretrained_checkpoint)

    # Reinitialize the vocabs for the new units and sessions
    model.unit_emb.extend_vocab([f"unit_{i}" for i in range(100)])
    model.unit_emb.subset_vocab([f"unit_{i}" for i in range(100)])
    model.session_emb.extend_vocab([f"session_{i}" for i in range(10)])
    model.session_emb.subset_vocab([f"session_{i}" for i in range(10)])

    # Create dummy input data
    batch_size = 2
    n_in = 10
    n_latent = (
        int(model.sequence_length / model.latent_step) * model.num_latents_per_step
    )
    n_out = 4

    inputs = {
        "input_unit_index": torch.randint(0, 100, (batch_size, n_in)),
        "input_timestamps": torch.rand(batch_size, n_in) * model.sequence_length,
        "input_token_type": torch.randint(0, 4, (batch_size, n_in)),
        "input_mask": torch.ones(batch_size, n_in, dtype=torch.bool),
        "latent_index": torch.arange(n_latent).repeat(batch_size, 1)
        % model.num_latents_per_step,
        "latent_timestamps": torch.linspace(0, model.sequence_length, n_latent).repeat(
            batch_size, 1
        ),
        "session_index": torch.zeros(batch_size, dtype=torch.long),
        "output_timestamps": torch.rand(batch_size, n_out) * model.sequence_length,
    }

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # Verify output shape
    assert outputs.shape == (batch_size, n_out, READOUT_DIM)


def _tiny_model() -> POYO:
    model = POYO(
        sequence_length=1.0,
        latent_step=0.25,
        num_latents_per_step=4,
        dim_out=READOUT_DIM,
        dim=32,
        depth=1,
        dim_head=16,
    )
    model.unit_emb.initialize_vocab(["u0", "u1"])
    model.session_emb.initialize_vocab(["s0"])
    return model


def _tiny_data() -> Data:
    return Data(
        spikes=IrregularTimeSeries(
            timestamps=np.array([0.1, 0.5, 0.9]),
            unit_index=np.array([0, 1, 0]),
            domain="auto",
        ),
        units=ArrayDict(id=np.array(["u0", "u1"])),
        session=Data(id="s0"),
        domain="auto",
    )


def test_tokenize_emits_float32_timestamps():
    model = _tiny_model()
    tokens = model.tokenize(_tiny_data())
    assert tokens["input_timestamps"].obj.dtype == np.float32
    assert tokens["latent_timestamps"].dtype == np.float32


def test_one_hot_embedding_matches_embedding():
    weight = torch.randn(4, 16, requires_grad=True)
    ref_weight = weight.detach().clone().requires_grad_(True)
    index = torch.randint(0, 4, (3, 50))
    grad = torch.randn(3, 50, 16)

    out = _one_hot_embedding(index, weight)
    ref = torch.nn.functional.embedding(index, ref_weight)
    torch.testing.assert_close(out, ref)

    out.backward(grad)
    ref.backward(grad)
    torch.testing.assert_close(weight.grad, ref_weight.grad)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires an Apple-Silicon GPU"
)
def test_forward_backward_on_mps_matches_cpu():
    torch.manual_seed(0)
    model = _tiny_model().eval()
    model_mps = copy.deepcopy(model).to("mps")
    batch = collate([model.tokenize(_tiny_data()) for _ in range(2)])
    output_timestamps = torch.rand(2, 8)

    model(**batch, output_timestamps=output_timestamps).sum().backward()
    batch_mps = {k: v.to("mps") for k, v in batch.items()}
    model_mps(
        **batch_mps, output_timestamps=output_timestamps.to("mps")
    ).sum().backward()

    params_mps = model_mps.parameters()
    for (name, p), p_mps in zip(model.named_parameters(), params_mps, strict=True):
        torch.testing.assert_close(
            p_mps.grad.cpu(), p.grad, atol=1e-4, rtol=1e-4, msg=name
        )
