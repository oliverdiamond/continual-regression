import argparse
from pathlib import Path

import jax
import optax
import orbax.checkpoint as ocp
from flax import nnx
from tqdm import tqdm

from network.base import BaseNetwork
from optimizer.adam import adam

# Reference: https://flax.readthedocs.io/en/latest/mnist_tutorial.html


def loss_fn(model: BaseNetwork, batch):
    predictions = model(batch["features"])
    loss = optax.l2_loss(predictions=predictions, targets=batch["targets"]).mean()
    return loss, predictions


@nnx.jit
def train_step(model: BaseNetwork, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
    """Train for a single step."""
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, _), grads = grad_fn(model, batch)
    metrics.update(loss=loss)
    optimizer.update(grads)


@nnx.jit
def eval_step(model: BaseNetwork, metrics: nnx.MultiMetric, batch):
    loss, logits = loss_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch["label"])


@nnx.jit
def pred_step(model: BaseNetwork, batch):
    logits = model(batch["image"])
    return logits.argmax(axis=1)


def train_offline(dataset, model, optimizer, metrics, metrics_history, num_epochs: int):
    # In the experiments that use fve days of data, this network is optimized for
    # 4000 epochs using the Adam optimizer with an L2 weight decay rate of lambda =
    # 0.003 (Note: this is L2 not AdamW) and a batch size of 512, in the offline phase.
    for epoch in tqdm(range(num_epochs)):
        dataset = get_train_dataset()
        for step, batch in enumerate(dataset):
            train_step(model, optimizer, metrics, batch)


def get_train_dataset():
    seed = 0
    key = jax.random.key(seed)
    for _ in range(100):
        yield {"features": jax.random.uniform(key, shape=(512, 384)), "targets": jax.random.uniform(key, shape=(512))}
        _, key = jax.random.split(key)


def checkpoint(model: nnx.Module, optimizer: nnx.Optimizer, ckpt_dir: Path):
    checkpointer = ocp.StandardCheckpointer()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    _, model_state = nnx.split(model)
    checkpointer.save((ckpt_dir / "model").absolute(), model_state)
    _, optimizer_state = nnx.split(optimizer)
    checkpointer.save((ckpt_dir / "optimizer").absolute(), optimizer_state)
    checkpointer.wait_until_finished()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", default=0.001, type=float)
    parser.add_argument("--b1", default=0.9, type=float)
    parser.add_argument("--b2", default=0.99, type=float)
    parser.add_argument("--eps", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=0.003, type=float)
    parser.add_argument("--num_epochs", default=4000, type=int)
    args = parser.parse_args()

    ckpt_dir = Path(f"checkpoints/pretrained_eta_{args.learning_rate}")

    if ckpt_dir.exists():
        print(f"{ckpt_dir} exists, skipping pretraining...")

    model = BaseNetwork(rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(
        model,
        adam(
            learning_rate=args.learning_rate,
            b1=args.b1,
            b2=args.b2,
            eps=args.eps,
            weight_decay=args.weight_decay,
        )
    )
    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
    )

    metrics_history = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }

    train_dataset = get_train_dataset()
    train_offline(
        train_dataset,
        model,
        optimizer,
        metrics,
        metrics_history,
        num_epochs=args.num_epochs,
    )
    checkpoint(model, optimizer, ckpt_dir)


if __name__ == "__main__":
    main()
