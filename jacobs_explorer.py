"""
python explore_cebra.py --quick          # ~1-2 min, sanity check
python explore_cebra.py                  # full run, ~10-20 min on CPU
python explore_cebra.py --device mps

Otherwise, setup is manually specified in the code.

To run the Allen code, data must be manually downloaded from https://figshare.com/s/60adb075234c2cc51fa3 and put in the ./data folder


"""

from __future__ import annotations

import argparse
import itertools
import pickle
import time
from os import makedirs

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor

import cebra.datasets
from cebra import CEBRA

# ===========================================================================
# Part 1: My own code version of https://cebra.ai/docs/demo_notebooks/Demo_Allen.html, except the evaluation part.

# (Could still use some cleanup and functionality)

# ===========================================================================


def allen_demo(args):
    cortex = "VISp"
    num_neurons = 800
    seed = 333

    model_conf = {}

    model_conf["train_steps"] = 10000 if not args.quick else 500
    model_conf["temperature"] = 1.0
    model_conf["lr"] = 3e-4
    model_conf["num_hidden_units"] = 128
    model_conf["output_dim"] = 128
    model_conf["verbose"] = True
    model_conf["device"] = args.device
    model_conf["time_offset"] = 1
    model_conf["batch_size"] = 512
    model_conf["conditional"] = "time_delta"
    model_conf["model_architecture"] = {
        "ca": f"offset{model_conf['time_offset']}-model",
        "np": f"offset{model_conf['time_offset']}-model" if args.quick else f"resample{model_conf['time_offset']}-model",
    }
    model_conf["model_architecture"]["joint"] = [model_conf["model_architecture"]["ca"], model_conf["model_architecture"]["np"]]

    train_embeddings, test_embeddings = {}, {}

    makedirs("plots", exist_ok=True)
    makedirs("jacobs_models", exist_ok=True)

    ca_train_set, ca_test_set, train_embeddings["ca"], test_embeddings["ca_test"] = create_train_test("ca", model_conf, cortex, num_neurons, seed)
    np_train_set, np_test_set, train_embeddings["np"], test_embeddings["np_test"] = create_train_test("np", model_conf, cortex, num_neurons, seed)

    plot_embeddings(train_embeddings, save=True, out=f"plots/allen_demo_embeddings{model_conf['train_steps']}_np_{model_conf['model_architecture']['np']}.png")

    joint_train_set, joint_test_set, train_embeddings["joint"], test_embeddings["joint_test"] = create_train_test(
        "joint", model_conf, cortex, num_neurons, seed
    )

    plot_embeddings(train_embeddings["joint"], save=True, out=f"plots/joint_allen_demo_embeddings{model_conf['train_steps']}.png")


def build_model_session_solver(data_loader, model_conf, multi=False):
    if multi:
        model = torch.nn.ModuleList()
        for dataset, mod in zip(data_loader.dataset.iter_sessions(), model_conf["model_architecture"]):
            model.append(
                cebra.models.init(mod, dataset.input_dimension, model_conf["num_hidden_units"], model_conf["output_dim"], normalize=True).to(
                    model_conf["device"]
                )
            )
            dataset.configure_for(model[-1])
        model.to(model_conf["device"])
    else:
        model = cebra.models.init(
            model_conf["model_architecture"],
            data_loader.dataset.input_dimension,
            model_conf["num_hidden_units"],
            model_conf["output_dim"],
            normalize=True,
        ).to(model_conf["device"])
        data_loader.dataset.configure_for(model)

    criterion = cebra.models.InfoNCE(temperature=model_conf["temperature"])
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), criterion.parameters()), lr=model_conf["lr"])
    if multi:
        return model, cebra.solver.MultiSessionSolver(
            model=model,
            criterion=criterion,
            tqdm_on=model_conf["verbose"],
            optimizer=optimizer,
        ).to(model_conf["device"])
    else:
        return model, cebra.solver.SingleSessionSolver(
            model=model,
            criterion=criterion,
            tqdm_on=model_conf["verbose"],
            optimizer=optimizer,
        ).to(model_conf["device"])


def create_train_test(name, model_conf, cortex="VISp", num_neurons=800, seed=333):
    new_conf = model_conf.copy()

    if name == "ca":
        train_set = cebra.datasets.init(f"allen-movie-one-ca-{cortex}-{num_neurons}-train-10-{seed}").to(model_conf["device"])
        test_set = cebra.datasets.init(f"allen-movie-one-ca-{cortex}-{num_neurons}-test-10-{seed}").to(model_conf["device"])
        new_conf["model_architecture"] = model_conf["model_architecture"]["ca"]
        is_multi = False
        mod_name = f"jacobs_models/movie_ca_{new_conf['model_architecture']}_{new_conf['train_steps']}"
    elif name == "np":
        train_set = cebra.datasets.init(f"allen-movie-one-neuropixel-{cortex}-{num_neurons}-train-10-{seed}").to(model_conf["device"])
        test_set = cebra.datasets.init(f"allen-movie-one-neuropixel-{cortex}-{num_neurons}-test-10-{seed}").to(model_conf["device"])
        new_conf["model_architecture"] = model_conf["model_architecture"]["np"]
        is_multi = False
        mod_name = f"jacobs_models/movie_np_{new_conf['model_architecture']}_{new_conf['train_steps']}"
    elif name == "joint":
        train_set = cebra.datasets.init(f"allen-movie-one-ca-neuropixel-{cortex}-{num_neurons}-train-10-{seed}").to(model_conf["device"])
        test_set = cebra.datasets.init(f"allen-movie-one-ca-neuropixel-{cortex}-{num_neurons}-test-10-{seed}").to(model_conf["device"])
        new_conf["model_architecture"] = model_conf["model_architecture"]["joint"]
        is_multi = True
        mod_name = f"jacobs_models/movie_joint_{str(new_conf['model_architecture'])}_{new_conf['train_steps']}"
    else:
        raise ValueError(f"Unknown dataset name: {name}")

    try:
        model = pickle.load(open(mod_name, "rb")).to(model_conf["device"])
        train_set.configure_for(model)

    except FileNotFoundError:
        if is_multi:
            data_loader = cebra.data.ContinuousMultiSessionDataLoader(
                train_set,
                num_steps=model_conf["train_steps"],
                batch_size=model_conf["batch_size"],
                conditional=model_conf["conditional"],
                time_offset=model_conf["time_offset"],
            ).to(model_conf["device"])

        else:
            data_loader = cebra.data.ContinuousDataLoader(
                train_set,
                num_steps=model_conf["train_steps"],
                batch_size=model_conf["batch_size"],
                conditional=model_conf["conditional"],
                time_offset=model_conf["time_offset"],
            ).to(model_conf["device"])

        model, solver = build_model_session_solver(data_loader, new_conf, is_multi)
        solver.fit(data_loader)
        pickle.dump(model, open(mod_name, "wb"))

    if is_multi:
        emb, test_emb = {}, {}
        i = 0
        for dataset, mod in zip(train_set.iter_sessions(), model):
            mod.eval()
            dataset.configure_for(mod)
            key = f"np_{i}" if str(dataset).find("neuropixel") != -1 else f"ca_{i}"
            emb[key] = mod(dataset[torch.arange(len(dataset))].to(model_conf["device"])).detach().cpu().numpy()
            i += 1
        i = 0
        for dataset, mod in zip(test_set.iter_sessions(), model):
            mod.eval()
            dataset.configure_for(mod)
            key = f"np_{i}" if str(dataset).find("neuropixel") != -1 else f"ca_{i}"
            test_emb[key] = mod(dataset[torch.arange(len(dataset))].to(model_conf["device"])).detach().cpu().numpy()
            i += 1
    else:
        model.eval()
        emb = model(train_set[torch.arange(len(train_set))].to(model_conf["device"])).detach().cpu().numpy()
        test_set.configure_for(model)
        test_emb = model(test_set[torch.arange(len(test_set))].to(model_conf["device"])).detach().cpu().numpy()

    return train_set, test_set, emb, test_emb


def plot_embeddings(embeddings, save=False, out="embeddings.png"):

    n = len(embeddings)
    fig = plt.figure(figsize=(6 * n, 5))

    for k, (label, emb) in enumerate(embeddings.items()):
        scale = 1 if label.find("ca") != -1 else 4
        ax = fig.add_subplot(1, n, k + 1)
        ax.scatter(emb[:, 0], emb[:, 1], cmap="magma", c=np.tile(np.repeat(np.arange(900), scale), 9), s=1)
        ax.set_title(label)
        ax.axis("off")

    if save:
        plt.savefig(out, dpi=140)

    plt.show()


# ===========================================================================
# PART 2: AI-written hippocampus demo
# ===========================================================================


def hippocampus_demo(args):
    np.random.seed(args.seed)
    iters = 500 if args.quick else 5000

    print("=" * 72)
    print("DATA")
    print("=" * 72)
    neural, behavior = load_hippocampus()
    describe("neural", neural)
    describe("behavior", behavior)
    print(f"\n{neural.shape[1]} neurons x {neural.shape[0]} time bins")
    print("behavior column 0 is position; columns 1-2 are direction indicators")

    neural_tr, neural_te = split(neural)
    behav_tr, behav_te = split(behavior)
    pos_tr, pos_te = behav_tr[:, 0], behav_te[:, 0]

    print("\n" + "=" * 72)
    print("TRAINING")
    print("=" * 72)
    embeddings, losses = {}, {}

    # (a) CEBRA-Behavior: positives drawn using the position label
    m_beh = build_model(args, "time_delta", iters)
    embeddings["behavior"] = fit(m_beh, neural_tr, behav_tr, "behavior")
    losses["behavior"] = m_beh.state_dict_["loss"]

    # (b) CEBRA-Time: fully self-supervised, no labels at all
    m_time = build_model(args, "time", iters)
    embeddings["time"] = fit(m_time, neural_tr, None, "time")
    losses["time"] = m_time.state_dict_["loss"]

    # (c) Control: same as (a) but labels shuffled. This is the honest floor.
    #     If it decodes as well as (a), the label wasn't doing any work.
    shuffled = behav_tr.copy()
    np.random.shuffle(shuffled)
    m_shuf = build_model(args, "time_delta", iters)
    embeddings["shuffled labels"] = fit(m_shuf, neural_tr, shuffled, "shuffled")
    losses["shuffled labels"] = m_shuf.state_dict_["loss"]

    print("\n" + "=" * 72)
    print("DECODING POSITION FROM HELD-OUT DATA")
    print("=" * 72)

    for label, model in [("CEBRA-Behavior", m_beh), ("CEBRA-Time", m_time), ("CEBRA-Shuffled (control)", m_shuf)]:
        decode_position(model.transform(neural_tr), pos_tr, model.transform(neural_te), pos_te, label)

    # Baseline: skip the embedding entirely, decode from raw spike counts.
    decode_position(neural_tr, pos_tr, neural_te, pos_te, "raw spikes (no embedding)")

    print("\nRead this table before looking at any picture. CEBRA-Behavior")
    print("should beat both the shuffled control and raw spikes. If it doesn't,")
    print("the embedding is decorative.")

    if not args.no_plot:
        plot_all(embeddings, pos_tr, losses)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="500 iterations instead of 5000")
    p.add_argument("--device", default="cpu", help="cpu | mps | cuda_if_available")
    p.add_argument("--output-dim", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def describe(name, x):
    """Shape/dtype/range of an array, so you always know what you're holding."""
    a = np.asarray(x)
    line = f"{name:<24} {str(a.shape):<16} {a.dtype}"
    if a.size and np.issubdtype(a.dtype, np.number):
        line += f"  min={a.min():.4g} max={a.max():.4g} mean={a.mean():.4g}"
    print(line)


def load_hippocampus():
    """Returns (neural, behavior) as float32 arrays.

    behavior columns are conventionally [position, moving_left, moving_right].
    Verify that against the dataset object printed below rather than trusting
    this docstring — CEBRA's dataset classes have changed over versions.
    """

    candidates = [
        "rat-hippocampus-single-achilles",
        "rat-hippocampus-single",
        "rat-hippocampus-achilles-3fold-trial-split-0",
    ]
    dataset = None
    for name in candidates:
        try:
            dataset = cebra.datasets.init(name)
            print(f"loaded dataset: {name}")
            break
        except Exception as exc:  # noqa: BLE001
            print(f"  '{name}' failed: {type(exc).__name__}: {exc}")
    if dataset is None:
        raise SystemExit('No dataset loaded. List what\'s available with:\n    python -c "import cebra.datasets as d; print(d.get_datalist())"')

    print("\ndataset attributes:")
    for attr in sorted(a for a in dir(dataset) if not a.startswith("_")):
        val = getattr(dataset, attr, None)
        if hasattr(val, "shape"):
            print(f"  .{attr:<20} shape={tuple(val.shape)}")

    neural = np.asarray(dataset.neural, dtype=np.float32)
    behavior = np.asarray(dataset.continuous_index, dtype=np.float32)
    return neural, behavior


def build_model(args, conditional, max_iterations):

    return CEBRA(
        model_architecture="offset10-model",
        batch_size=512,
        learning_rate=3e-4,
        temperature=1.0,
        output_dimension=args.output_dim,
        max_iterations=max_iterations,
        distance="cosine",
        conditional=conditional,
        device=args.device,
        verbose=True,
        time_offsets=10,
    )


def fit(model, neural, behavior=None, label=""):
    t0 = time.time()
    model.fit(neural) if behavior is None else model.fit(neural, behavior)
    emb = model.transform(neural)
    print(f"[{label}] fitted in {time.time() - t0:.1f}s  final loss {model.state_dict_['loss'][-1]:.4f}")
    describe(f"[{label}] embedding", emb)
    return emb


def decode_position(train_emb, train_pos, test_emb, test_pos, label=""):
    """kNN regression from embedding to position. Reports median absolute error
    in the same units as the position variable, plus R^2."""

    knn = KNeighborsRegressor(n_neighbors=36, metric="cosine")
    knn.fit(train_emb, train_pos)
    pred = knn.predict(test_emb)
    err = float(np.median(np.abs(pred - test_pos)))
    r2 = float(r2_score(test_pos, pred))
    print(f"{label:<32} median err = {err:.4f}   R2 = {r2:.3f}")
    return err, r2


def split(x, frac=0.8):
    """Contiguous split. Do NOT use a random split here — adjacent timepoints
    are near-identical, so random splitting leaks test data into training and
    every method looks excellent."""
    n = int(len(x) * frac)
    return x[:n], x[n:]


def plot_all(embeddings, position, losses, out="cebra_exploration.png"):

    n = len(embeddings)
    fig = plt.figure(figsize=(5 * n, 9))

    for k, (label, emb) in enumerate(embeddings.items()):
        ax = fig.add_subplot(2, n, k + 1, projection="3d")
        m = min(len(emb), len(position))
        ax.scatter(emb[:m, 0], emb[:m, 1], emb[:m, 2], c=position[:m], cmap="viridis", s=0.5, alpha=0.6)
        ax.set_title(label)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ax = fig.add_subplot(2, 1, 2)
    for label, loss in losses.items():
        ax.plot(loss, label=label)
    ax.set_xlabel("iteration")
    ax.set_ylabel("InfoNCE loss")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"\nsaved {out}")


def main():
    args = parse_args()
    allen_demo(args)


if __name__ == "__main__":
    main()
