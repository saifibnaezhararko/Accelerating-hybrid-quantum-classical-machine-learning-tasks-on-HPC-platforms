# PennyLane + Qiskit Aer: circuits, backend port, and two hybrid routes

Software Lead work folder. Five questions, answered with runnable scripts and
measured numbers rather than assertions:

1. Build a PennyLane circuit and run it on Qiskit Aer.
2. Test Evelyn's `mc1_iqp_cups` code.
3. Use Qiskit Aer as the backend for the existing lambeq pipeline.
4. `convert -> CNN -> hybrid quantum PennyLane`.
5. Run **MC1 through dimensionality reduction** into a narrow circuit, and get
   the lambeq route's accuracy at 2 qubits instead of 18.

Everything here is additive. **No contributor file was modified.** The only edits
outside this folder are two bug fixes in `src/qnlp_hpc/backends/registry.py`
(Software Lead's own file), both caused by findings below.

```
pennylane_aer/
├── README.md                      this file
├── aer_backend.py                 the Aer backend_config that actually works
├── hybrid_cnn_quantum.py          convert / CNN / quantum layer
├── mc1_reduced.py                 MC1 -> embedding -> PCA/TSVD/UMAP -> circuit
├── test_evelyn_iqp.py             25 tests for mc1_iqp_cups
├── test_mc1_reduced.py            27 tests for the reduced route
├── 01_pennylane_aer_circuit.py    PennyLane circuit on Aer, 4 configurations
├── 03_lambeq_aer_backend.py       Evelyn's IQP model ported onto Aer
├── 04_train_cnn_hybrid.py         hybrid training vs classical control
├── 05_performance_plots.py        figures, rendered from the JSON reports
├── 06_train_mc1_reduced.py        MC1 reduced route: 6 stages, multi-seed
└── outputs/                       JSON reports + run logs + PNG figures
```

Run anything with `PYTHONPATH=src` from the repo root.

---

## Headline: the two traps that make "lambeq on Aer" silently wrong

Both are silent. Neither raises anything you would notice in training output.

### Trap 1 — `qiskit.aer` is not analytic by default

`qml.device("qiskit.aer", wires=n)` with `shots=None` looks exact. It is not.
The default Aer backend is `aer_simulator`, on which pennylane-qiskit cannot
compute exact probabilities, so it **falls back to sampling** and warns once.

Measured (`01_pennylane_aer_circuit.py`, 4 qubits, deviation from `default.qubit`):

| configuration | max abs deviation | verdict |
|---|---|---|
| Aer, default backend, `shots=None` | **4.44e-03** | silently sampled |
| Aer, `aer_simulator_statevector` | **1.53e-16** | genuinely exact |
| Aer, statevector, 4096 shots | 6.40e-03 | ~1/sqrt(shots), as expected |

Fix: always pass `backend="aer_simulator_statevector"`.

### Trap 2 — lambeq's circuits post-select, so shots produce NaN

lambeq's `cups_reader` + `IQPAnsatz` circuits are dominated by post-selection.
For MC1 at `SENTENCE_QUBITS=2`:

| sentence length | qubits | post-selected | P(post-selection succeeds) |
|---|---|---|---|
| 3 words | 14 | 12 | ~2⁻¹² |
| 8 words | 18 | 16 | ~2⁻¹⁶ |

At 1024 shots, **zero shots survive** post-selection. The unnormalised
probabilities are exactly 0, lambeq normalises them, and `0/0` gives **NaN** —
a whole batch of NaN logits, not an exception.

So: **shot-based Aer cannot evaluate lambeq's circuits at all.** Only the exact
statevector path works. This has a real consequence for the project — the
"shots" knob is not available for these circuits, and neither is real hardware
without a different ansatz.

### The naming that causes both

lambeq's `backend_config` uses `device` for something that is not a device:

```python
backend_config = {
    "backend": "qiskit.aer",                  # the PennyLane plugin
    "device":  "aer_simulator_statevector",   # renamed by lambeq to `backend`,
}                                             # i.e. Aer's own backend name
```

Consequence: `{"device": "GPU"}` — the obvious way to ask for GPU, and what the
backend registry previously did — raises `Backend 'GPU' does not exist`. There
is no free key for the GPU flag, so the **only** route is a pre-configured
simulator instance:

```python
from qiskit_aer import AerSimulator
backend_config = {
    "backend": "qiskit.aer",
    "device":  AerSimulator(method="statevector", device="GPU"),
}
```

`aer_backend.aer_backend_config(gpu=True)` builds exactly this.

---

## 1. PennyLane circuit on Qiskit Aer

`01_pennylane_aer_circuit.py` — an IQP-shaped block (Hadamard → CRZ ladder →
rotations) run on four backends with PyTorch autograd. 4 qubits, this machine
(CPU-only Aer build):

| backend | forward | backward | gradients |
|---|---|---|---|
| `default.qubit` | 6.6 ms | 51 ms | yes |
| Aer, default backend | 202 ms | 4428 ms | yes (but sampled) |
| Aer, statevector | 127 ms | 3731 ms | yes |
| Aer, statevector + 4096 shots | 132 ms | 3635 ms | yes |

Aer is **~19× slower forward and ~73× slower backward** than `default.qubit` at
this size. Gradients flow through Aer in every case — the PyTorch↔Qiskit seam
works — but the backward cost is the thing GPU work has to attack.

## 2. Testing Evelyn's code

The repo suite (96 tests) reports **0% coverage of `mc1_iqp_cups`**, so nothing
was checking the model that produces the headline numbers. `test_evelyn_iqp.py`
adds **25 tests, all passing**, covering the claims the code makes about itself:
loader validation, split conservation and disjointness, held-out vocabulary
coverage, circuit/symbol construction, and the model contract.

Two tests worth calling out:

- **`test_pair_model_is_symmetric`** — the docstring calls it a symmetric
  classifier; both pair features (`|l−r|`, `l*r`) are swap-invariant, so
  `forward(a,b)` must equal `forward(b,a)`. It does. An asymmetric feature would
  break this.
- **`test_gradients_reach_the_lambeq_circuit_symbols`** — confirms the loss moves
  the *quantum* parameters, not just the classifier head. Without it the circuit
  could be a frozen random feature map and the model would still look like it
  trains.

**End-to-end run** (`outputs/02_evelyn_baseline_run.log`), full 120 epochs:

```
Trainable lambeq symbols: 49
Training completed in 1308.68 s
Selected epoch: 118 (minimum development loss)
Test accuracy: 1.0000
Test prediction counts: {0: 7, 1: 6}
```

**Evelyn's model is correct and works.** Test accuracy 1.0 with balanced
predictions, against 35% (below chance) for the earlier `spiders_reader`
baseline. The sentence-disjoint split plus `cups_reader` + `IQPAnsatz` fixed it.

### Issues found (for the owning contributors, not changed here)

| # | Issue | Impact |
|---|---|---|
| 1 | `scripts/train_mc1_iqp_cups.py` has no `sys.path` bootstrap | `ModuleNotFoundError: No module named 'qnlp_hpc'` on a clean checkout without an editable install. `scripts/train_mc1_spider.py` already has the fix; copy it. Workaround: `PYTHONPATH=src`. |
| 2 | `mc1_iqp_cups_gpu` duplicates `mc1_iqp_cups` almost verbatim | ~980 lines; the only differences are `GPU_DEVICE`, CUDA assertions, `device=` and three summary fields. Any fix to the CPU version must be applied twice. A `device` config field would remove the whole module. |
| 3 | Training never early-stops | Selected epoch 118 of 120 with patience 20 — dev loss was still creeping down, so the run is compute-bound, not converged. Worth more epochs or a stricter criterion. |
| 4 | `load_mc1` uses `rsplit(",", 2)` | A comma inside a sentence silently shifts the delimiter. Pinned by a test so the behaviour is known, not surprising, on future datasets. |

## 3. Qiskit Aer as the lambeq backend

`03_lambeq_aer_backend.py` performs the port CLAUDE.md assigns to the Quantum ML
Lead ("port TKet → Qiskit Aer, validate identical predictions"), against the
model that exists, without editing it.

The key structural point: **Evelyn's `IQPPairModel` does not simulate anything.**
It derives from `PytorchQuantumModel`, which contracts the circuit as a tensor
network. Swapping the base class to `PennyLaneModel` — same `forward` body, same
pair features, same head — routes the identical circuits through a real
simulator. Weights are copied **by symbol name** (`aer_backend.copy_weights`),
because the two model classes build their own symbol orderings.

8 MC1 test pairs, 18-qubit circuits, identical weights:

| backend | time | vs baseline | max abs logit diff | same predictions |
|---|---|---|---|---|
| tensor contraction (Evelyn's) | 3.33 s | 1.0× | — | — |
| PennyLane `default.qubit` | 6.83 s | 2.1× | 5.96e-08 | **yes** |
| **Qiskit Aer statevector** | **4.28 s** | **1.3×** | **5.96e-08** | **yes** |
| Aer, 1024 shots | 4.07 s | 1.2× | NaN | **no** |

5.96e-08 is float32 epsilon — that is exact agreement. **The port is validated:
Aer reproduces the baseline predictions exactly**, and at 18 qubits it costs only
1.3× the tensor contraction, so it is a viable quantum layer, not just a
demonstration.

### Registry bugs this found (fixed in `src/qnlp_hpc/backends/registry.py`)

- `pennylane-qiskit-aer` did not pin the statevector backend → Trap 1, and on
  post-selected circuits Trap 2 (NaN). Now uses `aer_simulator_statevector`.
- `pennylane-qiskit-aer-gpu` passed `device="GPU"`, which raises
  `Backend 'GPU' does not exist` — the GPU backend could never have worked. Now
  builds an `AerSimulator(method="statevector", device="GPU")` instance.

`scripts/check_backends.py --verify` after the fix: `pennylane-qiskit-aer` **ok,
gradients flow**.

## 4. convert → CNN → hybrid quantum PennyLane

The lambeq route ties circuit width to sentence length (18 qubits for 8 words),
which does not scale to TREC. `hybrid_cnn_quantum.py` takes the other route from
CLAUDE.md §3 — *embedding → quantum layer → classifier*:

```
convert   TrecDataset   CSV -> vocabulary (train only) -> padded id sequences
cnn       TextCNN       embedding -> Conv1d(2,3,4) -> max-pool -> n_qubits angles
quantum   TorchLayer    AngleEmbedding + StronglyEntanglingLayers -> <Z_i>
head      Linear(n_qubits, 2)
```

Circuit width becomes a **hyperparameter** (3–8 qubits, the NLP lead's target)
instead of a function of sentence length, and nothing is post-selected — so
unlike the lambeq circuits this one runs correctly under shots and on hardware.
The `tanh`→`π` scaling on the CNN output matters: unbounded activations would
wrap around the Bloch sphere and alias different sentences onto the same angle.

Reduced TREC (174 train / 44 test, DESC vs HUM, vocab 360), 4 qubits × 2 layers,
seed 0, 30 epochs:

| configuration | final test acc | best test acc | s/epoch | quantum params |
|---|---|---|---|---|
| classical control (CNN → linear) | 0.9091 | 0.9545 | 0.11 | 0 |
| hybrid, `default.qubit` | 0.8636 | 0.9091 | 0.70 | 24 |
| hybrid, Qiskit Aer | not trained | — | **1368 (projected)** | 24 |

The classical control is not decoration: without it, "the hybrid got 0.86" cannot
be distinguished from "the CNN did all the work". At this scale the quantum
bottleneck **costs** ~5 points of accuracy and ~6× the time — an honest result,
and the expected one, since a 4-qubit circuit is a narrower bottleneck than the
linear layer it replaces on a task a CNN already solves.

**Aer was not trained to convergence, because it cannot be on this CPU.** The
script instead runs real optimisation steps and measures them
(`benchmark_backend`), which is the honest thing to report and the number the
GPU work has to beat:

```
3 optimisation steps, batch size 2
mean step: 15.72 s  ->  7.86 s per sentence
projected: 22.8 min/epoch, 11.4 h for 30 epochs
gradient reaching circuit weights: 0.762808
```

The gradient check matters: it confirms Aer is genuinely training the circuit and
is merely slow, not broken. **~1950× slower per epoch than `default.qubit`** — this
is precisely the "more than one hour on a CPU-based simulator" problem CLAUDE.md
§1 sets out to solve, now reproduced on a model we control.

### Full CNN + TREC training on an Aer GPU

The script keeps the cheap benchmark as its default, but can now train the actual
Aer-backed hybrid model for complete epochs. `--only-aer` avoids repeating the
classical and `default.qubit` controls when only the GPU stage is needed:

```bash
PYTHONPATH=src python pennylane_aer/04_train_cnn_hybrid.py --gpu --only-aer \
  --aer-mode train --aer-epochs 1 --aer-batch-size 4 \
  --aer-learning-rate 0.005 --aer-diff-method parameter-shift
```

Start with one epoch because this route differentiates both the circuit weights
and the CNN-produced circuit inputs through an external simulator. Every history
row in `outputs/04_cnn_hybrid_quantum.json` records loss, train/test accuracy,
and measured epoch seconds. When `--gpu` is requested but Aer exposes no GPU,
the script exits instead of silently running the expensive job on CPU.

### A third compatibility finding: parameter-shift cannot batch

Running the hybrid on Aer initially failed:

```
NotImplementedError: Computing the gradient of broadcasted tapes with respect to
the broadcasted parameters using the parameter-shift rule ... is not supported (#4462)
```

`default.qubit` differentiates by **backprop through the simulator**, which
handles a broadcast batch fine. Aer has no backprop, so PennyLane falls back to
**parameter-shift**, which cannot differentiate a broadcasted tape. External
simulators must therefore be fed **one sample at a time**
(`HybridTextClassifier(broadcast=False)`, selected automatically).

This compounds badly and is the core scaling problem for the GPU work: Aer pays
2×n_params circuit evaluations per sample for the gradient **and** loses batch
parallelism. At 24 parameters that is ~49 circuit executions per sentence per
step.

---

## 5. MC1 through dimensionality reduction

`mc1_reduced.py` + `06_train_mc1_reduced.py`. Section 4 applied the
*embedding -> quantum layer -> classifier* route to TREC; this applies it to
**MC1**, so it can be compared directly against the lambeq baseline on the same
sentence-disjoint split:

```
sentence -> TF-IDF / bag-of-words / BERT -> PCA | TSVD | UMAP -> n_qubits angles
         -> AngleEmbedding + StronglyEntanglingLayers (re-uploading) -> <Z_i>
pair     -> |l - r|, l * r  (swap-invariant, as in Evelyn's IQPPairModel) -> head
```

Split membership is imported from `mc1_iqp_cups.config`, so the train/dev/test
boundary is Evelyn's, unchanged.

```
python pennylane_aer/06_train_mc1_reduced.py --qubits 2 --seeds 0-4 \
    --qubit-sweep 2,3,4,6,8 --keep-history --aer-epochs 40
```

`--only training,scaling,augmentation,width,aer-eval,aer-training` re-runs a
subset and merges it into the existing report — so stage 6 can be repeated on a
GPU node without spending an hour reproducing the rest.

### MC1's topic labels are recoverable from MC1 itself

MC1 labels a pair 1 when the two sentences share a topic and 0 when they do not.
That makes the sentence graph a **2-colouring problem** — label-1 edges force
equal colours, label-0 edges force opposite ones. It resolves into exactly 2
components covering all 83 sentences, and the resulting assignment reproduces
**all 100 shipped labels with 0 disagreements**. No lexicon was needed, and
`derive_sentence_topics` raises rather than guess if a future dataset is not
2-colourable.

This licenses forming *all* pairs within a split: MC1 ships 13 test pairs over
its 11 test sentences, but all **55** possible pairs carry a label the dataset
already determined. Nothing crosses the sentence-disjoint boundary, so this adds
supervision without adding leakage.

| split | sentences | shipped pairs | all pairs |
|---|---|---|---|
| train | 58 | 64 | **1,653** |
| development | 14 | 13 | 91 |
| test | 11 | 13 | **55** |

Reporting on 55 test pairs instead of 13 also raises the resolution of the
accuracy estimate from 7.7 points per pair to 1.8.

### Result: the lambeq route's accuracy at 2 qubits

2 qubits, 2 layers, 2 re-uploads, TF-IDF + PCA, 5 seeds, 60 epochs:

| model | quantum params | total params | test (55 pairs, 95% CI) | MC1's own 13 pairs | s/epoch |
|---|---|---|---|---|---|
| **quantum (2-qubit circuit)** | 12 | 126 | **1.0000 [1.0000, 1.0000]** | **1.0000** | 0.42 |
| classical control (Linear + tanh) | 0 | 120 | 1.0000 [1.0000, 1.0000] | 1.0000 | 0.06 |
| no bottleneck (head only) | 0 | 114 | 1.0000 [1.0000, 1.0000] | 1.0000 | 0.04 |

Against the lambeq baseline (§2) on the identical test pairs:

| | lambeq `cups_reader` + IQP | this route |
|---|---|---|
| qubits | 18 | **2** |
| post-selected qubits | 16 (P ~ 2⁻¹⁶) | **0** |
| MC1 13-pair test accuracy | 1.0000 (1 seed) | 1.0000 (**5/5 seeds**) |
| training time | 1308.7 s | **~25 s** |
| shots / real hardware | impossible (NaN) | works |

**Read the control column before the quantum one.** All three models saturate,
so the honest statement is that the quantum layer *matches* the controls — it
does not beat them. On a task a 114-parameter head already solves, a 2-qubit
circuit cannot demonstrate an advantage; what it demonstrates is that the
NISQ-shaped route loses nothing while removing 16 qubits and all post-selection.

### Two ablations, and one that changes the answer

**Angle scaling.** Reduced components must be mapped into the rotation range.
Scaling each component to unit variance (`per-component`) lifts the
lowest-variance components — which on MC1 carry syntax, not topic — up to the
amplitude of the topic-carrying one, so the circuit sees noise at signal
strength. A single `global` scale preserves the variance ordering PCA produced.

| scaling | 2 qubits | 4 qubits |
|---|---|---|
| global | **1.0000** | **0.9491** |
| per-component | 0.9673 | 0.7673 |

The gap widens with width exactly as the mechanism predicts — more retained
components means more low-variance directions to inflate.

**Augmentation.** Training on MC1's 64 shipped training pairs versus all 1,653
its training sentences allow, with the evaluation splits held fixed:

| training pairs | test (55) | MC1's own 13 pairs |
|---|---|---|
| 64 (shipped) | 0.7673 | 0.7846 |
| 1,653 (augmented) | **1.0000** | **1.0000** |

**This is the honest caveat on the comparison above.** Trained on the same 64
pairs the lambeq model saw, this route reaches 0.7846 on the 13 test pairs,
*below* the lambeq baseline's 1.0000. The advantage comes from the augmented
supervision and the cost profile — not from the reduced representation being
intrinsically stronger than `cups_reader` + IQP. Grammar buys sample efficiency;
this route buys width, shots, and speed.

### More qubits make it worse

Circuit width sweep, 5 seeds per point, classical control swept alongside:

| qubits | explained variance | quantum params | quantum test | classical test |
|---|---|---|---|---|
| **2** | 0.385 | 12 | **1.0000 [1.0000, 1.0000]** | 1.0000 |
| 3 | 0.494 | 18 | 0.8909 [0.7472, 1.0000] | 1.0000 |
| 4 | 0.596 | 24 | 0.9491 [0.8555, 1.0000] | 1.0000 |
| 6 | 0.753 | 36 | 0.6764 [0.4602, 0.8925] | 1.0000 |
| 8 | 0.871 | 48 | 0.6836 [0.4553, 0.9120] | 1.0000 |

Accuracy **falls** as width grows while explained variance *rises*, and the
classical control stays at 1.0000 throughout — so this is optimisation, not
representation. The per-epoch histories confirm it: at 4 qubits, seed 3's
training loss falls 0.67 -> 0.008 while its development loss climbs
0.70 -> 14.80. The extra circuit parameters memorise the 58 training sentences.

For the Quantum ML Lead: **2 qubits is the right width for MC1**, and any move
to a larger dataset should re-run this sweep rather than assume more qubits
help. Regularising the circuit (dropout on the quantum features, stronger weight
decay, fewer entangling layers) is the obvious next lever and is not tried here.

### Aer: shots work on this route

Weights trained on `default.qubit` are transferred to Aer and evaluated:

| backend | seconds | test (55) | agreement with `default.qubit` | max abs logit diff |
|---|---|---|---|---|
| Aer statevector, exact | 3.12 | 1.0000 | 1.0000 | 4.29e-06 |
| Aer statevector, 1024 shots | 3.00 | **1.0000** | **1.0000** | 9.45e-01 |
| Aer statevector, 8192 shots | 3.32 | **1.0000** | **1.0000** | 2.63e-01 |

This is the direct contrast with §3. On the lambeq circuits, 1024 shots produce
**NaN** — 2⁻¹⁶ post-selection means zero shots survive. Here nothing is
post-selected, so shots merely add sampling noise: the logits move by ~1 at 1024
shots and **every prediction still matches the exact run**. The decision
boundary sits far enough from the data to absorb it.

That makes this the configuration to try on real hardware, and the one where
`lightning.gpu` adjoint differentiation is worth retesting (§3's adjoint
blocker was post-selection).

### Aer training cost: SPSA is what makes it affordable

Because pair features are indexed from per-sentence circuit outputs, one epoch
costs **58 circuit evaluations — the number of distinct training sentences — no
matter that it covers all 1,653 training pairs.** Evelyn's model evaluates two
circuits per pair.

| diff_method | circuit evals / sentence | s / sentence | projected min / epoch |
|---|---|---|---|
| SPSA | 2 | 0.48 | **0.46** |
| parameter-shift | 24 | 3.33 | 3.22 |

Parameter-shift costs `2 x n_params` evaluations; SPSA costs 2 regardless, which
is **7.0x cheaper per epoch** at this width.

### Aer trained from scratch, to convergence

Section 4 could only *project* Aer's training cost — 11.4 h for 30 epochs, never
run. Here the run completes:

```
40 epochs, SPSA, full batch (all 1,653 pairs), 2 qubits
24.0 s/epoch  ->  16.0 min total
selected epoch 17 (minimum development loss)
test accuracy 1.0000   MC1's own 13 pairs 1.0000
```

**Qiskit Aer trained this model from random initialisation to the same 1.0000
the exact simulator reaches**, on a CPU, in 16 minutes.

The step size mattered more than the backend. A first run at lr 0.2 — a value
tuned against exact gradients — reached only 0.8182, with training loss falling
while development loss oscillated between 0.64 and 0.95. That looked like SPSA
being fundamentally noisier, but it was mistuning: on `default.qubit`, SPSA
reaches 1.0000 at lr 5e-2 and 0.9091 at 2e-1, while backprop reaches 1.0000 at
both. **A gradient estimated from two evaluations needs a smaller step than an
exact one.** `--aer-learning-rate` now defaults to 5e-2.

## 6. Figures

`05_performance_plots.py` renders the numbers above straight from
`outputs/*.json` — it re-runs no experiment, so it is cheap and stays correct
after any of 01/03/04 is re-executed (on a GPU node, for instance):

```
python pennylane_aer/05_performance_plots.py
```

| figure | source | what it shows |
|---|---|---|
| `05_perf_micro_backends.png` | 01 | forward/backward cost per backend, and the deviation-from-exact that exposes Trap 1 |
| `05_perf_lambeq_port.png` | 03 | 18-qubit port: Aer at 1.29x baseline with identical predictions, shots invalid |
| `05_perf_cnn_training.png` | 04 | loss and test-accuracy curves, hybrid vs classical control |
| `05_perf_epoch_cost.png` | 04 | seconds/epoch across devices — the 1,968x gap the GPU work has to close |
| `05_perf_mc1_accuracy.png` | 06 | quantum layer vs both controls, and the angle-scaling ablation |
| `05_perf_mc1_width.png` | 06 | accuracy vs circuit width against rising explained variance |
| `05_perf_mc1_aer.png` | 06 | Aer exact and shot-based accuracy, and SPSA vs parameter-shift cost |

Ratio-like spreads use a log axis, and on a log axis bar length is meaningless,
so those figures are dot plots rather than bars. Any run that produced NaN is
drawn in grey and labelled, never silently omitted.

---

## Consequences for the project

- **Adjoint differentiation is still unavailable** for the lambeq path.
  CLAUDE.md §3 lists it as a method to use, but lightning rejects adjoint on
  post-selected circuits, and Aer offers no backprop. Everything falls back to
  parameter-shift. The routes in §4 and §5 have no post-selection, so they are
  where adjoint on `lightning.gpu` is worth retesting first.
- **Post-selection, not qubit count, is the near-term ceiling** on the lambeq
  route. 18 qubits is easy for any simulator; 2⁻¹⁶ post-selection is what rules
  out shots and hardware. §5 shows the cost of removing it: the reduced route
  runs at 2 qubits with shots and on hardware, but needs augmented supervision
  to match `cups_reader` + IQP on MC1's own 64 training pairs. Grammar buys
  sample efficiency; reduction buys width, shots, and speed.
- **For the HPC Specialist:** `qiskit-aer` here exposes `['CPU']` only, so both
  GPU backends remain unverified. The GPU config is now at least *correct* —
  `aer_backend_config(gpu=True)` — so it can be tested on the node by running
  `03_lambeq_aer_backend.py --gpu`, `04_train_cnn_hybrid.py --gpu`, and
  `06_train_mc1_reduced.py --gpu --only aer-eval,aer-training`. The last is the
  cheapest GPU smoke test in the folder — it needs no lambeq parse and merges
  into the existing report.
- **Batch-size-1 on external simulators** should be assumed when planning GPU
  benchmarks; the win has to come from per-circuit speed, not batching, unless
  adjoint becomes available. §5 shows a second lever that is not batching:
  making circuit cost track *distinct sentences* rather than training examples.
- **SPSA needs a smaller step size than backprop.** Measured on `default.qubit`
  at 2 qubits, SPSA reaches 1.0000 at lr 5e-2 but only 0.9091 at 2e-1, where
  backprop reaches 1.0000 at both. A gradient estimated from two evaluations is
  noisy, and a step size tuned against exact gradients will overshoot it. Worth
  knowing before SPSA is used for the large HPC runs CLAUDE.md §4 plans.
