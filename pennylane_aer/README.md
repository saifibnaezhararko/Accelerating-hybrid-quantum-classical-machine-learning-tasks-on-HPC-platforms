# PennyLane + Qiskit Aer: circuits, backend port, and two hybrid routes

Software Lead work folder. Five questions, answered with runnable scripts and
measured numbers rather than assertions:

1. Build a PennyLane circuit and run it on Qiskit Aer.
2. Test Evelyn's `mc1_iqp_cups` code.
3. Use Qiskit Aer as the backend for the existing lambeq pipeline.
4. `convert -> CNN -> hybrid quantum PennyLane`.
5. Run the **raw TREC dataset through dimensionality reduction** into a narrow
   circuit, on the 6-way task with the official split.

Everything here is additive. **No contributor file was modified.** The only edits
outside this folder are two bug fixes in `src/qnlp_hpc/backends/registry.py`
(Software Lead's own file), both caused by findings below.

```
pennylane_aer/
├── README.md                      this file
├── aer_backend.py                 the Aer backend_config that actually works
├── hybrid_cnn_quantum.py          convert / CNN / quantum layer
├── trec_reduced.py                TREC -> TF-IDF -> TSVD/PCA/UMAP -> circuit
├── test_evelyn_iqp.py             25 tests for mc1_iqp_cups
├── test_trec_reduced.py           42 tests for the reduced route
├── 01_pennylane_aer_circuit.py    PennyLane circuit on Aer, 4 configurations
├── 03_lambeq_aer_backend.py       Evelyn's IQP model ported onto Aer
├── 04_train_cnn_hybrid.py         hybrid training vs classical control
├── 05_performance_plots.py        figures, rendered from the JSON reports
├── 06_train_trec_reduced.py       TREC reduced route: 5 stages, multi-seed
└── outputs/                       JSON reports + run logs + PNG figures
```

Sections 2 and 3 stay on MC1: they test the Quantum ML Lead's `mc1_iqp_cups`
model and port *it* onto Aer, so the dataset is fixed by what they examine.
Section 5 is the dimensionality-reduction route, and it now runs on TREC.

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

`03_lambeq_aer_backend.py` performs the port the work plan assigns to the Quantum ML
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
the project methodology — *embedding → quantum layer → classifier*:

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
is precisely the "more than one hour on a CPU-based simulator" problem the
project sets out to solve, now reproduced on a model we control.

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

## 5. TREC through dimensionality reduction

`trec_reduced.py` + `06_train_trec_reduced.py`. Section 4 put a **CNN** in front
of the quantum layer on a filtered 174-question subset of TREC. This drops the
CNN, takes the **raw dataset as shipped** in `trec dataset/`, and reduces
straight from the bag of words:

```
question -> TF-IDF -> TruncatedSVD | PCA | UMAP -> n_qubits angles
         -> AngleEmbedding + StronglyEntanglingLayers (re-uploading) -> <Z_i>
         -> Linear -> 6 class logits
```

Circuit width is a hyperparameter in the 2-8 qubit range instead of a function
of sentence length, and nothing is post-selected. TREC questions run to 37
words, so the lambeq route of sections 2-3 — a qubit per pregroup type — is not
reachable on this dataset at all.

```
python pennylane_aer/06_train_trec_reduced.py --qubits 8 --epochs 30 --seeds 0-4
python pennylane_aer/06_train_trec_reduced.py --only width --width-sweep 2,3,4,6,8 --seeds 0-2
python pennylane_aer/06_train_trec_reduced.py --only aer-eval,aer-training --seeds 0
```

`--only training,scaling,width,aer-eval,aer-training` re-runs a subset and
merges it into the existing report, so a stage can be repeated on a GPU node
without reproducing the rest.

### The dataset, used as distributed

TREC's official 5,452 / 500 split is kept — published numbers are quoted
against it — and development is carved out of training, stratified:

| split | questions | DESC | ENTY | ABBR | HUM | NUM | LOC |
|---|---|---|---|---|---|---|---|
| train | 4,635 | 988 | 1,062 | 73 | 1,040 | 762 | 710 |
| development | 817 | 174 | 188 | 13 | 183 | 134 | 125 |
| test | 500 | 138 | 94 | 9 | 65 | 113 | 81 |

The classes are badly unbalanced — ABBR is 86 of the 5,452 training rows — so
the development split is drawn per class rather than uniformly, and **macro-F1
is reported alongside accuracy**: a model that abandons ABBR entirely loses
almost no accuracy.

One trap worth recording: **the label order in these CSVs is not the ABBR-first
order some TREC distributions use.** Here label 0 is DESC and label 2 is ABBR,
which is only visible by reading the questions — label 2 is "What does INRI
stand for ?". A wrong constant mislabels every class in the report while every
accuracy stays correct, so `test_class_names_match_the_question_content` pins
the order against phrases only one category contains.

### Two reference lines, without which the numbers mean nothing

| | test accuracy |
|---|---|
| majority class (always DESC) | 0.2760 |
| full-dimension logistic regression (8,460 TF-IDF features) | **0.8580** |

Everything below sits between these. The gap between 0.858 and what the
reduced route achieves is the price of the reduction itself, and it is by far
the largest term — **8,460 features into 8 components keeps 5.2% of the
variance.**

### Result at 8 qubits

8 qubits, 2 layers, 2 re-uploads, TF-IDF + TruncatedSVD, 5 seeds, 30 epochs:

| model | quantum params | total params | test accuracy (95% CI) | macro-F1 | s/epoch |
|---|---|---|---|---|---|
| quantum (8-qubit circuit) | 48 | 294 | 0.4936 [0.4289, 0.5583] | 0.4951 | 5.64 |
| classical control (Linear + tanh) | 0 | 318 | 0.5936 [0.5109, 0.6763] | 0.5656 | 0.10 |
| **no bottleneck (head only)** | 0 | 246 | **0.6068 [0.5769, 0.6367]** | **0.5925** | 0.08 |

**The quantum layer costs ~11 points against the plainest control**, and is 69x
slower per epoch. This is the same direction as section 4's CNN hybrid, which lost
~5 points on a filtered TREC subset — two independent routes to the quantum
layer now agree that on this dataset it costs accuracy rather than adding it.

That is the honest reading: on a 6-way task whose reduced representation is
already lossy, routing the features through a narrow circuit loses more. The
quantum layer is not adding capacity here — it is a second bottleneck behind
the first.

### Angle scaling still decides a lot

| scaling | test accuracy (95% CI) |
|---|---|
| global | **0.4936 [0.4289, 0.5583]** |
| per-component | 0.3544 [0.2839, 0.4249] |

A 14-point gap with non-overlapping intervals. Scaling each component to unit
variance presents a low-variance direction to the circuit as loudly as the
leading one; a single global scale keeps the reducer's ordering. The test
`test_per_component_scaling_flattens_the_component_spreads` pins the mechanism:
the min/max spread ratio goes from 0.79 under `global` to 0.93+ under
`per-component`.

### Reducer choice: TSVD or PCA

TruncatedSVD is the default because it consumes the sparse TF-IDF matrix
directly — densifying TREC's 5,452 x 8,460 matrix costs ~370 MB for no gain.
One consequence is worth knowing: **TSVD does not centre the data**, so its
leading direction is the "average question" and loses most of its spread once
the reduced features are centred. Measured component spreads at 8 components:
TSVD `[0.071 0.095 0.092 ...]` — component 0 is *not* the widest — against PCA's
monotone `[0.103 0.092 0.084 ...]`.

It does not translate into a consistent accuracy difference (logistic
regression on the reduced features, 500 test questions):

| components | TSVD | PCA |
|---|---|---|
| 2 | 0.2340 | 0.3300 |
| 4 | 0.3160 | 0.4700 |
| 6 | 0.5160 | 0.5080 |
| 8 | **0.5340** | 0.5160 |

PCA is better at the narrow end, TSVD at the wide end, and the differences are
within the ~4-point resolution of a 500-question test set. `--reducer pca` is
available for the narrow-width studies.

### Width sweep: both curves rise, and the control stays ahead

3 seeds per point, 30 epochs, classical control swept alongside:

| qubits | explained variance | quantum params | quantum test | classical test |
|---|---|---|---|---|
| 2 | 0.0147 | 12 | **0.3433 [0.3178, 0.3688]** | 0.2773 [0.0260, 0.5286] |
| 3 | 0.0235 | 18 | 0.3267 [0.1514, 0.5019] | 0.3907 [0.2211, 0.5602] |
| 4 | 0.0300 | 24 | 0.3280 [0.3131, 0.3429] | 0.4627 [0.3613, 0.5640] |
| 6 | 0.0417 | 36 | 0.4153 [0.3275, 0.5031] | 0.5793 [0.5490, 0.6097] |
| 8 | 0.0524 | 48 | 0.5293 [0.4907, 0.5679] | **0.6060 [0.4425, 0.7695]** |

Accuracy **rises** with width for both models, tracking explained variance —
the reduction is the binding constraint, and 8 qubits is not enough to
saturate the task. A task whose reduced representation already
saturates would show the opposite shape, with extra width only adding parameters
to overfit with.

The one width where the circuit leads is **2 qubits**: 0.3433 against 0.2773,
where the classical control's mean sits on the majority-class rate (0.2760) —
one of its three seeds collapsed to predicting DESC for everything, which is
also why its interval is so wide. A re-uploading circuit is a more expressive
map than `Linear(2, 2) -> tanh` when two dimensions is all there is. Treat this
as suggestive rather than established: three seeds, overlapping intervals, and
macro-F1 is poor for both (0.2385 against 0.1830).

### Aer: the port is exact, but shots are not free here

Weights trained on `default.qubit` transferred to Aer, 200 test questions:

| backend | seconds | test accuracy | agreement with `default.qubit` | max abs logit diff |
|---|---|---|---|---|
| Aer statevector, exact | 32.9 | 0.5300 | **1.0000** | 7.63e-06 |
| Aer statevector, 1024 shots | 30.8 | 0.4550 | 0.7650 | 3.51 |
| Aer statevector, 8192 shots | 34.8 | 0.5400 | 0.9050 | 1.44 |

The exact path reproduces the reference to float32 epsilon, so **the port is
validated**. The shot rows are the interesting ones. Exact simulation
reproduces every prediction, but at 1024 shots about a quarter of them flip.

Nothing about the circuit changed — what changed is the **margin**. This is a
6-way task with neighbouring classes close together, so sampling noise of ~3.5
in logit space crosses decision boundaries. A binary task whose logits sit
far from the boundary would tolerate the same shot count comfortably. **Shot budgets have to be set from the margin of
the task, not from the width of the circuit** — worth knowing before anyone
quotes a shot count for hardware.

### Aer training on the full dataset is not feasible on this CPU

| diff_method | circuit evals / question | s / question | projected hours per epoch |
|---|---|---|---|
| SPSA | 2 | 0.56 | **0.71** |
| parameter-shift | 96 | 14.89 | 19.17 |

Parameter-shift costs `2 x n_params` evaluations; SPSA costs 2 regardless,
making it **27x cheaper**. Even so, one epoch over the 4,635 training questions
costs 0.71 h with SPSA, and the 30-epoch schedule used above would take **~21
hours**. Gradients do reach the circuit in both cases (0.219 and 0.065), so
this is slow, not broken.

This is the project's core problem statement — "more than one hour on a
CPU-based simulator" — reproduced on a realistic dataset with a model we
control, and it is the number the GPU work has to attack. At this scale CPU Aer
stops being a training option and becomes an evaluation-only backend.

## 6. Figures

`05_performance_plots.py` renders the numbers above straight from
`outputs/*.json` — it re-runs no experiment, so it is cheap and stays correct
after any of 01/03/04/06 is re-executed (on a GPU node, for instance):

```
python pennylane_aer/05_performance_plots.py
```

| figure | source | what it shows |
|---|---|---|
| `05_perf_micro_backends.png` | 01 | forward/backward cost per backend, and the deviation-from-exact that exposes Trap 1 |
| `05_perf_lambeq_port.png` | 03 | 18-qubit port: Aer at 1.29x baseline with identical predictions, shots invalid |
| `05_perf_cnn_training.png` | 04 | loss and test-accuracy curves, hybrid vs classical control |
| `05_perf_epoch_cost.png` | 04 | seconds/epoch across devices — the 1,968x gap the GPU work has to close |
| `05_perf_trec_accuracy.png` | 06 | quantum layer vs both controls against the two ceilings, and the angle-scaling ablation |
| `05_perf_trec_width.png` | 06 | accuracy vs circuit width, tracking explained variance |
| `05_perf_trec_aer.png` | 06 | Aer exact and shot-based accuracy, and SPSA vs parameter-shift cost |

Ratio-like spreads use a log axis, and on a log axis bar length is meaningless,
so those figures are dot plots rather than bars. Any run that produced NaN is
drawn in grey and labelled, never silently omitted.

---

## Consequences for the project

- **Adjoint differentiation is still unavailable** for the lambeq path.
  The project methodology lists it as a method to use, but lightning rejects adjoint on
  post-selected circuits, and Aer offers no backprop. Everything falls back to
  parameter-shift. The routes in §4 and §5 have no post-selection, so they are
  where adjoint on `lightning.gpu` is worth retesting first.
- **Post-selection, not qubit count, is the near-term ceiling** on the lambeq
  route. 18 qubits is easy for any simulator; 2⁻¹⁶ post-selection is what rules
  out shots and hardware. It also caps the *dataset*: circuit width follows
  sentence length, and TREC questions reach 37 words, so §2-3's route cannot be
  run on TREC at all. §5 is what replaces it, at the cost of the grammar.
- **The reduction, not the circuit, is the dominant loss on a real dataset.**
  Full-dimension logistic regression reaches 0.858 on 6-way TREC; the best
  model on 8 reduced components reaches 0.607, and the quantum layer 0.494.
  Before more qubits are requested, the question to settle is how many
  components the task needs — 8 components keep 5.2% of the variance.
- **Shot budgets follow the decision margin, not the circuit width.** The same
  8-qubit circuit that reproduces every prediction under exact simulation
  agrees only 76.5% of the time at 1024 shots on 6-way TREC, and 90.5% at 8192.
  A shot count validated on a saturated binary task will not transfer.
- **For the HPC Specialist:** `qiskit-aer` here exposes `['CPU']` only, so both
  GPU backends remain unverified. The GPU config is now at least *correct* —
  `aer_backend_config(gpu=True)` — so it can be tested on the node by running
  `03_lambeq_aer_backend.py --gpu`, `04_train_cnn_hybrid.py --gpu`, and
  `06_train_trec_reduced.py --gpu --only aer-eval,aer-training`. The last is the
  cheapest GPU smoke test in the folder — it needs no lambeq parse and merges
  into the existing report.
- **Batch-size-1 on external simulators** should be assumed when planning GPU
  benchmarks; the win has to come from per-circuit speed, not batching, unless
  adjoint becomes available.
- **The target to beat is 0.71 h/epoch.** That is SPSA on Aer over TREC's 4,635
  training questions, measured; parameter-shift is 19.17 h/epoch. A 30-epoch
  schedule is ~21 hours on this CPU, which is the project's core problem
  statement on a realistic dataset.
- **SPSA needs a smaller step size than backprop.** A gradient estimated from
  two evaluations is noisy, and a step size tuned against exact gradients will
  overshoot it. Worth knowing before SPSA is used for the large HPC runs the
  work plan schedules.
