# Learning Hamiltonians from data — minimal reproductions

Small standalone scripts reproducing Bertalan, Dietrich, Mezić & Kevrekidis,
*"On Learning Hamiltonian Systems from Data"*, Chaos **29**, 121107 (2019)
([arXiv:1907.12715](https://arxiv.org/abs/1907.12715)).

Reading notes on the paper: `~/paper-notes/bertalan2019-learning-hamiltonian-systems.md`.

Nothing here imports `humancompatible` — these are self-contained, numpy + torch
only, and are meant to be read and poked at rather than reused.

## The idea in one paragraph

Hamilton's equations say the observed vector field is the symplectic gradient of
an unknown scalar `H`:

```
ω ∇H(q,p) = ν(q,p),      ω = [[0, I], [−I, 0]]
```

That is a *linear* PDE in the unknown, and the data gives scattered samples of
`∇H` — so `H` can be recovered by least squares (a GP) or by minimising the
residual (a network). `H` is identified only up to an additive constant, which is
why every script pins `H(x₀) = H₀` at one reference point. When the observations
are not `(q,p)` but some distortion of them, an autoencoder learns the
coordinates at the same time, and the observed time derivatives are pushed into
the learned space by the chain rule.

## Files

| file | what it does |
|---|---|
| `common.py` | pendulum data (symplectic Euler + finite differences), tanh MLP, autograd helpers |
| `e1_gp.py` | Gaussian process recovery of `H`, Eqs. (6)–(8). No training loop |
| `e2_nn.py` | network recovery of `H` in canonical coordinates, Eqs. (10)–(11), plus the loss ablation |
| `e3_transformed.py` | linear and nonlinear distortions of `(q,p)`; autoencoder + `Ĥ` trained jointly, Eqs. (12)–(17) |
| `e4_video.py` | `H` from rendered pendulum video: frames → PCA(20) → dense AE → 4-d latent → `Ĥ` |

Everything writes PNGs into `figs/`.

## Running

```bash
conda run -n hc-dev python e1_gp.py
conda run -n hc-dev python e2_nn.py
conda run -n hc-dev python e2_nn.py --ablate f4     # also: f1, f2
conda run -n hc-dev python e3_transformed.py linear
conda run -n hc-dev python e3_transformed.py nonlinear
conda run -n hc-dev python e4_video.py
```

`e1` is instant; the rest are ~1–3 min each on CPU.

## What each one is meant to show

**`e1_gp.py`** — `H` drops out of one least-squares solve. Reports error over the
whole grid *and* restricted to grid points near data, because the paper's real
caveat is coverage: the recovered `H` is only meaningful where the trajectories
went. The kernel matrix is deliberately smooth and therefore numerically
rank-deficient, so the solve uses the substituted variable `u = K⁻¹h` and an
`rcond` cutoff rather than forming `K⁻¹`.

**`e2_nn.py`** — same problem, network version, and the ablation of Sec. III:
`f₄` (conservation) is formally implied by `f₁, f₂` and dropping it costs
nothing; but dropping `f₁` or `f₂` alone hurts badly even with `f₄` switched on
to stand in for it. A constraint being redundant *in theory* does not make it
redundant to the optimiser.

**`e3_transformed.py`** — the actual contribution. The observations are distorted
and the coordinates are learned along with `Ĥ`. Two structural terms matter:
`f₅` (reconstruction) forces encoder and decoder to be mutually inverse, and
`f₆ = (det Dθ̂⁻¹)⁻²` stops the latent space collapsing — without it, mapping all
data to a point and letting `Ĥ` be constant satisfies every residual. The script
also prints the composite map from true to learned coordinates, to check the
paper's identifiability observation: `(q,p)` is recoverable only up to a
symplectomorphism, but the optimiser nonetheless tends to keep `q` unmixed.

**`e4_video.py`** — a single video frame shows `q` but not `p`, so it is not a
state. The renderer drags a decaying tail of past positions behind the bob, which
puts velocity into each frame. Collapse is a much stronger attractor at this
dimension, so `f₆` is replaced by a std-dev penalty on `Ĥ` plus max-squared-error
terms, as the paper describes.

## Deliberate simplifications

- Pendulum only (the paper's examples are all pendulum-based anyway).
- Full-batch Adam, no minibatching, no validation split.
- `e4` renders 24×24 frames with a Gaussian blob rather than the paper's renderer,
  and uses PCA + dense layers, as the paper does (they name conv autoencoders as
  future work).
- Derivatives come from central differences on symplectic-Euler trajectories, so
  they carry `O(dt²)` error — deliberately, since "from data" is the point. There
  is no measurement-noise model, matching the paper, which does not study noise.
