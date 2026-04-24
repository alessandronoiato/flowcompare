# Where does the noise live?

## A unified view of Bayesian Flow Networks and Discrete Flow Matching

Bayesian Flow Networks (BFN; Graves et al. 2023) and Discrete Flow Matching
(DFM; Gat et al. 2024, Campbell et al. 2024) are two of the most prominent
recent entries in generative modelling of discrete sequences. For protein
sequences in particular, BFN has a strong case as the current SOTA for
unconditional generation (see InstaDeep's ProtBFN, Atkinson et al. 2024),
while DFM — together with its mask-interpolant specialisation that
recovers absorbing-state discrete diffusion — is the method du jour for
masked / fill-in-the-blank style protein generation, and is closely related
to MaskGIT (Chang et al. 2022) and D3PM (Austin et al. 2021).

They look very different. BFN interpolates continuous parameters on the
simplex, adds Gaussian noise to them, and derives a loss that is a
time-weighted $L_2$ distance between a predicted distribution and the
one-hot data. DFM interpolates discrete tokens directly, defines a
continuous-time Markov chain (CTMC) between data and a simple prior, and
recovers the familiar cross-entropy loss, reweighted by the interpolant
schedule.

The thesis of this repository is that, once framed correctly, **both are
instances of a single object**: a continuous-time process $q_t(z_t \mid x_1)$
between a prior $q_0$ and a data distribution $q_1$ on sequences,
parameterised by a neural network $\hat{x}_1 = f_\theta(z_t, t)$ trained to
predict the clean data. Everything they disagree on — loss form, sampler,
NFE / quality tradeoffs, compatibility with inpainting — falls out of *where
the noise lives* in the state space.

This document derives both methods under that single abstraction, records
the design predictions it makes, and lists the concrete quantities the
benchmark in this repo measures to test those predictions head-to-head.

## 1. The abstraction

A **sequence process** is a tuple
$(\mathcal{X}, \mathcal{Z}, q_0, \{q_t(\cdot \mid x_1)\}_{t \in [0,1]})$
where:

- $\mathcal{X} = \{1, \dots, K\}^L$ is the data space: sequences of length $L$
  over a vocabulary of size $K$.
- $\mathcal{Z}$ is the **state space** of the process; this is the object
  that BFN and DFM make different choices about.
- $q_0$ is a data-independent **prior** on $\mathcal{Z}$.
- $q_1(\cdot \mid x_1)$ places all mass on the embedding of $x_1$ into
  $\mathcal{Z}$.
- $q_t(\cdot \mid x_1)$ is any continuous path between the two (marginal
  over $x_1$) satisfying $q_0 = \mathbb{E}_{x_1} q_0$ and
  $q_1 = \delta_{x_1}$.

A neural network $f_\theta$ takes a sample $z_t \sim q_t(\cdot \mid x_1)$
and the scalar time $t$, and outputs $\hat{x}_1$-**prediction logits**.
Training minimises a continuous-time negative-ELBO:

$$
\mathcal{L}(\theta)
  = \mathbb{E}_{x_1 \sim \text{data}} \mathbb{E}_{t \sim U[0,1]}
    \mathbb{E}_{z_t \sim q_t(\cdot \mid x_1)}
    \left[ w(t) \cdot D\left(x_1, \, f_\theta(z_t, t)\right) \right]
$$

for some process-specific weight $w(t)$ and distance / loss $D$. Sampling
runs a discretisation of the implied backward process from $t = 0$ (prior)
to $t = 1$ (data), calling $f_\theta$ at each step.

Both BFN and DFM fit here. Let us write each one down.

## 2. BFN: noise on the parameters

BFN chooses
$\mathcal{Z} = \Delta_{K-1}^L$, the $L$-fold product of $K$-simplices. The
state $z_t = \theta_t$ is a sequence of categorical distributions — one
probability vector per position, each summing to 1.

The forward process is a **Gaussian channel on the parameters**. The sender
transmits noisy evidence about the clean data:

$$
y \sim \mathcal{N}\!\big(\beta(t) \cdot (K e(x_1) - \mathbf{1}),\ \beta(t) K I\big),
$$

where $e(x_1) \in \{0, 1\}^{KL}$ is the one-hot encoding of $x_1$ and
$\beta(t)$ is a monotonically increasing accuracy schedule. A Bayesian
update from the uniform prior yields

$$
\theta_t = \mathrm{softmax}(y)
\quad\text{(since}\ \log \theta_0 = -\log K \cdot \mathbf{1}\text{ is constant).}
$$

At $t = 0$, $\beta(0) = 0$ and $\theta_0 = \mathbf{1}/K$ exactly (uniform
over tokens). At $t = 1$, $\theta_1$ concentrates near $e(x_1)$.

The continuous-time bound (Graves 2023, Theorem 4.2, discrete case) is

$$
\boxed{\ \mathcal{L}^\infty_{\text{BFN}}(x_1)
  = K \cdot \mathbb{E}_{t, y}\left[\beta'(t)\ \lVert e(x_1) - \hat{e}_\theta(\theta_t, t)\rVert_2^2\right] \ }
$$

where $\hat{e}_\theta(\theta_t, t) = \mathrm{softmax}(f_\theta(\theta_t, t))$ is
the network's predicted categorical. Implemented as `bfn_loss` in
`flowcompare/processes/bfn.py`.

With $\beta(t) = \beta_1 t^2$ (the schedule used in ProtBFN and in this
repo), $\beta'(t) = 2 \beta_1 t$, so the loss weight grows linearly in $t$:
early in generation (where $\theta_t$ is close to uniform) the loss is
cheap; late in generation (where $\theta_t$ is near one-hot) it is
expensive. This matches the intuition that "teaching the network to get the
last 5% right is harder than teaching it to get the first 5% right".

**Sampling.** Start from $\theta_0 = \mathbf{1}/K$ and iterate $n$ times:

1. Compute $\hat{p}_i = \mathrm{softmax}(f_\theta(\theta_t, t))_i$.
2. Sample $\hat{x}_i \sim \text{Cat}(\hat{p}_i)$.
3. Sample a sender $y \sim \mathcal{N}(\Delta \beta \cdot (K e(\hat{x}) - \mathbf{1}),\ \Delta \beta K I)$.
4. Bayesian update: $\theta_{t+\Delta t} = \mathrm{softmax}(\log \theta_t + y)$.

This is Algorithm 10 of Graves 2023 and maps directly to `BFNProcess.step`
in `flowcompare/processes/bfn.py`. Each step makes one network call and
delivers a small, Gaussian-weighted amount of evidence; convergence to a
confident $\theta_1$ therefore needs *many* steps.

## 3. DFM (mask interpolant): noise on the tokens

DFM chooses $\mathcal{Z} = \{1, \dots, K+1\}^L$, the data space with an
extra `MASK` token adjoined. The mask interpolant defines
$q_t(\cdot \mid x_1)$ by letting each position independently equal $x_{1,i}$
with probability $\kappa(t) = t$ and `MASK` otherwise:

$$
q_t(z_i \mid x_1) = t \cdot \delta_{z_i = x_{1,i}} \ +\ (1 - t) \cdot \delta_{z_i = \mathrm{MASK}}.
$$

The prior $q_0$ is thus the all-`MASK` sequence; the terminal $q_1$ is
$\delta_{x_1}$. The forward dynamics are a particularly simple CTMC: no
position that is currently unmasked ever gets re-masked, and each masked
position transitions to $x_{1,i}$ at rate $1/(1 - t)$.

The continuous-time loss (Gat et al. 2024, Prop. 3.1; Campbell et al. 2024,
eq. 19) is the mask-indicator-weighted cross-entropy:

$$
\boxed{\ \mathcal{L}^\infty_{\text{DFM}}(x_1)
  = \mathbb{E}_{t, z_t}\!\left[\frac{1}{1 - t} \sum_{i : z_{t,i} = \mathrm{MASK}} \mathrm{CE}\!\left(f_\theta(z_t, t)_i,\ x_{1, i}\right)\right] \ }
$$

Implemented as `dfm_mask_loss` in `flowcompare/processes/dfm.py`. The
$1 / (1 - t)$ weight compensates for the fact that the *fraction* of
positions still masked shrinks with $t$: each remaining mask carries more
information about the bound as time progresses. The absorbing prior also
means that positions already correctly filled in contribute exactly zero
to the loss — the network is not asked to waste capacity re-predicting
them.

**Sampling.** Start from the all-`MASK` state and iterate $n$ times
(tau-leaping of the forward CTMC):

1. Compute $\hat{p}_i = \mathrm{softmax}(f_\theta(z_t, t))_i$ at every position.
2. For each position $i$ currently masked, with probability
   $\Delta t / (1 - t)$ (capped at 1 near $t = 1$), sample a token from
   $\hat{p}_i$ and replace `MASK` with it. Otherwise leave the position
   masked.

This is `DFMProcess.step`. A single step can unmask *any fraction* of the
sequence: if $\Delta t$ is large, many positions jump at once. Compared to
BFN's Gaussian accumulation, each step is much more expressive per NFE.

## 4. Side by side

| Question                          | BFN                                | DFM (mask interp.)                  |
|-----------------------------------|------------------------------------|-------------------------------------|
| State space $\mathcal{Z}$         | $\Delta_{K-1}^L$ (simplex per pos) | $\{1, \dots, K+1\}^L$ (tokens+MASK) |
| Prior $q_0$                       | $\mathbf{1}/K$ uniform             | all-MASK                            |
| Corruption                        | Gaussian on simplex parameters     | independent token-wise mask         |
| Loss form                         | weighted $L_2$, weight $K\beta'(t)$| weighted CE on MASK positions, weight $1/(1-t)$ |
| Loss at unmasked positions        | always nonzero                     | exactly zero                        |
| Update per sampling step          | $\Delta\theta \propto \Delta \beta$| potentially all positions unmask    |
| Inpainting / conditioning         | awkward (continuous state hides which positions are "known") | trivial (just never mask known positions) |
| Depends on a `MASK` token         | no                                 | yes                                 |

Two tradeoffs are especially sharp, both of which the benchmark here
measures:

**NFE vs. quality.** BFN's per-step update is a Gaussian draw whose
magnitude scales as $\Delta \beta = O(\Delta t)$ for a smooth schedule, so
information accumulates linearly and many steps are required to reach
confident $\theta_1$. DFM can flip multiple positions per step — at any
$t$ the unmasking rate is $1/(1-t)$ per masked position, and a single large
$\Delta t$ legitimately flips a $\Delta t / (1-t)$ fraction of them. We
therefore predict **DFM dominates at low NFE; BFN catches up at high NFE**,
with the crossover depending on $\beta_1$ and the schedule.

**Inpainting.** DFM's mask representation is exactly the `<mask>` token of
BERT-style MLMs. Conditional generation is "run the sampler, but never
unmask positions that you want to hold fixed". BFN has no such built-in
conditioning: to pin position $i$ to token $a$ you have to inject a
one-hot $\theta$ at that position and hope the network respects it. We
predict **DFM is strictly better on inpainting / fragment-completion
tasks**, and set up the eval suite (`flowcompare/eval/pareto.py`) to
measure this once we have trained checkpoints.

**Connection to masked diffusion and MaskGIT.** Absorbing-state D3PM
(Austin et al. 2021) is exactly DFM with the mask interpolant $\kappa(t) = t$
and a discrete-time schedule. MaskGIT (Chang et al. 2022) is the same
object with confidence-based unmasking order at sampling time rather than
random positions; the DFM CTMC sampler recovers MaskGIT when you choose
the "highest confidence first" unmasking policy instead of the independent
Bernoulli policy we use here. Both are special cases of the continuous-time
framework — which is part of the point: the framework makes the design
axes (interpolant, schedule, unmasking policy, continuous-vs-discrete
state) explicit instead of hiding them in training-time choices.

## 5. What the theory predicts (and what we measure)

The theory makes concrete predictions the eval suite checks:

1. **NFE Pareto curve.** On the same model architecture, same training data,
   same parameter count: DFM should dominate the NFE / quality Pareto
   frontier at small NFE, with BFN approaching as NFE grows. Measured by
   `flowcompare/eval/pareto.py` + ESMFold pLDDT as the quality metric.

2. **Validation ELBO.** At infinite NFE, both losses are upper bounds on
   $-\log p(x)$, so the achievable validation ELBO should be similar when
   parameters and compute are matched. Differences reveal optimisation
   asymmetries, not capacity ones. Measured by
   `flowcompare.eval.compute_held_out_loss`.

3. **Diversity.** Both should produce diverse samples from a well-trained
   model; absorbing-state DFM has a mild tendency to fall back on
   high-frequency amino acids early in the sampling trajectory, potentially
   reducing diversity relative to BFN's simplex-noise dynamics. Measured
   by `flowcompare.eval.mean_pairwise_identity`.

4. **Inpainting quality.** DFM trivially supports conditional generation;
   BFN does not. Measured by a future inpainting benchmark that freezes
   50% of known residues and measures pLDDT on the inpainted regions.

5. **Foldability (ESMFold pLDDT / pTM).** The headline biological metric.
   Reported as a function of NFE for each method.

None of these are actually measured yet — this repository establishes the
training + eval infrastructure that lets them be measured once the
research-grade training runs (UniRef50 + OAS, 1–4 GPU-days) complete.

## References

- Graves, Srivastava, Atkinson, Mandt. *Bayesian Flow Networks*. arXiv:2308.07037, 2023.
- Atkinson, Noguera, Liu, Lin, et al. *ProtBFN*. 2024.
- Gat, Remez, Shaul, Kreis, Chen, Synnaeve, Adi, Lipman. *Discrete Flow Matching*. arXiv:2407.15595, 2024.
- Campbell, Yim, Barzilay, Rainforth, Jaakkola. *Generative Flows on Discrete State-Spaces*. ICML 2024.
- Austin, Johnson, Ho, Tarlow, van den Berg. *Structured Denoising Diffusion Models in Discrete State-Spaces*. NeurIPS 2021.
- Chang, Zhang, Jiang, Liu, Freeman. *MaskGIT: Masked Generative Image Transformer*. CVPR 2022.
