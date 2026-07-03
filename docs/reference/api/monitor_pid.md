# Partial Information Decomposition

The `monitor.pid` module estimates the redundant and synergistic
information that two oscillator groups carry about the global phase state.
It is a time-series Williams-Beer `I_min` decomposition over circular phase
observables, not a single-snapshot synchrony score.

The implementation has a five-slot backend chain:

1. Rust
2. Mojo
3. Julia
4. Go
5. Python

Unavailable optional toolchains are reported explicitly by the benchmark
gate. Available backends must reproduce the Python reference within the
declared tolerance and must satisfy the same decomposition contracts.

---

## Mathematical Contract

Input is a finite real phase history `phases` with shape `(T, N)`, where
`T` is the number of timesteps and `N` is the number of oscillators. Each
timestep is reduced to three circular observables:

| Symbol | Observable |
|---|---|
| `Y_t` | global order-parameter phase over all oscillators |
| `A_t` | order-parameter phase of `group_a` |
| `B_t` | order-parameter phase of `group_b` |

The three series are wrapped into `[0, 2*pi)` and binned into `n_bins`
equal-width phase bins. With Williams and Beer specific information,

```text
I_spec(Y=y; S) = sum_s p(s|y) * log(p(y|s) / p(y))
```

the exported components are:

```text
I_red = sum_y p(y) * min(I_spec(Y=y; A), I_spec(Y=y; B))
I_syn = MI(A,B; Y) - MI(A; Y) - MI(B; Y) + I_red
```

`I_red` is the Williams-Beer `I_min` redundancy. `I_syn` is clipped at zero
after the floating-point estimate so small negative roundoff cannot cross the
public boundary. A single snapshot (`T = 1`) or an empty group carries no
distributional information and returns `0.0`.

---

## API

```python
from scpn_phase_orchestrator.monitor.pid import redundancy, synergy

red = redundancy(history, group_a=[0, 1, 2], group_b=[3, 4, 5], n_bins=32)
syn = synergy(history, group_a=[0, 1, 2], group_b=[3, 4, 5], n_bins=32)
```

### `redundancy`

Returns the information about the global phase target that is available from
both source groups. Positive redundancy means either group alone carries
overlapping information about the same target state.

### `synergy`

Returns the information about the global phase target that is available only
from the joint observation of both groups. Positive synergy means the groups
are complementary: neither group alone carries the full target information.

### Input validation

The public boundary rejects:

- boolean phase aliases,
- numeric-string phase aliases,
- complex phase samples, including object-dtype complex aliases,
- non-finite phase samples,
- non-vector or out-of-range group indices,
- numeric-string group indices,
- boolean, complex, non-integral, or non-finite group indices,
- `n_bins < 2`,
- boolean or non-integral `n_bins` values,
- backend scalar outputs that are numeric strings, non-finite, or negative.

One-dimensional phase input is interpreted as a single timestep. Because a
single timestep does not define a distribution, both public functions return
`0.0` in that case.

---

## Backend Chain

| Slot | Backend | Loader |
|---|---|---|
| 1 | Rust | `spo_kernel.pid_decomposition_rust` |
| 2 | Mojo | `mojo/pid_mojo` subprocess executable |
| 3 | Julia | `julia/pid.jl` through `juliacall` |
| 4 | Go | `go/libpid.so` through `ctypes` |
| 5 | Python | NumPy reference implementation |

The dispatcher tries the active backend first, then every resolved backend in
order, then the Python reference. Loader failures and unavailable optional
toolchains are allowed to fall back. Physics-contract failures at the public
boundary are rejected before a result can be accepted.

Direct Go, Julia, and Mojo bridge wrappers share the same typed backend input
validation before optional runtime loading. Their outputs are validated as
finite non-negative redundancy and synergy scalars, never numeric-string
aliases, before returning to the public dispatcher.

---

## Parity Gate

The release benchmark function is `benchmark_pid_polyglot_parity_gate`. Run it
directly with:

```bash
PYTHONPATH=.:src python benchmarks/pid_benchmark.py \
  --parity-gate \
  --n-steps 1500 \
  --n-bins 12 \
  --calls 1
```

The gate records all declared backend slots in canonical order
`rust`, `mojo`, `julia`, `go`, `python`. Acceptance requires:

- the Python reference record,
- one explicit record per declared backend slot,
- parity for every available backend,
- non-negative redundancy and synergy,
- positive synergy for a deterministic co-varying source pair,
- positive redundancy and vanishing synergy for a fully redundant source pair,
- a deterministic benchmark hash excluding wall-clock timing.

The benchmark evidence kind is `local_regression_non_isolated`; wall-clock
values are local, non-isolated regression evidence. It does not make production
timing claims unless the benchmark metadata records CPU isolation and host-load
controls.

The canonical reference suite exposes the same gate as `pid_polyglot`.

---

## Relationship To Transfer Entropy

Transfer entropy asks whether the past of one phase stream helps predict a
target stream beyond the target's own past. PID asks how two source groups
share information about a target distribution. Use transfer entropy for
directed pairwise influence, and use PID when redundant versus synergistic
group information is the quantity of interest.

::: scpn_phase_orchestrator.monitor.pid
