# Geometry Constraints

## Purpose

Geometry constraints project the coupling matrix K_nm into a feasible set after each modification. This enforces structural invariants that the supervisor's ControlActions cannot violate.

## Constraint Catalogue

| Constraint | Effect | Idempotent | Implementation |
|------------|--------|------------|----------------|
| `SymmetryConstraint` | `K' = 0.5 * (K + K^T)` | Yes | `coupling/geometry_constraints.py` |
| `NonNegativeConstraint` | `K' = max(K, 0)` | Yes | `coupling/geometry_constraints.py` |

Constraints are applied in order via `project_knm(knm, constraints)`. The composition of idempotent projections is not generally idempotent, but the alternating-projection pattern converges to the intersection of convex sets (Bauschke & Combettes, 2011).

## Projection API

```python
project_knm(knm: NDArray, constraints: list[GeometryConstraint]) -> NDArray
```

Returns a new array; the input is not mutated.

## Binding Spec Configuration

```yaml
geometry_prior:
  constraint_type: symmetric_non_negative
  params: {}
```

The `constraint_type` string is parsed as a keyword bag:

- Contains `"symmetric"` → `SymmetryConstraint` added.
- Contains `"non_negative"` or `"nonneg"` → `NonNegativeConstraint` added.

## Integration Point

When `binding_spec.geometry_prior` is present, the CLI `run` loop applies `project_knm` to the effective K_nm after imprint modulation and before each UPDE step.

## Properties

- **Symmetry**: after projection, `K_nm == K_mn` to machine precision.
- **Non-negativity**: after projection, `K_nm >= 0` elementwise.
- **Diagonal**: diagonal of K_nm is not modified by these constraints. Domainpacks that require zero self-coupling should set `diag(K) = 0` in the coupling builder.

## References

- `src/scpn_phase_orchestrator/coupling/geometry_constraints.py` — constraint classes.
- H. H. Bauschke & P. L. Combettes (2011). *Convex Analysis and Monotone Operator Theory in Hilbert Spaces*. Springer. — alternating projections convergence.
