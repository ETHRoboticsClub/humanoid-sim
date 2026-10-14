# humanoid-sim

- Python modules live directly in `src/`, installed as `humanoid_sim`.
- The shared boundary is device process data. Firmware owns drive demands and state;
  physics owns physical state. Keep requested, applied and measured quantities separate.
- Keep device models batched over environments. No host transfers in steady-state model math.
- Isaac Lab/Newton imports are lazy. CPU model tests must run without either installed.
- Use the exact upstream revisions in `runtime.lock.yaml`; no silent physics fallback.
- Runtime transport integration, Controls, MuJoCo backends and RL trainers are out of scope.
- Preserve unrelated work. Commit or push only when asked; never push directly to main.
- Firmware behavior inferred from sources must be labeled as modeled until verified against
  device documentation or independent traces. CPU tests do not establish GPU or hardware fidelity.
