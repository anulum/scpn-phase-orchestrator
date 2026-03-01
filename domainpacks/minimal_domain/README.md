# minimal_domain

Minimal test domain. 4 oscillators, 2 layers, Physical channel only.

All oscillators are in good_layers -- the engine should converge to high R.

## Run

```bash
spo run domainpacks/minimal_domain/binding_spec.yaml --steps 100
```

Or directly:

```bash
python domainpacks/minimal_domain/run.py
```
