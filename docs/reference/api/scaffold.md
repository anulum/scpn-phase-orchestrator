<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator — Scaffold API -->

# Scaffold API

The scaffold API creates reviewable SCPN domainpack starting points.

It serves two operator workflows.

First, `spo scaffold NAME` emits a deterministic minimal domainpack.

Second, `spo scaffold NAME --llm --description TEXT` converts an inert natural-language description into a reviewed binding proposal.

Both paths are intentionally conservative.

The output is a domainpack directory, not an activated runtime.

The generated `binding_spec.yaml` must still pass the normal binding loader, validator, review process, and operator approval flow before production use.

The LLM-guided path is `review-only` by design.

It never installs a domainpack globally.

It never applies actuation.

It never runs a simulation without a separate command.

It never treats provider output as trusted configuration.

It produces files that humans and automated validators can inspect.

## Public surface

The scaffold public surface is exposed from `scpn_phase_orchestrator.scaffold`.

The implementation lives in `scpn_phase_orchestrator.scaffold.llm`.

The CLI entry point is `spo scaffold`.

The minimal scaffold branch is implemented in `scpn_phase_orchestrator.runtime.cli.scaffold`.

The LLM-guided branch delegates proposal generation to `propose_domainpack_from_description`.

The dedicated API objects are listed below.

| Symbol | Role |
|---|---|
| `LLMScaffoldConfig` | Runtime bounds for guided scaffold generation. |
| `LLMScaffoldProposal` | Immutable result carrying YAML, validation evidence, provenance, and hashes. |
| `LLMScaffoldProvider` | Protocol implemented by JSON-only completion providers. |
| `StaticJSONScaffoldProvider` | Deterministic offline provider for captured proposal files. |
| `LocalHTTPScaffoldProvider` | HTTP provider for private chat-completion-compatible gateways. |
| `configured_llm_scaffold_provider` | Environment-backed provider factory that fails closed when incomplete. |
| `propose_domainpack_from_description` | Main proposal function for inert description to validated binding YAML. |

## What the scaffold API is for

Use the scaffold API when a new domain needs a valid SCPN binding skeleton.

Use it during onboarding to avoid hand-writing the first `binding_spec.yaml`.

Use it during research triage to turn a domain description into a review packet.

Use it during regulated deployment preparation to capture a deterministic proposal before review.

Use it to create reproducible examples for training, documentation, and support.

Use it to verify that a proposed oscillator mapping can pass the production binding validator.

Use it to separate proposal generation from runtime activation.

Use it to document why a domainpack exists and which evidence was used to initialise it.

Use it to build a safe handoff between exploratory modelling and operator-controlled execution.

Use it when the source description is useful but not trusted.

Do not use it as an automatic deployment path.

Do not use it to bypass binding review.

Do not use it to bypass safety-tier review.

Do not use it to write hardware commands.

Do not use it to apply an actuator setting.

Do not use it as a substitute for scientific validation of the chosen model.

Do not use it as a substitute for domain expert review.

## Minimal scaffold workflow

The minimal command is:

```bash
spo scaffold pump_lab
```

The command creates `domainpacks/pump_lab/`.

It writes `binding_spec.yaml` if the file is absent.

It writes `README.md` if the file is absent.

It leaves existing files in place.

The minimal `binding_spec.yaml` uses one physical oscillator.

It sets `safety_tier: research`.

It sets `sample_period_s: 0.01`.

It sets `control_period_s: 0.1`.

It declares one layer named `default`.

It declares one family named `default`.

It assigns channel `P`.

It assigns extractor type `physical`.

It sets a finite base coupling strength.

It includes empty `boundaries`.

It includes empty `actuators`.

It can be loaded with `load_binding_spec`.

It can be checked with `validate_binding_spec`.

It can drive a small local simulation after review.

The minimal path is useful for manual editing.

The minimal path is deterministic.

The minimal path does not require a provider.

The minimal path does not inspect external data.

The minimal path does not infer oscillator semantics.

The operator remains responsible for replacing placeholders with domain-specific terms.

## Guided scaffold workflow

The guided command is:

```bash
spo scaffold traffic_grid --llm --description "traffic signals and queue pressure"
```

The guided path requires a description.

If the description is missing, the CLI fails before contacting a provider.

If no provider is configured and no offline response file is supplied, the CLI fails closed.

For deterministic review, use an offline proposal file:

```bash
spo scaffold traffic_grid --llm --description "traffic signals" --llm-response-json proposal.json
```

The offline file must contain one JSON response accepted by `StaticJSONScaffoldProvider`.

The response can be either a direct proposal object or a wrapper from which completion text is extracted.

The extracted completion must be a `strict JSON object`.

Markdown is rejected.

Free-form prose is rejected.

Arrays at the top level are rejected.

Scalar top-level values are rejected.

Malformed JSON is rejected.

The proposal is normalised before YAML generation.

The generated YAML is checked through `load_binding_spec`.

The loaded binding is checked through `validate_binding_spec`.

Validation failures are recorded on the proposal.

The CLI writes the proposal even only after successful proposal construction.

The CLI writes `llm_scaffold_audit.json` for review evidence.

## Review-only safety model

The scaffold API is a proposal system.

A proposal is not a deployment.

A proposal is not an operational policy.

A proposal is not an approval record.

A proposal is not a live plant interface.

A proposal is not a hardware command.

A proposal is not a simulation result.

The generated files must be inspected before execution.

The generated binding must be validated before execution.

The generated safety tier must be approved before execution.

The generated actuator limits must be reviewed before execution.

The generated boundaries must be checked against real operating limits before execution.

The generated oscillator channels must be mapped to real sensors before execution.

The generated natural frequencies must be checked against domain evidence before execution.

The generated coupling constants must be calibrated before operational use.

The generated README must be edited to describe actual domain assumptions.

The generated audit record must be retained with the review packet.

## Trust boundary

The operator description is treated as inert project data.

The description is not executed.

The description is not interpolated into shell commands.

The description is not used to alter system instructions.

The description is wrapped in a dedicated data block inside the prompt.

The prompt tells the provider to return only one JSON object.

The provider response is untrusted.

The provider response must pass JSON parsing.

The provider response must pass schema-like normalisation.

The normalised payload must produce deterministic binding YAML.

The deterministic binding YAML must pass production binding loading.

The loaded binding must pass production validation.

The final proposal records hashes for audit comparison.

The final proposal records provider identity for provenance.

The final proposal records channels and periods for review.

The final proposal does not imply approval.

## Prompt override detection

The guided path rejects common prompt-override markers before provider invocation.

A description containing `ignore previous instructions` is rejected.

A description containing `disregard prior instructions` is rejected.

A description containing a synthetic `system:` role marker is rejected.

A description containing a synthetic `assistant:` role marker is rejected.

A description containing a synthetic `developer:` role marker is rejected.

A description containing a synthetic `tool:` role marker is rejected.

A description containing `role: system` is rejected.

A description containing XML-style role tags is rejected.

The rejection class is a `ValueError`.

The message includes `prompt-override`.

This is a defensive input screen, not a proof of semantic safety.

Operators must still review the resulting proposal.

Providers must still be isolated according to deployment policy.

Audit review must still compare the proposal to domain evidence.

## LLMScaffoldConfig

`LLMScaffoldConfig` defines generation limits.

The default timeout is thirty seconds.

The default maximum oscillator count is 128.

The default maximum description length is 8000 characters.

The default sample period is one second.

The default control period is ten seconds.

`timeout_s` must be positive.

`timeout_s` must be finite.

`max_oscillators` must be positive.

`max_description_chars` must be positive.

`default_sample_period_s` must be positive.

`default_sample_period_s` must be finite.

`default_control_period_s` must be positive.

`default_control_period_s` must be finite.

Invalid configuration raises `ValueError` during construction.

Configuration validation happens before provider use.

Configuration validation prevents undefined proposal bounds.

Configuration defaults make offline tests deterministic.

Configuration defaults make small examples easy to review.

Configuration should be tightened for constrained review contexts.

Configuration should not be used to hide poor domain modelling.

## LLMScaffoldProposal

`LLMScaffoldProposal` is the immutable proposal return value.

It carries `yaml_text`.

It carries `validation_errors`.

It carries `provenance`.

It carries `raw_response_sha256`.

It carries `description_sha256`.

`yaml_text` is the generated binding specification.

`validation_errors` is an ordered tuple of validator messages.

An empty tuple means the generated YAML loaded and validated cleanly.

A non-empty tuple means the proposal needs correction before use.

`provenance` records review metadata.

`raw_response_sha256` identifies the exact provider response text.

`description_sha256` identifies the exact input description text.

The hashes allow a review packet to prove which input and response produced a binding.

The hashes are not a substitute for content review.

The proposal method `to_audit_record()` returns a JSON-safe dictionary.

The audit record includes kind `llm_scaffold`.

The audit record includes the provider name.

The audit record includes the description hash.

The audit record includes the response hash.

The audit record includes validation errors.

The audit record includes oscillator count.

The audit record includes channels.

The audit record includes sample and control periods.

The audit record is what the CLI writes to `llm_scaffold_audit.json`.

## LLMScaffoldProvider

`LLMScaffoldProvider` is a protocol.

It requires a stable `name` property.

It requires a `complete(prompt: str) -> str` method.

The return value must be text.

The returned text must contain one JSON proposal object after extraction.

The provider abstraction keeps proposal validation independent from transport.

The provider abstraction makes offline review and live review use the same normaliser.

The provider abstraction does not grant trust to provider output.

Every provider result still passes the same strict parser.

Every provider result still passes the same normalisation rules.

Every provider result still passes the same binding validator.

## StaticJSONScaffoldProvider

`StaticJSONScaffoldProvider` is the preferred review and test provider.

It stores a response string.

It exposes a stable provider name.

Its default provider name is `static-json`.

Its `complete()` method requires a non-empty prompt.

Its `complete()` method returns the configured response string.

It does not perform network I/O.

It is deterministic.

It is suitable for regression tests.

It is suitable for saved review packets.

It is suitable for regulated workflows that require repeatable evidence.

It is suitable when the proposal has already been captured by a separate process.

It is not a validator by itself.

It is not a parser by itself.

It simply supplies text to the same production proposal path.

## LocalHTTPScaffoldProvider

`LocalHTTPScaffoldProvider` is the live HTTP provider implementation.

It requires a non-empty endpoint.

It requires a non-empty model identifier.

The endpoint scheme must be `http` or `https`.

The request body follows a chat-completion style message structure.

The system message asks for one strict JSON object.

The user message carries the scaffold prompt.

The temperature is set to zero.

Connection failures are wrapped as runtime failures.

Timeout failures are wrapped as runtime failures.

The provider extracts completion text from common response shapes.

It accepts a direct string payload.

It accepts a mapping with `text`.

It accepts a mapping with `content`.

It accepts a choices list containing a message content string.

It accepts a choices list containing a text string.

It rejects provider responses without completion text.

It does not validate the proposal itself.

The proposal path validates the extracted text after return.

Live use should be restricted to controlled provider endpoints.

Live use should still produce an audit record.

Live use should still require human review before execution.

## configured_llm_scaffold_provider

`configured_llm_scaffold_provider()` builds the default live provider.

It reads `SPO_LLM_ENDPOINT`.

It reads `SPO_LLM_MODEL`.

It applies the timeout from `LLMScaffoldConfig`.

It fails closed when the endpoint is absent.

It fails closed when the model identifier is absent.

It returns `LocalHTTPScaffoldProvider` when configuration is complete.

It does not contact the provider during construction.

It does not validate a description.

It does not parse a response.

It only builds the provider object used later by `propose_domainpack_from_description`.

For deterministic review, prefer `--llm-response-json` instead of live provider discovery.

For automated tests, prefer `StaticJSONScaffoldProvider`.

## propose_domainpack_from_description

`propose_domainpack_from_description()` is the primary guided API.

It accepts a description string.

It accepts `project_name`.

It accepts a provider.

It accepts optional configuration.

It validates the project name against `[a-zA-Z0-9_-]+`.

It strips surrounding whitespace from the description.

It rejects an empty description.

It rejects descriptions longer than the configured limit.

It rejects prompt-override markers.

It builds the scaffold prompt.

It calls the provider once.

It parses one strict JSON object.

It normalises the payload.

It emits deterministic binding YAML.

It validates the generated YAML through the production binding stack.

It records provenance.

It records description and response hashes.

It returns `LLMScaffoldProposal`.

It raises `ValueError` for invalid user input or invalid proposal content.

It raises runtime failures for provider transport failures.

It does not write files.

The CLI writes files after receiving the proposal.

This separation lets library callers decide where review artefacts belong.

## Proposal JSON contract

The proposal object must contain `name` or allow the project name default.

If `name` is present, it must match the requested project name.

The name must match `[a-zA-Z0-9_-]+`.

The proposal object must contain `oscillators`.

`oscillators` must be a sequence.

`oscillators` must not be empty.

`oscillators` must not exceed `max_oscillators`.

Each oscillator must contain `id`.

Each oscillator `id` must be a valid identifier.

Each oscillator id must be unique.

Each oscillator must contain `channel`.

Each channel must satisfy binding channel rules.

Each oscillator must contain `extractor_type`.

Each extractor type must be one of the binding extractor types.

Each oscillator may contain `omega`.

Missing `omega` defaults to `1.0`.

`omega` must be finite.

`sample_period_s` is optional.

Missing `sample_period_s` uses the configured default.

`sample_period_s` must be positive.

`sample_period_s` must be finite.

`control_period_s` is optional.

Missing `control_period_s` uses the configured default or the sample period if larger.

`control_period_s` must be positive.

`control_period_s` must be finite.

`control_period_s` must be greater than or equal to `sample_period_s`.

`safety_tier` is optional.

Missing `safety_tier` defaults to `research`.

`safety_tier` must be one of the binding safety tiers.

`coupling` is optional.

Missing `coupling` defaults to base strength `0.45` and decay `0.3`.

`coupling.base_strength` must be finite and non-negative.

`coupling.decay_alpha` must be finite and non-negative.

`boundaries` is optional.

Missing `boundaries` is treated as empty.

Each boundary must contain `name`.

Each boundary must contain `variable`.

Each boundary must contain `severity`.

Boundary severity must be one of the binding severities.

Boundary `lower` may be null.

Boundary `upper` may be null.

Finite lower and upper values must satisfy lower less than upper.

`actuators` is optional.

Missing `actuators` is treated as empty.

Each actuator must contain `name`.

Each actuator must contain `knob`.

Each actuator knob must be one of the binding knobs.

Each actuator must contain `scope`.

Each actuator must contain `limits`.

Actuator limits must contain exactly two finite numbers.

Actuator lower limit must be less than or equal to the upper limit.

Unknown proposal fields are ignored by the current normaliser.

Review processes should not rely on ignored fields.

## Generated YAML contract

The YAML emitter is deterministic for the normalised payload.

It emits a comment marking the binding as an assisted proposal.

It emits `name`.

It emits `version: "0.1.0"`.

It emits `safety_tier`.

It emits `sample_period_s`.

It emits `control_period_s`.

It emits one layer per oscillator.

Each layer name is the oscillator id.

Each layer index is the oscillator index.

Each layer contains exactly one oscillator id.

Each layer contains one natural frequency value.

Each layer references a generated family name.

Generated family names use the pattern `llm_INDEX_channel`.

Each family includes the channel.

Each family includes the extractor type.

Each family includes an empty config mapping.

The coupling block includes base strength.

The coupling block includes decay alpha.

The coupling block includes an empty templates mapping.

The drivers block includes physical, informational, and symbolic entries.

The objectives block marks all generated layers as good layers.

The objectives block sets empty bad layers.

The objectives block sets unit good and bad weights.

The boundaries block serialises normalised boundaries.

The actuator block serialises normalised actuators.

The amplitude block uses finite conservative defaults.

The YAML references `policy.yaml`.

The guided CLI path writes only the binding and audit file by default.

If a policy file is required for a downstream command, the operator must create and review it.

## Output files

Minimal scaffold writes `binding_spec.yaml`.

Minimal scaffold writes `README.md`.

Guided scaffold writes `binding_spec.yaml`.

Guided scaffold writes `README.md` if absent.

Guided scaffold writes `llm_scaffold_audit.json`.

Guided scaffold does not overwrite an existing README.

Guided scaffold overwrites the generated binding for the selected domain directory.

Guided scaffold overwrites the scaffold audit record for the selected domain directory.

The domain directory is always under `domainpacks/NAME`.

The domain name must not contain path separators.

The domain name must not contain `..` traversal.

The domain name must match the same identifier expression used by the proposal path.

The CLI prints the scaffolded path after success.

A successful print message is not a validation certificate.

The validation evidence is in the proposal and audit record.

## Audit record contract

`llm_scaffold_audit.json` is designed for review packet storage.

It is sorted and indented by the CLI.

It includes `kind`.

It includes `provider`.

It includes `input_family`.

It includes `description_sha256`.

It includes `raw_response_sha256`.

It includes `validation_errors`.

It includes `oscillator_count`.

It includes `channels`.

It includes `sample_period_s`.

It includes `control_period_s`.

The audit record does not include the raw description.

The audit record does not include the raw provider response.

The audit record should be stored beside the reviewed proposal.

The audit record lets a reviewer detect if a regenerated proposal came from different source text.

The audit record lets a reviewer compare provider identities across proposals.

The audit record lets a reviewer detect validation debt before execution.

## Failure modes

Invalid project names fail before provider invocation.

Empty descriptions fail before provider invocation.

Oversized descriptions fail before provider invocation.

Prompt-override markers fail before provider invocation.

Missing live provider configuration fails closed.

Invalid endpoint schemes fail during provider completion.

Provider connection failures fail as runtime errors.

Provider timeout failures fail as runtime errors.

Malformed provider JSON fails during strict parsing.

Top-level non-object JSON fails during strict parsing.

Missing oscillator lists fail during normalisation.

Empty oscillator lists fail during normalisation.

Duplicate oscillator ids fail during normalisation.

Invalid channels fail during normalisation.

Invalid extractor types fail during normalisation.

Non-finite natural frequencies fail during normalisation.

Invalid periods fail during normalisation.

Invalid safety tiers fail during normalisation.

Negative coupling values fail during normalisation.

Invalid boundary ranges fail during normalisation.

Invalid actuator limits fail during normalisation.

Binding loader failures are captured as validation errors.

Binding validator failures are captured as validation errors.

CLI exceptions are reported as command failures.

No runtime activation happens on failure.

## Operator review checklist

Confirm the domain name matches the intended project.

Confirm every oscillator has a real physical, informational, or symbolic source.

Confirm each channel belongs to the intended N-channel algebra mapping.

Confirm each extractor type matches the available data source.

Confirm each natural frequency is calibrated or explicitly placeholder.

Confirm the sample period matches the measurement cadence.

Confirm the control period matches the planned review cadence.

Confirm the safety tier is correct for the deployment context.

Confirm coupling base strength is justified.

Confirm coupling decay is justified.

Confirm boundaries describe real operating limits.

Confirm boundary severity is correct.

Confirm actuator knobs are actually controllable.

Confirm actuator scopes are correct.

Confirm actuator limits are conservative.

Confirm `validate_binding_spec` returns no errors.

Confirm a separate reviewed policy exists when downstream commands require it.

Confirm the README explains domain assumptions.

Confirm the audit record is retained.

Confirm no generated file is treated as approved without review.

## Scientific modelling notes

Scaffolding is not parameter identification.

Scaffolding is not frequency estimation.

Scaffolding is not coupling inference.

Scaffolding is not safety verification.

Scaffolding is not causal discovery.

Scaffolding is not a proof of synchronisation.

Scaffolding only creates a candidate binding structure.

The candidate structure must be compared with measurements.

The candidate coupling must be calibrated against domain traces.

The candidate boundaries must be derived from physical or operational limits.

The candidate actuators must be checked against real actuator dynamics.

The candidate safety tier must match the domain risk.

The candidate extractor types must match the signal processing path.

The candidate model must be benchmarked in its downstream workflow before release.

## Common domain use cases

Traffic control can map intersections, phases, pedestrian demand, and queue pressure to oscillators.

Cardiac review can map heart-rate, respiration, and coherence signals to review-only oscillators.

Power-grid review can map buses, frequency deviations, and protection zones to oscillators.

Network-security review can map event streams, latency, and anomaly channels to oscillators.

Industrial pump review can map vibration, pressure, flow, and valve state to oscillators.

Plasma review can map safety-factor margins, density channels, and edge-localised modes to oscillators.

BCI review can map EEG bands and behavioural event channels to oscillators.

Warehouse review can map conveyors, queues, and robot zones to oscillators.

Environmental review can map sensor stations, pollutant bands, and weather forcing to oscillators.

Market review can map sector factors, volatility, and liquidity pressure to oscillators.

Each domain still needs domain evidence.

The scaffold only provides a structured starting point.

## CLI examples

Create a minimal scaffold:

```bash
spo scaffold valve_tune
```

Create an offline guided scaffold:

```bash
spo scaffold traffic_grid --llm --description "intersection phases and queue pressure" --llm-response-json proposal.json
```

Validate the generated binding:

```bash
spo validate domainpacks/traffic_grid/binding_spec.yaml
```

Inspect resolved binding configuration:

```bash
spo inspect domainpacks/traffic_grid/binding_spec.yaml --json
```

Run only after review:

```bash
spo run domainpacks/traffic_grid/binding_spec.yaml --steps 100
```

Keep the generation command, proposal file, generated binding, and audit record together in the review packet.

## Library examples

Offline proposal generation can be used directly from Python.

```python
import json
from scpn_phase_orchestrator.scaffold import (
    StaticJSONScaffoldProvider,
    propose_domainpack_from_description,
)

provider = StaticJSONScaffoldProvider(json.dumps({
    "name": "traffic_grid",
    "oscillators": [
        {"id": "north_south", "channel": "I", "extractor_type": "event"},
        {"id": "queue_pressure", "channel": "P", "extractor_type": "physical"},
    ],
}))
proposal = propose_domainpack_from_description(
    "traffic signals and queue pressure",
    project_name="traffic_grid",
    provider=provider,
)
assert proposal.validation_errors == ()
```

The example uses offline provider input.

The example does not contact a live service.

The example does not write files.

The caller chooses where to store the YAML and audit record.

## Validation pipeline

The proposal path produces YAML.

The YAML is written to a temporary `binding_spec.yaml` during validation.

`load_binding_spec` parses the temporary file.

Binding parse failures are captured.

`validate_binding_spec` checks the loaded object.

Binding validator messages are captured.

The proposal carries the resulting error tuple.

The CLI test suite checks that guided output loads cleanly.

The scaffold validation suite checks that minimal output can drive the simulation pipeline.

This page is guarded by `tests/test_reference_api_scaffold.py`.

## Compatibility notes

The scaffold API is Python-owned.

It emits standard domainpack files consumed by the rest of the package.

It does not introduce a Rust, Go, Julia, or Mojo execution path.

No polyglot counterpart needs to be changed when only this reference page changes.

Downstream engines consume the generated binding through the normal binding loader.

If future polyglot generators consume scaffold proposals directly, they should honour the same review-only boundary.

If future benchmarks include scaffold throughput, benchmark docs should cite offline provider fixtures rather than live provider variability.

## Troubleshooting

If `spo scaffold NAME --llm` reports missing provider configuration, use `--llm-response-json` for deterministic review.

If the description is rejected for prompt-override markers, rewrite it as plain domain facts.

If JSON parsing fails, remove markdown fences and explanatory prose from the provider response.

If oscillator validation fails, check required `id`, `channel`, and `extractor_type` fields.

If channel validation fails, use recognised binding channel identifiers.

If extractor validation fails, use one of the supported extractor types.

If period validation fails, use positive finite numeric values.

If boundary validation fails, make lower and upper limits ordered or null where appropriate.

If actuator validation fails, use supported knobs and ordered finite limits.

If binding validation reports missing policy content, add a reviewed `policy.yaml` before downstream commands that require one.

If a generated binding seems scientifically wrong, discard it and regenerate from a better domain description or write the binding manually.

## API reference

::: scpn_phase_orchestrator.scaffold.llm
