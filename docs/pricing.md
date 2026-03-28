<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Pricing

SCPN Phase Orchestrator is dual-licensed: open-source for research and education,
commercial licenses for proprietary integration.

<style>
.pricing-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1.5rem;
  margin: 2rem 0;
}
.pricing-card {
  border: 2px solid #e0e0e0;
  border-radius: 12px;
  padding: 2rem;
  text-align: center;
  transition: transform 0.2s, box-shadow 0.2s;
}
.pricing-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 24px rgba(0,0,0,0.12);
}
.pricing-card.featured {
  border-color: #7c3aed;
  position: relative;
}
.pricing-card.featured::before {
  content: "EARLY ADOPTER";
  position: absolute;
  top: -12px;
  left: 50%;
  transform: translateX(-50%);
  background: #7c3aed;
  color: white;
  padding: 4px 16px;
  border-radius: 12px;
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.05em;
}
.pricing-card h3 {
  margin-top: 0;
  font-size: 1.4rem;
}
.pricing-price {
  font-size: 2.5rem;
  font-weight: 700;
  margin: 1rem 0 0.25rem;
}
.pricing-price-full {
  text-decoration: line-through;
  color: #999;
  font-size: 1.2rem;
}
.pricing-period {
  color: #666;
  font-size: 0.9rem;
  margin-bottom: 0.5rem;
}
.pricing-savings {
  color: #7c3aed;
  font-weight: 700;
  font-size: 0.9rem;
  margin-bottom: 1.5rem;
}
.pricing-features {
  text-align: left;
  list-style: none;
  padding: 0;
  margin: 1.5rem 0;
}
.pricing-features li {
  padding: 0.4rem 0;
  border-bottom: 1px solid #f0f0f0;
}
.pricing-features li::before {
  content: "\2713  ";
  color: #2d6a4f;
  font-weight: bold;
}
.pricing-btn {
  display: inline-block;
  padding: 12px 32px;
  border-radius: 8px;
  font-weight: 600;
  text-decoration: none;
  transition: background 0.2s;
}
.pricing-btn-primary {
  background: #5b21b6;
  color: white !important;
}
.pricing-btn-primary:hover {
  background: #4c1d95;
}
.pricing-btn-outline {
  border: 2px solid #5b21b6;
  color: #5b21b6 !important;
}
.pricing-btn-outline:hover {
  background: #f5f3ff;
}
.pricing-btn-enterprise {
  background: #1a237e;
  color: white !important;
}
.pricing-btn-enterprise:hover {
  background: #0d1642;
}
.pricing-btn-founding {
  background: #7c3aed;
  color: white !important;
}
.pricing-btn-founding:hover {
  background: #6d28d9;
}
.pricing-badge {
  display: inline-block;
  background: #ede9fe;
  color: #5b21b6;
  padding: 2px 10px;
  border-radius: 8px;
  font-size: 0.8rem;
  font-weight: 600;
  margin-bottom: 0.5rem;
}
</style>

!!! warning "Early Adopter Pricing — First 25 Customers"
    SPO commercial licenses are available at **introductory pricing**
    for the first 25 customers. These rates lock in permanently — when the
    program closes, standard pricing applies to new customers.

    **Spots remaining: 25 of 25**

<div class="pricing-grid">

<div class="pricing-card">
<h3>Community</h3>
<div class="pricing-price">Free</div>
<div class="pricing-period">Open Source — AGPL-3.0</div>
<div class="pricing-savings">&nbsp;</div>
<ul class="pricing-features">
<li>9 ODE engines (Kuramoto, Stuart-Landau, inertial, market, swarmalator, ...)</li>
<li>33 domainpacks (plasma, power grid, neuroscience, finance, robotics, ...)</li>
<li>16 dynamical monitors (chimera, EVS, Lyapunov, PAC, transfer entropy, ...)</li>
<li>Differentiable JAX backend (nn/ module)</li>
<li>Rust FFI acceleration (spo-kernel)</li>
<li>QueueWaves cascade failure detector</li>
<li>Closed-loop supervisory control</li>
<li>SHA256-chained deterministic audit replay</li>
<li>Full API documentation + 19 notebooks</li>
<li>Community support (GitHub Discussions)</li>
<li>Source modifications must remain open (AGPL)</li>
</ul>
<a href="https://pypi.org/project/scpn-phase-orchestrator/" class="pricing-btn pricing-btn-outline">pip install scpn-phase-orchestrator</a>
</div>

<div class="pricing-card featured">
<h3>Professional</h3>
<span class="pricing-badge">First 25 customers</span>
<div class="pricing-price-full">CHF 1,490 /yr</div>
<div class="pricing-price">CHF 490</div>
<div class="pricing-period">per seat / year — locked permanently</div>
<div class="pricing-savings">Save CHF 1,000/yr (67% off)</div>
<ul class="pricing-features">
<li>Everything in Community</li>
<li>Closed-source integration permitted</li>
<li>Priority email support (48h business hours)</li>
<li>Custom domainpack development (2/year)</li>
<li>Quarterly security advisories</li>
<li>QueueWaves production deployment support</li>
<li>No AGPL copyleft obligation</li>
</ul>
<a href="https://polar.sh/checkout/polar_c_FSPyudrXL66ZowVoDQukVXe7UgWftgvOfDgtl1eIOVJ" class="pricing-btn pricing-btn-founding">Buy Now — CHF 490/yr</a>
</div>

<div class="pricing-card">
<h3>Enterprise</h3>
<span class="pricing-badge">First 25 customers</span>
<div class="pricing-price-full">CHF 14,900 /yr</div>
<div class="pricing-price">CHF 4,900</div>
<div class="pricing-period">site license / year — locked permanently</div>
<div class="pricing-savings">Save CHF 10,000/yr (67% off)</div>
<ul class="pricing-features">
<li>Everything in Professional</li>
<li>Unlimited seats across organization</li>
<li>Priority email support (24h business hours)</li>
<li>Custom FPGA target integration</li>
<li>Formal safety certification reports (IEC 61508)</li>
<li>White-label and OEM licensing</li>
<li>On-premise deployment assistance</li>
<li>Joint development agreements</li>
</ul>
<a href="https://polar.sh/checkout/polar_c_qFpPwWDGR3H5EuxkGBAwVGaq5yjF6fBQ4dUx14Z3bNJ" class="pricing-btn pricing-btn-enterprise">Buy Now — CHF 4,900/yr</a>
</div>

</div>

<div style="text-align: center; margin: 1.5rem 0;">
<div class="pricing-card" style="display: inline-block; max-width: 420px; border-color: #dc2626; border-width: 3px;">
<span class="pricing-badge" style="background: #fecaca; color: #991b1b;">10 spots — never again</span>
<h3>Founding Member</h3>
<div class="pricing-price-full">CHF 1,490 /yr</div>
<div class="pricing-price">CHF 290</div>
<div class="pricing-period">per seat / year — locked for life</div>
<div class="pricing-savings">Save CHF 1,200/yr (81% off standard price — forever)</div>
<ul class="pricing-features">
<li>Everything in Professional</li>
<li>Lifetime price lock (CHF 290/yr — even when standard is CHF 1,490)</li>
<li>Direct access to lead developer (email + video calls)</li>
<li>Input on roadmap priorities</li>
<li>Name in CONTRIBUTORS.md + release notes</li>
<li>Free 30-day evaluation before commitment</li>
<li>Early access to new engines and domainpacks</li>
</ul>
<a href="https://polar.sh/checkout/polar_c_v5TMwF9nYtryQ6ch0SAczSPzmEm90SW5T8lY604yURg" class="pricing-btn pricing-btn-founding">Claim Founding Spot — CHF 290/yr</a>
<p style="font-size: 0.8rem; color: #666; margin-top: 0.5rem;">
Spots remaining: 10 of 10. When gone, this tier closes permanently.
</p>
</div>
</div>

---

## What You Get

### Closed-Loop Phase Control (Unique to SPO)

No other oscillator library has a supervisory control loop. SPO monitors
coherence in real time, detects regime transitions, and adapts coupling
parameters automatically. The supervisor achieves 26% faster recovery than
passive Kuramoto dynamics.

### 9 ODE Engines in One Package

Standard Kuramoto, Stuart-Landau (amplitude), inertial swing equation
(power grids), financial market regime detection, swarmalator dynamics
(robotics), stochastic resonance, geometric (torus-preserving), time-delayed
coupling, and simplicial 3-body interactions. Each backed by theory: 70+
cited papers.

### Differentiable Phase Dynamics (JAX)

KuramotoLayer and StuartLandauLayer as equinox modules. Gradient-based
coupling optimization, inverse Kuramoto (infer K from data), reservoir
computing, UDE-Kuramoto (physics + neural residual), oscillator Ising
machine. JIT-compilable, vmap-compatible, GPU-ready.

### 33 Domainpacks — Plug and Play

Plasma, power grids, neuroscience, finance, robotics, manufacturing,
aerospace, quantum, traffic, cardiac, circadian, sleep, epidemiology,
photonics, telecommunications — each with a validated `binding_spec.yaml`.
Run `spo scaffold <name>` to create your own.

---

## Feature Comparison

| Feature | Community | Professional | Enterprise |
|---------|:---------:|:------------:|:----------:|
| 9 ODE engines + Rust kernel | Yes | Yes | Yes |
| 33 domainpacks | Yes | Yes | Yes |
| 16 dynamical monitors | Yes | Yes | Yes |
| Differentiable JAX backend (nn/) | Yes | Yes | Yes |
| QueueWaves cascade detector | Yes | Yes | Yes |
| SHA256 audit replay | Yes | Yes | Yes |
| 19 notebooks + full docs | Yes | Yes | Yes |
| Closed-source use | No (AGPL) | **Yes** | **Yes** |
| Priority support | Community | **48h** | **24h** |
| Custom domainpacks | Self-serve | **2/year** | **Unlimited** |
| FPGA deployment support | Self-serve | **Guided** | **Custom targets** |
| Safety certification reports | — | — | **Yes** |
| OEM / white-label | — | — | **Yes** |
| On-premise deployment | — | — | **Yes** |

---

## Why Buy Now?

```
Standard pricing (after early adopter program closes):

    Professional:     CHF 1,490 /yr per seat
    Enterprise:       CHF 14,900 /yr site license

Early adopter pricing (first 25 customers — locked permanently):

    Professional:     CHF 490 /yr per seat      <- you save CHF 1,000/yr
    Enterprise:       CHF 4,900 /yr site license <- you save CHF 10,000/yr
    Founding Member:  CHF 290 /yr per seat       <- you save CHF 1,200/yr

    Example: Year 1 you pay CHF 490.
             Year 5 new customers pay CHF 1,490.
             You still pay CHF 490. Every year. Forever.
```

SPO is the only open-source framework with closed-loop supervisory control
over coupled oscillator dynamics. It combines Kuramoto theory with production
deployment (Rust FFI, FPGA, gRPC, Prometheus). The standard pricing reflects
this — CHF 1,490/yr is less than a single MATLAB Control System Toolbox license.

Early adopters get this at 67-81% off. Permanently.

---

## Academic Pricing

Free Professional license for .edu email addresses. Includes closed-source
rights for thesis work and research prototypes.

<a href="mailto:protoscience@anulum.li?subject=SPO%20Academic%20License&body=Institution:%0AResearch%20group:%0AUse%20case:" class="pricing-btn pricing-btn-outline">Apply for Academic License</a>

---

## FAQ

**Can I use the Community edition commercially?**
Yes, as long as your modifications are released under AGPL-3.0.
If you need proprietary code, choose Professional.

**What domains does SPO support?**
33 domainpacks ship out of the box. Professional includes 2 custom
domainpacks per year. Enterprise includes unlimited custom development.

**What is the early adopter program?**
The first 25 paying customers lock in current pricing permanently.
When the 25th customer signs up, prices increase to standard rates
for all new customers. Existing customers keep their locked rate.

**What is the Founding Member tier?**
10 spots at 81% off standard pricing — CHF 290/yr instead of CHF 1,490.
Includes direct developer access, roadmap input, and a free 30-day
evaluation. Once 10 spots fill, the tier closes permanently.

**Can I evaluate before purchasing?**
The Community edition is fully functional. Founding Members additionally
get 30 days free before their first payment.

**What payment methods do you accept?**
Credit card via Polar.sh, bank transfer (IBAN), or invoice (NET-30 for
Enterprise). All prices in CHF. EUR and GBP accepted at daily exchange rate.

**Do you offer refunds?**
30-day money-back guarantee on all tiers.

---

<p style="text-align: center; margin-top: 3rem;">
<strong>Ready to orchestrate phase dynamics at scale?</strong><br>
<a href="https://polar.sh/checkout/polar_c_FSPyudrXL66ZowVoDQukVXe7UgWftgvOfDgtl1eIOVJ" class="pricing-btn pricing-btn-primary" style="margin-top: 1rem;">Get Started — CHF 490/yr</a>
&nbsp;&nbsp;
<a href="mailto:protoscience@anulum.li?subject=SPO%20Inquiry" class="pricing-btn pricing-btn-outline" style="margin-top: 1rem;">Talk to Us</a>
</p>

---

*SCPN Phase Orchestrator is developed by [ANULUM](https://www.anulum.li) — domain-agnostic
coherence control from simulation to silicon.*

*Contact: [protoscience@anulum.li](mailto:protoscience@anulum.li) |
[www.anulum.li](https://www.anulum.li) |
ORCID: [0009-0009-3560-0851](https://orcid.org/0009-0009-3560-0851)*
