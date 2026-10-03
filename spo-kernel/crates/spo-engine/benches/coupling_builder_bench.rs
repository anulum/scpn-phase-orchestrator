// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Actual coupling construction benchmark

//! Measure Rust construction with the same scalar bits as the Python benchmark.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use spo_engine::coupling::CouplingBuilder;
use spo_types::CouplingConfig;
use std::time::Duration;

fn bench_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("coupling_builder_generic");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_millis(200));
    let config = CouplingConfig {
        base_strength: 0.45,
        decay_alpha: 0.3,
    };
    for n in [16usize, 64, 100] {
        println!(
            "construction_input n={n} base_bits={} decay_bits={}",
            config.base_strength.to_bits(),
            config.decay_alpha.to_bits()
        );
        for _ in 0..2 {
            let state = CouplingBuilder::build(n, &config).expect("valid finite parameters");
            for i in 0..n {
                for j in 0..n {
                    assert!(state.knm[i * n + j].is_finite());
                    assert_eq!(state.knm[i * n + j], state.knm[j * n + i]);
                    assert!(state.knm[i * n + j] >= 0.0);
                }
                assert_eq!(state.knm[i * n + i], 0.0);
            }
            assert!(state.alpha.iter().all(|v| *v == 0.0));
        }
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &size| {
            b.iter(|| {
                criterion::black_box(
                    CouplingBuilder::build(size, &config).expect("valid finite parameters"),
                )
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_construction);
criterion_main!(benches);
