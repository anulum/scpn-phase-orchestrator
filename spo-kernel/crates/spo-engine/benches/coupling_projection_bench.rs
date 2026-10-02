// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Phase Orchestrator — Coupling projection benchmark

//! Measure the actual public finite coupling projection after correctness checks.

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use spo_engine::coupling::project_knm;

fn bench_projection(c: &mut Criterion) {
    let mut group = c.benchmark_group("project_knm");
    for n in [16usize, 64, 256] {
        let raw: Vec<f64> = (0..n * n).map(|i| (i % 31) as f64 / 31.0 - 0.5).collect();
        println!(
            "projection_input n={} bits={:?}",
            n,
            raw.iter().map(|value| value.to_bits()).collect::<Vec<_>>()
        );
        let mut checked = raw.clone();
        project_knm(&mut checked, n).expect("finite coupling input");
        for i in 0..n {
            assert_eq!(checked[i * n + i], 0.0);
            for j in 0..n {
                assert_eq!(checked[i * n + j], checked[j * n + i]);
                assert!(checked[i * n + j].is_finite());
                assert!(checked[i * n + j] >= 0.0);
            }
        }
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &size| {
            b.iter_batched(
                || raw.clone(),
                |mut matrix| {
                    project_knm(&mut matrix, size).expect("finite coupling input");
                    criterion::black_box(matrix);
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_projection);
criterion_main!(benches);
