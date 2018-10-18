#[macro_use]
extern crate criterion;
#[macro_use]
extern crate lazy_static;
extern crate rand;
extern crate lab;

use criterion::Criterion;
use rand::Rng;
use rand::distributions::Standard;

lazy_static! {
    static ref LABS: Vec<lab::Lab> = {
        let rand_seed = [0u8;32];
        let mut rng: rand::StdRng = rand::SeedableRng::from_seed(rand_seed);
        let labs: Vec<[f32; 8]> = rng.sample_iter(&Standard).take(512).collect();
        labs.iter().map(|lab| lab::Lab { l: lab[0], a: lab[1], b: lab[2] }).collect()
    };
}

fn labs_to_rgbs(c: &mut Criterion) {
    c.bench_function("labs_to_rgbs", move |b| {
        b.iter(|| lab::labs_to_rgbs(&LABS))
    });
}

fn labs_to_rgbs_simd(c: &mut Criterion) {
    c.bench_function("labs_to_rgbs_simd", move |b| {
        b.iter(|| unsafe { lab::simd::labs_to_rgbs(&LABS) })
    });
}

criterion_group!(benches, labs_to_rgbs, labs_to_rgbs_simd);
criterion_main!(benches);
