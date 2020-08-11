#[macro_use]
extern crate criterion;
#[macro_use]
extern crate lazy_static;
extern crate lab;
extern crate rand;

use criterion::Criterion;
use rand::distributions::Standard;
use rand::Rng;

lazy_static! {
    static ref RGBS: Vec<[u8; 3]> = {
        let rand_seed = [0u8; 32];
        let mut rng: rand::StdRng = rand::SeedableRng::from_seed(rand_seed);
        rng.sample_iter(&Standard).take(512).collect()
    };
}

// fn rgb_to_lab(c: &mut Criterion) {
//     let rgb = RGBS[0];
//     c.bench_function("rgb_to_lab", move |b| {
//         b.iter(|| lab::Lab::from_rgb(&rgb))
//     });
// }

fn rgbs_to_labs(c: &mut Criterion) {
    c.bench_function("rgbs_to_labs", move |b| b.iter(|| lab::rgbs_to_labs(&RGBS)));
}

fn rgbs_to_labs_simd(c: &mut Criterion) {
    c.bench_function("rgbs_to_labs_simd", move |b| {
        b.iter(|| lab::simd::rgbs_to_labs(&RGBS))
    });
}

criterion_group!(benches, rgbs_to_labs, rgbs_to_labs_simd);
criterion_main!(benches);
