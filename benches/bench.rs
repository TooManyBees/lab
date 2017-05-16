#![feature(test)]

extern crate test;
extern crate rand;
extern crate lab;

use lab::Lab;
use test::Bencher;
use self::rand::Rng;

const BENCH_LENGTH: usize = 200;
#[bench]
fn convert_rgb_to_lab(b: &mut Bencher) {
    let rand_seed: &[_] = &[1, 2, 3, 4];
    let mut rng: rand::StdRng = rand::SeedableRng::from_seed(rand_seed);
    let mut rgbs = Vec::with_capacity(BENCH_LENGTH);
    for _ in 0..BENCH_LENGTH {
        rgbs.push(rng.gen::<[u8; 3]>());
    }

    b.iter(|| {
        rgbs.iter().map(|rgb| Lab::from_rgb(rgb)).collect::<Vec<Lab>>()
    });
}
