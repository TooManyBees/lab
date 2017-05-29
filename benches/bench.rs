#![feature(test)]

extern crate test;
extern crate rand;
extern crate lab;

use lab::Lab;
use test::Bencher;
use self::rand::Rng;

const BENCH_LENGTH: usize = 200;
const RAND_SEED: &[usize] = &[1, 2, 3, 4];

fn random_test_data<T, F>(closure: F) -> Vec<T>
    where F: Fn([u8; 3]) -> T {
    let mut rng: rand::StdRng = rand::SeedableRng::from_seed(RAND_SEED);
    let mut rgbs = Vec::with_capacity(BENCH_LENGTH);
    for _ in 0..BENCH_LENGTH {
        rgbs.push(closure(rng.gen()));
    }
    rgbs
}

#[bench]
fn convert_many_rgbs_to_lab_f32(b: &mut Bencher) {
    let rgbs = random_test_data(|rgb| rgb);

    b.iter(|| {
        rgbs.iter().map(|rgb| Lab::from_rgb(rgb)).collect::<Vec<Lab<f32>>>()
    });
}

#[bench]
fn convert_many_rgbs_to_lab_i8(b: &mut Bencher) {
    let rgbs = random_test_data(|rgb| rgb);

    b.iter(|| {
        rgbs.iter().map(|rgb| Lab::from_rgb(rgb)).collect::<Vec<Lab<i8>>>()
    });
}

#[bench]
fn distance_f32(b: &mut Bencher) {
    let rgbs = random_test_data(|rgb| Lab::from_rgb(&rgb));

    b.iter(|| {
        rgbs.windows(2).map(|labs| labs[0].squared_distance(&labs[1])).collect::<Vec<f32>>()
    });
}

#[bench]
fn distance_i8(b: &mut Bencher) {
    let rgbs = random_test_data(|rgb| Lab::from_rgb(&rgb));

    b.iter(|| {
        rgbs.windows(2).map(|labs| labs[0].squared_distance(&labs[1])).collect::<Vec<i8>>()
    });
}
