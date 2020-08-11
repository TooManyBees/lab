use crate::Lab;
use approx::{AbsDiffEq, RelativeEq};

impl AbsDiffEq<Lab> for Lab {
    type Epsilon = f32;

    fn default_epsilon() -> Self::Epsilon {
        std::f32::EPSILON
    }

    fn abs_diff_eq(&self, other: &Lab, epsilon: Self::Epsilon) -> bool {
        AbsDiffEq::abs_diff_eq(&self.l, &other.l, epsilon) &&
        AbsDiffEq::abs_diff_eq(&self.a, &other.a, epsilon) &&
        AbsDiffEq::abs_diff_eq(&self.b, &other.b, epsilon)
    }
}

impl RelativeEq<Lab> for Lab {
    fn default_max_relative() -> Self::Epsilon {
        std::f32::EPSILON
    }

    fn relative_eq(&self, other: &Lab, epsilon: Self::Epsilon, max_relative: Self::Epsilon) -> bool {
        RelativeEq::relative_eq(&self.l, &other.l, epsilon, max_relative) &&
        RelativeEq::relative_eq(&self.a, &other.a, epsilon, max_relative) &&
        RelativeEq::relative_eq(&self.b, &other.b, epsilon, max_relative)
    }
}
