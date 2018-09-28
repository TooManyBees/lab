"""Calculate XYZ↔sRGB conversion matrices.

See http://www.brucelindbloom.com/index.html?Eqn_XYZ_to_Lab.html for description
of the calculation.  Code uses rational numbers throughout the calculation to
avoid rounding errors creeping in.  Coefficients are converted to floating point
only at the end for printing.
"""

__author__ = 'Michal Nazarewicz <mina86@mina86.com>'

import collections
import fractions


def inverse(M):
    def signed_minor_det(row, col):
        a, b, c, d = [M[r][c]
                      for r in (0, 1, 2) if r != row
                      for c in (0, 1, 2) if c != col]
        res = a * d - b * c
        return res if (row ^ col) & 1 == 0 else -res

    signed_minors = [
        [signed_minor_det(row, col) for col in (0, 1, 2)] for row in (0, 1, 2)
    ]
    det = sum(M[0][col] * signed_minors[0][col] for col in (0, 1, 2))
    return [[signed_minors[col][row] / det for col in (0, 1, 2)]
            for row in (0, 1, 2)]


def main():
    xy = collections.namedtuple('xy', 'x y')

    # https://en.wikipedia.org/wiki/SRGB#The_sRGB_gamut
    r = xy(fractions.Fraction(64, 100), fractions.Fraction(33, 100))
    g = xy(fractions.Fraction(30, 100), fractions.Fraction(60, 100))
    b = xy(fractions.Fraction(15, 100), fractions.Fraction( 6, 100))

    # https://en.wikipedia.org/wiki/Illuminant_D65#Definition
    W = [
        fractions.Fraction(95047, 100000), 1, fractions.Fraction(108883, 100000)
    ]

    # http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
    matrix = [[c.x / c.y for c in (r, g, b)],
              [1, 1, 1],
              [(1 - c.x - c.y) / c.y for c in (r, g, b)]]
    inv = inverse(matrix)
    S = tuple(sum(W[c] * inv[r][c] for c in (0, 1, 2)) for r in (0, 1, 2))
    M = [[matrix[r][c] * S[c] for c in (0, 1, 2)] for r in (0, 1, 2)]

    print('[M] =')
    for row in M:
        print('  {:-20} {:-20} {:-20}'.format(
            *[v.numerator / v.denominator for v in row]))
    print()

    print('[M]^-1 =')
    for row in inverse(M):
        print('  {:-20} {:-20} {:-20}'.format(
            *[v.numerator / v.denominator for v in row]))
    print()


if __name__ == '__main__':
    main()
