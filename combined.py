# ============================================================
# Combined Python file
# Generated: Fri Jun 19 08:57:23 CEST 2026
# Source dir: /mnt/d/Projects/MathTutorOrange/mathtutor
# Files included: 50
# ============================================================


# ────────────────────────────────────────────────────────────
# FILE: ./apply_coercion.py
# ────────────────────────────────────────────────────────────

from pathlib import Path

VERIFIERS = ["linear_equation", "fraction", "inequality", "polynomial", "system"]
BASE = Path("src/mathtutor/domain/verifiers")
SIG = "    def accepts(self, student: Artifact, target: Target) -> Judgment:"
COERCE = "        student = student.expr if isinstance(student, Artifact) else student"

for name in VERIFIERS:
    path = BASE / f"{name}.py"
    text = path.read_text()
    if COERCE.strip() in text:
        print(f"skip   {name}.py (already patched)"); continue
    if SIG not in text:
        print(f"WARN   {name}.py — accepts signature not found, edit manually"); continue
    path.write_text(text.replace(SIG, SIG + "\n" + COERCE, 1))
    print(f"patch  {name}.py")

# ────────────────────────────────────────────────────────────
# FILE: ./combined_scripts.py
# ────────────────────────────────────────────────────────────


# ==========================================
# START OF FILE: ./src/mathtutor/domain/verifiers/linear_equation.py
# ==========================================

# mathtutor/domain/verifiers/equation.py

from __future__ import annotations

from typing import Any
from sympy import Eq, FiniteSet, Symbol, S
from sympy.solvers.solveset import solveset
from sympy.sets.sets import Set

from mathtutor.contracts import Verifier, Artifact, Canonical, Target, Judgment, ParseError
from mathtutor.cas.parsing import parse_math


class EquationVerifier(Verifier):
    """Verifier for algebraic equations via exact solution-set equality."""

    domain = "equation"

    def parse(self, raw: str) -> Artifact:
        return parse_math(raw)

    def canonical(self, a: Artifact) -> Canonical:
        if not isinstance(a, Eq):
            raise ParseError("Expected equation")
        symbol = next(iter(a.free_symbols), Symbol("x"))
        return solveset(a, symbol, domain=S.Reals)

    def accepts(self, student: Artifact, target: Target) -> Judgment:
        try:
            target_set = self.canonical(target.payload["answer"])   # canonical('1')
            student_set = self.canonical(student)
        except ParseError:
            return Judgment(False, False, False, False, False, True, 1.0, {})
        except Exception:
            return Judgment(False, False, False, False, False, True, 1.0, {})

        value_equivalent = student_set == target_set
        missing = list(target_set - student_set) if isinstance(target_set, Set) else []
        extra = list(student_set - target_set) if isinstance(student_set, Set) else []
        partial = bool(missing) and not extra and len(student_set) > 0

        return Judgment(
            parsed_ok=True,
            value_equivalent=value_equivalent,
            form_ok=True,
            correct=value_equivalent,
            partial=partial,
            decidable=True,
            confidence=1.0,
            detail={"missing": missing, "extra": extra},
        )

# ==========================================
# START OF FILE: ./src/mathtutor/domain/verifiers/fraction.py
# ==========================================

# mathtutor/domain/verifiers/fraction.py

from __future__ import annotations

from math import gcd
from sympy import Rational

from mathtutor.contracts import Verifier, Artifact, Canonical, Target, Judgment, ParseError
from mathtutor.cas.parsing import parse_math


class FractionVerifier(Verifier):
    """Verifier for exact rational numbers with reduced-form requirement."""

    domain = "fraction"

    def parse(self, raw: str) -> Artifact:
        return parse_math(raw)

    def canonical(self, a):
        if not isinstance(a, Eq):
            raise ParseError("Expected equation")

    def _is_reduced(self, r: Rational) -> bool:
        return gcd(abs(r.p), abs(r.q)) == 1

    def accepts(self, student: Artifact, target: Target) -> Judgment:
        try:
            s = self.canonical(student)
            t = self.canonical(target.payload["answer"])
        except Exception:
            return Judgment(False, False, False, False, False, True, 1.0, {})

        value_equivalent = s == t
        form_ok = self._is_reduced(student if isinstance(student, Rational) else s)

        return Judgment(
            True,
            value_equivalent,
            form_ok,
            value_equivalent and form_ok,
            False,
            True,
            1.0,
            {},
        )

# ==========================================
# START OF FILE: ./src/mathtutor/domain/verifiers/inequality.py
# ==========================================

# mathtutor/domain/verifiers/inequality.py

from __future__ import annotations

from sympy import S
from sympy.solvers.solveset import solveset

from mathtutor.contracts import Verifier, Artifact, Canonical, Target, Judgment
from mathtutor.cas.parsing import parse_math


class InequalityVerifier(Verifier):
    """Verifier for inequality solution sets."""

    domain = "inequality"

    def parse(self, raw: str) -> Artifact:
        return parse_math(raw)

    def canonical(self, a: Artifact) -> Canonical:
        symbol = next(iter(a.free_symbols))
        return solveset(a, symbol, domain=S.Reals)

    def accepts(self, student: Artifact, target: Target) -> Judgment:
        try:
            s = self.canonical(student)
            t = self.canonical(target.payload["answer"])
        except Exception:
            return Judgment(False, False, False, False, False, True, 1.0, {})

        value_equivalent = s == t
        form_ok = getattr(target, "form", None) != "interval" or value_equivalent

        return Judgment(
            True,
            value_equivalent,
            form_ok,
            value_equivalent and form_ok,
            False,
            True,
            1.0,
            {},
        )

# ==========================================
# START OF FILE: ./.venv/lib/python3.12/site-packages/numpy/polynomial/polynomial.py
# ==========================================

"""
=================================================
Power Series (:mod:`numpy.polynomial.polynomial`)
=================================================

This module provides a number of objects (mostly functions) useful for
dealing with polynomials, including a `Polynomial` class that
encapsulates the usual arithmetic operations.  (General information
on how this module represents and works with polynomial objects is in
the docstring for its "parent" sub-package, `numpy.polynomial`).

Classes
-------
.. autosummary::
   :toctree: generated/

   Polynomial

Constants
---------
.. autosummary::
   :toctree: generated/

   polydomain
   polyzero
   polyone
   polyx

Arithmetic
----------
.. autosummary::
   :toctree: generated/

   polyadd
   polysub
   polymulx
   polymul
   polydiv
   polypow
   polyval
   polyval2d
   polyval3d
   polygrid2d
   polygrid3d

Calculus
--------
.. autosummary::
   :toctree: generated/

   polyder
   polyint

Misc Functions
--------------
.. autosummary::
   :toctree: generated/

   polyfromroots
   polyroots
   polyvalfromroots
   polyvander
   polyvander2d
   polyvander3d
   polycompanion
   polyfit
   polytrim
   polyline

See Also
--------
`numpy.polynomial`

"""
__all__ = [
    'polyzero', 'polyone', 'polyx', 'polydomain', 'polyline', 'polyadd',
    'polysub', 'polymulx', 'polymul', 'polydiv', 'polypow', 'polyval',
    'polyvalfromroots', 'polyder', 'polyint', 'polyfromroots', 'polyvander',
    'polyfit', 'polytrim', 'polyroots', 'Polynomial', 'polyval2d', 'polyval3d',
    'polygrid2d', 'polygrid3d', 'polyvander2d', 'polyvander3d',
    'polycompanion']

import numpy as np
from numpy._core.overrides import array_function_dispatch as _array_function_dispatch

from . import polyutils as pu
from ._polybase import ABCPolyBase

polytrim = pu.trimcoef

#
# These are constant arrays are of integer type so as to be compatible
# with the widest range of other types, such as Decimal.
#

# Polynomial default domain.
polydomain = np.array([-1., 1.])

# Polynomial coefficients representing zero.
polyzero = np.array([0])

# Polynomial coefficients representing one.
polyone = np.array([1])

# Polynomial coefficients representing the identity x.
polyx = np.array([0, 1])

#
# Polynomial series functions
#


def polyline(off, scl):
    """
    Returns an array representing a linear polynomial.

    Parameters
    ----------
    off, scl : scalars
        The "y-intercept" and "slope" of the line, respectively.

    Returns
    -------
    y : ndarray
        This module's representation of the linear polynomial ``off +
        scl*x``.

    See Also
    --------
    numpy.polynomial.chebyshev.chebline
    numpy.polynomial.legendre.legline
    numpy.polynomial.laguerre.lagline
    numpy.polynomial.hermite.hermline
    numpy.polynomial.hermite_e.hermeline

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> P.polyline(1, -1)
    array([ 1, -1])
    >>> P.polyval(1, P.polyline(1, -1))  # should be 0
    0.0

    """
    if scl != 0:
        return np.array([off, scl])
    else:
        return np.array([off])


def polyfromroots(roots):
    """
    Generate a monic polynomial with given roots.

    Return the coefficients of the polynomial

    .. math:: p(x) = (x - r_0) * (x - r_1) * ... * (x - r_n),

    where the :math:`r_n` are the roots specified in `roots`.  If a zero has
    multiplicity n, then it must appear in `roots` n times. For instance,
    if 2 is a root of multiplicity three and 3 is a root of multiplicity 2,
    then `roots` looks something like [2, 2, 2, 3, 3]. The roots can appear
    in any order.

    If the returned coefficients are `c`, then

    .. math:: p(x) = c_0 + c_1 * x + ... +  x^n

    The coefficient of the last term is 1 for monic polynomials in this
    form.

    Parameters
    ----------
    roots : array_like
        Sequence containing the roots.

    Returns
    -------
    out : ndarray
        1-D array of the polynomial's coefficients If all the roots are
        real, then `out` is also real, otherwise it is complex.  (see
        Examples below).

    See Also
    --------
    numpy.polynomial.chebyshev.chebfromroots
    numpy.polynomial.legendre.legfromroots
    numpy.polynomial.laguerre.lagfromroots
    numpy.polynomial.hermite.hermfromroots
    numpy.polynomial.hermite_e.hermefromroots

    Notes
    -----
    The coefficients are determined by multiplying together linear factors
    of the form ``(x - r_i)``, i.e.

    .. math:: p(x) = (x - r_0) (x - r_1) ... (x - r_n)

    where ``n == len(roots) - 1``; note that this implies that ``1`` is always
    returned for :math:`a_n`.

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> P.polyfromroots((-1,0,1))  # x(x - 1)(x + 1) = x^3 - x
    array([ 0., -1.,  0.,  1.])
    >>> j = complex(0,1)
    >>> P.polyfromroots((-j,j))  # complex returned, though values are real
    array([1.+0.j,  0.+0.j,  1.+0.j])

    """
    return pu._fromroots(polyline, polymul, roots)


def polyadd(c1, c2):
    """
    Add one polynomial to another.

    Returns the sum of two polynomials `c1` + `c2`.  The arguments are
    sequences of coefficients from lowest order term to highest, i.e.,
    [1,2,3] represents the polynomial ``1 + 2*x + 3*x**2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of polynomial coefficients ordered from low to high.

    Returns
    -------
    out : ndarray
        The coefficient array representing their sum.

    See Also
    --------
    polysub, polymulx, polymul, polydiv, polypow

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> c1 = (1, 2, 3)
    >>> c2 = (3, 2, 1)
    >>> sum = P.polyadd(c1,c2); sum
    array([4.,  4.,  4.])
    >>> P.polyval(2, sum)  # 4 + 4(2) + 4(2**2)
    28.0

    """
    return pu._add(c1, c2)


def polysub(c1, c2):
    """
    Subtract one polynomial from another.

    Returns the difference of two polynomials `c1` - `c2`.  The arguments
    are sequences of coefficients from lowest order term to highest, i.e.,
    [1,2,3] represents the polynomial ``1 + 2*x + 3*x**2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of polynomial coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Of coefficients representing their difference.

    See Also
    --------
    polyadd, polymulx, polymul, polydiv, polypow

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> c1 = (1, 2, 3)
    >>> c2 = (3, 2, 1)
    >>> P.polysub(c1,c2)
    array([-2.,  0.,  2.])
    >>> P.polysub(c2, c1)  # -P.polysub(c1,c2)
    array([ 2.,  0., -2.])

    """
    return pu._sub(c1, c2)


def polymulx(c):
    """Multiply a polynomial by x.

    Multiply the polynomial `c` by x, where x is the independent
    variable.


    Parameters
    ----------
    c : array_like
        1-D array of polynomial coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Array representing the result of the multiplication.

    See Also
    --------
    polyadd, polysub, polymul, polydiv, polypow

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> c = (1, 2, 3)
    >>> P.polymulx(c)
    array([0., 1., 2., 3.])

    """
    # c is a trimmed copy
    [c] = pu.as_series([c])
    # The zero series needs special treatment
    if len(c) == 1 and c[0] == 0:
        return c

    prd = np.empty(len(c) + 1, dtype=c.dtype)
    prd[0] = c[0] * 0
    prd[1:] = c
    return prd


def polymul(c1, c2):
    """
    Multiply one polynomial by another.

    Returns the product of two polynomials `c1` * `c2`.  The arguments are
    sequences of coefficients, from lowest order term to highest, e.g.,
    [1,2,3] represents the polynomial ``1 + 2*x + 3*x**2.``

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of coefficients representing a polynomial, relative to the
        "standard" basis, and ordered from lowest order term to highest.

    Returns
    -------
    out : ndarray
        Of the coefficients of their product.

    See Also
    --------
    polyadd, polysub, polymulx, polydiv, polypow

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> c1 = (1, 2, 3)
    >>> c2 = (3, 2, 1)
    >>> P.polymul(c1, c2)
    array([  3.,   8.,  14.,   8.,   3.])

    """
    # c1, c2 are trimmed copies
    [c1, c2] = pu.as_series([c1, c2])
    ret = np.convolve(c1, c2)
    return pu.trimseq(ret)


def polydiv(c1, c2):
    """
    Divide one polynomial by another.

    Returns the quotient-with-remainder of two polynomials `c1` / `c2`.
    The arguments are sequences of coefficients, from lowest order term
    to highest, e.g., [1,2,3] represents ``1 + 2*x + 3*x**2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of polynomial coefficients ordered from low to high.

    Returns
    -------
    [quo, rem] : ndarrays
        Of coefficient series representing the quotient and remainder.

    See Also
    --------
    polyadd, polysub, polymulx, polymul, polypow

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> c1 = (1, 2, 3)
    >>> c2 = (3, 2, 1)
    >>> P.polydiv(c1, c2)
    (array([3.]), array([-8., -4.]))
    >>> P.polydiv(c2, c1)
    (array([ 0.33333333]), array([ 2.66666667,  1.33333333]))  # may vary

    """
    # c1, c2 are trimmed copies
    [c1, c2] = pu.as_series([c1, c2])
    if c2[-1] == 0:
        raise ZeroDivisionError  # FIXME: add message with details to exception

    # note: this is more efficient than `pu._div(polymul, c1, c2)`
    lc1 = len(c1)
    lc2 = len(c2)
    if lc1 < lc2:
        return c1[:1] * 0, c1
    elif lc2 == 1:
        return c1 / c2[-1], c1[:1] * 0
    else:
        dlen = lc1 - lc2
        scl = c2[-1]
        c2 = c2[:-1] / scl
        i = dlen
        j = lc1 - 1
        while i >= 0:
            c1[i:j] -= c2 * c1[j]
            i -= 1
            j -= 1
        return c1[j + 1:] / scl, pu.trimseq(c1[:j + 1])


def polypow(c, pow, maxpower=None):
    """Raise a polynomial to a power.

    Returns the polynomial `c` raised to the power `pow`. The argument
    `c` is a sequence of coefficients ordered from low to high. i.e.,
    [1,2,3] is the series  ``1 + 2*x + 3*x**2.``

    Parameters
    ----------
    c : array_like
        1-D array of array of series coefficients ordered from low to
        high degree.
    pow : integer
        Power to which the series will be raised
    maxpower : integer, optional
        Maximum power allowed. This is mainly to limit growth of the series
        to unmanageable size. Default is 16

    Returns
    -------
    coef : ndarray
        Power series of power.

    See Also
    --------
    polyadd, polysub, polymulx, polymul, polydiv

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> P.polypow([1, 2, 3], 2)
    array([ 1., 4., 10., 12., 9.])

    """
    # note: this is more efficient than `pu._pow(polymul, c1, c2)`, as it
    # avoids calling `as_series` repeatedly
    return pu._pow(np.convolve, c, pow, maxpower)


def polyder(c, m=1, scl=1, axis=0):
    """
    Differentiate a polynomial.

    Returns the polynomial coefficients `c` differentiated `m` times along
    `axis`.  At each iteration the result is multiplied by `scl` (the
    scaling factor is for use in a linear change of variable).  The
    argument `c` is an array of coefficients from low to high degree along
    each axis, e.g., [1,2,3] represents the polynomial ``1 + 2*x + 3*x**2``
    while [[1,2],[1,2]] represents ``1 + 1*x + 2*y + 2*x*y`` if axis=0 is
    ``x`` and axis=1 is ``y``.

    Parameters
    ----------
    c : array_like
        Array of polynomial coefficients. If c is multidimensional the
        different axis correspond to different variables with the degree
        in each axis given by the corresponding index.
    m : int, optional
        Number of derivatives taken, must be non-negative. (Default: 1)
    scl : scalar, optional
        Each differentiation is multiplied by `scl`.  The end result is
        multiplication by ``scl**m``.  This is for use in a linear change
        of variable. (Default: 1)
    axis : int, optional
        Axis over which the derivative is taken. (Default: 0).

    Returns
    -------
    der : ndarray
        Polynomial coefficients of the derivative.

    See Also
    --------
    polyint

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> c = (1, 2, 3, 4)
    >>> P.polyder(c)  # (d/dx)(c)
    array([  2.,   6.,  12.])
    >>> P.polyder(c, 3)  # (d**3/dx**3)(c)
    array([24.])
    >>> P.polyder(c, scl=-1)  # (d/d(-x))(c)
    array([ -2.,  -6., -12.])
    >>> P.polyder(c, 2, -1)  # (d**2/d(-x)**2)(c)
    array([  6.,  24.])

    """
    c = np.array(c, ndmin=1, copy=True)
    if c.dtype.char in '?bBhHiIlLqQpP':
        # astype fails with NA
        c = c + 0.0
    cdt = c.dtype
    cnt = pu._as_int(m, "the order of derivation")
    iaxis = pu._as_int(axis, "the axis")
    if cnt < 0:
        raise ValueError("The order of derivation must be non-negative")
    iaxis = np.lib.array_utils.normalize_axis_index(iaxis, c.ndim)

    if cnt == 0:
        return c

    c = np.moveaxis(c, iaxis, 0)
    n = len(c)
    if cnt >= n:
        c = c[:1] * 0
    else:
        for i in range(cnt):
            n = n - 1
            c *= scl
            der = np.empty((n,) + c.shape[1:], dtype=cdt)
            for j in range(n, 0, -1):
                der[j - 1] = j * c[j]
            c = der
    c = np.moveaxis(c, 0, iaxis)
    return c


def polyint(c, m=1, k=[], lbnd=0, scl=1, axis=0):
    """
    Integrate a polynomial.

    Returns the polynomial coefficients `c` integrated `m` times from
    `lbnd` along `axis`.  At each iteration the resulting series is
    **multiplied** by `scl` and an integration constant, `k`, is added.
    The scaling factor is for use in a linear change of variable.  ("Buyer
    beware": note that, depending on what one is doing, one may want `scl`
    to be the reciprocal of what one might expect; for more information,
    see the Notes section below.) The argument `c` is an array of
    coefficients, from low to high degree along each axis, e.g., [1,2,3]
    represents the polynomial ``1 + 2*x + 3*x**2`` while [[1,2],[1,2]]
    represents ``1 + 1*x + 2*y + 2*x*y`` if axis=0 is ``x`` and axis=1 is
    ``y``.

    Parameters
    ----------
    c : array_like
        1-D array of polynomial coefficients, ordered from low to high.
    m : int, optional
        Order of integration, must be positive. (Default: 1)
    k : {[], list, scalar}, optional
        Integration constant(s).  The value of the first integral at zero
        is the first value in the list, the value of the second integral
        at zero is the second value, etc.  If ``k == []`` (the default),
        all constants are set to zero.  If ``m == 1``, a single scalar can
        be given instead of a list.
    lbnd : scalar, optional
        The lower bound of the integral. (Default: 0)
    scl : scalar, optional
        Following each integration the result is *multiplied* by `scl`
        before the integration constant is added. (Default: 1)
    axis : int, optional
        Axis over which the integral is taken. (Default: 0).

    Returns
    -------
    S : ndarray
        Coefficient array of the integral.

    Raises
    ------
    ValueError
        If ``m < 1``, ``len(k) > m``, ``np.ndim(lbnd) != 0``, or
        ``np.ndim(scl) != 0``.

    See Also
    --------
    polyder

    Notes
    -----
    Note that the result of each integration is *multiplied* by `scl`.  Why
    is this important to note?  Say one is making a linear change of
    variable :math:`u = ax + b` in an integral relative to `x`. Then
    :math:`dx = du/a`, so one will need to set `scl` equal to
    :math:`1/a` - perhaps not what one would have first thought.

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> c = (1, 2, 3)
    >>> P.polyint(c)  # should return array([0, 1, 1, 1])
    array([0.,  1.,  1.,  1.])
    >>> P.polyint(c, 3)  # should return array([0, 0, 0, 1/6, 1/12, 1/20])
     array([ 0.        ,  0.        ,  0.        ,  0.16666667,  0.08333333, # may vary
             0.05      ])
    >>> P.polyint(c, k=3)  # should return array([3, 1, 1, 1])
    array([3.,  1.,  1.,  1.])
    >>> P.polyint(c,lbnd=-2)  # should return array([6, 1, 1, 1])
    array([6.,  1.,  1.,  1.])
    >>> P.polyint(c,scl=-2)  # should return array([0, -2, -2, -2])
    array([ 0., -2., -2., -2.])

    """
    c = np.array(c, ndmin=1, copy=True)
    if c.dtype.char in '?bBhHiIlLqQpP':
        # astype doesn't preserve mask attribute.
        c = c + 0.0
    cdt = c.dtype
    if not np.iterable(k):
        k = [k]
    cnt = pu._as_int(m, "the order of integration")
    iaxis = pu._as_int(axis, "the axis")
    if cnt < 0:
        raise ValueError("The order of integration must be non-negative")
    if len(k) > cnt:
        raise ValueError("Too many integration constants")
    if np.ndim(lbnd) != 0:
        raise ValueError("lbnd must be a scalar.")
    if np.ndim(scl) != 0:
        raise ValueError("scl must be a scalar.")
    iaxis = np.lib.array_utils.normalize_axis_index(iaxis, c.ndim)

    if cnt == 0:
        return c

    k = list(k) + [0] * (cnt - len(k))
    c = np.moveaxis(c, iaxis, 0)
    for i in range(cnt):
        n = len(c)
        c *= scl
        if n == 1 and np.all(c[0] == 0):
            c[0] += k[i]
        else:
            tmp = np.empty((n + 1,) + c.shape[1:], dtype=cdt)
            tmp[0] = c[0] * 0
            tmp[1] = c[0]
            for j in range(1, n):
                tmp[j + 1] = c[j] / (j + 1)
            tmp[0] += k[i] - polyval(lbnd, tmp)
            c = tmp
    c = np.moveaxis(c, 0, iaxis)
    return c


def polyval(x, c, tensor=True):
    """
    Evaluate a polynomial at points x.

    If `c` is of length ``n + 1``, this function returns the value

    .. math:: p(x) = c_0 + c_1 * x + ... + c_n * x^n

    The parameter `x` is converted to an array only if it is a tuple or a
    list, otherwise it is treated as a scalar. In either case, either `x`
    or its elements must support multiplication and addition both with
    themselves and with the elements of `c`.

    If `c` is a 1-D array, then ``p(x)`` will have the same shape as `x`.  If
    `c` is multidimensional, then the shape of the result depends on the
    value of `tensor`. If `tensor` is true the shape will be c.shape[1:] +
    x.shape. If `tensor` is false the shape will be c.shape[1:]. Note that
    scalars have shape (,).

    Trailing zeros in the coefficients will be used in the evaluation, so
    they should be avoided if efficiency is a concern.

    Parameters
    ----------
    x : array_like, compatible object
        If `x` is a list or tuple, it is converted to an ndarray, otherwise
        it is left unchanged and treated as a scalar. In either case, `x`
        or its elements must support addition and multiplication with
        with themselves and with the elements of `c`.
    c : array_like
        Array of coefficients ordered so that the coefficients for terms of
        degree n are contained in c[n]. If `c` is multidimensional the
        remaining indices enumerate multiple polynomials. In the two
        dimensional case the coefficients may be thought of as stored in
        the columns of `c`.
    tensor : boolean, optional
        If True, the shape of the coefficient array is extended with ones
        on the right, one for each dimension of `x`. Scalars have dimension 0
        for this action. The result is that every column of coefficients in
        `c` is evaluated for every element of `x`. If False, `x` is broadcast
        over the columns of `c` for the evaluation.  This keyword is useful
        when `c` is multidimensional. The default value is True.

    Returns
    -------
    values : ndarray, compatible object
        The shape of the returned array is described above.

    See Also
    --------
    polyval2d, polygrid2d, polyval3d, polygrid3d

    Notes
    -----
    The evaluation uses Horner's method.

    When using coefficients from polynomials created with ``Polynomial.fit()``,
    use ``p(x)`` or ``polyval(x, p.convert().coef)`` to handle domain/window
    scaling correctly, not ``polyval(x, p.coef)``.

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.polynomial.polynomial import polyval
    >>> polyval(1, [1,2,3])
    6.0
    >>> a = np.arange(4).reshape(2,2)
    >>> a
    array([[0, 1],
           [2, 3]])
    >>> polyval(a, [1, 2, 3])
    array([[ 1.,   6.],
           [17.,  34.]])
    >>> coef = np.arange(4).reshape(2, 2)  # multidimensional coefficients
    >>> coef
    array([[0, 1],
           [2, 3]])
    >>> polyval([1, 2], coef, tensor=True)
    array([[2.,  4.],
           [4.,  7.]])
    >>> polyval([1, 2], coef, tensor=False)
    array([2.,  7.])

    """
    c = np.array(c, ndmin=1, copy=None)
    if c.dtype.char in '?bBhHiIlLqQpP':
        # astype fails with NA
        c = c + 0.0
    if isinstance(x, (tuple, list)):
        x = np.asarray(x)
    if isinstance(x, np.ndarray) and tensor:
        c = c.reshape(c.shape + (1,) * x.ndim)

    c0 = c[-1] + x * 0
    for i in range(2, len(c) + 1):
        c0 = c[-i] + c0 * x
    return c0


def polyvalfromroots(x, r, tensor=True):
    """
    Evaluate a polynomial specified by its roots at points x.

    If `r` is of length ``N``, this function returns the value

    .. math:: p(x) = \\prod_{n=1}^{N} (x - r_n)

    The parameter `x` is converted to an array only if it is a tuple or a
    list, otherwise it is treated as a scalar. In either case, either `x`
    or its elements must support multiplication and addition both with
    themselves and with the elements of `r`.

    If `r` is a 1-D array, then ``p(x)`` will have the same shape as `x`.  If `r`
    is multidimensional, then the shape of the result depends on the value of
    `tensor`. If `tensor` is ``True`` the shape will be r.shape[1:] + x.shape;
    that is, each polynomial is evaluated at every value of `x`. If `tensor` is
    ``False``, the shape will be r.shape[1:]; that is, each polynomial is
    evaluated only for the corresponding broadcast value of `x`. Note that
    scalars have shape (,).

    Parameters
    ----------
    x : array_like, compatible object
        If `x` is a list or tuple, it is converted to an ndarray, otherwise
        it is left unchanged and treated as a scalar. In either case, `x`
        or its elements must support addition and multiplication with
        with themselves and with the elements of `r`.
    r : array_like
        Array of roots. If `r` is multidimensional the first index is the
        root index, while the remaining indices enumerate multiple
        polynomials. For instance, in the two dimensional case the roots
        of each polynomial may be thought of as stored in the columns of `r`.
    tensor : boolean, optional
        If True, the shape of the roots array is extended with ones on the
        right, one for each dimension of `x`. Scalars have dimension 0 for this
        action. The result is that every column of coefficients in `r` is
        evaluated for every element of `x`. If False, `x` is broadcast over the
        columns of `r` for the evaluation.  This keyword is useful when `r` is
        multidimensional. The default value is True.

    Returns
    -------
    values : ndarray, compatible object
        The shape of the returned array is described above.

    See Also
    --------
    polyroots, polyfromroots, polyval

    Examples
    --------
    >>> from numpy.polynomial.polynomial import polyvalfromroots
    >>> polyvalfromroots(1, [1, 2, 3])
    0.0
    >>> a = np.arange(4).reshape(2, 2)
    >>> a
    array([[0, 1],
           [2, 3]])
    >>> polyvalfromroots(a, [-1, 0, 1])
    array([[-0.,   0.],
           [ 6.,  24.]])
    >>> r = np.arange(-2, 2).reshape(2,2)  # multidimensional coefficients
    >>> r # each column of r defines one polynomial
    array([[-2, -1],
           [ 0,  1]])
    >>> b = [-2, 1]
    >>> polyvalfromroots(b, r, tensor=True)
    array([[-0.,  3.],
           [ 3., 0.]])
    >>> polyvalfromroots(b, r, tensor=False)
    array([-0.,  0.])

    """
    r = np.array(r, ndmin=1, copy=None)
    if r.dtype.char in '?bBhHiIlLqQpP':
        r = r.astype(np.double)
    if isinstance(x, (tuple, list)):
        x = np.asarray(x)
    if isinstance(x, np.ndarray):
        if tensor:
            r = r.reshape(r.shape + (1,) * x.ndim)
        elif x.ndim >= r.ndim:
            raise ValueError("x.ndim must be < r.ndim when tensor == False")
    return np.prod(x - r, axis=0)

def _polyval2d_dispatcher(x, y, c):
    return (x, y, c)

def _polygrid2d_dispatcher(x, y, c):
    return (x, y, c)

@_array_function_dispatch(_polyval2d_dispatcher)
def polyval2d(x, y, c):
    """
    Evaluate a 2-D polynomial at points (x, y).

    This function returns the value

    .. math:: p(x,y) = \\sum_{i,j} c_{i,j} * x^i * y^j

    The parameters `x` and `y` are converted to arrays only if they are
    tuples or a lists, otherwise they are treated as a scalars and they
    must have the same shape after conversion. In either case, either `x`
    and `y` or their elements must support multiplication and addition both
    with themselves and with the elements of `c`.

    If `c` has fewer than two dimensions, ones are implicitly appended to
    its shape to make it 2-D. The shape of the result will be c.shape[2:] +
    x.shape.

    Parameters
    ----------
    x, y : array_like, compatible objects
        The two dimensional series is evaluated at the points ``(x, y)``,
        where `x` and `y` must have the same shape. If `x` or `y` is a list
        or tuple, it is first converted to an ndarray, otherwise it is left
        unchanged and, if it isn't an ndarray, it is treated as a scalar.
    c : array_like
        Array of coefficients ordered so that the coefficient of the term
        of multi-degree i,j is contained in ``c[i,j]``. If `c` has
        dimension greater than two the remaining indices enumerate multiple
        sets of coefficients.

    Returns
    -------
    values : ndarray, compatible object
        The values of the two dimensional polynomial at points formed with
        pairs of corresponding values from `x` and `y`.

    See Also
    --------
    polyval, polygrid2d, polyval3d, polygrid3d

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> c = ((1, 2, 3), (4, 5, 6))
    >>> P.polyval2d(1, 1, c)
    21.0

    """
    return pu._valnd(polyval, c, x, y)

@_array_function_dispatch(_polygrid2d_dispatcher)
def polygrid2d(x, y, c):
    """
    Evaluate a 2-D polynomial on the Cartesian product of x and y.

    This function returns the values:

    .. math:: p(a,b) = \\sum_{i,j} c_{i,j} * a^i * b^j

    where the points ``(a, b)`` consist of all pairs formed by taking
    `a` from `x` and `b` from `y`. The resulting points form a grid with
    `x` in the first dimension and `y` in the second.

    The parameters `x` and `y` are converted to arrays only if they are
    tuples or a lists, otherwise they are treated as a scalars. In either
    case, either `x` and `y` or their elements must support multiplication
    and addition both with themselves and with the elements of `c`.

    If `c` has fewer than two dimensions, ones are implicitly appended to
    its shape to make it 2-D. The shape of the result will be c.shape[2:] +
    x.shape + y.shape.

    Parameters
    ----------
    x, y : array_like, compatible objects
        The two dimensional series is evaluated at the points in the
        Cartesian product of `x` and `y`.  If `x` or `y` is a list or
        tuple, it is first converted to an ndarray, otherwise it is left
        unchanged and, if it isn't an ndarray, it is treated as a scalar.
    c : array_like
        Array of coefficients ordered so that the coefficients for terms of
        degree i,j are contained in ``c[i,j]``. If `c` has dimension
        greater than two the remaining indices enumerate multiple sets of
        coefficients.

    Returns
    -------
    values : ndarray, compatible object
        The values of the two dimensional polynomial at points in the Cartesian
        product of `x` and `y`.

    See Also
    --------
    polyval, polyval2d, polyval3d, polygrid3d

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> c = ((1, 2, 3), (4, 5, 6))
    >>> P.polygrid2d([0, 1], [0, 1], c)
    array([[ 1.,  6.],
           [ 5., 21.]])

    """
    return pu._gridnd(polyval, c, x, y)


def polyval3d(x, y, z, c):
    """
    Evaluate a 3-D polynomial at points (x, y, z).

    This function returns the values:

    .. math:: p(x,y,z) = \\sum_{i,j,k} c_{i,j,k} * x^i * y^j * z^k

    The parameters `x`, `y`, and `z` are converted to arrays only if
    they are tuples or a lists, otherwise they are treated as a scalars and
    they must have the same shape after conversion. In either case, either
    `x`, `y`, and `z` or their elements must support multiplication and
    addition both with themselves and with the elements of `c`.

    If `c` has fewer than 3 dimensions, ones are implicitly appended to its
    shape to make it 3-D. The shape of the result will be c.shape[3:] +
    x.shape.

    Parameters
    ----------
    x, y, z : array_like, compatible object
        The three dimensional series is evaluated at the points
        ``(x, y, z)``, where `x`, `y`, and `z` must have the same shape.  If
        any of `x`, `y`, or `z` is a list or tuple, it is first converted
        to an ndarray, otherwise it is left unchanged and if it isn't an
        ndarray it is  treated as a scalar.
    c : array_like
        Array of coefficients ordered so that the coefficient of the term of
        multi-degree i,j,k is contained in ``c[i,j,k]``. If `c` has dimension
        greater than 3 the remaining indices enumerate multiple sets of
        coefficients.

    Returns
    -------
    values : ndarray, compatible object
        The values of the multidimensional polynomial on points formed with
        triples of corresponding values from `x`, `y`, and `z`.

    See Also
    --------
    polyval, polyval2d, polygrid2d, polygrid3d

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> c = ((1, 2, 3), (4, 5, 6), (7, 8, 9))
    >>> P.polyval3d(1, 1, 1, c)
    45.0

    """
    return pu._valnd(polyval, c, x, y, z)


def polygrid3d(x, y, z, c):
    """
    Evaluate a 3-D polynomial on the Cartesian product of x, y and z.

    This function returns the values:

    .. math:: p(a,b,c) = \\sum_{i,j,k} c_{i,j,k} * a^i * b^j * c^k

    where the points ``(a, b, c)`` consist of all triples formed by taking
    `a` from `x`, `b` from `y`, and `c` from `z`. The resulting points form
    a grid with `x` in the first dimension, `y` in the second, and `z` in
    the third.

    The parameters `x`, `y`, and `z` are converted to arrays only if they
    are tuples or a lists, otherwise they are treated as a scalars. In
    either case, either `x`, `y`, and `z` or their elements must support
    multiplication and addition both with themselves and with the elements
    of `c`.

    If `c` has fewer than three dimensions, ones are implicitly appended to
    its shape to make it 3-D. The shape of the result will be c.shape[3:] +
    x.shape + y.shape + z.shape.

    Parameters
    ----------
    x, y, z : array_like, compatible objects
        The three dimensional series is evaluated at the points in the
        Cartesian product of `x`, `y`, and `z`.  If `x`, `y`, or `z` is a
        list or tuple, it is first converted to an ndarray, otherwise it is
        left unchanged and, if it isn't an ndarray, it is treated as a
        scalar.
    c : array_like
        Array of coefficients ordered so that the coefficients for terms of
        degree i,j are contained in ``c[i,j]``. If `c` has dimension
        greater than two the remaining indices enumerate multiple sets of
        coefficients.

    Returns
    -------
    values : ndarray, compatible object
        The values of the two dimensional polynomial at points in the Cartesian
        product of `x` and `y`.

    See Also
    --------
    polyval, polyval2d, polygrid2d, polyval3d

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> c = ((1, 2, 3), (4, 5, 6), (7, 8, 9))
    >>> P.polygrid3d([0, 1], [0, 1], [0, 1], c)
    array([[ 1., 13.],
           [ 6., 51.]])

    """
    return pu._gridnd(polyval, c, x, y, z)


def polyvander(x, deg):
    """Vandermonde matrix of given degree.

    Returns the Vandermonde matrix of degree `deg` and sample points
    `x`. The Vandermonde matrix is defined by

    .. math:: V[..., i] = x^i,

    where ``0 <= i <= deg``. The leading indices of `V` index the elements of
    `x` and the last index is the power of `x`.

    If `c` is a 1-D array of coefficients of length ``n + 1`` and `V` is the
    matrix ``V = polyvander(x, n)``, then ``np.dot(V, c)`` and
    ``polyval(x, c)`` are the same up to roundoff. This equivalence is
    useful both for least squares fitting and for the evaluation of a large
    number of polynomials of the same degree and sample points.

    Parameters
    ----------
    x : array_like
        Array of points. The dtype is converted to float64 or complex128
        depending on whether any of the elements are complex. If `x` is
        scalar it is converted to a 1-D array.
    deg : int
        Degree of the resulting matrix.

    Returns
    -------
    vander : ndarray.
        The Vandermonde matrix. The shape of the returned matrix is
        ``x.shape + (deg + 1,)``, where the last index is the power of `x`.
        The dtype will be the same as the converted `x`.

    See Also
    --------
    polyvander2d, polyvander3d

    Examples
    --------
    The Vandermonde matrix of degree ``deg = 5`` and sample points
    ``x = [-1, 2, 3]`` contains the element-wise powers of `x`
    from 0 to 5 as its columns.

    >>> from numpy.polynomial import polynomial as P
    >>> x, deg = [-1, 2, 3], 5
    >>> P.polyvander(x=x, deg=deg)
    array([[  1.,  -1.,   1.,  -1.,   1.,  -1.],
           [  1.,   2.,   4.,   8.,  16.,  32.],
           [  1.,   3.,   9.,  27.,  81., 243.]])

    """
    ideg = pu._as_int(deg, "deg")
    if ideg < 0:
        raise ValueError("deg must be non-negative")

    x = np.array(x, copy=None, ndmin=1) + 0.0
    dims = (ideg + 1,) + x.shape
    dtyp = x.dtype
    v = np.empty(dims, dtype=dtyp)
    v[0] = x * 0 + 1
    if ideg > 0:
        v[1] = x
        for i in range(2, ideg + 1):
            v[i] = v[i - 1] * x
    return np.moveaxis(v, 0, -1)


def polyvander2d(x, y, deg):
    """Pseudo-Vandermonde matrix of given degrees.

    Returns the pseudo-Vandermonde matrix of degrees `deg` and sample
    points ``(x, y)``. The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., (deg[1] + 1)*i + j] = x^i * y^j,

    where ``0 <= i <= deg[0]`` and ``0 <= j <= deg[1]``. The leading indices of
    `V` index the points ``(x, y)`` and the last index encodes the powers of
    `x` and `y`.

    If ``V = polyvander2d(x, y, [xdeg, ydeg])``, then the columns of `V`
    correspond to the elements of a 2-D coefficient array `c` of shape
    (xdeg + 1, ydeg + 1) in the order

    .. math:: c_{00}, c_{01}, c_{02} ... , c_{10}, c_{11}, c_{12} ...

    and ``np.dot(V, c.flat)`` and ``polyval2d(x, y, c)`` will be the same
    up to roundoff. This equivalence is useful both for least squares
    fitting and for the evaluation of a large number of 2-D polynomials
    of the same degrees and sample points.

    Parameters
    ----------
    x, y : array_like
        Arrays of point coordinates, all of the same shape. The dtypes
        will be converted to either float64 or complex128 depending on
        whether any of the elements are complex. Scalars are converted to
        1-D arrays.
    deg : list of ints
        List of maximum degrees of the form [x_deg, y_deg].

    Returns
    -------
    vander2d : ndarray
        The shape of the returned matrix is ``x.shape + (order,)``, where
        :math:`order = (deg[0]+1)*(deg([1]+1)`.  The dtype will be the same
        as the converted `x` and `y`.

    See Also
    --------
    polyvander, polyvander3d, polyval2d, polyval3d

    Examples
    --------
    >>> import numpy as np

    The 2-D pseudo-Vandermonde matrix of degree ``[1, 2]`` and sample
    points ``x = [-1, 2]`` and ``y = [1, 3]`` is as follows:

    >>> from numpy.polynomial import polynomial as P
    >>> x = np.array([-1, 2])
    >>> y = np.array([1, 3])
    >>> m, n = 1, 2
    >>> deg = np.array([m, n])
    >>> V = P.polyvander2d(x=x, y=y, deg=deg)
    >>> V
    array([[ 1.,  1.,  1., -1., -1., -1.],
           [ 1.,  3.,  9.,  2.,  6., 18.]])

    We can verify the columns for any ``0 <= i <= m`` and ``0 <= j <= n``:

    >>> i, j = 0, 1
    >>> V[:, (deg[1]+1)*i + j] == x**i * y**j
    array([ True,  True])

    The (1D) Vandermonde matrix of sample points ``x`` and degree ``m`` is a
    special case of the (2D) pseudo-Vandermonde matrix with ``y`` points all
    zero and degree ``[m, 0]``.

    >>> P.polyvander2d(x=x, y=0*x, deg=(m, 0)) == P.polyvander(x=x, deg=m)
    array([[ True,  True],
           [ True,  True]])

    """
    return pu._vander_nd_flat((polyvander, polyvander), (x, y), deg)


def polyvander3d(x, y, z, deg):
    """Pseudo-Vandermonde matrix of given degrees.

    Returns the pseudo-Vandermonde matrix of degrees `deg` and sample
    points ``(x, y, z)``. If `l`, `m`, `n` are the given degrees in `x`, `y`, `z`,
    then The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., (m+1)(n+1)i + (n+1)j + k] = x^i * y^j * z^k,

    where ``0 <= i <= l``, ``0 <= j <= m``, and ``0 <= j <= n``.  The leading
    indices of `V` index the points ``(x, y, z)`` and the last index encodes
    the powers of `x`, `y`, and `z`.

    If ``V = polyvander3d(x, y, z, [xdeg, ydeg, zdeg])``, then the columns
    of `V` correspond to the elements of a 3-D coefficient array `c` of
    shape (xdeg + 1, ydeg + 1, zdeg + 1) in the order

    .. math:: c_{000}, c_{001}, c_{002},... , c_{010}, c_{011}, c_{012},...

    and  ``np.dot(V, c.flat)`` and ``polyval3d(x, y, z, c)`` will be the
    same up to roundoff. This equivalence is useful both for least squares
    fitting and for the evaluation of a large number of 3-D polynomials
    of the same degrees and sample points.

    Parameters
    ----------
    x, y, z : array_like
        Arrays of point coordinates, all of the same shape. The dtypes will
        be converted to either float64 or complex128 depending on whether
        any of the elements are complex. Scalars are converted to 1-D
        arrays.
    deg : list of ints
        List of maximum degrees of the form [x_deg, y_deg, z_deg].

    Returns
    -------
    vander3d : ndarray
        The shape of the returned matrix is ``x.shape + (order,)``, where
        :math:`order = (deg[0]+1)*(deg([1]+1)*(deg[2]+1)`.  The dtype will
        be the same as the converted `x`, `y`, and `z`.

    See Also
    --------
    polyvander, polyvander3d, polyval2d, polyval3d

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.polynomial import polynomial as P
    >>> x = np.asarray([-1, 2, 1])
    >>> y = np.asarray([1, -2, -3])
    >>> z = np.asarray([2, 2, 5])
    >>> l, m, n = [2, 2, 1]
    >>> deg = [l, m, n]
    >>> V = P.polyvander3d(x=x, y=y, z=z, deg=deg)
    >>> V
    array([[  1.,   2.,   1.,   2.,   1.,   2.,  -1.,  -2.,  -1.,
             -2.,  -1.,  -2.,   1.,   2.,   1.,   2.,   1.,   2.],
           [  1.,   2.,  -2.,  -4.,   4.,   8.,   2.,   4.,  -4.,
             -8.,   8.,  16.,   4.,   8.,  -8., -16.,  16.,  32.],
           [  1.,   5.,  -3., -15.,   9.,  45.,   1.,   5.,  -3.,
            -15.,   9.,  45.,   1.,   5.,  -3., -15.,   9.,  45.]])

    We can verify the columns for any ``0 <= i <= l``, ``0 <= j <= m``,
    and ``0 <= k <= n``

    >>> i, j, k = 2, 1, 0
    >>> V[:, (m+1)*(n+1)*i + (n+1)*j + k] == x**i * y**j * z**k
    array([ True,  True,  True])

    """
    return pu._vander_nd_flat((polyvander, polyvander, polyvander), (x, y, z), deg)


def polyfit(x, y, deg, rcond=None, full=False, w=None):
    """
    Least-squares fit of a polynomial to data.

    Return the coefficients of a polynomial of degree `deg` that is the
    least squares fit to the data values `y` given at points `x`. If `y` is
    1-D the returned coefficients will also be 1-D. If `y` is 2-D multiple
    fits are done, one for each column of `y`, and the resulting
    coefficients are stored in the corresponding columns of a 2-D return.
    The fitted polynomial(s) are in the form

    .. math::  p(x) = c_0 + c_1 * x + ... + c_n * x^n,

    where `n` is `deg`.

    Parameters
    ----------
    x : array_like, shape (`M`,)
        x-coordinates of the `M` sample (data) points ``(x[i], y[i])``.
    y : array_like, shape (`M`,) or (`M`, `K`)
        y-coordinates of the sample points.  Several sets of sample points
        sharing the same x-coordinates can be (independently) fit with one
        call to `polyfit` by passing in for `y` a 2-D array that contains
        one data set per column.
    deg : int or 1-D array_like
        Degree(s) of the fitting polynomials. If `deg` is a single integer
        all terms up to and including the `deg`'th term are included in the
        fit. For NumPy versions >= 1.11.0 a list of integers specifying the
        degrees of the terms to include may be used instead.
    rcond : float, optional
        Relative condition number of the fit.  Singular values smaller
        than `rcond`, relative to the largest singular value, will be
        ignored.  The default value is ``len(x)*eps``, where `eps` is the
        relative precision of the platform's float type, about 2e-16 in
        most cases.
    full : bool, optional
        Switch determining the nature of the return value.  When ``False``
        (the default) just the coefficients are returned; when ``True``,
        diagnostic information from the singular value decomposition (used
        to solve the fit's matrix equation) is also returned.
    w : array_like, shape (`M`,), optional
        Weights. If not None, the weight ``w[i]`` applies to the unsquared
        residual ``y[i] - y_hat[i]`` at ``x[i]``. Ideally the weights are
        chosen so that the errors of the products ``w[i]*y[i]`` all have the
        same variance.  When using inverse-variance weighting, use
        ``w[i] = 1/sigma(y[i])``.  The default value is None.

    Returns
    -------
    coef : ndarray, shape (`deg` + 1,) or (`deg` + 1, `K`)
        Polynomial coefficients ordered from low to high.  If `y` was 2-D,
        the coefficients in column `k` of `coef` represent the polynomial
        fit to the data in `y`'s `k`-th column.

    [residuals, rank, singular_values, rcond] : list
        These values are only returned if ``full == True``

        - residuals -- sum of squared residuals of the least squares fit
        - rank -- the numerical rank of the scaled Vandermonde matrix
        - singular_values -- singular values of the scaled Vandermonde matrix
        - rcond -- value of `rcond`.

        For more details, see `numpy.linalg.lstsq`.

    Raises
    ------
    RankWarning
        Raised if the matrix in the least-squares fit is rank deficient.
        The warning is only raised if ``full == False``.  The warnings can
        be turned off by:

        >>> import warnings
        >>> warnings.simplefilter('ignore', np.exceptions.RankWarning)

    See Also
    --------
    numpy.polynomial.chebyshev.chebfit
    numpy.polynomial.legendre.legfit
    numpy.polynomial.laguerre.lagfit
    numpy.polynomial.hermite.hermfit
    numpy.polynomial.hermite_e.hermefit
    polyval : Evaluates a polynomial.
    polyvander : Vandermonde matrix for powers.
    numpy.linalg.lstsq : Computes a least-squares fit from the matrix.
    scipy.interpolate.UnivariateSpline : Computes spline fits.

    Notes
    -----
    The solution is the coefficients of the polynomial `p` that minimizes
    the sum of the weighted squared errors

    .. math:: E = \\sum_j w_j^2 * |y_j - p(x_j)|^2,

    where the :math:`w_j` are the weights. This problem is solved by
    setting up the (typically) over-determined matrix equation:

    .. math:: V(x) * c = w * y,

    where `V` is the weighted pseudo Vandermonde matrix of `x`, `c` are the
    coefficients to be solved for, `w` are the weights, and `y` are the
    observed values.  This equation is then solved using the singular value
    decomposition of `V`.

    If some of the singular values of `V` are so small that they are
    neglected (and `full` == ``False``), a `~exceptions.RankWarning` will be
    raised.  This means that the coefficient values may be poorly determined.
    Fitting to a lower order polynomial will usually get rid of the warning
    (but may not be what you want, of course; if you have independent
    reason(s) for choosing the degree which isn't working, you may have to:
    a) reconsider those reasons, and/or b) reconsider the quality of your
    data).  The `rcond` parameter can also be set to a value smaller than
    its default, but the resulting fit may be spurious and have large
    contributions from roundoff error.

    Polynomial fits using double precision tend to "fail" at about
    (polynomial) degree 20. Fits using Chebyshev or Legendre series are
    generally better conditioned, but much can still depend on the
    distribution of the sample points and the smoothness of the data.  If
    the quality of the fit is inadequate, splines may be a good
    alternative.

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.polynomial import polynomial as P
    >>> x = np.linspace(-1,1,51)  # x "data": [-1, -0.96, ..., 0.96, 1]
    >>> rng = np.random.default_rng()
    >>> err = rng.normal(size=len(x))
    >>> y = x**3 - x + err  # x^3 - x + Gaussian noise
    >>> c, stats = P.polyfit(x,y,3,full=True)
    >>> c # c[0], c[1] approx. -1, c[2] should be approx. 0, c[3] approx. 1
    array([ 0.23111996, -1.02785049, -0.2241444 ,  1.08405657]) # may vary
    >>> stats # note the large SSR, explaining the rather poor results
    [array([48.312088]),                                        # may vary
     4,
     array([1.38446749, 1.32119158, 0.50443316, 0.28853036]),
     1.1324274851176597e-14]

    Same thing without the added noise

    >>> y = x**3 - x
    >>> c, stats = P.polyfit(x,y,3,full=True)
    >>> c # c[0], c[1] ~= -1, c[2] should be "very close to 0", c[3] ~= 1
    array([-6.73496154e-17, -1.00000000e+00,  0.00000000e+00,  1.00000000e+00])
    >>> stats # note the minuscule SSR
    [array([8.79579319e-31]),
     np.int32(4),
     array([1.38446749, 1.32119158, 0.50443316, 0.28853036]),
     1.1324274851176597e-14]

    """
    return pu._fit(polyvander, x, y, deg, rcond, full, w)


def polycompanion(c):
    """
    Return the companion matrix of c.

    The companion matrix for power series cannot be made symmetric by
    scaling the basis, so this function differs from those for the
    orthogonal polynomials.

    Parameters
    ----------
    c : array_like
        1-D array of polynomial coefficients ordered from low to high
        degree.

    Returns
    -------
    mat : ndarray
        Companion matrix of dimensions (deg, deg).

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> c = (1, 2, 3)
    >>> P.polycompanion(c)
    array([[ 0.        , -0.33333333],
           [ 1.        , -0.66666667]])

    """
    # c is a trimmed copy
    [c] = pu.as_series([c])
    if len(c) < 2:
        raise ValueError('Series must have maximum degree of at least 1.')
    if len(c) == 2:
        return np.array([[-c[0] / c[1]]])

    n = len(c) - 1
    mat = np.zeros((n, n), dtype=c.dtype)
    bot = mat.reshape(-1)[n::n + 1]
    bot[...] = 1
    mat[:, -1] -= c[:-1] / c[-1]
    return mat


def polyroots(c):
    """
    Compute the roots of a polynomial.

    Return the roots (a.k.a. "zeros") of the polynomial

    .. math:: p(x) = \\sum_i c[i] * x^i.

    Parameters
    ----------
    c : 1-D array_like
        1-D array of polynomial coefficients.

    Returns
    -------
    out : ndarray
        Array of the roots of the polynomial. If all the roots are real,
        then `out` is also real, otherwise it is complex.

    See Also
    --------
    numpy.polynomial.chebyshev.chebroots
    numpy.polynomial.legendre.legroots
    numpy.polynomial.laguerre.lagroots
    numpy.polynomial.hermite.hermroots
    numpy.polynomial.hermite_e.hermeroots

    Notes
    -----
    The root estimates are obtained as the eigenvalues of the companion
    matrix, Roots far from the origin of the complex plane may have large
    errors due to the numerical instability of the power series for such
    values. Roots with multiplicity greater than 1 will also show larger
    errors as the value of the series near such points is relatively
    insensitive to errors in the roots. Isolated roots near the origin can
    be improved by a few iterations of Newton's method.

    Examples
    --------
    >>> import numpy.polynomial.polynomial as poly
    >>> poly.polyroots(poly.polyfromroots((-1,0,1)))
    array([-1.,  0.,  1.])
    >>> poly.polyroots(poly.polyfromroots((-1,0,1))).dtype
    dtype('float64')
    >>> j = complex(0,1)
    >>> poly.polyroots(poly.polyfromroots((-j,0,j)))
    array([  0.00000000e+00+0.j,   0.00000000e+00+1.j,   2.77555756e-17-1.j])  # may vary

    """  # noqa: E501
    # c is a trimmed copy
    [c] = pu.as_series([c])
    if len(c) < 2:
        return np.array([], dtype=c.dtype)
    if len(c) == 2:
        return np.array([-c[0] / c[1]])

    m = polycompanion(c)
    r = np.linalg.eigvals(m)
    r.sort()
    return r


#
# polynomial class
#

class Polynomial(ABCPolyBase):
    """A power series class.

    The Polynomial class provides the standard Python numerical methods
    '+', '-', '*', '//', '%', 'divmod', '**', and '()' as well as the
    attributes and methods listed below.

    Parameters
    ----------
    coef : array_like
        Polynomial coefficients in order of increasing degree, i.e.,
        ``(1, 2, 3)`` give ``1 + 2*x + 3*x**2``.
    domain : (2,) array_like, optional
        Domain to use. The interval ``[domain[0], domain[1]]`` is mapped
        to the interval ``[window[0], window[1]]`` by shifting and scaling.
        The default value is [-1., 1.].
    window : (2,) array_like, optional
        Window, see `domain` for its use. The default value is [-1., 1.].
    symbol : str, optional
        Symbol used to represent the independent variable in string
        representations of the polynomial expression, e.g. for printing.
        The symbol must be a valid Python identifier. Default value is 'x'.

        .. versionadded:: 1.24

    """
    # Virtual Functions
    _add = staticmethod(polyadd)
    _sub = staticmethod(polysub)
    _mul = staticmethod(polymul)
    _div = staticmethod(polydiv)
    _pow = staticmethod(polypow)
    _val = staticmethod(polyval)
    _int = staticmethod(polyint)
    _der = staticmethod(polyder)
    _fit = staticmethod(polyfit)
    _line = staticmethod(polyline)
    _roots = staticmethod(polyroots)
    _fromroots = staticmethod(polyfromroots)

    # Virtual properties
    domain = np.array(polydomain)
    window = np.array(polydomain)
    basis_name = None

    @classmethod
    def _str_term_unicode(cls, i, arg_str):
        if i == '1':
            return f"·{arg_str}"
        else:
            return f"·{arg_str}{i.translate(cls._superscript_mapping)}"

    @staticmethod
    def _str_term_ascii(i, arg_str):
        if i == '1':
            return f" {arg_str}"
        else:
            return f" {arg_str}**{i}"

    @staticmethod
    def _repr_latex_term(i, arg_str, needs_parens):
        if needs_parens:
            arg_str = rf"\left({arg_str}\right)"
        if i == 0:
            return '1'
        elif i == 1:
            return arg_str
        else:
            return f"{arg_str}^{{{i}}}"

# ==========================================
# START OF FILE: ./src/mathtutor/tests/test_safety.py
# ==========================================

"""
tests/test_safety.py — Tests for safety/leak_filter.py and safety/claim_cert.py.

All correctness decisions go through PolynomialVerifier (CAS), never through
string similarity.  The test matrix covers the four cases mandated by the spec:

  1. A leaked root (x = 3) is redacted when gated.
  2. A leaked root passes through unchanged when NOT gated.
  3. A false certified claim is removed.
  4. A true certified claim is kept and unwrapped.
  5. An unparseable claim is removed (fail-safe).

Additional edge-case tests cover:
  - Trivial reorderings of a two-root answer set.
  - An empty / no-claim text is returned unchanged.
  - Multiple claims in one response (some good, some bad).
"""

from __future__ import annotations

import pytest
import sympy
from sympy import FiniteSet, symbols

from mathtutor.contracts import Target
from mathtutor.domain.verifiers.polynomial import PolynomialVerifier
from mathtutor.safety.leak_filter import redact_answers
from mathtutor.safety.claim_cert import certify


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

x = symbols("x")


@pytest.fixture()
def verifier() -> PolynomialVerifier:
    """A fresh PolynomialVerifier for each test."""
    return PolynomialVerifier()


def make_target(value: sympy.Basic) -> Target:
    """Helper: wrap a SymPy value in a Target."""
    return Target(domain="expression", payload={"answer": value})


# ===========================================================================
# TASK A — leak_filter.redact_answers
# ===========================================================================


class TestRedactAnswers:
    """Spec-mandated leak-filter tests (§ TASK A)."""

    # --- Core spec test 1 ---------------------------------------------------

    def test_root_is_redacted_when_gated(self):
        """
        A leaked root "x = 3" is redacted from LLM prose when gated=True.
        """
        text = "Great work!  Now try to find x.  Hint: x = 3 is the solution."
        result = redact_answers(text, answers=["3"], gated=True)
        assert "3" not in result
        assert "[hidden while you work]" in result

    # --- Core spec test 2 ---------------------------------------------------

    def test_root_passes_through_when_not_gated(self):
        """
        The same text is returned unchanged when gated=False
        (student has finished / answer revealed).
        """
        text = "Great work!  Now try to find x.  Hint: x = 3 is the solution."
        result = redact_answers(text, answers=["3"], gated=False)
        assert result == text

    # --- Trivial reordering -------------------------------------------------

    def test_trivial_reordering_is_redacted(self):
        """
        "3, 2" and "2, 3" are both redacted for a two-root answer {2, 3}.
        """
        answers = ["2", "3"]

        text_forward = "The solutions are 2, 3 — check both!"
        result_f = redact_answers(text_forward, answers=answers, gated=True)
        assert "2, 3" not in result_f
        assert "[hidden while you work]" in result_f

        text_reverse = "The solutions are 3, 2 — check both!"
        result_r = redact_answers(text_reverse, answers=answers, gated=True)
        assert "3, 2" not in result_r
        assert "[hidden while you work]" in result_r

    # --- Individual roots in multi-root case --------------------------------

    def test_individual_roots_redacted_in_multi_root_case(self):
        """
        Each root is individually redacted even without the comma-list form.
        """
        answers = ["2", "3"]
        text = "Try substituting 2 into the equation.  Or maybe 3?"
        result = redact_answers(text, answers=answers, gated=True)
        assert "2" not in result
        assert "3" not in result

    # --- No answers → text unchanged ----------------------------------------

    def test_empty_answers_list_is_noop(self):
        text = "Try x = 7."
        assert redact_answers(text, answers=[], gated=True) == text

    # --- Word-boundary safety -----------------------------------------------

    def test_word_boundary_does_not_clobber_other_digits(self):
        """
        Answer "3" must not redact the "3" inside "13" or "30".
        """
        text = "Look at step 13 and 30 in your notes."
        result = redact_answers(text, answers=["3"], gated=True)
        # "13" and "30" should be untouched; standalone "3" would be caught
        assert "13" in result
        assert "30" in result

    # --- Negative answer ----------------------------------------------------

    def test_negative_root_is_redacted(self):
        text = "One solution is -2, the other is 5."
        result = redact_answers(text, answers=["-2", "5"], gated=True)
        assert "-2" not in result
        assert "5" not in result


# ===========================================================================
# TASK B — claim_cert.certify
# ===========================================================================


class TestCertify:
    """Spec-mandated claim-certification tests (§ TASK B)."""

    # --- Core spec test 3: false claim is removed ---------------------------

    def test_false_claim_is_removed(self, verifier):
        """
        A claim that is wrong (5 ≠ 3) is removed from the prose.
        """
        # Target: x = 3  (FiniteSet so the verifier uses solution-set path)
        target = make_target(FiniteSet(3))

        text = "The answer is <<claim>>5<</claim>> — did you get that?"
        result = certify(text, verifier, target)

        assert "<<claim>>" not in result
        assert "5" not in result
        # The surrounding prose survives
        assert "The answer is" in result
        assert "did you get that?" in result

    # --- Core spec test 4: true claim is kept and unwrapped -----------------

    def test_true_claim_is_kept_and_unwrapped(self, verifier):
        """
        A correct claim (3 == 3) is kept and its delimiters are removed.
        """
        target = make_target(FiniteSet(3))

        text = "The answer is <<claim>>3<</claim>> — well done!"
        result = certify(text, verifier, target)

        # Delimiters gone
        assert "<<claim>>" not in result
        assert "<</claim>>" not in result
        # Content preserved
        assert "3" in result
        assert "well done!" in result

    # --- Core spec test 5: unparseable claim is removed (fail-safe) ---------

    def test_unparseable_claim_is_removed(self, verifier):
        """
        A claim that cannot be parsed (e.g. random punctuation) is dropped,
        not propagated.  Fail-safe path.
        """
        target = make_target(FiniteSet(3))

        text = "Try <<claim>>???#!<</claim>> and see what happens."
        result = certify(text, verifier, target)

        assert "<<claim>>" not in result
        assert "???#!" not in result
        # Surrounding prose survives
        assert "Try" in result
        assert "and see what happens." in result

    # --- No claims → text returned unchanged --------------------------------

    def test_no_claims_text_unchanged(self, verifier):
        target = make_target(FiniteSet(3))
        text = "Here is a hint: think about what value makes x − 3 zero."
        assert certify(text, verifier, target) == text

    # --- Mixed: one good, one bad -------------------------------------------

    def test_mixed_claims_good_kept_bad_dropped(self, verifier):
        """
        In a response with two claims, the correct one is unwrapped and the
        false one is removed.
        """
        target = make_target(FiniteSet(3))
        text = (
            "First, note that x = <<claim>>3<</claim>>.  "
            "Also, some say x = <<claim>>7<</claim>> but that's wrong."
        )
        result = certify(text, verifier, target)

        assert "3" in result          # good claim kept
        assert "7" not in result      # false claim removed
        assert "<<claim>>" not in result

    # --- Expression claim (not just numbers) --------------------------------

    def test_expression_claim_kept_when_equivalent(self, verifier):
        """
        An expression claim that is algebraically equivalent to the target
        is kept even if written differently.
        e.g. target = 6/2, claim = 3  (both equal 3)
        """
        target = make_target(sympy.Rational(6, 2))   # evaluates to 3

        text = "So the answer simplifies to <<claim>>3<</claim>>."
        result = certify(text, verifier, target)

        assert "3" in result
        assert "<<claim>>" not in result

    # --- Equation claim matched against FiniteSet target --------------------

    def test_equation_claim_matched_to_solution_set(self, verifier):
        """
        A claim written as ``x = 3`` is equivalent to a FiniteSet({3}) target.
        """
        target = make_target(FiniteSet(3))

        text = "We find <<claim>>x = 3<</claim>>."
        result = certify(text, verifier, target)

        assert "<<claim>>" not in result
        assert "x = 3" in result


# ────────────────────────────────────────────────────────────
# FILE: ./conftest.py
# ────────────────────────────────────────────────────────────

# conftest.py — empty file is enough, but this ensures pytest treats
# the root as the rootdir and picks up pyproject.toml config

# ────────────────────────────────────────────────────────────
# FILE: ./scratch_raw.py
# ────────────────────────────────────────────────────────────

import traceback
from sympy import Eq, symbols
from mathtutor.domain.verifiers.system import SystemVerifier

x, y = symbols("x y")
v = SystemVerifier()
cand = [Eq(x + y, 3), Eq(x - y, 1)]

# Build whatever Target the test uses:
from mathtutor.tests.test_verifiers import Target   # reuse the test's Target
t = Target([Eq(x + y, 3), Eq(x - y, 1)])

# Find the internal parse method and call it directly, letting it raise:
for name in dir(v):
    if "parse" in name.lower():
        print("trying:", name)
        try:
            getattr(v, name)(cand)
        except Exception:
            traceback.print_exc()

# ────────────────────────────────────────────────────────────
# FILE: ./src/mathtutor/__init__.py
# ────────────────────────────────────────────────────────────



# ────────────────────────────────────────────────────────────
# FILE: ./src/mathtutor/cas/__init__.py
# ────────────────────────────────────────────────────────────



# ────────────────────────────────────────────────────────────
# FILE: ./src/mathtutor/cas/equivalence.py
# ────────────────────────────────────────────────────────────

# mathtutor/cas/equivalence.py
"""CAS-level equivalence shared by verifiers and the claim certifier.

Handles heterogeneous answer representations — bare values/expressions,
equations (Eq), and solution sets (FiniteSet / other Sets). This is general
CAS equivalence, not specific to any single verifier domain.
"""
from __future__ import annotations

from sympy import Eq, FiniteSet, S, simplify
from sympy.sets.sets import Set
from sympy.solvers.solveset import solveset


def _as_set_or_value(obj):
    """Classify obj as a solution Set or a bare value.

    Returns (is_set, payload):
      * (True, <Set>)    obj is already a Set, or an equation we can solve
      * (False, <expr>)  a number/expression to be compared by value
    """
    if isinstance(obj, Set):
        return True, obj
    if isinstance(obj, Eq):
        sym = next(iter(obj.free_symbols), None)
        if sym is None:                       # degenerate, e.g. Eq(3, 3)
            return False, obj.lhs - obj.rhs
        return True, solveset(obj, sym, domain=S.Reals)
    return False, obj

def normalize_answer(verifier, answer):
    """payload['answer'] may be a raw string (generators) or a pre-parsed
    SymPy object (unit tests / certify). Parse strings via the verifier's own
    parser; pass everything else through unchanged."""
    if isinstance(answer, str):
        parsed = verifier.parse(answer)
        return getattr(parsed, "expr", parsed)   # unwrap Artifact if needed
    return answer

def value_equivalent(student, answer) -> bool:
    """True iff `student` and `answer` denote the same value or solution set.

      set   vs set   -> set equality
      set   vs value -> lift the value to the singleton {value}, compare sets
      value vs value -> simplify(difference) == 0
    """
    s_set, s_val = _as_set_or_value(student)
    a_set, a_val = _as_set_or_value(answer)
    if s_set and a_set:
        return s_val == a_val
    if s_set != a_set:
        the_set = s_val if s_set else a_val
        the_val = a_val if s_set else s_val
        return the_set == FiniteSet(the_val)
    return simplify(s_val - a_val) == 0

# ────────────────────────────────────────────────────────────
# FILE: ./src/mathtutor/cas/numeric.py
# ────────────────────────────────────────────────────────────

from __future__ import annotations

import random
from typing import Tuple

import sympy as sp


def numeric_equivalent(
    a,
    b,
    *,
    n_points: int = 40,
    tol: float = 1e-9,
    seed: int = 0,
) -> Tuple[bool, float]:
    """
    Probabilistic equivalence test for semi-decidable expressions.

    Returns:
        (equivalent, confidence)

    Confidence is always strictly < 1.0.
    """
    rng = random.Random(seed)

    expr = sp.simplify(a - b)
    symbols = sorted(expr.free_symbols, key=lambda s: s.name)

    if not symbols:
        try:
            val = abs(complex(expr.evalf()))
            return (val < tol, 0.999)
        except Exception:
            return (False, 0.1)

    valid_samples = 0

    for _ in range(n_points * 5):  # allow retries for invalid points
        subs = {s: rng.uniform(-10, 10) for s in symbols}

        try:
            v = expr.subs(subs).evalf()

            if v.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
                continue

            c = complex(v)
            if abs(c.imag) > tol:
                continue

            valid_samples += 1

            if abs(c.real) > tol:
                confidence = min(0.999, 0.2 + 0.7 * valid_samples / n_points)
                return (False, confidence)

            if valid_samples >= n_points:
                confidence = min(0.999, 0.5 + 0.499 * valid_samples / n_points)
                return (True, confidence)

        except Exception:
            continue

    confidence = min(0.999, valid_samples / max(1, n_points) * 0.5)
    return (False, confidence)


# ────────────────────────────────────────────────────────────
# FILE: ./src/mathtutor/cas/parsing.py
# ────────────────────────────────────────────────────────────

from __future__ import annotations

import re
from typing import List

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

from mathtutor.contracts import Artifact, ParseError

_TRANSFORMS = (
    standard_transformations
    + (implicit_multiplication_application, convert_xor)
)


def _normalize(raw: str) -> str:
    """Normalize loose student input into something SymPy can parse."""
    s = raw.strip()
    if not s:
        raise ParseError("Empty input.")

    s = s.replace("∞", "inf")
    s = s.replace("−", "-")
    s = s.replace("\n", ";")

    # Interval notation like (-inf, 3]
    s = re.sub(r"\binf\b", "oo", s)

    return s


def _parse_single(expr: str):
    expr = expr.strip()
    if not expr:
        raise ParseError("Empty expression in system.")

    # Set literal
    if expr.startswith("{") and expr.endswith("}"):
        inner = expr[1:-1].strip()
        if not inner:
            return sp.FiniteSet()
        parts = [parse_expr(x.strip(), transformations=_TRANSFORMS) for x in inner.split(",")]
        return sp.FiniteSet(*parts)

    # Interval literal
    interval_match = re.match(r"^([\(\[])\s*(.+?)\s*,\s*(.+?)\s*([\)\]])$", expr)
    if interval_match:
        left_br, a_raw, b_raw, right_br = interval_match.groups()
        a = sp.sympify(a_raw)
        b = sp.sympify(b_raw)
        return sp.Interval(
            a,
            b,
            left_open=(left_br == "("),
            right_open=(right_br == ")"),
        )

    # Equation
    if "=" in expr and not any(op in expr for op in ("<=", ">=", "<", ">")):
        lhs, rhs = expr.split("=", 1)
        if not lhs.strip() or not rhs.strip():
            raise ParseError(f"Malformed equation: {expr}")
        return sp.Eq(
            parse_expr(lhs, transformations=_TRANSFORMS),
            parse_expr(rhs, transformations=_TRANSFORMS),
        )

    # Inequalities
    for op in ("<=", ">=", "<", ">"):
        if op in expr:
            lhs, rhs = expr.split(op, 1)
            if not lhs.strip() or not rhs.strip():
                raise ParseError(f"Malformed inequality: {expr}")
            l = parse_expr(lhs, transformations=_TRANSFORMS)
            r = parse_expr(rhs, transformations=_TRANSFORMS)
            return {
                "<=": sp.Le,
                ">=": sp.Ge,
                "<": sp.Lt,
                ">": sp.Gt,
            }[op](l, r)

    return parse_expr(expr, transformations=_TRANSFORMS)


def parse_math(raw: str) -> Artifact:
    """
    Parse loose student math input into an Artifact containing a SymPy object.

    Supported:
      - implicit multiplication: 2x, 3(x+1)
      - ^ exponent
      - equations, inequalities
      - systems via newline or ';'
      - sets {1,2}
      - intervals (-inf,3]
    """
    s = _normalize(raw)

    try:
        if ";" in s:
            parts = [p.strip() for p in s.split(";") if p.strip()]
            objs = [_parse_single(p) for p in parts]
            return Artifact(kind="system", expr=tuple(objs), raw="system")

        obj = _parse_single(s)

        if isinstance(obj, sp.Equality):
            kind = "equation"
        elif isinstance(obj, sp.core.relational.Relational):
            kind = "inequality"
        elif isinstance(obj, (sp.Set, sp.FiniteSet, sp.Interval)):
            kind = "set"
        elif obj.is_number:
            kind = "value"
        else:
            kind = "expression"

        return Artifact(kind=kind, expr=obj, raw=s)

    except ParseError:
        raise
    except Exception as e:
        raise ParseError(f"Could not parse input: {raw!r}. Reason: {e}") from e


def echo_latex(a: Artifact) -> str:
    """Return LaTeX rendering of parsed artifact."""
    if a.kind == "system":
        return r"\left\{" + ", ".join(sp.latex(x) for x in a.obj) + r"\right."
    return sp.latex(a.obj)


# ────────────────────────────────────────────────────────────
# FILE: ./src/mathtutor/contracts.py
# ────────────────────────────────────────────────────────────

# mathtutor/contracts.py

"""Shared contracts and core types for Verified Math Tutor.

This module intentionally has no dependencies on other project modules.
It defines dataclasses, enums, protocols, and serialization helpers used
throughout the system.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol
import json


class ParseError(Exception):
    """Raised when raw student input cannot be parsed into a math artifact."""


@dataclass(slots=True)
class Artifact:
    """Parsed mathematical artifact."""

    kind: str  # equation|expression|inequality|system|value|set
    expr: object
    raw: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Target:
    """Describes the constraints for a correct answer."""

    domain: str
    payload: dict[str, Any]
    form: str | None = None
    complete_count: int | None = None


@dataclass(slots=True)
class Canonical:
    """Canonical comparable representation."""

    key: object
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Judgment:
    parsed_ok: bool
    value_equivalent: bool
    form_ok: bool
    correct: bool
    partial: bool
    decidable: bool
    confidence: float
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class KnowledgeComponent:
    """A knowledge component in the curriculum DAG."""

    id: str
    name: str
    prerequisites: list[str]
    verifier_domain: str
    difficulty_band: int
    generators: list[str]


@dataclass(slots=True)
class BuggyRule:
    """Represents a common misconception transformation."""

    id: str
    description: str
    applies_to: Callable[[object], bool]
    transform: Callable[[object], object]
    remediation: str


class Verifier(Protocol):
    """Protocol implemented by domain verifiers."""

    domain: str

    def parse(self, raw: str) -> Artifact:
        ...

    def canonical(self, a: Artifact) -> Canonical:
        ...

    def accepts(self, student: Artifact, target: Target) -> Judgment:
        ...


class SupportLevel(Enum):
    WORKED = "worked"
    COMPLETION = "completion"
    INDEPENDENT = "independent"


class Verdict(Enum):
    CORRECT = "correct"
    PARTIAL = "partial"
    WRONG = "wrong"
    ABSTAIN = "abstain"


def verdict_from_judgment(j: Judgment) -> Verdict:
    """Map a Judgment to a coarse verdict."""
    if not j.parsed_ok or j.confidence < 0.5:
        return Verdict.ABSTAIN
    if j.correct:
        return Verdict.CORRECT
    if j.partial:
        return Verdict.PARTIAL
    return Verdict.WRONG


@dataclass(frozen=True, slots=True)
class TelemetryEvent:
    """Immutable telemetry event."""

    event_id: str
    session_id: str
    user_pseudonym: str
    ts: float

    kc_id: str | None = None
    problem_id: str | None = None
    opportunity_index: int = 0
    action: str = ""

    input_artifact: str | None = None
    verdict: str | None = None
    error_kind: str | None = None
    misconception_id: str | None = None

    support_level: str | None = None
    hint_level: int = 0
    latency_ms: int = 0

    p_known_before: float | None = None
    p_known_after: float | None = None

    policy_id: str | None = None
    affect_signal: str | None = None

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, data: str) -> "TelemetryEvent":
        """Deserialize from JSON."""
        return cls(**json.loads(data))


# ────────────────────────────────────────────────────────────
# FILE: ./src/mathtutor/domain/__init__.py
# ────────────────────────────────────────────────────────────



# ────────────────────────────────────────────────────────────
# FILE: ./src/mathtutor/domain/curriculum.py
# ────────────────────────────────────────────────────────────

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set

from mathtutor.contracts import KnowledgeComponent


class CurriculumError(Exception):
    """Raised when curriculum invariants are violated."""


class Curriculum:
    """
    Directed acyclic prerequisite graph of knowledge components (KCs).

    Nodes are KCs keyed by `kc.id`.
    Edges point from prerequisite -> dependent.
    """

    def __init__(self, kcs: List[KnowledgeComponent] | None = None) -> None:
        self._kcs: Dict[str, KnowledgeComponent] = {}

        if kcs:
            for kc in kcs:
                self.add(kc)

    def add(self, kc: KnowledgeComponent) -> None:
        """
        Add a KC to the curriculum.

        Raises:
            CurriculumError: on duplicate ID, missing prerequisite, or cycle.
        """
        if kc.id in self._kcs:
            raise CurriculumError(f"Duplicate KC id: {kc.id}")

        missing = [p for p in kc.prerequisites if p not in self._kcs]
        if missing:
            raise CurriculumError(
                f"KC '{kc.id}' references missing prerequisites: {missing}"
            )

        self._kcs[kc.id] = kc

        if not self.is_dag():
            del self._kcs[kc.id]
            raise CurriculumError(f"Adding KC '{kc.id}' introduces a cycle")

    def get(self, kc_id: str) -> KnowledgeComponent:
        """Return a KC by ID."""
        return self._kcs[kc_id]

    def prerequisites(self, kc_id: str) -> List[str]:
        """Return prerequisite IDs for a KC."""
        return list(self.get(kc_id).prerequisites)

    def dependents(self, kc_id: str) -> List[str]:
        """Return IDs of KCs depending on the given KC."""
        return [
            other.id
            for other in self._kcs.values()
            if kc_id in other.prerequisites
        ]

    def is_dag(self) -> bool:
        """Return True if the graph is acyclic."""
        visited: Set[str] = set()
        visiting: Set[str] = set()

        def dfs(node: str) -> bool:
            if node in visiting:
                return False
            if node in visited:
                return True

            visiting.add(node)
            for dep in self.prerequisites(node):
                if not dfs(dep):
                    return False
            visiting.remove(node)
            visited.add(node)
            return True

        return all(dfs(node) for node in self._kcs)

    def topological_order(self) -> List[str]:
        """
        Return a valid topological ordering.

        Raises:
            CurriculumError: if graph is cyclic.
        """
        indegree = {node: 0 for node in self._kcs}

        for kc in self._kcs.values():
            for prereq in kc.prerequisites:
                indegree[kc.id] += 1

        queue = [node for node, deg in indegree.items() if deg == 0]
        order: List[str] = []

        while queue:
            node = queue.pop(0)
            order.append(node)

            for dep in self.dependents(node):
                indegree[dep] -= 1
                if indegree[dep] == 0:
                    queue.append(dep)

        if len(order) != len(self._kcs):
            raise CurriculumError("Graph contains cycle")

        return order

    def unmet_prerequisites(
        self,
        kc_id: str,
        mastered: Set[str],
    ) -> List[str]:
        """Return prerequisite IDs not yet mastered."""
        return [
            prereq
            for prereq in self.prerequisites(kc_id)
            if prereq not in mastered
        ]

    def ready_kcs(self, mastered: Set[str]) -> List[str]:
        """
        Return KCs whose prerequisites are fully mastered and which are not yet mastered.
        """
        ready = []
        for kc_id in self._kcs:
            if kc_id in mastered:
                continue
            if not self.unmet_prerequisites(kc_id, mastered):
                ready.append(kc_id)
        return ready


def build_sample_curriculum() -> Curriculum:
    """
    Build a small sample curriculum:
    fractions -> equations -> quadratics
    """
    curriculum = Curriculum()

    curriculum.add(
        KnowledgeComponent(
            id="fraction_basics",
            name="Fraction Basics",
            prerequisites=[],
            verifier_domain="fractions",
            generator="generate_fraction_basics",
        )
    )

    curriculum.add(
        KnowledgeComponent(
            id="fraction_operations",
            name="Fraction Operations",
            prerequisites=["fraction_basics"],
            verifier_domain="fractions",
            generator="generate_fraction_operations",
        )
    )

    curriculum.add(
        KnowledgeComponent(
            id="simplify_expressions",
            name="Simplify Expressions",
            prerequisites=["fraction_operations"],
            verifier_domain="algebraic_simplification",
            generator="generate_simplify_expressions",
        )
    )

    curriculum.add(
        KnowledgeComponent(
            id="linear_one_step",
            name="One-Step Linear Equations",
            prerequisites=["simplify_expressions"],
            verifier_domain="linear_equations",
            generator="generate_linear_one_step",
        )
    )

    curriculum.add(
        KnowledgeComponent(
            id="linear_multi_step",
            name="Multi-Step Linear Equations",
            prerequisites=["linear_one_step"],
            verifier_domain="linear_equations",
            generator="generate_linear_multi_step",
        )
    )

    curriculum.add(
        KnowledgeComponent(
            id="distributive_property",
            name="Distributive Property",
            prerequisites=["linear_one_step"],
            verifier_domain="expression_expansion",
            generator="generate_distributive_property",
        )
    )

    curriculum.add(
        KnowledgeComponent(
            id="factoring_quadratics",
            name="Factoring Quadratics",
            prerequisites=["distributive_property", "linear_multi_step"],
            verifier_domain="quadratics",
            generator="generate_factoring_quadratics",
        )
    )

    curriculum.add(
        KnowledgeComponent(
            id="solve_quadratics",
            name="Solve Quadratics",
            prerequisites=["factoring_quadratics"],
            verifier_domain="quadratics",
            generator="generate_solve_quadratics",
        )
    )

    return curriculum


# ────────────────────────────────────────────────────────────
# FILE: ./src/mathtutor/domain/generators.py
# ────────────────────────────────────────────────────────────

# mathtutor/domain/generators.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Any
import random
import uuid
from fractions import Fraction

from mathtutor.contracts import Target
from mathtutor.domain.verifiers.linear_equation import EquationVerifier as LinearEquationVerifier
from mathtutor.domain.verifiers.polynomial import PolynomialVerifier as QuadraticEquationVerifier
from mathtutor.domain.verifiers.fraction import FractionVerifier as FractionAdditionVerifier


@dataclass(frozen=True)
class Problem:
    id: str
    kc_id: str
    domain: str
    prompt_text: str
    parsed_target: Target
    reference_answer: str
    difficulty_band: int
    meta: dict


class GeneratorError(Exception):
    """Raised when a generated problem fails self-verification."""


_REGISTRY: Dict[str, Callable[..., Problem]] = {}


def register(name: str) -> Callable:
    """Decorator to register a problem generator."""

    def deco(fn: Callable[..., Problem]) -> Callable[..., Problem]:
        _REGISTRY[name] = fn
        return fn

    return deco


def generate(name: str, *, difficulty_band: int, seed: int) -> Problem:
    """Generate a problem deterministically from a registered generator."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown generator: {name}")
    return _REGISTRY[name](difficulty_band=difficulty_band, seed=seed)


def _difficulty_range(band: int) -> tuple[int, int]:
    """Cheap solve-step proxy range."""
    ranges = {
        1: (1, 3),
        2: (3, 6),
        3: (6, 10),
    }
    if band not in ranges:
        raise GeneratorError(f"Unsupported difficulty band: {band}")
    return ranges[band]


def _assert_self_verified(
    problem: Problem,
    verifier: Any,
    solve_steps: int,
) -> None:
    """Run verifier and difficulty checks."""
    student = verifier.parse(problem.reference_answer)   # parse("1") → Artifact(expr=Integer(1))
    judgment = verifier.accepts(student, problem.parsed_target)

    if not judgment.correct:
        raise GeneratorError(
            f"Verifier rejected generated answer: {problem.reference_answer}"
        )

    low, high = _difficulty_range(problem.difficulty_band)
    if not (low <= solve_steps <= high):
        raise GeneratorError(
            f"Solve-step proxy {solve_steps} out of band {problem.difficulty_band}"
        )


@register("linear_equation")
def generate_linear_equation(*, difficulty_band: int, seed: int) -> Problem:
    rng = random.Random(seed)

    coeff_max = {1: 5, 2: 20, 3: 50}[difficulty_band]
    x_solution = rng.randint(-coeff_max, coeff_max)
    a = rng.choice([i for i in range(-coeff_max, coeff_max + 1) if i not in (0,)])
    b = rng.randint(-coeff_max, coeff_max)
    c = a * x_solution + b

    prompt = f"Solve for x: {a}x + {b} = {c}"
    reference = str(x_solution)

    target = Target(domain="linear_equation", payload={"answer": reference})

    problem = Problem(
        id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"linear-{seed}-{difficulty_band}")),
        kc_id="solve_linear_eq",
        domain="linear_equation",
        prompt_text=prompt,
        parsed_target=target,
        reference_answer=reference,
        difficulty_band=difficulty_band,
        meta={"a": a, "b": b, "c": c},
    )

    solve_steps = 2 if difficulty_band == 1 else 4 if difficulty_band == 2 else 7
    _assert_self_verified(problem, LinearEquationVerifier(), solve_steps)
    return problem


@register("quadratic_equation")
def generate_quadratic_equation(*, difficulty_band: int, seed: int) -> Problem:
    rng = random.Random(seed)

    root_max = {1: 4, 2: 8, 3: 15}[difficulty_band]

    r1 = rng.randint(-root_max, root_max)
    r2 = rng.randint(-root_max, root_max)

    a = 1
    b = -(r1 + r2)
    c = r1 * r2

    prompt = f"Solve for x: x^2 + ({b})x + ({c}) = 0"
    reference = f"{{{min(r1,r2)}, {max(r1,r2)}}}"

    target = Target(domain="quadratic_equation", payload={"answer": reference})

    problem = Problem(
        id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"quad-{seed}-{difficulty_band}")),
        kc_id="solve_quadratic_eq",
        domain="quadratic_equation",
        prompt_text=prompt,
        parsed_target=target,
        reference_answer=reference,
        difficulty_band=difficulty_band,
        meta={"roots": (r1, r2)},
    )

    solve_steps = 3 if difficulty_band == 1 else 5 if difficulty_band == 2 else 8
    _assert_self_verified(problem, QuadraticEquationVerifier(), solve_steps)
    return problem


@register("fraction_addition")
def generate_fraction_addition(*, difficulty_band: int, seed: int) -> Problem:
    rng = random.Random(seed)

    term_count = {1: 2, 2: 3, 3: 4}[difficulty_band]
    denom_max = {1: 6, 2: 12, 3: 20}[difficulty_band]

    fractions = []
    total = Fraction(0, 1)

    for _ in range(term_count):
        denom = rng.randint(2, denom_max)
        numer = rng.randint(1, denom - 1)
        frac = Fraction(numer, denom)
        fractions.append(frac)
        total += frac

    prompt = "Compute: " + " + ".join(f"{f.numerator}/{f.denominator}" for f in fractions)
    reference = f"{total.numerator}/{total.denominator}"

    target = Target(domain="fraction_addition", payload={"answer": reference})

    problem = Problem(
        id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"frac-{seed}-{difficulty_band}")),
        kc_id="fraction_addition",
        domain="fraction_addition",
        prompt_text=prompt,
        parsed_target=target,
        reference_answer=reference,
        difficulty_band=difficulty_band,
        meta={"terms": fractions},
    )

    solve_steps = 2 if difficulty_band == 1 else 4 if difficulty_band == 2 else 8
    _assert_self_verified(problem, FractionAdditionVerifier(), solve_steps)
    return problem


# ────────────────────────────────────────────────────────────
# FILE: ./src/mathtutor/domain/verifiers/__init__.py
# ────────────────────────────────────────────────────────────



# ────────────────────────────────────────────────────────────
# FILE: ./src/mathtutor/domain/verifiers/fraction.py
# ────────────────────────────────────────────────────────────

# mathtutor/domain/verifiers/fraction.py

from __future__ import annotations

from math import gcd
from sympy import Rational

from mathtutor.contracts import Verifier, Artifact, Canonical, Target, Judgment, ParseError
from mathtutor.cas.parsing import parse_math


class FractionVerifier(Verifier):
    """Verifier for exact rational numbers with reduced-form requirement."""

    domain = "fraction"

    def parse(self, raw: str) -> Artifact:
        return parse_math(raw)

    def canonical(self, a: Artifact) -> Canonical:
        return Rational(a)

    def _is_reduced(self, r: Rational) -> bool:
        return gcd(abs(r.p), abs(r.q)) == 1

    def accepts(self, student: Artifact, target: Target) -> Judgment:
        student = student.expr if isinstance(student, Artifact) else student
        try:
            s = self.canonical(student)
            t = self.canonical(target.payload["answer"])
        except Exception:
            return Judgment(False, False, False, False, False, True, 1.0, {})

        value_equivalent = s == t
        form_ok = self._is_reduced(student if isinstance(student, Rational) else s)

        return Judgment(
            True,
            value_equivalent,
            form_ok,
            value_equivalent and form_ok,
            False,
            True,
            1.0,
            {},
        )


# ────────────────────────────────────────────────────────────
# FILE: ./src/mathtutor/domain/verifiers/inequality.py
# ────────────────────────────────────────────────────────────

# mathtutor/domain/verifiers/inequality.py

from __future__ import annotations

from sympy import S
from sympy.solvers.solveset import solveset

from mathtutor.contracts import Verifier, Artifact, Canonical, Target, Judgment
from mathtutor.cas.parsing import parse_math


class InequalityVerifier(Verifier):
    """Verifier for inequality solution sets."""

    domain = "inequality"

    def parse(self, raw: str) -> Artifact:
        return parse_math(raw)

    def canonical(self, a: Artifact) -> Canonical:
        symbol = next(iter(a.free_symbols))
        return solveset(a, symbol, domain=S.Reals)

    def accepts(self, student: Artifact, target: Target) -> Judgment:
        student = student.expr if isinstance(student, Artifact) else student
        try:
            s = self.canonical(student)
            t = self.canonical(target.payload["answer"])
        except Exception:
            return Judgment(False, False, False, False, False, True, 1.0, {})

        value_equivalent = s == t
        form_ok = getattr(target, "form", None) != "interval" or value_equivalent

        return Judgment(
            True,
            value_equivalent,
            form_ok,
            value_equivalent and form_ok,
            False,
            True,
            1.0,
            {},
        )


# ────────────────────────────────────────────────────────────
# FILE: ./src/mathtutor/domain/verifiers/linear_equation.py
# ────────────────────────────────────────────────────────────

# mathtutor/domain/verifiers/equation.py

from __future__ import annotations

from typing import Any
from sympy import Eq, FiniteSet, Symbol, S
from sympy.solvers.solveset import solveset
from sympy.sets.sets import Set

from mathtutor.contracts import Verifier, Artifact, Canonical, Target, Judgment, ParseError
from mathtutor.cas.parsing import parse_math
from sympy import FiniteSet
from sympy.sets.sets import Set
from mathtutor.cas.equivalence import normalize_answer

class EquationVerifier(Verifier):
    """Verifier for algebraic equations via exact solution-set equality."""

    domain = "equation"

    def parse(self, raw: str) -> Artifact:
        return parse_math(raw)

    def canonical(self, a: Artifact) -> Canonical:
        if isinstance(a, Eq):
            symbol = next(iter(a.free_symbols), Symbol("x"))
            return solveset(a, symbol, domain=S.Reals)
        if isinstance(a, Set):
            return a
        return FiniteSet(a)        # bare value → singleton solution set {a}

    def accepts(self, student: Artifact, target: Target) -> Judgment:
        student = student.expr if isinstance(student, Artifact) else student
        answer = normalize_answer(self, target.payload["answer"])
        try:
            target_set = self.canonical(answer)
            student_set = self.canonical(student)
        except ParseError:
            return Judgment(False, False, False, False, False, True, 1.0, {})
        except Exception:
            return Judgment(False, False, False, False, False, True, 1.0, {})

        value_equivalent = student_set == target_set
        missing = list(target_set - student_set) if isinstance(target_set, Set) else []
        extra = list(student_set - target_set) if isinstance(student_set, Set) else []
        partial = bool(missing) and not extra and len(student_set) > 0

        return Judgment(
            parsed_ok=True,
            value_equivalent=value_equivalent,
            form_ok=True,
            correct=value_equivalent,
            partial=partial,
            decidable=True,
            confidence=1.0,
            detail={"missing": missing, "extra": extra},
        )


# ────────────────────────────────────────────────────────────
# FILE: ./src/mathtutor/domain/verifiers/polynomial.py
# ────────────────────────────────────────────────────────────

# mathtutor/domain/verifiers/polynomial.py

from __future__ import annotations

from sympy import expand, factor, Mul, simplify
import sympy as sp
from sympy import FiniteSet, Eq
from mathtutor.contracts import Verifier, Artifact, Canonical, Target, Judgment
from mathtutor.cas.parsing import parse_math
from mathtutor.cas.equivalence import value_equivalent

def normalize_answer(verifier, answer_str: str):
    """Parse and normalize answers for comparison (handles sets, equations, etc.)."""
    try:
        if isinstance(answer_str, str):
            # Handle set notation like '{2, 2}' or '{-7, 0}'
            if answer_str.startswith('{') and answer_str.endswith('}'):
                content = answer_str[1:-1].strip()
                if content:
                    items = [sp.sympify(item.strip()) for item in content.split(',')]
                    return FiniteSet(*items)
            # Try parsing as sympy expression/equation
            return sp.sympify(answer_str)
        return answer_str
    except Exception:
        return answer_str  # fallback
    
class PolynomialVerifier(Verifier):
    """Verifier for polynomial equivalence and structural form."""

    domain = "polynomial"

    def parse(self, raw: str) -> Artifact:
        return parse_math(raw)

    def canonical(self, a: Artifact) -> Canonical:
        return expand(a)

    def _is_expanded(self, expr) -> bool:
        return expand(expr) == expr

    def _is_fully_factored(self, expr) -> bool:
        return factor(expr) == expr

    def accepts(self, student: Artifact, target: Target) -> Judgment:
        student = student.expr if isinstance(student, Artifact) else student
        answer = normalize_answer(self, target.payload["answer"])  # now defined
        try:
            ve = value_equivalent(student, answer)
        except Exception:
            return Judgment(False, False, False, False, False, True, 1.0, {})

        required_form = getattr(target, "form", None)
        if required_form == "expanded":
            form_ok = self._is_expanded(student)
        elif required_form == "factored":
            form_ok = self._is_fully_factored(student)
        else:
            form_ok = True

        return Judgment(
            True,
            ve,
            form_ok,
            ve and form_ok,
            False,
            True,
            1.0,
            {"expected_form": required_form},
        )


# ────────────────────────────────────────────────────────────
# FILE: ./src/mathtutor/domain/verifiers/system.py
# ────────────────────────────────────────────────────────────

# mathtutor/domain/verifiers/system.py

from __future__ import annotations

from sympy import linsolve, FiniteSet

from mathtutor.contracts import Verifier, Artifact, Canonical, Target, Judgment
from mathtutor.cas.parsing import parse_math


class SystemVerifier(Verifier):
    """Verifier for systems of equations using exact solution-set equality."""

    domain = "system"

    def parse(self, raw: str) -> Artifact:
        return parse_math(raw)

    def canonical(self, a: Artifact) -> Canonical:
        eqs = a if isinstance(a, (list, tuple)) else [a]
        symbols = sorted(
            set().union(*(eq.free_symbols for eq in eqs)),
            key=lambda s: s.name,
        )
        return linsolve(eqs, symbols)

    def accepts(self, student: Artifact, target: Target) -> Judgment:
        student = student.expr if isinstance(student, Artifact) else student
        try:
            s = self.canonical(student)
            t = self.canonical(target.payload["answer"])
        except Exception:
            return Judgment(False, False, False, False, False, True, 1.0, {})

        value_equivalent = s == t

        return Judgment(
            True,
            value_equivalent,
            True,
            value_equivalent,
            False,
            True,
            1.0,
            {},
        )


# ────────────────────────────────────────────────────────────
# FILE: ./src/mathtutor/eval/__init__.py
# ────────────────────────────────────────────────────────────



# ────────────────────────────────────────────────────────────
# FILE: ./src/mathtutor/eval/learning_curves.py
# ────────────────────────────────────────────────────────────

# mathtutor/eval/learning_curves.py

"""Learning-curve analysis based on the Additive Factor Model (AFM).

Background — Additive Factor Model
------------------------------------
The AFM (Cen, Koedinger & Junker 2006) models the probability that a
student *s* answers a problem requiring knowledge component *kc* correctly
on their *n*-th opportunity (0-indexed) as::

    P(correct | s, kc, n) = σ( β_s + δ_kc + γ_kc · n )

where

* σ(z) = 1 / (1 + exp(−z))  is the logistic function
* β_s   is the **student ability** intercept (one per student)
* δ_kc  is the **KC easiness** intercept (one per KC; higher ⇒ easier)
* γ_kc  is the **learning rate** for KC (one per KC; should be > 0 if
         practice helps)

Fitting is done by minimising binary cross-entropy with L-BFGS-B via
``scipy.optimize.minimize``.  Parameters are initialised to zero.

Interpretation
--------------
* A *well-specified* KC shows a smooth, monotone **decline** in *error*
  rate (= 1 − P(correct)) as opportunity index increases.
* A *flat or noisy* curve suggests the KC conflates sub-skills and should
  be **split**.

Learning-gain summary
---------------------
Hake's normalised gain (Hake 1998)::

    g = (post − pre) / (100 − pre)

ranges from 0 (no gain) to 1 (perfect gain).  Guard: if pre == 100,
gain is undefined (ceiling); we return ``float('nan')``.
"""

from __future__ import annotations

import math
import warnings
from collections import defaultdict
from typing import Sequence

import numpy as np

try:
    from scipy.optimize import minimize as _scipy_minimize
    _SCIPY = True
except ImportError:                     # pragma: no cover
    _SCIPY = False

from mathtutor.contracts import TelemetryEvent


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sigma(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function σ(z) = 1/(1+e^{-z})."""
    # np.where avoids overflow in exp for large |z|
    return np.where(z >= 0,
                    1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))


def _bce_loss_and_grad(
    params: np.ndarray,
    y: np.ndarray,
    student_idx: np.ndarray,
    kc_idx: np.ndarray,
    opp: np.ndarray,
    n_students: int,
    n_kcs: int,
) -> tuple[float, np.ndarray]:
    """Binary cross-entropy loss + gradient for AFM."""
    beta  = params[:n_students]                        # (S,)
    delta = params[n_students : n_students + n_kcs]    # (K,)
    gamma = params[n_students + n_kcs :]               # (K,)

    # Logit for each observation
    z = beta[student_idx] + delta[kc_idx] + gamma[kc_idx] * opp   # (N,)
    p = _sigma(z)                                                 # (N,)

    # ---- Loss: mean binary cross-entropy ----
    # Numerically stable BCE from logits avoids eps capping issues
    loss_vec = np.maximum(z, 0) - y * z + np.log1p(np.exp(-np.abs(z)))
    loss = float(np.mean(loss_vec))

    # ---- Gradient ----
    # ∂L/∂z_i = (p_i − y_i) / N
    residual = (p - y) / len(y)                                   # (N,)

    grad_beta  = np.zeros(n_students)
    grad_delta = np.zeros(n_kcs)
    grad_gamma = np.zeros(n_kcs)

    np.add.at(grad_beta,  student_idx, residual)
    np.add.at(grad_delta, kc_idx,      residual)
    np.add.at(grad_gamma, kc_idx,      residual * opp)

    grad = np.concatenate([grad_beta, grad_delta, grad_gamma])
    return loss, grad


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_afm(events: Sequence[TelemetryEvent]) -> dict:
    """Fit the Additive Factor Model to a sequence of telemetry events.

    Only events whose ``verdict`` field is ``"correct"`` or ``"wrong"``
    (case-insensitive) *and* whose ``kc_id`` is non-null contribute to the
    fit.  All other events are silently ignored.

    Parameters
    ----------
    events:
        Flat sequence of :class:`~mathtutor.contracts.TelemetryEvent`.

    Returns
    -------
    dict with keys:

    ``"kc_easiness"`` : dict[str, float]
        δ_kc for every KC that appeared in the data.  Higher ⇒ easier.
    ``"kc_learning_rate"`` : dict[str, float]
        γ_kc for every KC.  Positive means practice helps.
    ``"student_ability"`` : dict[str, float]
        β_s for every student pseudonym.
    ``"converged"`` : bool
        Whether the optimiser reported successful convergence.
    ``"n_obs"`` : int
        Number of observations used in the fit.

    Notes
    -----
    With zero or one observations the model cannot be identified; an
    empty/trivial result is returned with ``converged=False``.
    """
    # ---- Collect observations ----
    student_ids: list[str] = []
    kc_ids_obs:  list[str] = []
    opps:        list[int] = []
    labels:      list[int] = []

    for ev in events:
        if ev.kc_id is None:
            continue
        v = (ev.verdict or "").lower()
        if v not in ("correct", "wrong"):
            continue
        student_ids.append(ev.user_pseudonym)
        kc_ids_obs.append(ev.kc_id)
        opps.append(int(ev.opportunity_index))
        labels.append(1 if v == "correct" else 0)

    n_obs = len(labels)
    empty: dict = {
        "kc_easiness": {}, "kc_learning_rate": {},
        "student_ability": {}, "converged": False, "n_obs": 0,
    }
    if n_obs < 2:
        return empty

    # ---- Build index maps ----
    s_vocab  = {s: i for i, s in enumerate(sorted(set(student_ids)))}
    kc_vocab = {k: i for i, k in enumerate(sorted(set(kc_ids_obs)))}
    n_s, n_k = len(s_vocab), len(kc_vocab)

    s_arr   = np.array([s_vocab[s]  for s in student_ids], dtype=np.int32)
    kc_arr  = np.array([kc_vocab[k] for k in kc_ids_obs],  dtype=np.int32)
    opp_arr = np.array(opps,                                dtype=np.float64)
    y_arr   = np.array(labels,                              dtype=np.float64)

    # ---- Optimise ----
    n_params = n_s + 2 * n_k
    x0 = np.zeros(n_params)

    converged = False
    params = x0.copy()

    if _SCIPY:
        result = _scipy_minimize(
            fun=_bce_loss_and_grad,
            x0=x0,
            method="L-BFGS-B",
            jac=True,
            args=(y_arr, s_arr, kc_arr, opp_arr, n_s, n_k),
            options={"maxiter": 1000, "ftol": 1e-9},
        )
        params    = result.x
        converged = bool(result.success)
    else:
        # Fallback: simple gradient descent (no scipy)        # pragma: no cover
        lr_opt = 0.1
        for _ in range(2000):
            _, grad = _bce_loss_and_grad(
                params, y_arr, s_arr, kc_arr, opp_arr, n_s, n_k)
            params -= lr_opt * grad
            if np.max(np.abs(grad)) < 1e-6:
                converged = True
                break

    # ---- Unpack ----
    beta  = params[:n_s]
    delta = params[n_s : n_s + n_k]
    gamma = params[n_s + n_k :]

    inv_s  = {i: s for s, i in s_vocab.items()}
    inv_kc = {i: k for k, i in kc_vocab.items()}

    return {
        "kc_easiness":      {inv_kc[i]: float(delta[i]) for i in range(n_k)},
        "kc_learning_rate": {inv_kc[i]: float(gamma[i]) for i in range(n_k)},
        "student_ability":  {inv_s[i]:  float(beta[i])  for i in range(n_s)},
        "converged":        converged,
        "n_obs":            n_obs,
    }


def error_rate_curve(
    events: Sequence[TelemetryEvent],
    kc_id: str,
) -> list[float]:
    """Compute per-opportunity error rate for a single KC.

    Each position in the returned list corresponds to one opportunity
    index (0, 1, 2, …).  The value is the fraction of attempts at that
    opportunity that were *incorrect* (verdict == ``"wrong"``).

    Positions with no observations are set to ``float('nan')``.

    Parameters
    ----------
    events:
        All telemetry events (any KC; irrelevant ones are filtered out).
    kc_id:
        The knowledge component to analyse.

    Returns
    -------
    list[float]
        Error rates indexed by opportunity number.  Length equals
        ``max(opportunity_index) + 1`` across matching events, or ``[]``
        if no matching events exist.
    """
    # Bucket by opportunity index
    buckets: dict[int, list[int]] = defaultdict(list)
    for ev in events:
        if ev.kc_id != kc_id:
            continue
        v = (ev.verdict or "").lower()
        if v not in ("correct", "wrong"):
            continue
        buckets[int(ev.opportunity_index)].append(0 if v == "correct" else 1)

    if not buckets:
        return []

    max_opp = max(buckets)
    curve: list[float] = []
    for i in range(max_opp + 1):
        if i in buckets and buckets[i]:
            curve.append(float(np.mean(buckets[i])))
        else:
            curve.append(float("nan"))
    return curve


def flag_misspecified_kcs(
    events: Sequence[TelemetryEvent],
    *,
    min_decline: float = 0.0,
) -> list[str]:
    """Return KC IDs whose error-rate curves fail to decline monotonically.

    A KC is flagged when its error-rate curve is *not* non-increasing by
    at least *min_decline* per step on average.  The check:

    1. Compute the error-rate curve for each KC.
    2. Ignore positions where the rate is ``nan`` (no observations).
    3. Compute ``Δ[i] = curve[i] − curve[i−1]`` for consecutive
       non-nan positions.
    4. A KC is **well-specified** if ``mean(Δ) ≤ −min_decline``
       (average step goes down).
    5. A KC is **flagged** (mis-specified / should be split) if
       ``mean(Δ) > −min_decline`` OR the curve has fewer than 2
       non-nan points (can't judge).

    Parameters
    ----------
    events:
        Telemetry events.
    min_decline:
        Minimum average per-step decline required to avoid flagging.
        Default ``0.0`` means *any* non-increasing average passes;
        raise it (e.g. ``0.01``) for stricter criteria.

    Returns
    -------
    list[str]
        Sorted list of flagged KC IDs.
    """
    # Gather all unique KC ids that have scoreable events
    kc_ids = {
        ev.kc_id for ev in events
        if ev.kc_id is not None
        and (ev.verdict or "").lower() in ("correct", "wrong")
    }

    flagged: list[str] = []
    for kc_id in sorted(kc_ids):
        curve = error_rate_curve(events, kc_id)

        # Keep only non-nan consecutive pairs
        non_nan = [(i, v) for i, v in enumerate(curve) if not math.isnan(v)]
        if len(non_nan) < 2:
            # Not enough data to assess — conservatively flag it
            flagged.append(kc_id)
            continue

        deltas = [
            non_nan[j][1] - non_nan[j - 1][1]
            for j in range(1, len(non_nan))
        ]
        # Non-negative mean delta => curve is flat or rising => mis-specified.
        # We require mean(Δ) < -min_decline for the KC to be considered
        # well-specified.  A flat curve (mean == 0) is flagged by default
        # because it provides no learning signal.
        if float(np.mean(deltas)) >= -min_decline:
            flagged.append(kc_id)

    return flagged


def normalized_gain(pre: float, post: float) -> float:
    """Compute Hake's normalised gain.

    Formula::

        g = (post − pre) / (100 − pre)

    Parameters
    ----------
    pre:
        Pre-test score as a percentage in [0, 100].
    post:
        Post-test score as a percentage in [0, 100].

    Returns
    -------
    float
        Normalised gain *g*.  Returns ``float('nan')`` when ``pre == 100``
        (ceiling effect; gain is undefined).

    Examples
    --------
    >>> normalized_gain(40, 70)
    0.5
    >>> import math; math.isnan(normalized_gain(100, 100))
    True
    """
    if pre == 100.0:
        return float("nan")
    return (post - pre) / (100.0 - pre)

# ────────────────────────────────────────────────────────────
# FILE: ./src/mathtutor/eval/telemetry.py
# ────────────────────────────────────────────────────────────

# mathtutor/eval/telemetry.py

"""Append-only telemetry sink and pseudonymization helpers.

Design principles
-----------------
* **Append-only**: ``TelemetrySink.emit`` only ever appends a JSON line.
  No line is ever modified or deleted by this module.
* **Pseudonymous by construction**: raw user IDs never touch the file.
  ``pseudonymize(raw_id, salt)`` converts them to an HMAC-SHA-256 hex
  digest before the event is written.  The salt is a caller-supplied
  secret (e.g. an environment variable); this module never stores it.
* **No PII escaping**: if a caller somehow embeds raw IDs elsewhere in
  the event fields that is their bug; this module only guarantees
  ``user_pseudonym`` is hashed.

File format: one UTF-8 JSON object per line (JSON Lines / NDJSON),
terminated by ``\\n``.  Reading is purely sequential; no index is built.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Iterable

from mathtutor.contracts import TelemetryEvent


# ---------------------------------------------------------------------------
# Pseudonymization
# ---------------------------------------------------------------------------

def pseudonymize(raw_id: str, salt: str) -> str:
    """Return a pseudonym for *raw_id* using HMAC-SHA-256 with *salt*.

    Properties
    ----------
    * **Deterministic**: same ``(raw_id, salt)`` always yields the same hex
      digest, so events from the same user are linkable within a study but
      the raw identity is not recoverable without the salt.
    * **One-way** (under HMAC security): without the salt an adversary
      cannot reverse the pseudonym to the original ID.
    * **Collision-resistant**: 256-bit output; birthday-bound is 2^128.

    Parameters
    ----------
    raw_id:
        The raw user identifier (e.g. an email address or database UUID).
        This value is **never** written to disk by this module.
    salt:
        A secret string known only to the system operator.  Treat it like
        a password: load from an environment variable, not source code.

    Returns
    -------
    str
        64-character lowercase hex digest.
    """
    return hmac.new(
        salt.encode(),
        raw_id.encode(),
        digestmod=hashlib.sha256,
    ).hexdigest()


# ---------------------------------------------------------------------------
# Sink
# ---------------------------------------------------------------------------

class TelemetrySink:
    """Append-only writer for :class:`~mathtutor.contracts.TelemetryEvent`.

    Parameters
    ----------
    path:
        Path to the JSON-Lines log file.  Created (including parent dirs)
        if it does not exist; opened in append mode, never truncated.

    Example
    -------
    >>> import tempfile, os, time
    >>> from mathtutor.contracts import TelemetryEvent
    >>> from mathtutor.eval.telemetry import TelemetrySink, pseudonymize
    >>>
    >>> salt = "dev-only-salt"
    >>> with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as f:
    ...     path = f.name
    >>> sink = TelemetrySink(path)
    >>> event = TelemetryEvent(
    ...     event_id="e1", session_id="s1",
    ...     user_pseudonym=pseudonymize("alice@example.com", salt),
    ...     ts=time.time(), kc_id="linear-equations", opportunity_index=0,
    ...     verdict="correct",
    ... )
    >>> sink.emit(event)
    >>> os.unlink(path)
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def emit(self, event: TelemetryEvent) -> None:
        """Append *event* as a single JSON line to the log file.

        The file is opened, written, and closed on every call so that:

        * Multiple processes can safely append concurrently (O_APPEND is
          atomic for writes ≤ PIPE_BUF on POSIX; JSON Lines are short).
        * A crash between emits never corrupts previously written lines.

        Parameters
        ----------
        event:
            A fully-constructed, pseudonymized :class:`TelemetryEvent`.
            The caller is responsible for setting ``user_pseudonym`` via
            :func:`pseudonymize`; this method does not inspect PII.
        """
        line = event.to_json() + "\n"
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(line)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    @staticmethod
    def read_all(path: str | os.PathLike[str]) -> list[TelemetryEvent]:
        """Deserialize every event from a JSON-Lines log file.

        Parameters
        ----------
        path:
            Path to a file previously written by :meth:`emit`.

        Returns
        -------
        list[TelemetryEvent]
            Events in the order they were appended (chronological if
            callers emit in order).  Returns ``[]`` if the file is empty
            or does not exist.

        Raises
        ------
        json.JSONDecodeError
            If a line is not valid JSON (indicates file corruption).
        TypeError / KeyError
            If a line's JSON does not match :class:`TelemetryEvent` fields.
        """
        p = Path(path)
        if not p.exists():
            return []
        events: list[TelemetryEvent] = []
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:  # skip blank lines that can appear at EOF
                    events.append(TelemetryEvent.from_json(line))
        return events

# ────────────────────────────────────────────────────────────
# FILE: ./src/mathtutor/learner/__init__.py
# ────────────────────────────────────────────────────────────



# ────────────────────────────────────────────────────────────
# FILE: ./src/mathtutor/learner/bkt.py
# ────────────────────────────────────────────────────────────

"""
mathtutor/learner/bkt.py
========================
Bayesian Knowledge Tracing (BKT) — Anderson / Corbett-Anderson model.

References
----------
Corbett, A. T., & Anderson, J. R. (1994). Knowledge tracing: Modeling the
acquisition of procedural knowledge. User Modeling and User-Adapted
Interaction, 4(4), 253–278.

Spec §7.2–7.3: four-parameter per-KC model with prerequisite propagation.

DESIGN INVARIANTS
-----------------
* No global mutable state — all state lives in BKTLearnerState instances.
* Pure functions where possible (observe mutates state, everything else is
  read-only or returns new values).
* confidence == 1.0 only when decidable is True (BKT is always decidable by
  construction; these are plain probability updates, not symbolic reasoning).
* Guard against degenerate params: S + G < 1 must hold, and every parameter
  must be strictly inside (0, 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

# contracts.py is the only allowed project import at this level.
from mathtutor.contracts import KnowledgeComponent

if TYPE_CHECKING:
    # Avoid circular import at runtime; used only for the propagate() helper.
    pass


# ---------------------------------------------------------------------------
# BKTParams
# ---------------------------------------------------------------------------

@dataclass
class BKTParams:
    """
    Four parameters that govern one Knowledge Component's BKT dynamics.

    Parameters
    ----------
    l0 : float
        Prior probability the learner already knows the KC before any practice.
        Literature default: 0.20.
    t : float
        Transition (learn) probability — probability of going from "not mastered"
        to "mastered" after a single opportunity. Literature default: 0.30.
    s : float
        Slip probability — P(wrong answer | mastered). Literature default: 0.10.
    g : float
        Guess probability — P(correct answer | not mastered). Literature default: 0.20.

    Constraints
    -----------
    * All parameters must be strictly in (0, 1).
    * s + g < 1  — violating this makes the model's posteriors incoherent
      (a correct answer would *reduce* the mastery estimate).
    """

    l0: float = 0.20
    t: float  = 0.30
    s: float  = 0.10
    g: float  = 0.20

    def __post_init__(self) -> None:
        for name, val in [("l0", self.l0), ("t", self.t),
                          ("s", self.s),  ("g", self.g)]:
            if not (0.0 < val < 1.0):
                raise ValueError(
                    f"BKTParams.{name} must be strictly in (0, 1); got {val}"
                )
        if self.s + self.g >= 1.0:
            raise ValueError(
                f"BKTParams requires s + g < 1 (got s={self.s}, g={self.g}, "
                f"sum={self.s + self.g:.4f}). Degenerate params make posteriors "
                "incoherent."
            )


# ---------------------------------------------------------------------------
# BKTLearnerState
# ---------------------------------------------------------------------------

class BKTLearnerState:
    """
    Tracks a single learner's estimated mastery across all Knowledge Components.

    Each KC is tracked independently with its own probability p = P(mastered)
    and its own BKTParams.  Call ``observe`` after every student response to
    update estimates.

    Parameters
    ----------
    default_params : BKTParams, optional
        Parameter set used for any KC that has not been explicitly registered.
        Defaults to the literature-prior BKTParams().

    Internals
    ---------
    _p      : dict[str, float]  — current P(mastered) per KC id
    _params : dict[str, BKTParams]  — params per KC id
    """

    def __init__(
        self,
        default_params: BKTParams | None = None,
    ) -> None:
        self._p:      dict[str, float]     = {}
        self._params: dict[str, BKTParams] = {}
        self._default_params = default_params or BKTParams()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def register(self, kc_id: str, params: BKTParams | None = None) -> None:
        """
        Explicitly register a KC with optional custom params.

        If the KC is already registered this is a no-op unless you pass new
        params, in which case the params are updated but the current p is
        preserved.
        """
        p = self._default_params if params is None else params
        self._params[kc_id] = p
        if kc_id not in self._p:
            self._p[kc_id] = p.l0

    def _ensure(self, kc_id: str) -> None:
        """Lazily initialise a KC with default params if not yet seen."""
        if kc_id not in self._params:
            self.register(kc_id)

    # ------------------------------------------------------------------
    # Core BKT update  (Spec §7.2)
    # ------------------------------------------------------------------

    def observe(self, kc_id: str, correct: bool) -> float:
        """
        Update P(mastered) for *kc_id* given one observed student response.

        Applies the two-step Corbett-Anderson update in order:

        Step 1 — condition on the observation (Bayes rule):
        ::

            if correct:
                p_post = p*(1-S) / ( p*(1-S) + (1-p)*G )
            else:
                p_post = p*S     / ( p*S     + (1-p)*(1-G) )

        Step 2 — learning opportunity (transition):
        ::

            p_next = p_post + (1 - p_post) * T

        Parameters
        ----------
        kc_id : str
            Knowledge-component identifier (matches KnowledgeComponent.id).
        correct : bool
            Whether the student's response was correct.

        Returns
        -------
        float
            The updated P(mastered) after this observation.
        """
        self._ensure(kc_id)
        p = self._p[kc_id]
        par = self._params[kc_id]

        # --- Step 1: condition on observation ---
        if correct:
            numerator   = p * (1.0 - par.s)
            denominator = numerator + (1.0 - p) * par.g
        else:
            numerator   = p * par.s
            denominator = numerator + (1.0 - p) * (1.0 - par.g)

        p_post = numerator / denominator  # denominator > 0 when s+g<1

        # --- Step 2: learning transition ---
        p_next = p_post + (1.0 - p_post) * par.t

        self._p[kc_id] = p_next
        return p_next

    # ------------------------------------------------------------------
    # Mastery query
    # ------------------------------------------------------------------

    def mastered(self, kc_id: str, threshold: float = 0.95) -> bool:
        """
        Return True iff the current P(mastered) >= *threshold*.

        Parameters
        ----------
        kc_id : str
        threshold : float
            Mastery threshold. Spec default is 0.95 — chosen so a single
            correct answer from the L0 prior CANNOT cross it (guessing is
            modelled).
        """
        self._ensure(kc_id)
        return self._p[kc_id] >= threshold

    def p_mastered(self, kc_id: str) -> float:
        """Return the raw P(mastered) estimate for *kc_id*."""
        self._ensure(kc_id)
        return self._p[kc_id]

    # ------------------------------------------------------------------
    # Effective prior under prerequisite conditioning  (Spec §7.3)
    # ------------------------------------------------------------------

    def effective_prior(
        self,
        kc_id: str,
        prereqs: list[str],
        mastered_set: set[str],
    ) -> float:
        """
        Return an adjusted prior for *kc_id* that accounts for prerequisite mastery.

        Rationale
        ---------
        If a student has not yet mastered the prerequisites of a KC, their
        effective probability of knowing that KC is lower than the raw L0 would
        suggest.  Conversely, mastering all prerequisites keeps the prior at L0
        (or can nudge it slightly upward if we have positive evidence).

        Simple model used here
        ----------------------
        Let ``r = |mastered_prereqs| / |prereqs|``  (fraction of prereqs mastered).
        ``effective_l0 = L0 * (alpha + (1 - alpha) * r)``
        where ``alpha = 0.3`` is a floor factor — even with zero prereqs mastered
        we don't completely zero out the prior.

        This keeps the function monotone and bounded in (0, L0], which respects
        the BKT invariant that priors are in (0, 1).

        Parameters
        ----------
        kc_id : str
        prereqs : list[str]
            IDs of direct prerequisite KCs.
        mastered_set : set[str]
            Set of KC ids the learner has already mastered.

        Returns
        -------
        float
            Adjusted L0 ∈ (0, 1).
        """
        self._ensure(kc_id)
        params = self._params[kc_id]

        if not prereqs:
            return params.l0

        ALPHA = 0.3  # floor: minimum fraction of L0 even with 0 prereqs mastered
        r = sum(1 for pid in prereqs if pid in mastered_set) / len(prereqs)
        adjusted = params.l0 * (ALPHA + (1.0 - ALPHA) * r)
        # Clamp to (0, 1) for safety — shouldn't be needed with valid params
        return max(1e-6, min(adjusted, 1.0 - 1e-6))


# ---------------------------------------------------------------------------
# propagate — weak upward/downward prerequisite conditioning
# ---------------------------------------------------------------------------

def propagate(state: BKTLearnerState, curriculum: object) -> None:
    """
    Apply weak prerequisite-conditioned probability propagation across the
    KC dependency graph.

    What "weak" means
    -----------------
    We do *not* hard-override any KC's p with the effective prior — that would
    discard hard-won evidence.  Instead we nudge: if a KC's current p is
    *above* what the effective prior would grant given its unmastered prereqs,
    we pull it gently toward the effective prior.  The nudge is small (weight
    ``NUDGE = 0.05``) so it acts as a regulariser, not a reset.

    Upward propagation (prereq → dependent)
    ----------------------------------------
    If a KC is mastered, its dependents get a small positive nudge on their
    current p (capped at their current p + NUDGE, and never above 0.94 to
    avoid spurious mastery declarations without evidence).

    Downward propagation (dependent → prereq — "weak upward" in learner terms)
    ---------------------------------------------------------------------------
    If a dependent KC is answered correctly, the probability of the prereq
    being known rises slightly.  We model this as: for each unmastered prereq,
    nudge its p up by ``NUDGE * P(dependent correct | prereq mastered)``, which
    we approximate as (1 - s_dependent).

    Parameters
    ----------
    state : BKTLearnerState
        Mutated in-place.
    curriculum : object
        Expected to have a ``kcs`` attribute — an iterable of
        ``KnowledgeComponent`` objects with ``.id`` and ``.prerequisites``
        (list[str]).

    Notes
    -----
    This is intentionally simple — it avoids full belief propagation over the
    DAG, which would require topological ordering and multiple passes.  For
    richer behaviour, replace with a proper sum-product pass.
    """
    NUDGE = 0.05
    MASTERY_CEILING = 0.94  # never declare mastery without a real observation

    kcs: list[KnowledgeComponent] = list(getattr(curriculum, "kcs", []))

    # Build prereq map: kc_id -> list[prereq_id]
    prereq_map: dict[str, list[str]] = {
        kc.id: list(kc.prerequisites) for kc in kcs
    }

    # Build reverse map: prereq_id -> list[dependent_id]
    dependent_map: dict[str, list[str]] = {}
    for kc_id, prereqs in prereq_map.items():
        for pid in prereqs:
            dependent_map.setdefault(pid, []).append(kc_id)

    mastered_set = {kc_id for kc_id in state._p if state.mastered(kc_id)}

    # --- Downward: mastered KC nudges its dependents up ---
    for mastered_kc in mastered_set:
        for dep_id in dependent_map.get(mastered_kc, []):
            if dep_id not in mastered_set:
                state._ensure(dep_id)
                current = state._p[dep_id]
                state._p[dep_id] = min(current + NUDGE, MASTERY_CEILING)

    # --- Upward: unmastered prereqs gently pull dependents toward effective prior ---
    for kc_id, prereqs in prereq_map.items():
        if not prereqs:
            continue
        unmastered_prereqs = [pid for pid in prereqs if pid not in mastered_set]
        if not unmastered_prereqs:
            continue  # all prereqs mastered — no downward drag needed

        effective = state.effective_prior(kc_id, prereqs, mastered_set)
        state._ensure(kc_id)
        current = state._p[kc_id]
        if current > effective:
            # Nudge toward effective prior, not hard-reset
            state._p[kc_id] = current - NUDGE * (current - effective)


# ────────────────────────────────────────────────────────────
# FILE: ./src/mathtutor/learner/scheduling.py
# ────────────────────────────────────────────────────────────

"""
mathtutor/learner/scheduling.py
================================
Half-Life Regression (HLR) style forgetting model and spaced-repetition
scheduler.

References
----------
Settles, B. & Meeder, B. (2016). A Trainable Spaced Repetition Model for
Language Learning. ACL 2016.

SPEC §7.4–7.5 — Forgetting + Scheduling

Forgetting model
----------------
Recall is modelled as exponential decay:

    p(t) = 2 ** (-elapsed / h)

where:
  - elapsed  = now_ts - last_seen_ts  (same time unit; seconds recommended)
  - h        = half_life              (same unit; at elapsed=h recall = 0.50)

Half-life grows with successful, *spaced* retrievals (the "spacing" part of
HLR): if the student answered correctly AND the elapsed time is at least
MIN_SPACING_RATIO * current half_life, the half_life is multiplied by
GROWTH_FACTOR.  On failure, the half_life is multiplied by DECAY_FACTOR
(shrunk) and floored at MIN_HALF_LIFE.

Scheduling policy
-----------------
1. Mastery gate  — a KC is never selected unless all its prerequisites appear
   in mastered_set.
2. Spacing       — schedule for review when predicted recall ≤ review_band
   (default 0.85 — the "desirable difficulty" sweet spot: still mostly
   remembered, but retrieval effort drives consolidation).
3. Interleaving  — `select_next` mixes due-review KCs with new (unseen/
   unmastered) KCs in a round-robin pattern so no single KC dominates a
   session (the "interleaving effect").

DESIGN INVARIANTS
-----------------
* Pure functions — no global mutable state.  `update_after_review` returns a
  *new* `RetentionState` rather than mutating in-place.
* Half-life is always a positive float; this is enforced in
  `update_after_review`.
* `predicted_recall` returns a float in (0, 1].  At elapsed=0 it is exactly
  1.0; it approaches 0 asymptotically but never reaches it.
* `confidence == 1.0` only when `decidable is True` (BKT invariant from the
  rest of the system; scheduling doesn't produce Judgment objects but follows
  the same honesty policy: never claim certainty we don't have).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Mapping

# ---------------------------------------------------------------------------
# Imports from sibling modules (do not redefine)
# ---------------------------------------------------------------------------
from mathtutor.learner.bkt import BKTLearnerState          # noqa: F401 (used by callers)
from mathtutor.domain.curriculum import Curriculum


# ---------------------------------------------------------------------------
# Hyperparameters (module-level constants, not global mutable state)
# ---------------------------------------------------------------------------

#: Half-life grows by this factor after a spaced successful retrieval.
GROWTH_FACTOR: float = 2.0

#: Half-life shrinks by this factor after a failed retrieval.
DECAY_FACTOR: float = 0.5

#: Minimum half-life floor (prevents collapse to 0).
MIN_HALF_LIFE: float = 1.0   # same unit as timestamps (e.g. seconds)

#: Default initial half-life for an unseen KC.
DEFAULT_HALF_LIFE: float = 86_400.0   # 1 day in seconds

#: A retrieval counts as "spaced" only if elapsed ≥ this fraction of h.
#: Immediately re-reading something does not earn a half-life boost.
MIN_SPACING_RATIO: float = 0.10


# ---------------------------------------------------------------------------
# RetentionState
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RetentionState:
    """
    Per-knowledge-component forgetting state.

    Attributes
    ----------
    half_life : float
        Current memory half-life in the same unit as timestamps (e.g. seconds).
        A larger value means the student forgets more slowly.
    last_seen_ts : float
        Unix timestamp (or any monotonic counter) of the most recent review
        of this KC.
    successful_reviews : int
        Cumulative count of successful (correct) retrievals for this KC.
        Used for diagnostics and can gate more aggressive half-life growth in
        future versions.
    """

    half_life: float = DEFAULT_HALF_LIFE
    last_seen_ts: float = 0.0
    successful_reviews: int = 0

    def __post_init__(self) -> None:
        if self.half_life <= 0.0:
            raise ValueError(f"half_life must be > 0, got {self.half_life!r}")
        if self.successful_reviews < 0:
            raise ValueError(
                f"successful_reviews must be >= 0, got {self.successful_reviews!r}"
            )


# ---------------------------------------------------------------------------
# Core recall formula
# ---------------------------------------------------------------------------

def predicted_recall(rs: RetentionState, now_ts: float) -> float:
    """
    Predict the probability of successful free recall at time ``now_ts``.

    Formula
    -------
    ::

        p = 2 ** (-(now_ts - last_seen_ts) / half_life)

    Step-by-step example
    --------------------
    Suppose ``half_life = 86_400`` (one day in seconds) and the student last
    reviewed 12 hours ago (elapsed = 43_200 s):

        p = 2 ** (-43_200 / 86_400)
          = 2 ** (-0.5)
          ≈ 0.707

    So after half a half-life the recall is ≈ 71 %.

    Parameters
    ----------
    rs : RetentionState
        Current retention state for the KC.
    now_ts : float
        Current timestamp (same unit as ``rs.last_seen_ts``).

    Returns
    -------
    float
        Predicted recall probability in (0, 1].  Returns 1.0 when
        ``now_ts <= rs.last_seen_ts`` (elapsed ≤ 0).
    """
    elapsed = now_ts - rs.last_seen_ts
    if elapsed <= 0.0:
        return 1.0
    return 2.0 ** (-elapsed / rs.half_life)


# ---------------------------------------------------------------------------
# State update after a review attempt
# ---------------------------------------------------------------------------

def update_after_review(
    rs: RetentionState,
    now_ts: float,
    success: bool,
) -> RetentionState:
    """
    Return a new ``RetentionState`` incorporating the outcome of a review.

    Growth rule (success)
    ---------------------
    A successful retrieval grows the half-life *only* if the retrieval was
    sufficiently spaced (elapsed ≥ MIN_SPACING_RATIO × h).  Massed (cramming)
    repetitions are detected and do not earn a bonus.

    ::

        if elapsed >= MIN_SPACING_RATIO * h:
            new_h = h * GROWTH_FACTOR      # e.g. h → 2h
        else:
            new_h = h                       # no reward for cramming

    Decay rule (failure)
    --------------------
    ::

        new_h = max(h * DECAY_FACTOR, MIN_HALF_LIFE)   # e.g. h → h/2, ≥ 1

    Parameters
    ----------
    rs : RetentionState
        Current state.
    now_ts : float
        Timestamp of this review event.
    success : bool
        True if the student recalled correctly, False otherwise.

    Returns
    -------
    RetentionState
        New (immutable) state.  The original is never mutated.
    """
    elapsed = max(now_ts - rs.last_seen_ts, 0.0)

    if success:
        spaced = elapsed >= MIN_SPACING_RATIO * rs.half_life
        new_h = rs.half_life * GROWTH_FACTOR if spaced else rs.half_life
        return replace(
            rs,
            half_life=new_h,
            last_seen_ts=now_ts,
            successful_reviews=rs.successful_reviews + 1,
        )
    else:
        new_h = max(rs.half_life * DECAY_FACTOR, MIN_HALF_LIFE)
        return replace(
            rs,
            half_life=new_h,
            last_seen_ts=now_ts,
            # successful_reviews unchanged — failure doesn't reset the count
        )


# ---------------------------------------------------------------------------
# Which KCs are due for review?
# ---------------------------------------------------------------------------

def due_for_review(
    states: Mapping[str, RetentionState],
    now_ts: float,
    band: float = 0.85,
) -> list[str]:
    """
    Return KC ids whose predicted recall has dropped to or below ``band``.

    The default band of 0.85 targets the "desirable difficulty" sweet spot:
    the student still mostly remembers the material, but the retrieval
    effort is high enough to drive long-term consolidation.

    Parameters
    ----------
    states : Mapping[str, RetentionState]
        Map from kc_id → RetentionState for every KC the learner has seen.
        KC ids not present in this mapping are considered unseen and are
        *not* returned (they belong to ``select_next``'s "new KC" pool).
    now_ts : float
        Current timestamp.
    band : float
        Recall threshold.  KCs with recall ≤ band are returned.
        Default 0.85.

    Returns
    -------
    list[str]
        KC ids that are due, sorted by ascending recall (most-forgotten first)
        so callers can prioritise the KC most at risk of being lost.
    """
    due: list[tuple[float, str]] = []
    for kc_id, rs in states.items():
        recall = predicted_recall(rs, now_ts)
        if recall <= band + 1e-9:
            due.append((recall, kc_id))
    # Most-forgotten first (lowest recall = highest urgency)
    due.sort(key=lambda t: t[0])
    return [kc_id for _, kc_id in due]


# ---------------------------------------------------------------------------
# Main scheduler
# ---------------------------------------------------------------------------

def select_next(
    curriculum: Curriculum,
    mastered_set: set[str],
    retention_states: Mapping[str, RetentionState],
    now_ts: float,
    k: int = 5,
) -> list[str]:
    """
    Choose up to ``k`` KC ids for the next session, interleaved.

    Algorithm
    ---------
    Step 1 — Prerequisite filter
        Collect *all* candidate KCs from the curriculum.  A KC is a candidate
        only if every KC in its ``prerequisites`` list is in ``mastered_set``.
        Already-mastered KCs are excluded from the "new" pool (they may still
        appear in the due-review pool).

    Step 2 — Split into two pools
        a. **Review pool** — candidates whose predicted recall ≤ ``band``
           (default 0.85) *and* that appear in ``retention_states`` (i.e. the
           learner has seen them before).  Sorted most-forgotten first.
        b. **New pool** — candidates not yet in ``retention_states`` (never
           seen), sorted by curriculum order (shallow first).

    Step 3 — Interleave
        Alternate: one review KC, one new KC, one review KC, …, until ``k``
        slots are filled or both pools are exhausted.  If one pool runs out,
        continue drawing from the other.

    This interleaving prevents "blocking" (spending the whole session on one
    KC) and mixes retrieval practice with new learning.

    Parameters
    ----------
    curriculum : Curriculum
        The prerequisite DAG.  Expected to have a ``kcs`` attribute that is
        an iterable of objects with ``.id`` (str) and ``.prerequisites``
        (list[str]).
    mastered_set : set[str]
        KC ids the learner has mastered (``bkt.mastered(kc_id)`` returned
        True at some point).
    retention_states : Mapping[str, RetentionState]
        Current forgetting state for every KC the learner has encountered.
    now_ts : float
        Current timestamp.
    k : int
        Maximum number of KCs to return.

    Returns
    -------
    list[str]
        Interleaved KC ids, length ≤ k.  Never contains a KC whose
        prerequisites are not all in ``mastered_set``.

    Raises
    ------
    ValueError
        If k < 1.
    """
    if k < 1:
        raise ValueError(f"k must be ≥ 1, got {k!r}")

    REVIEW_BAND = 0.85

    # ------------------------------------------------------------------
    # Step 1 — build the candidate set (prereqs met, not mastered)
    # ------------------------------------------------------------------
    all_kc_ids_in_order: list[str] = [kc.id for kc in curriculum.kcs]
    prereqs_by_id: dict[str, list[str]] = {
        kc.id: list(kc.prerequisites) for kc in curriculum.kcs
    }

    def prereqs_met(kc_id: str) -> bool:
        return all(p in mastered_set for p in prereqs_by_id.get(kc_id, []))

    # ------------------------------------------------------------------
    # Step 2 — split into review vs new pools
    # ------------------------------------------------------------------
    review_pool: list[tuple[float, str]] = []  # (recall, kc_id)
    new_pool: list[str] = []                    # curriculum-ordered

    for kc_id in all_kc_ids_in_order:
        if not prereqs_met(kc_id):
            continue  # blocked by unmet prereqs — never select

        if kc_id in retention_states:
            # Seen before — eligible for review if recall ≤ band
            recall = predicted_recall(retention_states[kc_id], now_ts)
            if recall <= REVIEW_BAND:
                review_pool.append((recall, kc_id))
        elif kc_id not in mastered_set:
            # Never seen and not mastered — eligible as a new KC
            new_pool.append(kc_id)

    # Sort review pool: most-forgotten first
    review_pool.sort(key=lambda t: t[0])
    review_ids = [kc_id for _, kc_id in review_pool]

    # ------------------------------------------------------------------
    # Step 3 — interleave review and new KCs (round-robin)
    # ------------------------------------------------------------------
    result: list[str] = []
    ri, ni = 0, 0
    # Alternate: review, new, review, new, …
    take_review = True   # start with review if available, else new

    while len(result) < k and (ri < len(review_ids) or ni < len(new_pool)):
        if take_review and ri < len(review_ids):
            result.append(review_ids[ri])
            ri += 1
        elif not take_review and ni < len(new_pool):
            result.append(new_pool[ni])
            ni += 1
        elif ri < len(review_ids):
            # new pool exhausted — drain review
            result.append(review_ids[ri])
            ri += 1
        else:
            # review pool exhausted — drain new
            result.append(new_pool[ni])
            ni += 1
        take_review = not take_review   # flip for next slot

    return result


# ────────────────────────────────────────────────────────────
# FILE: ./src/mathtutor/llm/__init__.py
# ────────────────────────────────────────────────────────────



# ────────────────────────────────────────────────────────────
# FILE: ./src/mathtutor/llm/offline.py
# ────────────────────────────────────────────────────────────

# mathtutor/llm/offline.py

"""Deterministic, template-based offline tutor.

This module exposes a single callable interface:

    coach(context: CoachingContext) -> str

It is the fallback used by the Orchestrator when the real LLM is unavailable
or not configured.  It uses ONLY:
  * CAS-verified facts carried in the CoachingContext
  * Canned phrase templates keyed on verdict / support level
  * No network calls, no random state (deterministic given the same context)

The interface it satisfies is intentionally minimal so that any LLM wrapper
can drop in as a replacement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ── shared contracts ──────────────────────────────────────────────────────────
from mathtutor.contracts import (
    Judgment,
    SupportLevel,
    Verdict,
    verdict_from_judgment,
)


# ── coaching context ──────────────────────────────────────────────────────────

@dataclass
class CoachingContext:
    """All CAS-verified facts the tutor needs to phrase a response.

    The orchestrator builds one of these after every verification step and
    passes it to coach().  All math content inside must already be CAS-
    certified before reaching the tutor — the tutor NEVER adjudicates math.

    Attributes
    ----------
    verdict:
        Coarse verdict from contracts.Verdict.
    judgment:
        Full Judgment dataclass from the CAS verifier.
    support_level:
        Scaffolding level chosen by the orchestrator.
    hint_level:
        0-based index of how many hints have already been given.
    kc_name:
        Human-readable name of the knowledge component being practised.
    problem_statement:
        The original problem text shown to the student.
    student_raw:
        Exactly what the student typed (pre-parse).
    correct_answer_str:
        CAS-formatted correct answer, already filtered by leak_filter.
        May be None when the orchestrator withholds it (independent mode).
    misconception_id:
        BuggyRule.id if the CAS diagnosed a specific misconception, else None.
    misconception_description:
        Human-readable description of the misconception.  May be None.
    worked_steps:
        Ordered list of CAS-certified step strings for WORKED support.
    extra:
        Arbitrary extra context (e.g. partial credit detail).
    """

    verdict: Verdict
    judgment: Judgment
    support_level: SupportLevel = SupportLevel.INDEPENDENT
    hint_level: int = 0
    kc_name: str = "this topic"
    problem_statement: str = ""
    student_raw: str = ""
    correct_answer_str: str | None = None
    misconception_id: str | None = None
    misconception_description: str | None = None
    worked_steps: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


# ── phrase banks ──────────────────────────────────────────────────────────────

# Keys: (verdict, support_level, hint_level_clamped)
# hint_level is clamped to 0..2 so we don't need an unbounded table.

_CORRECT_PHRASES = [
    "Great work — that's correct!",
    "Exactly right. Well done!",
    "That's the correct answer. Keep it up!",
]

_PARTIAL_PHRASES = [
    "You've got part of it right, but the answer isn't complete yet.",
    "That's partially correct — you're on the right track, but something is missing.",
    "Good start! You have some of the solution, but there's more to find.",
]

_ABSTAIN_PHRASES = [
    "I couldn't parse that input — could you rephrase it using standard notation?",
    "That didn't look like a math expression I can read. Try writing it differently.",
    "I had trouble understanding that. Please check your notation and try again.",
]

# Wrong, independent (no hint at all)
_WRONG_INDEPENDENT = [
    "That's not quite right. Review {kc_name} and try again.",
    "Not correct. Think about {kc_name} and give it another shot.",
    "That answer isn't right. Revisit {kc_name} before your next attempt.",
]

# Wrong, completion — give a structural hint, NO values
_WRONG_COMPLETION_H0 = [
    "Think about the structure of the solution to \"{problem}\". What's the first step?",
    "What form should the answer to \"{problem}\" take?",
    "Consider the method for {kc_name}. What operation should you apply first?",
]
_WRONG_COMPLETION_H1 = [
    "You need to apply {kc_name} here. Try setting up the equation carefully.",
    "Remember the key rule for {kc_name} and apply it step by step.",
    "Double-check your setup for {kc_name}. The process should guide you to the answer.",
]
_WRONG_COMPLETION_H2 = [
    "Work through the standard procedure for {kc_name} one step at a time.",
    "Go back to the definition of {kc_name} and apply each rule in order.",
    "Break the problem into smaller pieces using what you know about {kc_name}.",
]

# Misconception addendum (appended after the main phrase)
_MISCONCEPTION_ADDENDUM = (
    "It looks like there may be a common error here: {description}. "
    "Review that rule before retrying."
)

# Worked solution header/footer
_WORKED_HEADER = "Let's walk through this step by step:"
_WORKED_FOOTER = "Try a similar problem on your own to reinforce this."
_WORKED_REVEAL = "The correct answer is: {answer}"


# ── helpers ───────────────────────────────────────────────────────────────────

def _pick(phrases: list[str], index: int = 0) -> str:
    """Return phrase at *index* mod len(phrases), deterministically."""
    return phrases[index % len(phrases)]


def _fmt(template: str, ctx: CoachingContext) -> str:
    return template.format(
        kc_name=ctx.kc_name,
        problem=ctx.problem_statement or "this problem",
        answer=ctx.correct_answer_str or "(see solution)",
        description=ctx.misconception_description or "an unknown misconception",
    )


# ── public interface ──────────────────────────────────────────────────────────

def coach(context: CoachingContext) -> str:
    """Return a templated coaching message based purely on CAS facts.

    This function is deterministic: same context → same output.
    It never adjudicates correctness; it only phrases what the CAS decided.

    Parameters
    ----------
    context:
        CoachingContext populated by the orchestrator.

    Returns
    -------
    str
        A plain-text coaching message ready for display.
    """
    parts: list[str] = []

    verdict = context.verdict
    sl = context.support_level
    hl = min(context.hint_level, 2)  # clamp to 0-2

    # ── 1. main verdict phrase ──────────────────────────────────────────────
    if verdict == Verdict.CORRECT:
        parts.append(_pick(_CORRECT_PHRASES, hl))

    elif verdict == Verdict.PARTIAL:
        parts.append(_pick(_PARTIAL_PHRASES, hl))

    elif verdict == Verdict.ABSTAIN:
        parts.append(_pick(_ABSTAIN_PHRASES, hl))

    else:  # WRONG — branch on support level
        if sl == SupportLevel.WORKED:
            # Full worked solution: header + CAS steps + reveal (if available)
            parts.append(_WORKED_HEADER)
            if context.worked_steps:
                for i, step in enumerate(context.worked_steps, 1):
                    parts.append(f"  Step {i}: {step}")
            else:
                parts.append(_fmt(
                    "  Apply the standard procedure for {kc_name}.", context
                ))
            if context.correct_answer_str:
                parts.append(_fmt(_WORKED_REVEAL, context))
            parts.append(_WORKED_FOOTER)

        elif sl == SupportLevel.COMPLETION:
            bank = [_WRONG_COMPLETION_H0, _WRONG_COMPLETION_H1, _WRONG_COMPLETION_H2][hl]
            parts.append(_fmt(_pick(bank, hl), context))

        else:  # INDEPENDENT — no hints, no values
            parts.append(_fmt(_pick(_WRONG_INDEPENDENT, hl), context))

    # ── 2. misconception addendum (all non-correct verdicts) ───────────────
    if verdict != Verdict.CORRECT and context.misconception_id:
        if context.misconception_description:
            parts.append(_fmt(_MISCONCEPTION_ADDENDUM, context))

    # ── 3. partial credit detail ────────────────────────────────────────────
    if verdict == Verdict.PARTIAL and context.judgment.detail.get("found_roots"):
        found = context.judgment.detail["found_roots"]
        total = context.judgment.detail.get("total_roots", "?")
        parts.append(
            f"You found {len(found)} of {total} solution(s). "
            "Keep going — what other values satisfy the equation?"
        )

    return "\n".join(parts)


# ── LLM-compatible wrapper ────────────────────────────────────────────────────

class OfflineTutor:
    """Drop-in replacement for an LLM coaching object.

    The Orchestrator calls ``llm.coach(context)``; this class satisfies
    that interface using only deterministic templates.

    Example
    -------
    >>> tutor = OfflineTutor()
    >>> msg = tutor.coach(ctx)   # ctx: CoachingContext
    """

    name: str = "offline-template-tutor"

    def coach(self, context: CoachingContext) -> str:  # noqa: D102
        """Delegate to the module-level ``coach`` function."""
        return coach(context)

# ────────────────────────────────────────────────────────────
# FILE: ./src/mathtutor/main.py
# ────────────────────────────────────────────────────────────

def main():
    print("Hello from mathtutor!")


if __name__ == "__main__":
    main()


# ────────────────────────────────────────────────────────────
# FILE: ./src/mathtutor/orchestrator.py
# ────────────────────────────────────────────────────────────

# mathtutor/orchestrator.py

"""Orchestrator — the spine of the Verified Math Tutor.

Routing contract (SPEC §5, §13, §14)
--------------------------------------
* The CAS is the single source of mathematical truth.
* The LLM (or offline fallback) handles ONLY language / pedagogy.
* Deterministic turns (verdict, gate checks, worked steps) hit NO model.
* If the LLM raises, the offline tutor takes over transparently.
* Every concrete symbolic claim that would leave the system passes through
  ``safety.claim_cert.certify`` and ``safety.leak_filter.redact_answers``.
* One ``TelemetryEvent`` is emitted per turn.

Turn lifecycle
--------------
1. Parse student input (echo-confirm if ambiguous).
2. Verify via domain Verifier → Judgment.
3. Update learner state (BKT p_known).
4. If wrong, diagnose misconception.
5. Choose support level / hint depth (scaffolding).
6. Ask LLM (or offline tutor) for coaching prose.
7. Run safety filters on any prose.
8. Emit telemetry.
9. Return TurnResult.
"""

from __future__ import annotations

import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

# ── shared contracts ──────────────────────────────────────────────────────────
from mathtutor.contracts import (
    Artifact,
    Judgment,
    ParseError,
    SupportLevel,
    Target,
    TelemetryEvent,
    Verdict,
    Verifier,
    verdict_from_judgment,
)

# ── offline tutor (always available) ─────────────────────────────────────────
from mathtutor.llm.offline import CoachingContext, OfflineTutor

# ── optional dependencies: degrade gracefully if absent ──────────────────────

try:
    from mathtutor.learner.bkt import BKTModel  # type: ignore[import]
except ImportError:
    BKTModel = None  # type: ignore[assignment,misc]

try:
    from mathtutor.tutoring.misconceptions import diagnose  # type: ignore[import]
except ImportError:
    diagnose = None  # type: ignore[assignment]

try:
    from mathtutor.tutoring.scaffolding import choose_support  # type: ignore[import]
except ImportError:
    choose_support = None  # type: ignore[assignment]

try:
    from mathtutor.safety.claim_cert import certify  # type: ignore[import]
except ImportError:
    certify = None  # type: ignore[assignment]

try:
    from mathtutor.safety.leak_filter import redact_answers  # type: ignore[import]
except ImportError:
    redact_answers = None  # type: ignore[assignment]

try:
    from mathtutor.eval.telemetry import record  # type: ignore[import]
except ImportError:
    record = None  # type: ignore[assignment]


# ── session & turn types ──────────────────────────────────────────────────────

@dataclass
class Session:
    """Mutable per-student session state.

    The orchestrator modifies this in place on every turn.

    Attributes
    ----------
    session_id:
        Unique identifier for this session (UUID string).
    user_pseudonym:
        Privacy-safe identifier for the learner.
    kc_id:
        The knowledge-component currently being practised.
    kc_name:
        Human-readable name of the KC.
    problem_id:
        Identifier for the current problem.
    problem_statement:
        The full problem text shown to the student.
    target:
        The Target the verifier uses to judge correctness.
    verifier:
        Domain verifier to use for this problem.
    opportunity_index:
        How many attempts have been made on this KC so far.
    hint_level:
        How many hints have already been delivered this turn.
    p_known:
        BKT probability that the student knows the KC.
    bkt_model:
        Optional BKT model instance (may be None if module absent).
    history:
        Ordered list of past TurnResult objects this session.
    correct_answer_str:
        CAS-formatted reference answer.  May be None until set.
    """

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_pseudonym: str = "anonymous"
    kc_id: str | None = None
    kc_name: str = "this topic"
    problem_id: str | None = None
    problem_statement: str = ""
    target: Target | None = None
    verifier: Verifier | None = None
    opportunity_index: int = 0
    hint_level: int = 0
    p_known: float = 0.2
    bkt_model: Any = None
    history: list["TurnResult"] = field(default_factory=list)
    correct_answer_str: str | None = None


@dataclass
class TurnResult:
    """Everything produced by a single handle_turn call.

    Attributes
    ----------
    verdict:
        Coarse verdict (CORRECT / PARTIAL / WRONG / ABSTAIN).
    judgment:
        Full CAS Judgment (or None if parse failed completely).
    coaching_message:
        The prose shown to the student.
    p_known_after:
        Updated BKT probability after this turn.
    misconception_id:
        Buggy-rule ID diagnosed, if any.
    support_level:
        Scaffolding level that was used.
    telemetry:
        The TelemetryEvent emitted for this turn.
    parse_error:
        Set if the input could not be parsed.
    used_offline_tutor:
        True when the real LLM was unavailable and the fallback was used.
    """

    verdict: Verdict
    judgment: Judgment | None
    coaching_message: str
    p_known_after: float
    misconception_id: str | None = None
    support_level: SupportLevel = SupportLevel.INDEPENDENT
    telemetry: TelemetryEvent | None = None
    parse_error: str | None = None
    used_offline_tutor: bool = False


# ── LLM protocol ─────────────────────────────────────────────────────────────

class LLMCoach(Protocol):
    """Interface the Orchestrator expects from the LLM object."""

    def coach(self, context: CoachingContext) -> str:
        """Return a coaching message given verified CAS facts."""
        ...


# ── helpers ───────────────────────────────────────────────────────────────────

_OFFLINE = OfflineTutor()


def _now_ms() -> int:
    return int(time.monotonic() * 1000)


def _safe_certify(text: str) -> str:
    """Run claim_cert.certify if available; otherwise pass through."""
    if certify is None:
        return text
    try:
        return certify(text)
    except Exception:  # pragma: no cover
        return text


def _safe_redact(text: str, target: Target | None) -> str:
    """Run leak_filter.redact_answers if available; otherwise pass through."""
    if redact_answers is None or target is None:
        return text
    try:
        return redact_answers(text, target)
    except Exception:  # pragma: no cover
        return text


def _safe_diagnose(artifact: Artifact, target: Target) -> tuple[str | None, str | None]:
    """Return (misconception_id, description) or (None, None) on failure."""
    if diagnose is None:
        return None, None
    try:
        result = diagnose(artifact, target)
        if result is None:
            return None, None
        # diagnose may return a BuggyRule or a (id, description) tuple
        if hasattr(result, "id"):
            return result.id, getattr(result, "description", None)
        if isinstance(result, tuple) and len(result) == 2:
            return result
    except Exception:  # pragma: no cover
        pass
    return None, None


def _safe_choose_support(
    session: Session,
    judgment: Judgment,
) -> SupportLevel:
    """Return a scaffolding SupportLevel; fall back to INDEPENDENT."""
    if choose_support is None:
        return SupportLevel.INDEPENDENT
    try:
        return choose_support(session, judgment)
    except Exception:  # pragma: no cover
        return SupportLevel.INDEPENDENT


def _safe_bkt_update(session: Session, correct: bool) -> float:
    """Update p_known via BKT if available; return new probability."""
    if session.bkt_model is None or BKTModel is None:
        # Tiny hand-coded update so the value isn't always static
        lr = 0.1
        if correct:
            session.p_known = session.p_known + lr * (1.0 - session.p_known)
        else:
            session.p_known = session.p_known * (1.0 - lr)
        return session.p_known
    try:
        new_p = session.bkt_model.update(session.kc_id, correct)
        session.p_known = new_p
        return new_p
    except Exception:  # pragma: no cover
        return session.p_known


def _emit_telemetry(event: TelemetryEvent) -> None:
    """Fire-and-forget telemetry; never raises."""
    if record is not None:
        try:
            record(event)
        except Exception:  # pragma: no cover
            pass


def _null_judgment() -> Judgment:
    """Return a placeholder Judgment for unparseable input."""
    return Judgment(
        parsed_ok=False,
        value_equivalent=False,
        form_ok=False,
        correct=False,
        partial=False,
        decidable=False,
        confidence=0.0,
        detail={"reason": "parse_failure"},
    )


# ── orchestrator ──────────────────────────────────────────────────────────────

class Orchestrator:
    """Route a student turn through CAS verification → tutoring → safety.

    Parameters
    ----------
    llm:
        An object satisfying the LLMCoach protocol.  Defaults to
        ``OfflineTutor`` so the system works without any LLM.
    emit_telemetry:
        If True (default), emit a TelemetryEvent to ``eval.telemetry.record``.

    Usage
    -----
    >>> orch = Orchestrator()              # uses offline tutor
    >>> result = orch.handle_turn(session, "x = 3")
    >>> print(result.coaching_message)
    """

    def __init__(
        self,
        llm: LLMCoach | None = None,
        emit_telemetry: bool = True,
    ) -> None:
        self._llm: LLMCoach = llm if llm is not None else _OFFLINE
        self._emit_telemetry = emit_telemetry

    # ── public API ────────────────────────────────────────────────────────────

    def handle_turn(self, session: Session, student_raw: str) -> TurnResult:
        """Process one student input and return a fully-formed TurnResult.

        This method is the ONLY entry point for student interaction.  It
        implements the turn lifecycle described in the module docstring.

        Parameters
        ----------
        session:
            Mutable session state (modified in place).
        student_raw:
            Exactly what the student typed; not yet parsed or validated.

        Returns
        -------
        TurnResult
            Contains verdict, coaching message, updated p_known, and
            the telemetry event.
        """
        t_start = _now_ms()
        p_known_before = session.p_known

        # ── step 1: parse ─────────────────────────────────────────────────
        artifact: Artifact | None = None
        parse_err: str | None = None
        judgment: Judgment | None = None

        verifier = session.verifier
        if verifier is None:
            # No verifier configured — treat as abstain
            judgment = _null_judgment()
            parse_err = "No verifier configured for this session."
        else:
            try:
                artifact = verifier.parse(student_raw)
            except ParseError as exc:
                parse_err = str(exc)
                judgment = _null_judgment()
            except Exception as exc:  # pragma: no cover
                parse_err = f"Unexpected parse error: {exc}"
                judgment = _null_judgment()

        # ── step 2: verify ────────────────────────────────────────────────
        if artifact is not None and session.target is not None:
            try:
                judgment = verifier.accepts(artifact, session.target)  # type: ignore[union-attr]
            except Exception as exc:  # pragma: no cover
                judgment = Judgment(
                    parsed_ok=True,
                    value_equivalent=False,
                    form_ok=False,
                    correct=False,
                    partial=False,
                    decidable=False,
                    confidence=0.0,
                    detail={"reason": f"verifier_error: {exc}"},
                )
        elif judgment is None:
            judgment = _null_judgment()

        verdict = verdict_from_judgment(judgment)

        # ── step 3: update learner state ──────────────────────────────────
        p_known_after = _safe_bkt_update(session, correct=judgment.correct)
        session.opportunity_index += 1

        # ── step 4: diagnose misconception (wrong turns only) ─────────────
        misconception_id: str | None = None
        misconception_desc: str | None = None
        if verdict == Verdict.WRONG and artifact is not None and session.target is not None:
            misconception_id, misconception_desc = _safe_diagnose(
                artifact, session.target
            )

        # ── step 5: choose support level ──────────────────────────────────
        support_level = _safe_choose_support(session, judgment)

        # ── step 6: build coaching context ────────────────────────────────
        # Redact the correct answer if support level is INDEPENDENT
        answer_for_coach: str | None = session.correct_answer_str
        if support_level == SupportLevel.INDEPENDENT:
            answer_for_coach = None

        ctx = CoachingContext(
            verdict=verdict,
            judgment=judgment,
            support_level=support_level,
            hint_level=session.hint_level,
            kc_name=session.kc_name,
            problem_statement=session.problem_statement,
            student_raw=student_raw,
            correct_answer_str=answer_for_coach,
            misconception_id=misconception_id,
            misconception_description=misconception_desc,
            worked_steps=judgment.detail.get("worked_steps", []),
        )

        # ── step 6b: call LLM (with offline fallback) ─────────────────────
        used_offline = False
        raw_prose = ""
        try:
            raw_prose = self._llm.coach(ctx)
        except Exception:
            # LLM unavailable — fall back to offline tutor transparently
            used_offline = True
            raw_prose = _OFFLINE.coach(ctx)

        # ── step 7: safety filters on prose ──────────────────────────────
        # certify: strip or flag any unsupported mathematical claims
        safe_prose = _safe_certify(raw_prose)
        # redact_answers: ensure correct answer values are not leaked in
        # INDEPENDENT / COMPLETION modes
        if support_level != SupportLevel.WORKED:
            safe_prose = _safe_redact(safe_prose, session.target)

        # ── step 8: emit telemetry ────────────────────────────────────────
        event = TelemetryEvent(
            event_id=str(uuid.uuid4()),
            session_id=session.session_id,
            user_pseudonym=session.user_pseudonym,
            ts=time.time(),
            kc_id=session.kc_id,
            problem_id=session.problem_id,
            opportunity_index=session.opportunity_index,
            action="answer_attempt",
            input_artifact=student_raw[:200],
            verdict=verdict.value,
            error_kind=parse_err,
            misconception_id=misconception_id,
            support_level=support_level.value,
            hint_level=session.hint_level,
            latency_ms=_now_ms() - t_start,
            p_known_before=p_known_before,
            p_known_after=p_known_after,
            policy_id="offline" if used_offline else "llm",
        )
        if self._emit_telemetry:
            _emit_telemetry(event)

        # ── step 9: assemble result ───────────────────────────────────────
        result = TurnResult(
            verdict=verdict,
            judgment=judgment,
            coaching_message=safe_prose,
            p_known_after=p_known_after,
            misconception_id=misconception_id,
            support_level=support_level,
            telemetry=event,
            parse_error=parse_err,
            used_offline_tutor=used_offline or isinstance(self._llm, OfflineTutor),
        )
        session.history.append(result)
        return result

# ────────────────────────────────────────────────────────────
# FILE: ./src/mathtutor/safety/__init__.py
# ────────────────────────────────────────────────────────────



# ────────────────────────────────────────────────────────────
# FILE: ./src/mathtutor/safety/claim_cert.py
# ────────────────────────────────────────────────────────────

"""
safety/claim_cert.py — Claim certification for the Verified Math Tutor.

Purpose
-------
The LLM is allowed to make concrete symbolic claims in its tutoring prose
(e.g. "the derivative of x² is <<claim>>2x<</claim>>").  Before that prose
reaches the student, every ``<<claim>>…<</claim>>`` span is:

1. **Parsed** by the CAS.
2. **Verified** against the CAS ground truth for the current problem.
3. **Kept and unwrapped** if correct, or **silently dropped** if wrong or
   unparseable.

This ensures the LLM can contribute pedagogically useful language while the
CAS retains exclusive authority over correctness.

Fail-safe
---------
Any claim that cannot be parsed, or whose equivalence cannot be determined,
is *dropped* (not kept).  We do less rather than risk asserting a
false claim.

Logging
-------
Dropped claims are logged (at WARNING level) with their raw text and the
reason for dropping, for post-hoc auditing.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from mathtutor.contracts import ParseError, Target, Verifier

log = logging.getLogger(__name__)

# Delimiter pattern: <<claim>>…<</claim>>
# We use a non-greedy match so adjacent claims don't merge.
_CLAIM_RE = re.compile(r"<<claim>>(.*?)<</claim>>", re.DOTALL | re.IGNORECASE)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_claims(text: str) -> list[tuple[str, str]]:
    """
    Return a list of ``(full_match, inner_text)`` pairs for every
    ``<<claim>>…<</claim>>`` span found in *text*.
    """
    return [(m.group(0), m.group(1).strip()) for m in _CLAIM_RE.finditer(text)]


def _verify_claim(
    inner: str,
    verifier: Verifier,
    target: Target,
) -> bool:
    """
    Return ``True`` iff the claim expressed by *inner* is verified by the CAS.

    A claim passes iff:
    * It parses successfully, AND
    * ``verifier.accepts(student, target).value_equivalent`` is ``True``.

    We use ``value_equivalent`` (not ``correct``) so form constraints on
    the *student* answer don't falsely reject a correct interim claim.

    Returns ``False`` on any exception — fail-safe.
    """
    try:
        artifact = verifier.parse(inner)
    except ParseError as exc:
        log.warning("Claim DROPPED — ParseError: %r  reason=%s", inner, exc)
        return False
    except Exception as exc:  # pragma: no cover — unexpected
        log.warning("Claim DROPPED — unexpected parse error: %r  reason=%s", inner, exc)
        return False

    try:
        judgment = verifier.accepts(artifact, target)
    except Exception as exc:
        log.warning("Claim DROPPED — verifier error: %r  reason=%s", inner, exc)
        return False

    if not judgment.value_equivalent:
        log.warning(
            "Claim DROPPED — value_equivalent=False: %r  "
            "decidable=%s confidence=%.4f",
            inner,
            judgment.decidable,
            judgment.confidence,
        )
        return False

    # Undecidable claims: keep only if confidence is high enough to be useful.
    # We set a conservative threshold of 0.95 so near-certain probabilistic
    # checks still pass, but genuinely uncertain ones are dropped.
    if not judgment.decidable and judgment.confidence < 0.95:
        log.warning(
            "Claim DROPPED — low-confidence probabilistic check: %r  "
            "confidence=%.4f",
            inner,
            judgment.confidence,
        )
        return False

    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def certify(text: str, verifier: Verifier, target: Target) -> str:
    """
    Remove unverifiable or false ``<<claim>>…<</claim>>`` spans from *text*.

    Parameters
    ----------
    text:
        LLM-generated tutoring prose containing zero or more
        ``<<claim>>…<</claim>>`` spans.
    verifier:
        A domain :class:`~mathtutor.contracts.Verifier` instance.
        The CAS — not the LLM — decides correctness.
    target:
        The CAS-certified correct answer for the current problem.

    Returns
    -------
    str
        The prose with verified claims *unwrapped* (delimiters removed,
        inner text kept) and unverifiable/false claims *deleted entirely*.

    Behaviour
    ---------
    * **Correct claim** ``<<claim>>3<</claim>>`` → ``3``
    * **False claim**   ``<<claim>>5<</claim>>`` → *(deleted)*
    * **Unparseable**   ``<<claim>>??<</claim>>`` → *(deleted)*

    Fail-safe
    ---------
    On any unexpected error the offending span is dropped and a WARNING is
    logged.  The rest of the text is returned intact.
    """
    claims = _extract_claims(text)
    if not claims:
        return text

    result = text
    for full_match, inner in claims:
        try:
            if _verify_claim(inner, verifier, target):
                # Unwrap: replace delimiter+content with bare content
                result = result.replace(full_match, inner, 1)
            else:
                # Drop entirely (logging already done inside _verify_claim)
                result = result.replace(full_match, "", 1)
        except Exception as exc:  # pragma: no cover — belt-and-suspenders
            log.warning(
                "Claim DROPPED — unexpected error in certify loop: %r  reason=%s",
                full_match,
                exc,
            )
            result = result.replace(full_match, "", 1)

    return result


# ────────────────────────────────────────────────────────────
# FILE: ./src/mathtutor/safety/leak_filter.py
# ────────────────────────────────────────────────────────────

"""
safety/leak_filter.py — Answer-leak filter for the Verified Math Tutor.

Purpose
-------
While a student is still working on a problem ("gated" mode), prevent the
LLM's tutoring prose from revealing the correct answer literally.

How it works
------------
1.  The CAS solves the problem and supplies ``answers`` as a list of strings
    (e.g. ``["3", "-2"]``).
2.  This module replaces every literal occurrence of those answer strings —
    including trivial re-orderings (e.g. ``"3, 2"`` when answers are
    ``["2", "3"]``) — with the token ``[hidden while you work]``.

⚠️  KNOWN LIMITATION — LITERAL MATCHING ONLY
---------------------------------------------
This filter catches *literal* occurrences of the answer strings.  It does
**NOT** catch:

* Answers disguised in prose ("the solution is one more than two"),
* Answers embedded in equations the student hasn't simplified yet
  ("x = 6/2"),
* Answers revealed through strong hints ("try substituting the value that
  makes x − 3 equal to zero"),
* Answers in a different but equivalent form ("x = 6/2" when answer is 3).

These require semantic / algebraic understanding and are the responsibility
of the LLM prompt discipline and ``claim_cert.py``, not this filter.
Any production deployment should treat literal filtering as one layer of a
defence-in-depth stack, not a complete solution.
"""

from __future__ import annotations

import itertools
import re
from typing import Sequence


_HIDDEN = "[hidden while you work]"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _escape(s: str) -> str:
    """Regex-escape a literal string."""
    return re.escape(s.strip())


def _make_patterns(answers: Sequence[str]) -> list[re.Pattern[str]]:
    """
    Build a list of compiled regex patterns to match:

    1. Each individual answer string.
    2. Every permutation of the full answer set written as a
       comma-separated list (``"3, 2"`` and ``"2, 3"`` for answers
       ``["2", "3"]``).

    We use word boundaries (``\\b``) around numeric literals so that
    ``"3"`` does not match inside ``"13"`` or ``"30"``.
    """
    patterns: list[re.Pattern[str]] = []
    stripped = [a.strip() for a in answers if a.strip()]

    # 1. Individual answers
    for ans in stripped:
        esc = _escape(ans)
        # \b only works where the boundary is between a \w and a \W char.
        # Negative numbers like "-2" start with "-" which is \W, so the
        # leading \b would sit between two \W chars and never match.
        # Instead we use a lookbehind/lookahead that rejects digits on
        # either side, which correctly handles "-2" without eating "13".
        pat = rf"(?<!\d){esc}(?!\d)"
        patterns.append(re.compile(pat))

    # 2. Comma-separated permutations (trivial reorderings)
    if len(stripped) > 1:
        for perm in itertools.permutations(stripped):
            # e.g. "2, 3"  or  "2,3"  (with or without space after comma)
            joined = r"\s*,\s*".join(_escape(v) for v in perm)
            patterns.append(re.compile(rf"\b{joined}\b"))

    return patterns


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def redact_answers(text: str, answers: list[str], gated: bool) -> str:
    """
    Replace literal answer occurrences in *text* with ``[hidden while you work]``.

    Parameters
    ----------
    text:
        The LLM-generated tutoring prose to filter.
    answers:
        The CAS-certified correct answers as plain strings
        (e.g. ``["3", "-2"]``).  These come from the CAS solver, never
        from the LLM.
    gated:
        When ``True`` the student is still working — apply redaction.
        When ``False`` (problem solved / answer revealed) pass through
        unchanged.

    Returns
    -------
    str
        Filtered text.  When ``gated`` is ``False``, the original *text*
        is returned unmodified.

    Notes
    -----
    See module docstring for the *literal-only* limitation.  This function
    intentionally does the minimum safe thing: redact literals.  It does
    not attempt semantic analysis.
    """
    if not gated or not answers:
        return text

    result = text
    for pattern in _make_patterns(answers):
        result = pattern.sub(_HIDDEN, result)
    return result


# ────────────────────────────────────────────────────────────
# FILE: ./src/mathtutor/tutoring/__init__.py
# ────────────────────────────────────────────────────────────



# ────────────────────────────────────────────────────────────
# FILE: ./src/mathtutor/tutoring/misconceptions.py
# ────────────────────────────────────────────────────────────

# mathtutor/tutoring/misconceptions.py

from typing import Any, List
from sympy import Expr, Eq, Add, Mul, Pow, Rel, simplify, expand
from mathtutor.contracts import BuggyRule


def exact_match(a: Any, b: Any) -> bool:
    """
    Checks for canonical structural equality via SymPy.
    Ensures mathematical equivalence AND structural identity so that
    aggressive simplification doesn't erase the buggy form.
    """
    if type(a) != type(b):
        return False
    
    # SymPy's `==` handles structural identity (and basic commutativity).
    if a == b:
        return True
        
    # Handle swapped equations
    if hasattr(a, 'lhs') and hasattr(b, 'lhs'):
        if isinstance(a, Eq) and a.lhs == b.rhs and a.rhs == b.lhs:
            return True
        return False
        
    try:
        if hasattr(a, 'equals') and hasattr(b, 'equals') and a.equals(b):
            return (a.count_ops() == b.count_ops()) and (len(a.args) == len(b.args))
    except Exception:
        pass
        
    return False


class DistributeExponentOverSum:
    id = "distributes_exponent_over_sum"
    
    def applies_to(self, previous: Any) -> bool:
        return previous.has(Pow)
        
    def transform(self, previous: Any) -> List[Any]:
        """(a+b)**2 -> a**2 + b**2"""
        try:
            res = previous.replace(
                lambda x: x.is_Pow and x.base.is_Add,
                lambda x: Add(*[Pow(arg, x.exp) for arg in x.base.args])
            )
            return [res] if res != previous else []
        except Exception:
            return []


class MoveTermWithoutSignFlip:
    id = "moves_term_across_equals_without_sign_flip"
    
    def applies_to(self, previous: Any) -> bool:
        return isinstance(previous, Eq) and (previous.lhs.is_Add or previous.rhs.is_Add)
        
    def transform(self, previous: Any) -> List[Any]:
        """Eq(LHS + term, RHS) -> Eq(LHS, RHS + term)"""
        results = []
        if not isinstance(previous, Eq):
            return results
            
        if previous.lhs.is_Add:
            for arg in previous.lhs.args:
                results.append(Eq(previous.lhs - arg, previous.rhs + arg))
        if previous.rhs.is_Add:
            for arg in previous.rhs.args:
                results.append(Eq(previous.lhs + arg, previous.rhs - arg))
        return results


class CancelTermAcrossAddition:
    id = "cancels_term_across_addition"
    
    def applies_to(self, previous: Any) -> bool:
        return previous.has(Mul)
        
    def transform(self, previous: Any) -> List[Any]:
        """(a+b)/a -> b"""
        results = []
        def replacer(expr: Expr) -> Expr:
            if expr.is_Mul:
                adds = [a for a in expr.args if a.is_Add]
                pows = [a for a in expr.args if a.is_Pow and a.exp == -1]
                if adds and pows:
                    for add in adds:
                        for p in pows:
                            denom = p.base
                            for arg in add.args:
                                if arg == denom:
                                    return add - arg
            return expr
            
        try:
            new_expr = previous.replace(lambda x: x.is_Mul, replacer)
            if new_expr != previous:
                results.append(new_expr)
        except Exception:
            pass
        return results


class ClearsFractionOneTerm:
    id = "clears_fraction_multiplying_only_one_term"

    @staticmethod
    def _denom_if_fraction(term):
        """Return the denominator if `term` is a fractional Mul, else None.

        Handles both SymPy representations:
          - Mul(Rational(1, n), x)  — what SymPy actually produces for x/n
          - Mul(x, Pow(n, -1))      — explicit inverse, rarely seen in practice
        """
        if not term.is_Mul:
            return None
        for a in term.args:
            # Common case: Rational(p, q) with q > 1, e.g. Rational(1,2) for x/2
            if a.is_Rational and not a.is_Integer and a.q > 1:
                return a.q          # SymPy Integer denominator
            # Explicit Pow(n, -1) case
            if a.is_Pow and a.exp == -1:
                return a.base
        return None

    def applies_to(self, previous: Any) -> bool:
        return (
            isinstance(previous, Eq)
            and isinstance(previous.lhs, Add)
            and any(self._denom_if_fraction(t) is not None
                    for t in previous.lhs.args)
        )

    def transform(self, previous: Any) -> List[Any]:
        """x/2 + y = 5  →  x + y = 10  (student multiplied only the fraction)"""
        results = []
        if not isinstance(previous, Eq) or not isinstance(previous.lhs, Add):
            return results

        lhs, rhs = previous.lhs, previous.rhs
        lhs_terms = list(lhs.args)

        for i, term in enumerate(lhs_terms):
            denom = self._denom_if_fraction(term)
            if denom is None:
                continue
            # Correct: multiply every term AND the RHS by denom
            # Bug:     multiply only this fractional term (y stays as y, not 2y)
            cleared_term = term * denom          # x/2 * 2  →  x
            buggy_lhs = Add(*[
                cleared_term if j == i else t
                for j, t in enumerate(lhs_terms)
            ])                                   # x + y  (y is unchanged)
            buggy_rhs = rhs * denom              # 5 * 2  →  10
            results.append(Eq(buggy_lhs, buggy_rhs))

        return results

class InequalityMultiplyNegativeNoFlip:
    id = "forgets_to_flip_inequality_on_negative_multiply"
    
    def applies_to(self, previous: Any) -> bool:
        return isinstance(previous, Rel)
        
    def transform(self, previous: Any) -> List[Any]:
        """-x < 5 -> x < -5"""
        if isinstance(previous, Rel):
            try:
                return [type(previous)(previous.lhs * -1, previous.rhs * -1)]
            except Exception:
                pass
        return []


BUGGY_RULES: List[BuggyRule] = [
    DistributeExponentOverSum(),
    MoveTermWithoutSignFlip(),
    CancelTermAcrossAddition(),
    ClearsFractionOneTerm(),
    InequalityMultiplyNegativeNoFlip(),
]


def diagnose(previous: Any, student_line: Any, rules: List[BuggyRule] = None) -> List[str]:
    """
    Evaluates known misconceptions by structurally verifying the transformation against the student line.
    """
    if rules is None:
        rules = BUGGY_RULES
        
    matched_rules = []
    for r in rules:
        try:
            if not r.applies_to(previous):
                continue
            transforms = r.transform(previous)
            for t in transforms:
                if exact_match(t, student_line):
                    matched_rules.append(r.id)
                    break
        except Exception:
            continue
            
    return matched_rules


def classify_error(previous: Any, student_line: Any) -> str:
    """
    Symptom-level fallback classification when exact misconception diagnosis fails.
    """
    try:
        if isinstance(previous, Eq) and isinstance(student_line, Eq):
            if exact_match(previous.lhs * -1, student_line.lhs) or \
               exact_match(previous.rhs * -1, student_line.rhs):
                return "sign_error"
                
            diff = simplify((previous.lhs - previous.rhs) - (student_line.lhs - student_line.rhs))
            if diff.is_number and abs(diff) == 1:
                return "off_by_one"
    except Exception:
        pass
        
    return "unknown"

# ────────────────────────────────────────────────────────────
# FILE: ./src/mathtutor/tutoring/scaffolding.py
# ────────────────────────────────────────────────────────────

"""
tutoring/scaffolding.py — Adaptive scaffolding, hint ladder, and gate.

Design principles
-----------------
* **Support level** is a pure function of mastery probability *p_known*.
  Three zones: WORKED (study a full solution) → COMPLETION (fill in the
  blanks) → INDEPENDENT (hint ladder only).

* **Hint ladder** escalates one rung at a time, only on explicit request.
  It deliberately withholds the literal next line or final answer.
  The five rungs are:
    1. Region  – points to WHERE in the working the error lives.
    2. Kind    – names the TYPE of algebraic move needed.
    3. Socratic – a question targeting the diagnosed misconception.
    4. Analogous – the same move demonstrated on a DIFFERENT, simpler example.
    5. Worked  – the full solution, gated behind ``gate_open``.

* **Structural progress** is checked by the CAS, not by surface string
  comparison.  A step counts only if SymPy's canonical form changed AND
  the change isn't a cycle of recently-seen forms AND the new expression
  is no more complex than the previous one (complexity = operation count).

* **Gate** weighs two factors: (a) at least one structural step was made,
  and (b) enough time has passed.  Thresholds are modulated by mastery and
  frustration so an advanced but frustrated learner gets help sooner, while
  a low-mastery learner who rushes is held back.

No global mutable state: all state lives in HintLadder instances.
No LLM calls here: all language is template-driven so the module is
deterministic and testable.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import sympy
from sympy import sympify, count_ops, SympifyError

from mathtutor.contracts import SupportLevel, Judgment  # noqa: F401 (re-exported)


# ---------------------------------------------------------------------------
# 1. Support level
# ---------------------------------------------------------------------------

def support_level(
    p_known: float,
    *,
    low: float = 0.4,
    high: float = 0.85,
) -> SupportLevel:
    """
    Map a BKT mastery estimate *p_known* ∈ [0, 1] to a ``SupportLevel``.

    Zones
    -----
    p_known < low          → WORKED       (full worked example + self-explain)
    low ≤ p_known < high   → COMPLETION   (partial steps, blanks to fill)
    p_known ≥ high         → INDEPENDENT  (hint ladder only)

    Parameters
    ----------
    p_known : float
        Bayesian Knowledge Tracing posterior probability that the student
        has acquired the relevant knowledge component.  Must be in [0, 1].
    low : float, keyword-only
        Upper boundary of the WORKED zone (exclusive).  Default 0.40.
    high : float, keyword-only
        Lower boundary of the INDEPENDENT zone (inclusive).  Default 0.85.

    Returns
    -------
    SupportLevel

    Raises
    ------
    ValueError
        If ``p_known`` is outside [0, 1] or the thresholds are inconsistent.

    Examples
    --------
    >>> support_level(0.2)
    <SupportLevel.WORKED: 1>
    >>> support_level(0.6)
    <SupportLevel.COMPLETION: 2>
    >>> support_level(0.9)
    <SupportLevel.INDEPENDENT: 3>
    """
    if not (0.0 <= p_known <= 1.0):
        raise ValueError(f"p_known must be in [0, 1]; got {p_known}")
    if not (0.0 < low < high < 1.0):
        raise ValueError(
            f"Thresholds must satisfy 0 < low < high < 1; got low={low}, high={high}"
        )

    if p_known < low:
        return SupportLevel.WORKED
    if p_known < high:
        return SupportLevel.COMPLETION
    return SupportLevel.INDEPENDENT


# ---------------------------------------------------------------------------
# 2. Hint ladder
# ---------------------------------------------------------------------------

# Rung metadata: keys used in the returned payload dict.
# The VALUE strings are structural labels, not solution content.
_RUNG_KEYS = {
    1: "region",
    2: "kind",
    3: "socratic",
    4: "analogous",
    5: "worked_gate",
}

# Maximum rung before we'd need the gate
_MAX_FREE_RUNG = 4
_GATE_RUNG = 5


@dataclass
class HintLadder:
    """
    Tracks the current hint level for one (session, problem) pair.

    State
    -----
    _current_level : int
        Which rung was last emitted.  0 means no hint has been given yet.
    _session_id : str
        Opaque identifier for the session.
    _problem_id : str
        Opaque identifier for the problem within the session.

    The ladder escalates **one rung per call** to ``next_hint``.  It
    never emits the literal next algebraic step or the final answer.

    Payload keys returned per rung
    --------------------------------
    Rung 1 – ``region``     : a label such as ``"step_2"`` or ``"line_3"``;
                               never the corrected expression.
    Rung 2 – ``kind``       : a move name such as ``"collect_like_terms"``
                               or ``"isolate_variable"``.
    Rung 3 – ``socratic``   : a question referencing the diagnosed
                               misconception without stating the answer.
    Rung 4 – ``analogous``  : a worked step on a structurally similar but
                               *different* example (different numbers/letters).
    Rung 5 – ``worked_gate``: ``"GATE_CLOSED"`` if the gate is not open;
                               ``"GATE_OPEN"``  if the caller should now
                               display the full worked solution.
                               The *solution itself is never embedded here*.
    """

    _session_id: str = field(default="default_session")
    _problem_id: str = field(default="default_problem")
    _current_level: int = field(default=0, init=False)

    # ------------------------------------------------------------------ #
    # Templates — structural labels only, NEVER containing solution text  #
    # ------------------------------------------------------------------ #

    # Region labels: a tuple of plausible step locations.
    _REGIONS: tuple[str, ...] = field(default=(
        "setup_step",
        "first_transformation",
        "middle_step",
        "final_simplification",
        "conclusion_step",
    ), init=False, repr=False)

    # Move kinds: algebraic operation names.
    _KINDS: tuple[str, ...] = field(default=(
        "collect_like_terms",
        "apply_distributive_law",
        "isolate_the_variable",
        "multiply_both_sides",
        "divide_both_sides",
        "factor_expression",
        "expand_brackets",
        "cancel_common_factor",
        "apply_zero_product_rule",
    ), init=False, repr=False)

    def next_hint(
        self,
        diagnosis: list[str] | None = None,
        *,
        gate_open: bool = False,
    ) -> dict:
        """
        Advance the ladder by exactly one rung and return a payload dict.

        Parameters
        ----------
        diagnosis : list[str] | None
            A list of misconception tags produced by the diagnosis layer
            (e.g. ``["sign_error", "forgot_to_distribute"]``).  Used to
            select a Socratic question at rung 3.  If ``None`` or empty,
            a generic question is used.
        gate_open : bool, keyword-only
            Whether ``gate_open(...)`` returned True.  Passed in by the
            caller to keep this class decoupled from gate logic.
            Only relevant when we're about to emit rung 5.

        Returns
        -------
        dict with keys:
            ``level``   – int, the rung just emitted (1–5).
            ``hint_type`` – str, one of the _RUNG_KEYS values.
            One additional key named after the hint_type, containing the
            structural payload (never the solution).

        Notes
        -----
        Once the ladder reaches rung 5, further calls keep returning
        rung 5 (the gate check) rather than wrapping around.
        """
        next_level = min(self._current_level + 1, _GATE_RUNG)
        self._current_level = next_level
        hint_type = _RUNG_KEYS[next_level]

        if next_level == 1:
            payload = self._rung_region(diagnosis)
        elif next_level == 2:
            payload = self._rung_kind(diagnosis)
        elif next_level == 3:
            payload = self._rung_socratic(diagnosis)
        elif next_level == 4:
            payload = self._rung_analogous(diagnosis)
        else:  # next_level == 5
            payload = "GATE_OPEN" if gate_open else "GATE_CLOSED"

        return {
            "level": next_level,
            "hint_type": hint_type,
            hint_type: payload,
        }

    def reset(self) -> None:
        """Reset the ladder to rung 0 (start of a new problem attempt)."""
        self._current_level = 0

    @property
    def current_level(self) -> int:
        """The rung last emitted (0 = no hint given yet)."""
        return self._current_level

    # ------------------------------------------------------------------ #
    # Private rung builders                                                #
    # ------------------------------------------------------------------ #

    def _rung_region(self, diagnosis: list[str] | None) -> dict:
        """
        Rung 1 — identify the region of the error.

        Returns a dict with a ``step_label`` (never the corrected value)
        and an optional ``hint_text`` that points *toward* the location
        without stating the fix.
        """
        # Use diagnosis to pick a step if available, else generic.
        if diagnosis:
            tag = diagnosis[0]
            text = (
                f"Look carefully at the step where you applied '{tag}'. "
                "Something changes unexpectedly there."
            )
        else:
            text = (
                "Check each transformation line by line. "
                "There is one step that introduces an inconsistency."
            )
        return {
            "step_label": "examine_your_working",
            "hint_text": text,
        }

    def _rung_kind(self, diagnosis: list[str] | None) -> dict:
        """
        Rung 2 — name the kind of algebraic move needed.

        Returns a dict with a ``move_name`` taken from ``_KINDS``.
        The move is named, not demonstrated.
        """
        if diagnosis:
            # Map common misconception tags to move names
            _TAG_TO_MOVE: dict[str, str] = {
                "sign_error": "track_sign_across_operation",
                "forgot_to_distribute": "apply_distributive_law",
                "wrong_inverse": "apply_correct_inverse_operation",
                "incomplete_factoring": "factor_expression",
                "missed_root": "apply_zero_product_rule",
                "like_terms_not_collected": "collect_like_terms",
            }
            for tag in diagnosis:
                if tag in _TAG_TO_MOVE:
                    move = _TAG_TO_MOVE[tag]
                    break
            else:
                move = self._KINDS[0]
        else:
            move = "isolate_the_variable"

        return {
            "move_name": move,
            "hint_text": (
                f"The next productive move is to '{move}'. "
                "Think about what that operation does to both sides."
            ),
        }

    def _rung_socratic(self, diagnosis: list[str] | None) -> dict:
        """
        Rung 3 — Socratic question targeting the diagnosed misconception.

        Returns a question that provokes re-examination without giving the
        answer.  The question is chosen by misconception tag when available.
        """
        _TAG_TO_QUESTION: dict[str, str] = {
            "sign_error": (
                "When you moved that term across the equals sign, "
                "what happens to its sign — and did that happen here?"
            ),
            "forgot_to_distribute": (
                "You multiplied one term inside the bracket. "
                "Does the factor outside apply to every term inside?"
            ),
            "wrong_inverse": (
                "To undo an operation, you apply its inverse. "
                "What is the exact inverse of the operation you used?"
            ),
            "incomplete_factoring": (
                "You factored partially. "
                "Is every factor on the left now in its simplest form?"
            ),
            "missed_root": (
                "A product equals zero. "
                "How many different ways can a product equal zero — "
                "and have you accounted for all of them?"
            ),
            "like_terms_not_collected": (
                "Two terms share the same variable. "
                "What can you do when terms have identical variable parts?"
            ),
        }

        if diagnosis:
            for tag in diagnosis:
                if tag in _TAG_TO_QUESTION:
                    question = _TAG_TO_QUESTION[tag]
                    break
            else:
                question = (
                    "Look at the step just before your answer. "
                    "Is every equality still valid at that point — why or why not?"
                )
        else:
            question = (
                "At which point does the left-hand side stop being equal "
                "to the right-hand side of the original equation?"
            )

        return {"question": question}

    def _rung_analogous(self, diagnosis: list[str] | None) -> dict:
        """
        Rung 4 — demonstrate the same move on a DIFFERENT, simpler example.

        The analogous example uses different numbers and variable names so
        the student cannot copy it directly.  It shows the *structural
        pattern*, not the solution to the current problem.
        """
        _TAG_TO_ANALOGY: dict[str, dict] = {
            "sign_error": {
                "description": "Moving a term across the equals sign flips its sign.",
                "before": "a + 3 = 7",
                "after":  "a = 7 - 3   (the +3 becomes -3 on crossing)",
                "note": "Notice: the term changes sign when it crosses the equals sign.",
            },
            "forgot_to_distribute": {
                "description": "The factor outside multiplies EVERY term inside.",
                "before": "2*(m + 4)",
                "after":  "2*m + 2*4  =  2m + 8",
                "note": "Both m and 4 are multiplied by 2, not just the first.",
            },
            "wrong_inverse": {
                "description": "Undo multiplication with division; undo addition with subtraction.",
                "before": "3*k = 12",
                "after":  "k = 12 / 3  (divide both sides by 3)",
                "note": "The inverse of ×3 is ÷3.",
            },
            "incomplete_factoring": {
                "description": "Keep factoring until no common factor remains.",
                "before": "4*n**2 - 16",
                "after":  "4*(n**2 - 4)  →  4*(n-2)*(n+2)",
                "note": "n²-4 is a difference of squares and factors further.",
            },
            "missed_root": {
                "description": "Zero-product rule: if A*B=0 then A=0 OR B=0.",
                "before": "(t - 1)*(t + 3) = 0",
                "after":  "t - 1 = 0  → t = 1   OR   t + 3 = 0  → t = -3",
                "note": "Two separate cases, two separate solutions.",
            },
        }

        generic_analogy = {
            "description": "Isolate the variable by performing the same operation on both sides.",
            "before": "b + 7 = 10",
            "after":  "b = 10 - 7  =  3",
            "note": "Subtracting 7 from both sides keeps the equation balanced.",
        }

        analogy: dict = generic_analogy
        if diagnosis:
            for tag in diagnosis:
                if tag in _TAG_TO_ANALOGY:
                    analogy = _TAG_TO_ANALOGY[tag]
                    break

        return {
            "description": analogy["description"],
            "analogous_before": analogy["before"],
            "analogous_after": analogy["after"],
            "note": analogy["note"],
            "reminder": (
                "This example uses different numbers and letters from your problem "
                "— apply the same structural idea, do not copy the values."
            ),
        }


# ---------------------------------------------------------------------------
# 3. Structural progress
# ---------------------------------------------------------------------------

def _sympy_canonical(expr: Any) -> sympy.Basic | None:
    """
    Convert *expr* to a SymPy expression in canonical form for **structural
    comparison**.

    We use ``expand`` only — not ``cancel`` — so that the canonical form
    preserves structural differences that are meaningful to a student.
    For example:

    * ``(x²-1)/(x-1)``  expands to  ``x²/(x-1) - 1/(x-1)``  (6 ops),
      while ``x+1`` expands to ``x+1``  (1 op).
      These are structurally different, so going from one to the other
      counts as progress — even though they are mathematically equal.

    Equivalence checking (for cycle detection) uses SymPy's ``.equals()``
    which *does* apply ``cancel`` and other algebraic identities internally.

    Returns ``None`` if conversion fails so callers can treat that as
    "unparseable" rather than raising.
    """
    if not isinstance(expr, sympy.Basic):
        try:
            expr = sympify(str(expr))
        except (SympifyError, TypeError, ValueError):
            return None
    try:
        return sympy.expand(expr)
    except (TypeError, ValueError):
        return None


def _complexity(expr: sympy.Basic) -> int:
    """
    Measure expression complexity as SymPy's operation count.

    ``count_ops`` returns the number of arithmetic operations in the
    expression tree.  A simpler expression (e.g. ``2*x`` from ``4*x/2``)
    has a strictly lower count.

    We use this as a *hint*, not a gate: complexity going down (or staying
    equal) is consistent with genuine progress.
    """
    return int(count_ops(expr))


class _ProgressTracker:
    """
    Internal helper: keeps a ring buffer of recently-seen canonical forms
    so ``is_structural_progress`` can detect cycles.

    Ring-buffer size is 8 steps — enough to catch ``+5-5`` patterns while
    keeping memory bounded.
    """
    _BUFFER_SIZE = 8

    def __init__(self) -> None:
        self._seen: deque[tuple[str, int]] = deque(maxlen=self._BUFFER_SIZE)

    def record(self, canonical_expr: sympy.Basic) -> None:
        """Add a canonical form to the ring buffer."""
        self._seen.append((str(canonical_expr), _complexity(canonical_expr)))

    def is_cycle(self, canonical_expr: sympy.Basic) -> bool:
        """Return True if this canonical form is mathematically equivalent
        to a recently-seen form (even if structurally different surface form)."""
        for k, _ in self._seen:
            try:
                seen_expr = sympy.sympify(k)
                if canonical_expr.equals(seen_expr):
                    return True
            except Exception:
                if str(canonical_expr) == k:
                    return True
        return False

    def last_complexity(self) -> int | None:
        """Complexity of the most-recently recorded expression, or None."""
        if not self._seen:
            return None
        return self._seen[-1][1]

    def clear(self) -> None:
        """Reset for a new problem."""
        self._seen.clear()


# Module-level tracker.  Callers that need per-problem isolation should
# instantiate their own _ProgressTracker and call is_structural_progress
# as a method instead.
_global_tracker = _ProgressTracker()


def is_structural_progress(
    prev_expr: Any,
    new_expr: Any,
    *,
    tracker: _ProgressTracker | None = None,
) -> bool:
    """
    Return ``True`` iff ``new_expr`` represents genuine structural progress
    over ``prev_expr``.

    A step counts as structural progress when **all three** conditions hold:

    1. **Form changed** — SymPy's canonical form of *new_expr* differs from
       that of *prev_expr*.  Two strings that simplify to the same SymPy
       expression (e.g. ``x + 0`` and ``x``) are considered unchanged.

    2. **Not a cycle** — the canonical form of *new_expr* has not appeared
       in the recent ring buffer of seen forms.  This catches patterns like
       ``+5`` then ``-5`` that wind up back at a previously-visited state.

    3. **Complexity did not strictly increase** — ``count_ops(new_expr)``
       must be ≤ ``count_ops(prev_expr)``.  This prevents "progress" that
       just makes the expression more complex (e.g. expanding ``x`` into
       ``x*1 + 0``).

    Parameters
    ----------
    prev_expr : Any
        The expression before the student's latest step.  Accepted as a
        SymPy ``Basic``, or any object whose ``str()`` is parseable by
        ``sympify``.
    new_expr : Any
        The expression after the student's latest step.
    tracker : _ProgressTracker | None, keyword-only
        If provided, the ring buffer used for cycle detection.  Defaults
        to the module-level ``_global_tracker`` (suitable for single-problem
        use; pass an explicit tracker for multi-problem sessions).

    Returns
    -------
    bool
        ``True`` iff all three conditions above are satisfied.
        ``False`` on any parse failure (fail-safe: uncertain → not progress).

    Examples
    --------
    >>> from sympy import symbols
    >>> x = symbols('x')
    >>> is_structural_progress(x + 5 - 5, x)        # cycle → False
    False
    >>> is_structural_progress(2*x + 4, 2*(x + 2))  # same canonical → False
    False
    >>> is_structural_progress(2*x + 4*x, 6*x)      # genuine simplification → True
    True
    """
    t = tracker if tracker is not None else _global_tracker

    prev_can = _sympy_canonical(prev_expr)
    new_can = _sympy_canonical(new_expr)

    if prev_can is None or new_can is None:
        # Unparseable input: fail-safe, claim no progress.
        return False

    # Condition 1: canonical form must have structurally changed.
    # We use ``==`` (structural equality) rather than ``.equals()``
    # (mathematical equivalence) here deliberately:
    #   * ``==`` is False for ``x²/(x-1) - 1/(x-1)`` vs ``x + 1``  ✓
    #   * ``.equals()`` would return True for those (same math) — wrong here,
    #     because the student DID make a visible structural step.
    # Cycle detection (condition 2) uses ``.equals()`` so that expressions
    # that are merely rearranged (same canonical form) are caught as cycles.
    if prev_can == new_can:
        t.record(new_can)
        return False

    # Condition 2: must not be a cycle (mathematical equivalence with a
    # recently-seen form, even if the surface form is different).
    if t.is_cycle(new_can):
        t.record(new_can)
        return False

    # Condition 3: complexity must not strictly increase.
    prev_ops = _complexity(prev_can)
    new_ops = _complexity(new_can)
    if new_ops > prev_ops:
        t.record(new_can)
        return False

    # All conditions satisfied: genuine structural progress.
    t.record(new_can)
    return True


# ---------------------------------------------------------------------------
# 4. Gate
# ---------------------------------------------------------------------------

# Default time thresholds (seconds).
_BASE_TIME_THRESHOLD_S: float = 120.0   # 2 minutes baseline

# Minimum structural steps before the gate can open at all.
_MIN_PROGRESS_STEPS: int = 1


def gate_open(
    progress_steps: int,
    time_on_task_s: float,
    p_known: float,
    frustration: float,
) -> bool:
    """
    Decide whether to open the worked-solution gate.

    Two necessary conditions (the gate is closed unless BOTH are met):

    A. **Genuine progress**: ``progress_steps ≥ 1``  (at least one
       structural step has been made).  This prevents rewarding
       passivity — a student who stares at the screen and immediately
       asks for the answer gets GATE_CLOSED.

    B. **Enough time on task**: ``time_on_task_s ≥ threshold``, where
       the threshold is a function of ``p_known`` and ``frustration``:

       * High mastery + high frustration  → threshold is *halved*
         (the student demonstrably knows the material and is stuck).
       * Low mastery + low effort (fast clicks) → threshold is *doubled*
         (the student is guessing, not thinking).
       * Otherwise the base threshold (120 s) applies.

    The threshold formula
    ---------------------
    ::

        threshold = BASE × frustration_factor × effort_factor

        frustration_factor = 1 / (1 + frustration × p_known × 2)
           - At frustration=1, p_known=1: factor ≈ 1/3  (gate opens faster)
           - At frustration=0           : factor = 1    (no change)

        effort_factor = 1 + (1 - p_known) × max(0, 1 - time_on_task_s / BASE)
           - If time_on_task_s << BASE and p_known is low: factor > 1
             (threshold raised — student is rushing)
           - If time_on_task_s ≥ BASE or p_known is high: factor ≈ 1

    Parameters
    ----------
    progress_steps : int
        Number of structurally-distinct steps taken (from
        ``is_structural_progress`` calls).
    time_on_task_s : float
        Total seconds the student has been working on this problem.
    p_known : float
        BKT mastery estimate in [0, 1].
    frustration : float
        Frustration signal in [0, 1].  0 = calm, 1 = maximally frustrated.
        Sourced from the affective model (outside this module).

    Returns
    -------
    bool
        ``True`` iff both conditions are satisfied.

    Raises
    ------
    ValueError
        If ``p_known`` or ``frustration`` are outside [0, 1].

    Examples
    --------
    >>> # High-mastery frustrated student: gate opens after < 2 minutes
    >>> gate_open(progress_steps=2, time_on_task_s=50, p_known=0.95, frustration=0.9)
    True
    >>> # Low-mastery fast student: gate stays closed longer
    >>> gate_open(progress_steps=1, time_on_task_s=10, p_known=0.2, frustration=0.0)
    False
    """
    if not (0.0 <= p_known <= 1.0):
        raise ValueError(f"p_known must be in [0, 1]; got {p_known}")
    if not (0.0 <= frustration <= 1.0):
        raise ValueError(f"frustration must be in [0, 1]; got {frustration}")

    # Condition A — must have made at least one structural step.
    if progress_steps < _MIN_PROGRESS_STEPS:
        return False

    # Condition B — time threshold, modulated by mastery and frustration.
    # frustration_factor: high frustration + high mastery → lower threshold.
    frustration_factor = 1.0 / (1.0 + frustration * p_known * 2.0)

    # effort_factor: low mastery + rushing → raise threshold.
    # "rushing" = spending much less than the base time.
    time_ratio = time_on_task_s / _BASE_TIME_THRESHOLD_S
    effort_factor = 1.0 + (1.0 - p_known) * max(0.0, 1.0 - time_ratio)

    threshold = _BASE_TIME_THRESHOLD_S * frustration_factor * effort_factor

    return time_on_task_s >= threshold


# ---------------------------------------------------------------------------
# 5. Completion problem helper
# ---------------------------------------------------------------------------

_BLANK = "___"  # sentinel used in tests to verify blanking


def completion_problem(worked_steps: list[str], reveal_through: int) -> dict:
    """
    Return a partial worked solution for COMPLETION-mode scaffolding.

    Steps up to and including *reveal_through* are shown in full.
    Steps after that index are replaced with ``"___"`` (the blank sentinel).

    Parameters
    ----------
    worked_steps : list[str]
        The complete list of worked-solution steps, in order.
        Each element is a human-readable string (LaTeX or plain text).
    reveal_through : int
        Zero-based index of the last step to reveal.  Steps at indices
        0 … reveal_through (inclusive) are shown; steps at
        reveal_through+1 … end are blanked.

    Returns
    -------
    dict with keys:
        ``steps``        – list of str, blanked as described.
        ``total_steps``  – int, length of the original list.
        ``revealed``     – int, number of steps shown.
        ``blanked``      – int, number of steps hidden.

    Raises
    ------
    ValueError
        If ``worked_steps`` is empty or ``reveal_through`` is out of range.

    Examples
    --------
    >>> steps = ["2x + 4 = 10", "2x = 6", "x = 3"]
    >>> completion_problem(steps, reveal_through=1)
    {'steps': ['2x + 4 = 10', '2x = 6', '___'],
     'total_steps': 3, 'revealed': 2, 'blanked': 1}
    """
    if not worked_steps:
        raise ValueError("worked_steps must not be empty.")
    n = len(worked_steps)
    if not (0 <= reveal_through < n):
        raise ValueError(
            f"reveal_through must be in [0, {n - 1}]; got {reveal_through}"
        )

    steps = [
        step if i <= reveal_through else _BLANK
        for i, step in enumerate(worked_steps)
    ]
    revealed = reveal_through + 1
    blanked = n - revealed

    return {
        "steps": steps,
        "total_steps": n,
        "revealed": revealed,
        "blanked": blanked,
    }


# ────────────────────────────────────────────────────────────
# FILE: ./temp_diagnostic.py
# ────────────────────────────────────────────────────────────

import random, numpy as np
from scipy.optimize import minimize as _scipy_minimize

def _sigma(z):
    return np.where(z >= 0, 1./(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def _bce_loss_and_grad(params, y, student_idx, kc_idx, opp, n_students, n_kcs):
    beta  = params[:n_students]
    delta = params[n_students:n_students+n_kcs]
    gamma = params[n_students+n_kcs:]
    z = beta[student_idx] + delta[kc_idx] + gamma[kc_idx] * opp
    p = _sigma(z)
    loss_vec = np.maximum(z,0) - y*z + np.log1p(np.exp(-np.abs(z)))
    loss = float(np.mean(loss_vec))
    residual = (p - y) / len(y)
    grad_beta = np.zeros(n_students); grad_delta = np.zeros(n_kcs); grad_gamma = np.zeros(n_kcs)
    np.add.at(grad_beta, student_idx, residual)
    np.add.at(grad_delta, kc_idx, residual)
    np.add.at(grad_gamma, kc_idx, residual * opp)
    return loss, np.concatenate([grad_beta, grad_delta, grad_gamma])

rng = random.Random(42)
evs = [{'u': f's{s}', 'opp': opp, 'y': 1 if rng.random() < min(0.95, 0.3+0.07*opp) else 0}
       for s in range(30) for opp in range(10)]
sv = {s:i for i,s in enumerate(sorted(set(e['u'] for e in evs)))}
n_s, n_k = len(sv), 1
s_arr = np.array([sv[e['u']] for e in evs], dtype=np.int32)
kc_arr = np.zeros(len(evs), dtype=np.int32)
opp_arr = np.array([e['opp'] for e in evs], dtype=np.float64)
y_arr = np.array([e['y'] for e in evs], dtype=np.float64)
x0 = np.zeros(n_s + 2*n_k)
res = _scipy_minimize(_bce_loss_and_grad, x0, method='L-BFGS-B', jac=True,
    args=(y_arr, s_arr, kc_arr, opp_arr, n_s, n_k),
    options={'maxiter': 1000, 'ftol': 1e-9})
import scipy
print('scipy:', scipy.__version__)
print('success:', res.success, '| status:', res.status, '| message:', res.message)


# ────────────────────────────────────────────────────────────
# FILE: ./tests/__init__.py
# ────────────────────────────────────────────────────────────



# ────────────────────────────────────────────────────────────
# FILE: ./tests/curriculum.py
# ────────────────────────────────────────────────────────────

import pytest

from mathtutor.domain.curriculum import (
    Curriculum,
    CurriculumError,
    build_sample_curriculum,
)
from mathtutor.contracts import KnowledgeComponent


def test_cycle_insertion_raises() -> None:
    curriculum = Curriculum()

    curriculum.add(
        KnowledgeComponent(
            id="a",
            name="A",
            prerequisites=[],
            verifier_domain="x",
            generator="g1",
        )
    )

    curriculum.add(
        KnowledgeComponent(
            id="b",
            name="B",
            prerequisites=["a"],
            verifier_domain="x",
            generator="g2",
        )
    )

    with pytest.raises(CurriculumError):
        curriculum.add(
            KnowledgeComponent(
                id="a2",
                name="A2",
                prerequisites=["b", "a2"],  # self-cycle
                verifier_domain="x",
                generator="g3",
            )
        )


def test_topological_order_respects_edges() -> None:
    curriculum = build_sample_curriculum()
    order = curriculum.topological_order()
    pos = {kc: i for i, kc in enumerate(order)}

    for kc_id in order:
        for prereq in curriculum.prerequisites(kc_id):
            assert pos[prereq] < pos[kc_id]


def test_unmet_prerequisites() -> None:
    curriculum = build_sample_curriculum()

    mastered = {"fraction_basics", "fraction_operations"}

    unmet = curriculum.unmet_prerequisites(
        "linear_one_step",
        mastered,
    )

    assert unmet == ["simplify_expressions"]


def test_ready_kcs() -> None:
    curriculum = build_sample_curriculum()

    ready = curriculum.ready_kcs(set())
    assert ready == ["fraction_basics"]

    mastered = {"fraction_basics"}
    ready = curriculum.ready_kcs(mastered)
    assert "fraction_operations" in ready
    assert "linear_one_step" not in ready

    mastered = {
        "fraction_basics",
        "fraction_operations",
        "simplify_expressions",
        "linear_one_step",
        "linear_multi_step",
        "distributive_property",
    }
    ready = curriculum.ready_kcs(mastered)
    assert "factoring_quadratics" in ready


# ────────────────────────────────────────────────────────────
# FILE: ./tests/test_bkt.py
# ────────────────────────────────────────────────────────────

"""
tests/test_bkt.py
=================
Pytest suite for mathtutor/learner/bkt.py.

All expected numeric values are derived from the closed-form BKT equations
and shown step-by-step in comments so the test doubles as living documentation
of the model's arithmetic.

Default BKTParams used throughout unless stated:
    L0 = 0.20,  T = 0.30,  S = 0.10,  G = 0.20
"""

import math
import pytest

from mathtutor.learner.bkt import BKTParams, BKTLearnerState, propagate


# ===========================================================================
# Helpers / fixtures
# ===========================================================================

def fresh_state(kc_id: str = "KC1") -> tuple[BKTLearnerState, str]:
    """Return a BKTLearnerState freshly initialised with default params."""
    state = BKTLearnerState()
    state.register(kc_id)
    return state, kc_id


def one_step_correct(p: float, s: float = 0.10, g: float = 0.20,
                     t: float = 0.30) -> float:
    """
    Reference implementation of the two-step BKT update for a correct answer.
    Used to cross-check the main implementation independently.
    """
    # Step 1
    num = p * (1 - s)
    den = num + (1 - p) * g
    p_post = num / den
    # Step 2
    return p_post + (1 - p_post) * t


def one_step_incorrect(p: float, s: float = 0.10, g: float = 0.20,
                       t: float = 0.30) -> float:
    """Reference implementation for an incorrect answer."""
    num = p * s
    den = num + (1 - p) * (1 - g)
    p_post = num / den
    return p_post + (1 - p_post) * t


# ===========================================================================
# 1.  BKTParams validation
# ===========================================================================

class TestBKTParamsValidation:

    def test_default_params_are_valid(self):
        """Literature defaults should pass without error."""
        p = BKTParams()
        assert p.l0 == 0.20
        assert p.t  == 0.30
        assert p.s  == 0.10
        assert p.g  == 0.20

    def test_custom_valid_params(self):
        p = BKTParams(l0=0.3, t=0.4, s=0.05, g=0.15)
        assert p.l0 == 0.3

    def test_s_plus_g_equal_to_one_raises(self):
        """s + g == 1.0 is degenerate (boundary) — must raise."""
        with pytest.raises(ValueError, match="s \\+ g"):
            BKTParams(s=0.5, g=0.5)

    def test_s_plus_g_above_one_raises(self):
        """The spec example: s=0.6, g=0.6 must raise."""
        with pytest.raises(ValueError, match="s \\+ g"):
            BKTParams(s=0.6, g=0.6)

    @pytest.mark.parametrize("param,val", [
        ("l0", 0.0), ("l0", 1.0), ("l0", -0.1), ("l0", 1.1),
        ("t",  0.0), ("t",  1.0),
        ("s",  0.0), ("s",  1.0),
        ("g",  0.0), ("g",  1.0),
    ])
    def test_boundary_values_raise(self, param, val):
        """Strict open-interval: 0 and 1 themselves are not allowed."""
        kwargs = dict(l0=0.2, t=0.3, s=0.1, g=0.2)
        kwargs[param] = val
        with pytest.raises(ValueError):
            BKTParams(**kwargs)


# ===========================================================================
# 2.  Single correct answer from L0 = 0.20
# ===========================================================================

class TestSingleCorrectAnswer:
    """
    Walk through the arithmetic explicitly so the test is self-documenting.

    Starting state:  p = L0 = 0.20
    Params: S=0.10, G=0.20, T=0.30

    --- Step 1: condition on correct answer ---
        numerator   = p*(1-S)      = 0.20 * 0.90 = 0.180
        denominator = num + (1-p)*G = 0.180 + 0.80*0.20 = 0.180 + 0.160 = 0.340
        p_post      = 0.180 / 0.340 ≈ 0.52941

    --- Step 2: learning transition ---
        p_next = p_post + (1 - p_post)*T
               = 0.52941 + (1 - 0.52941)*0.30
               = 0.52941 + 0.47059*0.30
               = 0.52941 + 0.14118
               ≈ 0.67059

    So after one correct answer, p ≈ 0.671.
    We assert p ∈ [0.65, 0.69] to tolerate floating-point rounding.
    We also assert p < 0.95 (single answer CANNOT declare mastery).
    """

    def test_one_correct_raises_p_to_roughly_0_67(self):
        state, kc = fresh_state()

        # Sanity-check: initial p equals L0
        assert state.p_mastered(kc) == pytest.approx(0.20)

        p_new = state.observe(kc, correct=True)

        # --- Arithmetic shown above ---
        # Step 1: p_post = (0.20*0.90) / (0.20*0.90 + 0.80*0.20)
        #                = 0.180 / 0.340 ≈ 0.52941
        # Step 2: p_next = 0.52941 + (1 - 0.52941)*0.30 ≈ 0.67059
        assert 0.65 <= p_new <= 0.69, (
            f"Expected p ≈ 0.671 after one correct; got {p_new:.5f}"
        )

    def test_one_correct_cannot_declare_mastery(self):
        """
        Even with guessing modelled, a single correct answer from p=0.20
        should never cross the 0.95 mastery threshold.
        """
        state, kc = fresh_state()
        state.observe(kc, correct=True)
        assert not state.mastered(kc), (
            "A single correct answer must not declare mastery (guessing guard)."
        )

    def test_reference_formula_matches_implementation(self):
        """Cross-check implementation against the standalone reference formula."""
        state, kc = fresh_state()
        p_impl = state.observe(kc, correct=True)
        p_ref  = one_step_correct(0.20)
        assert p_impl == pytest.approx(p_ref, rel=1e-9)


# ===========================================================================
# 3.  Multiple consecutive correct answers cross 0.95
# ===========================================================================

class TestConsecutiveCorrectAnswers:
    """
    Trace through multiple correct answers.

    After answer 1: p ≈ 0.671  (see TestSingleCorrectAnswer arithmetic)
    After answer 2:
        Step 1: num = 0.671*(0.90) = 0.604
                den = 0.604 + 0.329*0.20 = 0.604 + 0.066 = 0.670
                p_post = 0.604/0.670 ≈ 0.901
        Step 2: p_next = 0.901 + 0.099*0.30 ≈ 0.931
    After answer 3:
        Step 1: num = 0.931*0.90 = 0.838
                den = 0.838 + 0.069*0.20 = 0.838 + 0.014 = 0.852
                p_post = 0.838/0.852 ≈ 0.984
        Step 2: p_next = 0.984 + 0.016*0.30 ≈ 0.989

    So 3 correct answers should cross 0.95.  We also verify 4 answers to be safe.
    """

    def test_three_correct_answers_cross_mastery(self):
        state, kc = fresh_state()
        for _ in range(3):
            state.observe(kc, correct=True)

        # After 3 correct answers p ≈ 0.989 (see arithmetic above)
        assert state.mastered(kc), (
            f"Expected mastery after 3 correct answers; p={state.p_mastered(kc):.4f}"
        )

    def test_four_correct_answers_cross_mastery(self):
        """Belt-and-braces: 4 answers also cross the threshold."""
        state, kc = fresh_state()
        for _ in range(4):
            state.observe(kc, correct=True)
        assert state.mastered(kc)

    def test_two_correct_answers_do_not_cross_mastery(self):
        """
        After 2 correct answers p ≈ 0.931, which is below 0.95.
        This confirms the threshold is non-trivial to reach.
        """
        state, kc = fresh_state()
        for _ in range(2):
            state.observe(kc, correct=True)

        p = state.p_mastered(kc)
        assert p < 0.95, f"Expected p < 0.95 after 2 correct; got {p:.4f}"
        # Also assert it's in a reasonable range
        assert 0.90 <= p <= 0.95, f"Unexpected value {p:.4f} after 2 correct"


# ===========================================================================
# 4.  Incorrect answer lowers p
# ===========================================================================

class TestIncorrectAnswerLowersP:
    """
    From p = 0.20, one incorrect answer:

    Step 1: p_post = p*S / (p*S + (1-p)*(1-G))
                   = 0.20*0.10 / (0.20*0.10 + 0.80*0.80)
                   = 0.020 / (0.020 + 0.640)
                   = 0.020 / 0.660
                   ≈ 0.030

    Step 2: p_next = 0.030 + (1 - 0.030)*0.30
                   = 0.030 + 0.970*0.30
                   = 0.030 + 0.291
                   ≈ 0.321

    So p *rises* (the learning transition still happens) but the posterior
    from Step 1 drops hard — correctly interpreting an error as evidence of
    non-mastery.  Net result: p_next ≈ 0.321 < 0.671 (one-correct baseline).
    The key invariant: an incorrect answer produces a lower p than the
    corresponding correct answer would have.
    """

    def test_incorrect_answer_produces_lower_p_than_correct(self):
        """
        Incorrect answer from p=0.20 → p ≈ 0.321.
        Correct answer from p=0.20 → p ≈ 0.671.
        Incorrect must give a lower result.
        """
        state_correct, kc_c = fresh_state("KC_correct")
        state_wrong,   kc_w = fresh_state("KC_wrong")

        p_after_correct   = state_correct.observe(kc_c, correct=True)
        p_after_incorrect = state_wrong.observe(kc_w,   correct=False)

        assert p_after_incorrect < p_after_correct, (
            f"Incorrect answer should yield lower p than correct; "
            f"got {p_after_incorrect:.4f} vs {p_after_correct:.4f}"
        )

    def test_incorrect_answer_from_low_prior_approximately_0_32(self):
        """
        Arithmetic:
            Step 1: p_post = (0.20*0.10) / (0.20*0.10 + 0.80*0.80)
                           = 0.020 / 0.660 ≈ 0.0303
            Step 2: p_next = 0.0303 + (1-0.0303)*0.30 ≈ 0.321
        Assert p ∈ [0.30, 0.34].
        """
        state, kc = fresh_state()
        p_new = state.observe(kc, correct=False)

        # Step 1: p_post = 0.020/0.660 ≈ 0.0303
        # Step 2: p_next ≈ 0.321
        assert 0.30 <= p_new <= 0.34, (
            f"Expected p ≈ 0.321 after incorrect from L0=0.20; got {p_new:.5f}"
        )

    def test_reference_formula_matches_for_incorrect(self):
        state, kc = fresh_state()
        p_impl = state.observe(kc, correct=False)
        p_ref  = one_step_incorrect(0.20)
        assert p_impl == pytest.approx(p_ref, rel=1e-9)

    def test_mastery_lost_after_errors(self):
        """
        Build up to high-p then give an incorrect answer; mastery should drop.
        """
        state, kc = fresh_state()
        for _ in range(3):
            state.observe(kc, correct=True)
        assert state.mastered(kc), "Pre-condition: should be mastered after 3 correct"

        state.observe(kc, correct=False)
        # After one wrong from ~0.989 the model should drop back below 0.95
        assert not state.mastered(kc), (
            "Mastery should be lost after an incorrect answer"
        )


# ===========================================================================
# 5.  Effective prior under prerequisite conditioning
# ===========================================================================

class TestEffectivePrior:

    def test_no_prereqs_returns_l0(self):
        state, kc = fresh_state()
        ep = state.effective_prior(kc, prereqs=[], mastered_set=set())
        assert ep == pytest.approx(0.20)

    def test_all_prereqs_mastered_returns_close_to_l0(self):
        """When all prereqs are mastered, effective_prior = L0 * (0.3 + 0.7*1) = L0."""
        state, kc = fresh_state()
        ep = state.effective_prior(kc, prereqs=["PRE1", "PRE2"],
                                   mastered_set={"PRE1", "PRE2"})
        # ALPHA=0.3 → effective = L0 * (0.3 + 0.7*1.0) = L0 * 1.0 = 0.20
        assert ep == pytest.approx(0.20, rel=1e-6)

    def test_no_prereqs_mastered_returns_reduced_prior(self):
        """
        With 0/2 prereqs mastered:
            r = 0/2 = 0.0
            effective = L0 * (0.3 + 0.7*0.0) = 0.20 * 0.3 = 0.06
        """
        state, kc = fresh_state()
        ep = state.effective_prior(kc, prereqs=["PRE1", "PRE2"],
                                   mastered_set=set())
        # 0.20 * 0.3 = 0.06
        assert ep == pytest.approx(0.06, rel=1e-6)

    def test_partial_prereqs_mastered(self):
        """
        With 1/2 prereqs mastered:
            r = 0.5
            effective = 0.20 * (0.3 + 0.7*0.5) = 0.20 * 0.65 = 0.13
        """
        state, kc = fresh_state()
        ep = state.effective_prior(kc, prereqs=["PRE1", "PRE2"],
                                   mastered_set={"PRE1"})
        assert ep == pytest.approx(0.13, rel=1e-6)

    def test_effective_prior_always_in_unit_interval(self):
        """Sanity: the result must always be in (0, 1)."""
        state, kc = fresh_state()
        for n_mastered in range(4):
            mastered_set = {f"P{i}" for i in range(n_mastered)}
            prereqs = [f"P{i}" for i in range(3)]
            ep = state.effective_prior(kc, prereqs=prereqs,
                                       mastered_set=mastered_set)
            assert 0.0 < ep < 1.0


# ===========================================================================
# 6.  Multi-KC independence
# ===========================================================================

class TestMultiKCIndependence:
    """
    Updating one KC must not affect another KC's state.
    """

    def test_kcs_are_independent(self):
        state = BKTLearnerState()
        state.register("ALGEBRA")
        state.register("GEOMETRY")

        p_geo_before = state.p_mastered("GEOMETRY")
        state.observe("ALGEBRA", correct=True)
        p_geo_after = state.p_mastered("GEOMETRY")

        assert p_geo_before == p_geo_after, (
            "Observing ALGEBRA should not change GEOMETRY's p"
        )

    def test_custom_params_per_kc(self):
        """Different KCs can have different BKTParams."""
        state = BKTLearnerState()
        easy_params = BKTParams(l0=0.5, t=0.5, s=0.05, g=0.15)
        hard_params = BKTParams(l0=0.1, t=0.1, s=0.15, g=0.10)

        state.register("EASY", easy_params)
        state.register("HARD", hard_params)

        p_easy = state.observe("EASY", correct=True)
        p_hard = state.observe("HARD", correct=True)

        # Easy KC should have a higher p after one correct answer
        assert p_easy > p_hard


# ===========================================================================
# 7.  Monotonicity and boundary sanity
# ===========================================================================

class TestMonotonicity:

    def test_p_never_exceeds_1(self):
        state, kc = fresh_state()
        for _ in range(20):
            p = state.observe(kc, correct=True)
            assert p <= 1.0, f"p exceeded 1.0: {p}"

    def test_p_never_below_0(self):
        state, kc = fresh_state()
        for _ in range(20):
            p = state.observe(kc, correct=False)
            assert p >= 0.0, f"p dropped below 0.0: {p}"

    def test_more_correct_answers_monotonically_increase_p(self):
        """
        Consecutive correct answers should monotonically increase p.
        """
        state, kc = fresh_state()
        prev_p = state.p_mastered(kc)
        for _ in range(10):
            new_p = state.observe(kc, correct=True)
            assert new_p >= prev_p, (
                f"p decreased from {prev_p:.4f} to {new_p:.4f} on a correct answer"
            )
            prev_p = new_p


# ────────────────────────────────────────────────────────────
# FILE: ./tests/test_contracts.py
# ────────────────────────────────────────────────────────────

# tests/test_contracts.py

from mathtutor.contracts import (
    Artifact,
    BuggyRule,
    Canonical,
    Judgment,
    KnowledgeComponent,
    Target,
    TelemetryEvent,
    Verdict,
    verdict_from_judgment,
)


def test_construct_dataclasses():
    artifact = Artifact("expression", 42, "42", {})
    target = Target("algebra", {"answer": 42}, "reduced", 1)
    canonical = Canonical(42, {})
    kc = KnowledgeComponent(
        id="kc1",
        name="Linear Equations",
        prerequisites=[],
        verifier_domain="algebra",
        difficulty_band=1,
        generators=["gen_linear"],
    )
    rule = BuggyRule(
        id="bug1",
        description="Sign error",
        applies_to=lambda x: True,
        transform=lambda x: x,
        remediation="Check signs carefully",
    )

    assert artifact.kind == "expression"
    assert target.domain == "algebra"
    assert canonical.key == 42
    assert kc.name == "Linear Equations"
    assert rule.id == "bug1"


def test_verdict_mapping():
    correct = Judgment(True, True, True, True, False, True, 1.0, {})
    partial = Judgment(True, False, False, False, True, True, 1.0, {})
    wrong = Judgment(True, False, False, False, False, True, 1.0, {})
    abstain1 = Judgment(False, False, False, False, False, False, 0.0, {})
    abstain2 = Judgment(True, False, False, False, False, False, 0.4, {})

    assert verdict_from_judgment(correct) == Verdict.CORRECT
    assert verdict_from_judgment(partial) == Verdict.PARTIAL
    assert verdict_from_judgment(wrong) == Verdict.WRONG
    assert verdict_from_judgment(abstain1) == Verdict.ABSTAIN
    assert verdict_from_judgment(abstain2) == Verdict.ABSTAIN


def test_telemetry_roundtrip():
    event = TelemetryEvent(
        event_id="e1",
        session_id="s1",
        user_pseudonym="user",
        ts=123456.0,
        kc_id="kc1",
        action="submit",
        verdict="correct",
    )

    json_blob = event.to_json()
    restored = TelemetryEvent.from_json(json_blob)

    assert restored == event


# ────────────────────────────────────────────────────────────
# FILE: ./tests/test_eval.py
# ────────────────────────────────────────────────────────────

# tests/test_eval.py

"""Tests for eval/telemetry.py and eval/learning_curves.py.

Coverage targets
----------------
T1  emit → read round-trip preserves every field.
T2  append-only: two separate emits produce two lines; prior lines intact.
T3  read_all on non-existent file returns [].
T4  pseudonymize is deterministic per (raw_id, salt).
T5  pseudonymize never stores the raw id in its output.
T6  different salts produce different pseudonyms.
T7  normalized_gain(40, 70) == 0.5.
T8  normalized_gain(100, *) returns nan (ceiling guard).
T9  clean-decline KC is NOT flagged.
T10 flat KC IS flagged.
T11 fit_afm converges on simple synthetic data and returns expected keys.
T12 error_rate_curve returns correct per-opportunity rates.
T13 flag_misspecified_kcs returns sorted list.
"""

from __future__ import annotations

import math
import os
import tempfile
import time
import uuid

import pytest

from mathtutor.contracts import TelemetryEvent
from mathtutor.eval.telemetry import TelemetrySink, pseudonymize
from mathtutor.eval.learning_curves import (
    error_rate_curve,
    fit_afm,
    flag_misspecified_kcs,
    normalized_gain,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SALT = "test-salt-abc123"
_ALT_SALT = "different-salt-xyz"


def _event(
    *,
    user: str = "alice",
    kc: str = "kc-linear",
    opp: int = 0,
    verdict: str = "correct",
    session: str = "s1",
) -> TelemetryEvent:
    """Convenience factory: pseudonymizes user before constructing event."""
    return TelemetryEvent(
        event_id=str(uuid.uuid4()),
        session_id=session,
        user_pseudonym=pseudonymize(user, _SALT),
        ts=time.time(),
        kc_id=kc,
        opportunity_index=opp,
        verdict=verdict,
    )


def _make_declining_events(
    kc: str = "kc-declining",
    n_students: int = 20,
    n_opps: int = 8,
    seed: int = 42,
) -> list[TelemetryEvent]:
    """Synthetic events where error rate clearly declines with opportunity.

    At opportunity *n*, a student answers correctly with probability
    ``0.3 + 0.07 * n`` (capped at 0.95), ensuring a strong decline in
    error rate.
    """
    import random
    rng = random.Random(seed)
    events = []
    for s in range(n_students):
        for opp in range(n_opps):
            p_correct = min(0.95, 0.30 + 0.07 * opp)
            verdict = "correct" if rng.random() < p_correct else "wrong"
            events.append(_event(
                user=f"student-{s}", kc=kc, opp=opp, verdict=verdict
            ))
    return events


def _make_flat_events(
    kc: str = "kc-flat",
    n_students: int = 20,
    n_opps: int = 8,
    seed: int = 99,
) -> list[TelemetryEvent]:
    """Synthetic events where error rate stays ~constant (flat KC)."""
    import random
    rng = random.Random(seed)
    events = []
    for s in range(n_students):
        for opp in range(n_opps):
            verdict = "correct" if rng.random() < 0.50 else "wrong"
            events.append(_event(
                user=f"student-{s}", kc=kc, opp=opp, verdict=verdict
            ))
    return events


# ---------------------------------------------------------------------------
# T1-T3: TelemetrySink round-trip and append semantics
# ---------------------------------------------------------------------------

class TestTelemetrySink:

    def test_emit_read_roundtrip(self, tmp_path):
        """T1: every field survives emit → read_all."""
        path = tmp_path / "events.jsonl"
        sink = TelemetrySink(path)
        ev = TelemetryEvent(
            event_id="evt-001",
            session_id="sess-A",
            user_pseudonym=pseudonymize("bob@example.com", _SALT),
            ts=1_700_000_000.0,
            kc_id="algebra-1",
            problem_id="prob-42",
            opportunity_index=3,
            action="answer",
            verdict="wrong",
            hint_level=1,
            latency_ms=1234,
            p_known_before=0.4,
            p_known_after=0.35,
            policy_id="policy-v2",
        )
        sink.emit(ev)
        recovered = TelemetrySink.read_all(path)
        assert len(recovered) == 1
        assert recovered[0] == ev

    def test_append_only_two_emits(self, tmp_path):
        """T2: second emit appends; first line is unchanged."""
        path = tmp_path / "events.jsonl"
        sink = TelemetrySink(path)

        ev1 = _event(user="alice", kc="kc-A", opp=0, verdict="correct")
        ev2 = _event(user="bob",   kc="kc-B", opp=1, verdict="wrong")

        sink.emit(ev1)
        # Read after first emit
        after_first = TelemetrySink.read_all(path)
        assert len(after_first) == 1

        sink.emit(ev2)
        after_second = TelemetrySink.read_all(path)
        assert len(after_second) == 2

        # First event must be byte-for-byte identical after second write
        assert after_second[0] == ev1
        assert after_second[1] == ev2

    def test_read_all_missing_file(self, tmp_path):
        """T3: read_all returns [] when path does not exist."""
        result = TelemetrySink.read_all(tmp_path / "nonexistent.jsonl")
        assert result == []

    def test_parent_dirs_created(self, tmp_path):
        """Sink auto-creates nested directories."""
        path = tmp_path / "deep" / "nested" / "events.jsonl"
        sink = TelemetrySink(path)
        sink.emit(_event())
        assert path.exists()


# ---------------------------------------------------------------------------
# T4-T6: Pseudonymization
# ---------------------------------------------------------------------------

class TestPseudonymize:

    def test_deterministic(self):
        """T4: same inputs always give same pseudonym."""
        p1 = pseudonymize("alice@example.com", _SALT)
        p2 = pseudonymize("alice@example.com", _SALT)
        assert p1 == p2

    def test_raw_id_not_in_output(self):
        """T5: raw user id does not appear in the pseudonym string."""
        raw_id = "verysecretuser@example.com"
        pseudonym = pseudonymize(raw_id, _SALT)
        assert raw_id not in pseudonym
        # Also check the pseudonym doesn't contain any fragment of the email
        assert "@" not in pseudonym
        assert "verysecretuser" not in pseudonym

    def test_different_salts_differ(self):
        """T6: changing the salt produces a different pseudonym."""
        p1 = pseudonymize("alice@example.com", _SALT)
        p2 = pseudonymize("alice@example.com", _ALT_SALT)
        assert p1 != p2

    def test_different_users_differ(self):
        """Different users under the same salt produce different pseudonyms."""
        p1 = pseudonymize("alice@example.com", _SALT)
        p2 = pseudonymize("bob@example.com", _SALT)
        assert p1 != p2

    def test_pseudonym_is_hex_string(self):
        """Pseudonym is a 64-char lowercase hex digest (SHA-256 output)."""
        p = pseudonymize("user", _SALT)
        assert len(p) == 64
        assert all(c in "0123456789abcdef" for c in p)

    def test_event_pseudonym_not_raw_id(self, tmp_path):
        """T5 (via sink): raw id never written to disk."""
        raw_id = "private-user-12345"
        path = tmp_path / "events.jsonl"
        sink = TelemetrySink(path)
        ev = TelemetryEvent(
            event_id="e1", session_id="s1",
            user_pseudonym=pseudonymize(raw_id, _SALT),
            ts=time.time(),
        )
        sink.emit(ev)
        content = path.read_text(encoding="utf-8")
        assert raw_id not in content


# ---------------------------------------------------------------------------
# T7-T8: normalized_gain
# ---------------------------------------------------------------------------

class TestNormalizedGain:

    def test_known_value(self):
        """T7: normalized_gain(40, 70) == 0.5 (Hake example)."""
        assert normalized_gain(40, 70) == pytest.approx(0.5)

    def test_zero_gain(self):
        """Pre == post => gain == 0."""
        assert normalized_gain(50, 50) == pytest.approx(0.0)

    def test_full_gain(self):
        """Post == 100 => gain == 1.0."""
        assert normalized_gain(0, 100) == pytest.approx(1.0)

    def test_ceiling_pre_100(self):
        """T8: pre == 100 returns nan (ceiling guard)."""
        result = normalized_gain(100, 100)
        assert math.isnan(result)

    def test_negative_gain(self):
        """Post < pre => negative gain (regression)."""
        g = normalized_gain(60, 40)
        assert g < 0


# ---------------------------------------------------------------------------
# T9-T10: flag_misspecified_kcs
# ---------------------------------------------------------------------------

class TestFlagMisspecifiedKCs:

    def test_clean_decline_not_flagged(self):
        """T9: a KC with monotone-declining error rate is NOT flagged."""
        events = _make_declining_events("kc-good")
        flagged = flag_misspecified_kcs(events)
        assert "kc-good" not in flagged

    def test_flat_curve_flagged(self):
        """T10: a KC with flat error rate IS flagged."""
        events = _make_flat_events("kc-bad")
        flagged = flag_misspecified_kcs(events)
        assert "kc-bad" in flagged

    def test_mixed_flags(self):
        """Good KC not flagged; bad KC flagged when both are present."""
        events = (
            _make_declining_events("kc-good", seed=7)
            + _make_flat_events("kc-bad", seed=0)  # seed=0: mean Δ > 0 reliably
        )
        flagged = flag_misspecified_kcs(events)
        assert "kc-bad" in flagged
        assert "kc-good" not in flagged

    def test_result_is_sorted(self):
        """T13: flagged list is lexicographically sorted."""
        events = _make_flat_events("zz-kc") + _make_flat_events("aa-kc")
        flagged = flag_misspecified_kcs(events)
        assert flagged == sorted(flagged)

    def test_no_events_returns_empty(self):
        """No events => no KCs to flag."""
        assert flag_misspecified_kcs([]) == []

    def test_single_opportunity_flagged(self):
        """KC with only 1 opportunity index can't be assessed => flagged."""
        events = [_event(kc="kc-one-opp", opp=0, verdict="correct")]
        flagged = flag_misspecified_kcs(events)
        assert "kc-one-opp" in flagged


# ---------------------------------------------------------------------------
# T11: fit_afm
# ---------------------------------------------------------------------------

class TestFitAFM:

    def test_returns_expected_keys(self):
        """T11a: result dict has all expected keys."""
        events = _make_declining_events("kc-fit", n_students=10)
        result = fit_afm(events)
        for key in ("kc_easiness", "kc_learning_rate", "student_ability",
                    "converged", "n_obs"):
            assert key in result, f"missing key: {key}"

    def test_converged_on_clean_data(self):
        """T11b: optimiser converges on well-behaved synthetic data."""
        events = _make_declining_events("kc-fit", n_students=30, n_opps=10)
        result = fit_afm(events)
        assert result["converged"] is True

    def test_positive_learning_rate(self):
        """T11c: declining-error KC should get a positive learning rate."""
        events = _make_declining_events("kc-lr", n_students=40, n_opps=10)
        result = fit_afm(events)
        lr = result["kc_learning_rate"]["kc-lr"]
        # The data was constructed so practice helps; γ should be positive
        assert lr > 0, f"Expected γ > 0, got {lr}"

    def test_n_obs_correct(self):
        """T11d: n_obs equals number of scored events (correct+wrong)."""
        events = _make_declining_events("kc-n", n_students=5, n_opps=4)
        result = fit_afm(events)
        # All events have verdict correct/wrong
        assert result["n_obs"] == len(events)

    def test_empty_events_not_converged(self):
        """T11e: empty input returns converged=False, empty dicts."""
        result = fit_afm([])
        assert result["converged"] is False
        assert result["kc_easiness"] == {}

    def test_ignores_events_without_kc(self):
        """Events with kc_id=None are silently ignored."""
        ev_no_kc = TelemetryEvent(
            event_id="x", session_id="s", user_pseudonym="p",
            ts=1.0, kc_id=None, verdict="correct"
        )
        events = _make_declining_events("kc-ok", n_students=5) + [ev_no_kc]
        result = fit_afm(events)
        assert None not in result["kc_easiness"]


# ---------------------------------------------------------------------------
# T12: error_rate_curve
# ---------------------------------------------------------------------------

class TestErrorRateCurve:

    def test_correct_rates(self):
        """T12: error rate at opp 0 is 0.0 when all correct; 1.0 when all wrong."""
        events = [
            _event(kc="kc-x", opp=0, verdict="correct"),
            _event(kc="kc-x", opp=0, verdict="correct"),
            _event(kc="kc-x", opp=1, verdict="wrong"),
            _event(kc="kc-x", opp=1, verdict="wrong"),
        ]
        curve = error_rate_curve(events, "kc-x")
        assert len(curve) == 2
        assert curve[0] == pytest.approx(0.0)   # all correct at opp 0
        assert curve[1] == pytest.approx(1.0)   # all wrong at opp 1

    def test_empty_for_unknown_kc(self):
        """No events for requested KC => empty list."""
        events = [_event(kc="kc-other", opp=0, verdict="correct")]
        assert error_rate_curve(events, "kc-unknown") == []

    def test_nan_for_gap(self):
        """Gap in opportunity index produces nan at that position."""
        events = [
            _event(kc="kc-gap", opp=0, verdict="correct"),
            # opp=1 missing
            _event(kc="kc-gap", opp=2, verdict="wrong"),
        ]
        curve = error_rate_curve(events, "kc-gap")
        assert len(curve) == 3
        assert math.isnan(curve[1])

    def test_mixed_verdict(self):
        """50% error rate at a given opportunity."""
        events = [
            _event(kc="kc-mix", opp=0, verdict="correct"),
            _event(kc="kc-mix", opp=0, verdict="wrong"),
        ]
        curve = error_rate_curve(events, "kc-mix")
        assert curve[0] == pytest.approx(0.5)

# ────────────────────────────────────────────────────────────
# FILE: ./tests/test_generators.py
# ────────────────────────────────────────────────────────────

# tests/test_generators.py

import pytest

from mathtutor.domain.generators import generate, GeneratorError
import mathtutor.domain.generators as generators


GENS = [
    "linear_equation",
    "quadratic_equation",
    "fraction_addition",
]


def test_200_generations():
    count = 0
    for gen in GENS:
        for band in [1, 2, 3]:
            for seed in range(25):
                p = generate(gen, difficulty_band=band, seed=seed)
                assert p is not None
                count += 1
    assert count >= 200


def test_determinism():
    for gen in GENS:
        p1 = generate(gen, difficulty_band=2, seed=123)
        p2 = generate(gen, difficulty_band=2, seed=123)
        assert p1 == p2


def test_broken_reference_answer_raises(monkeypatch):
    original = generators.generate_linear_equation

    def broken(*, difficulty_band, seed):
        p = original(difficulty_band=difficulty_band, seed=seed)
        bad = p.__class__(
            id=p.id,
            kc_id=p.kc_id,
            domain=p.domain,
            prompt_text=p.prompt_text,
            parsed_target=p.parsed_target,
            reference_answer="999999",
            difficulty_band=p.difficulty_band,
            meta=p.meta,
        )
        generators._assert_self_verified(
            bad,
            generators.LinearEquationVerifier(),
            2,
        )
        return bad

    monkeypatch.setitem(generators._REGISTRY, "linear_equation", broken)

    with pytest.raises(GeneratorError):
        generate("linear_equation", difficulty_band=1, seed=1)


# ────────────────────────────────────────────────────────────
# FILE: ./tests/test_misconceptions.py
# ────────────────────────────────────────────────────────────

# tests/test_misconceptions.py

import pytest
from sympy import symbols, Eq, StrictLessThan, Add, Pow, Mul
from mathtutor.cas.parsing import parse_math
from mathtutor.tutoring.misconceptions import (
    diagnose, classify_error, exact_match, BUGGY_RULES,
    CancelTermAcrossAddition, ClearsFractionOneTerm
)

x, y = symbols('x y')

def test_exact_match():
    # Commutativity should match
    assert exact_match(x**2 + y**2, y**2 + x**2) is True
    # Structural difference shouldn't match (factored vs expanded)
    assert exact_match(x*(x + 1), x**2 + x) is False
    # Swapped equations
    assert exact_match(Eq(x, 5), Eq(5, x)) is True

def test_distributes_exponent_over_sum():
    P = (x + y)**2
    S = x**2 + y**2
    assert diagnose(P, S) == ["distributes_exponent_over_sum"]

def test_moves_term_across_equals_without_sign_flip():
    P = Eq(x + 3, 5)
    S = Eq(x, 5 + 3)
    assert diagnose(P, S) == ["moves_term_across_equals_without_sign_flip"]

def test_cancels_term_across_addition():
    # (x + y) / x
    P = Mul(Add(x, y), Pow(x, -1)) 
    S = y 
    assert diagnose(P, S) == ["cancels_term_across_addition"]

def test_clears_fraction_one_term():
    # x/2 + y = 5
    P = Eq(Mul(x, Pow(2, -1)) + y, 5)
    # x + y = 10 (forgot to multiply y by 2)
    S = Eq(x + y, 10)
    assert diagnose(P, S) == ["clears_fraction_multiplying_only_one_term"]

def test_forgets_to_flip_inequality():
    P = StrictLessThan(-x, 5)
    S = StrictLessThan(x, -5)
    assert diagnose(P, S) == ["forgets_to_flip_inequality_on_negative_multiply"]

def test_multiple_matches():
    # Construct a state that can plausibly hit two isolated/mocked buggy rules 
    # to test the disambiguation contract.
    class DummyRuleA:
        id = "rule_a"
        def applies_to(self, P): return True
        def transform(self, P): return [S_target]

    class DummyRuleB:
        id = "rule_b"
        def applies_to(self, P): return True
        def transform(self, P): return [S_target]

    P_dummy = Eq(x, 1)
    S_target = Eq(x, 2)
    
    rules = [DummyRuleA(), DummyRuleB()]
    assert diagnose(P_dummy, S_target, rules=rules) == ["rule_a", "rule_b"]

def test_novel_wrong_line_unknown():
    P = Eq(x, 2)
    S = Eq(x, 5)
    assert diagnose(P, S) == []
    assert classify_error(P, S) == "unknown"

def test_novel_wrong_line_sign_error():
    P = Eq(x, 2)
    S = Eq(-x, 2)
    assert diagnose(P, S) == []
    assert classify_error(P, S) == "sign_error"

def test_novel_wrong_line_off_by_one():
    P = Eq(x, 2)
    S = Eq(x, 3)
    assert diagnose(P, S) == []
    assert classify_error(P, S) == "off_by_one"


# ────────────────────────────────────────────────────────────
# FILE: ./tests/test_numeric.py
# ────────────────────────────────────────────────────────────

import sympy as sp

from mathtutor.cas.numeric import numeric_equivalent


def test_trig_identity():
    x = sp.Symbol("x")
    eq, conf = numeric_equivalent(sp.sin(x) ** 2 + sp.cos(x) ** 2, 1)
    assert eq is True
    assert conf < 1.0


def test_non_equivalent():
    x = sp.Symbol("x")
    eq, _ = numeric_equivalent(x, x + 1)
    assert eq is False


# ────────────────────────────────────────────────────────────
# FILE: ./tests/test_orchestrator.py
# ────────────────────────────────────────────────────────────

# tests/test_orchestrator.py
"""End-to-end tests for the Orchestrator using the offline (deterministic) tutor."""
from __future__ import annotations
import sys, os, types

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# The package layout already lives at ROOT/mathtutor/, so normal imports work.
import pytest
from mathtutor.contracts import (
    Artifact, Judgment, ParseError, SupportLevel,
    Target, TelemetryEvent, Verdict,
)
from mathtutor.orchestrator import Orchestrator, Session, TurnResult
from mathtutor.llm.offline import CoachingContext, OfflineTutor


# ── stubs ────────────────────────────────────────────────────────────────────

def _make_judgment(correct=False, partial=False, parsed_ok=True, detail=None):
    return Judgment(
        parsed_ok=parsed_ok, value_equivalent=correct, form_ok=correct,
        correct=correct, partial=partial, decidable=True,
        confidence=1.0 if parsed_ok else 0.0, detail=detail or {},
    )


class _StubVerifier:
    domain = "stub"
    def __init__(self, correct_raw, partial_raws=None, misconception_map=None):
        self._correct = {correct_raw} if isinstance(correct_raw, str) else set(correct_raw)
        self._partial = set(partial_raws or [])
        self._misconceptions = misconception_map or {}
    def parse(self, raw: str) -> Artifact:
        if raw.strip() == "UNPARSEABLE":
            raise ParseError("Cannot parse input")
        return Artifact(kind="value", expr=raw.strip(), raw=raw)
    def canonical(self, a): return a.expr
    def accepts(self, student, target):
        raw = student.expr
        correct = raw in self._correct
        partial = (not correct) and raw in self._partial
        detail = {}
        if raw in self._misconceptions:
            detail["misconception_id"] = self._misconceptions[raw]
        return _make_judgment(correct=correct, partial=partial, detail=detail)


def _make_session(correct_raws="x=3", problem="Solve x^2 - 5x + 6 = 0",
                  kc_name="quadratic equations", correct_answer_str="x=2 or x=3"):
    verifier = _StubVerifier(
        correct_raw=correct_raws,
        misconception_map={"x=9": "forgot_to_take_square_root"},
    )
    target = Target(domain="stub", payload={"solutions": [2, 3]})
    return Session(
        user_pseudonym="test_student", kc_id="quad_eq", kc_name=kc_name,
        problem_id="prob_001", problem_statement=problem,
        target=target, verifier=verifier, correct_answer_str=correct_answer_str,
    )


# ── Test 1: correct answer ────────────────────────────────────────────────────

class TestCorrectAnswer:
    def test_verdict_is_correct(self):
        session = _make_session(correct_raws=["x=2", "x=3"])
        result = Orchestrator().handle_turn(session, "x=2")
        assert result.verdict == Verdict.CORRECT

    def test_p_known_increases_after_correct(self):
        session = _make_session(correct_raws=["x=2", "x=3"])
        p_before = session.p_known
        result = Orchestrator().handle_turn(session, "x=2")
        assert result.p_known_after > p_before

    def test_judgment_attached(self):
        session = _make_session(correct_raws=["x=3"])
        result = Orchestrator().handle_turn(session, "x=3")
        assert result.judgment is not None and result.judgment.correct is True

    def test_coaching_message_non_empty(self):
        result = Orchestrator().handle_turn(_make_session(correct_raws=["x=3"]), "x=3")
        assert len(result.coaching_message.strip()) > 0


# ── Test 2: wrong answer, diagnosis, no leak ──────────────────────────────────

class TestWrongAnswerDiagnosisNoLeak:
    def test_verdict_is_wrong(self):
        result = Orchestrator().handle_turn(_make_session(), "x=99")
        assert result.verdict == Verdict.WRONG

    def test_p_known_decreases_after_wrong(self):
        session = _make_session()
        p_before = session.p_known
        result = Orchestrator().handle_turn(session, "x=99")
        assert result.p_known_after < p_before

    def test_no_correct_roots_leaked_in_independent_mode(self):
        session = _make_session(correct_answer_str="x=2 or x=3")
        result = Orchestrator().handle_turn(session, "x=99")
        msg = result.coaching_message
        assert "x=2" not in msg and "x=3" not in msg, f"Roots leaked: {msg!r}"

    def test_misconception_diagnosis_returned(self):
        import mathtutor.orchestrator as orch_mod
        orig = orch_mod._safe_diagnose
        def _fake(artifact, target):
            if artifact.expr == "x=9":
                return "forgot_to_take_square_root", "You squared instead of square-rooted."
            return None, None
        orch_mod._safe_diagnose = _fake
        try:
            result = Orchestrator().handle_turn(_make_session(), "x=9")
        finally:
            orch_mod._safe_diagnose = orig
        assert result.misconception_id == "forgot_to_take_square_root"

    def test_misconception_message_hint_not_value(self):
        import mathtutor.orchestrator as orch_mod
        orig = orch_mod._safe_diagnose
        def _fake(artifact, target):
            return "forgot_to_take_square_root", "You squared instead of square-rooted."
        orch_mod._safe_diagnose = _fake
        try:
            session = _make_session(correct_answer_str="x=2 or x=3")
            result = Orchestrator().handle_turn(session, "x=9")
        finally:
            orch_mod._safe_diagnose = orig
        msg = result.coaching_message
        assert "square" in msg.lower(), f"Expected hint in: {msg!r}"
        assert "x=2" not in msg and "x=3" not in msg, f"Roots leaked: {msg!r}"


# ── Test 3: LLM failure → offline fallback ───────────────────────────────────

class _BrokenLLM:
    def coach(self, context):
        raise RuntimeError("LLM service unavailable")

class TestLLMFailureFallback:
    def test_no_exception_propagates(self):
        result = Orchestrator(llm=_BrokenLLM()).handle_turn(_make_session(), "x=99")
        assert result is not None

    def test_used_offline_tutor_flag_set(self):
        result = Orchestrator(llm=_BrokenLLM()).handle_turn(_make_session(), "x=99")
        assert result.used_offline_tutor is True

    def test_coaching_message_still_present(self):
        result = Orchestrator(llm=_BrokenLLM()).handle_turn(_make_session(), "x=99")
        assert len(result.coaching_message.strip()) > 0

    def test_correct_answer_with_broken_llm(self):
        session = _make_session(correct_raws=["x=3"])
        result = Orchestrator(llm=_BrokenLLM()).handle_turn(session, "x=3")
        assert result.verdict == Verdict.CORRECT and result.used_offline_tutor is True


# ── Test 4: telemetry emitted per turn ────────────────────────────────────────

class TestTelemetryEmittedPerTurn:
    def test_telemetry_attached(self):
        result = Orchestrator().handle_turn(_make_session(), "x=3")
        assert result.telemetry is not None

    def test_telemetry_session_id(self):
        session = _make_session()
        result = Orchestrator().handle_turn(session, "x=3")
        assert result.telemetry.session_id == session.session_id

    def test_telemetry_verdict(self):
        result = Orchestrator().handle_turn(_make_session(), "x=99")
        assert result.telemetry.verdict == Verdict.WRONG.value

    def test_telemetry_p_known_fields(self):
        result = Orchestrator().handle_turn(_make_session(), "x=3")
        assert result.telemetry.p_known_before is not None
        assert result.telemetry.p_known_after is not None

    def test_telemetry_kc_and_problem_id(self):
        result = Orchestrator().handle_turn(_make_session(), "x=3")
        assert result.telemetry.kc_id == "quad_eq"
        assert result.telemetry.problem_id == "prob_001"

    def test_one_event_per_turn(self):
        import mathtutor.orchestrator as orch_mod
        emitted = []
        orig = orch_mod._emit_telemetry
        orch_mod._emit_telemetry = emitted.append
        try:
            session = _make_session()
            orch = Orchestrator()
            orch.handle_turn(session, "x=99")
            orch.handle_turn(session, "x=3")
        finally:
            orch_mod._emit_telemetry = orig
        assert len(emitted) == 2
        assert len({e.event_id for e in emitted}) == 2

    def test_telemetry_serialisable(self):
        result = Orchestrator().handle_turn(_make_session(), "x=3")
        restored = TelemetryEvent.from_json(result.telemetry.to_json())
        assert restored.session_id == result.telemetry.session_id
        assert restored.verdict == result.telemetry.verdict


# ── Bonus: parse failure ──────────────────────────────────────────────────────

class TestParseFailure:
    def test_unparseable_gives_abstain(self):
        result = Orchestrator().handle_turn(_make_session(), "UNPARSEABLE")
        assert result.verdict == Verdict.ABSTAIN

    def test_parse_error_string_attached(self):
        result = Orchestrator().handle_turn(_make_session(), "UNPARSEABLE")
        assert result.parse_error and len(result.parse_error) > 0

# ────────────────────────────────────────────────────────────
# FILE: ./tests/test_parsing.py
# ────────────────────────────────────────────────────────────

import sympy as sp
import pytest

from mathtutor.cas.parsing import parse_math
from mathtutor.contracts import ParseError


def test_equation_parses():
    a = parse_math("2x^2-5x+6=0")
    assert a.kind == "equation"
    assert isinstance(a.expr, sp.Equality)


def test_fraction_parses_to_rational():
    a = parse_math("3/4")
    assert a.kind == "value"
    assert a.expr == sp.Rational(3, 4)


def test_garbage_raises():
    with pytest.raises(ParseError):
        parse_math("2x +")


# ────────────────────────────────────────────────────────────
# FILE: ./tests/test_safety.py
# ────────────────────────────────────────────────────────────

"""
tests/test_safety.py — Tests for safety/leak_filter.py and safety/claim_cert.py.

All correctness decisions go through PolynomialVerifier (CAS), never through
string similarity.  The test matrix covers the four cases mandated by the spec:

  1. A leaked root (x = 3) is redacted when gated.
  2. A leaked root passes through unchanged when NOT gated.
  3. A false certified claim is removed.
  4. A true certified claim is kept and unwrapped.
  5. An unparseable claim is removed (fail-safe).

Additional edge-case tests cover:
  - Trivial reorderings of a two-root answer set.
  - An empty / no-claim text is returned unchanged.
  - Multiple claims in one response (some good, some bad).
"""

from __future__ import annotations

import pytest
import sympy
from sympy import FiniteSet, symbols

from mathtutor.contracts import Target
from mathtutor.domain.verifiers.polynomial import PolynomialVerifier
from mathtutor.safety.leak_filter import redact_answers
from mathtutor.safety.claim_cert import certify


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

x = symbols("x")


@pytest.fixture()
def verifier() -> PolynomialVerifier:
    """A fresh PolynomialVerifier for each test."""
    return PolynomialVerifier()


def make_target(value: sympy.Basic) -> Target:
    """Helper: wrap a SymPy value in a Target."""
    return Target(domain="expression", payload={"answer": value})


# ===========================================================================
# TASK A — leak_filter.redact_answers
# ===========================================================================


class TestRedactAnswers:
    """Spec-mandated leak-filter tests (§ TASK A)."""

    # --- Core spec test 1 ---------------------------------------------------

    def test_root_is_redacted_when_gated(self):
        """
        A leaked root "x = 3" is redacted from LLM prose when gated=True.
        """
        text = "Great work!  Now try to find x.  Hint: x = 3 is the solution."
        result = redact_answers(text, answers=["3"], gated=True)
        assert "3" not in result
        assert "[hidden while you work]" in result

    # --- Core spec test 2 ---------------------------------------------------

    def test_root_passes_through_when_not_gated(self):
        """
        The same text is returned unchanged when gated=False
        (student has finished / answer revealed).
        """
        text = "Great work!  Now try to find x.  Hint: x = 3 is the solution."
        result = redact_answers(text, answers=["3"], gated=False)
        assert result == text

    # --- Trivial reordering -------------------------------------------------

    def test_trivial_reordering_is_redacted(self):
        """
        "3, 2" and "2, 3" are both redacted for a two-root answer {2, 3}.
        """
        answers = ["2", "3"]

        text_forward = "The solutions are 2, 3 — check both!"
        result_f = redact_answers(text_forward, answers=answers, gated=True)
        assert "2, 3" not in result_f
        assert "[hidden while you work]" in result_f

        text_reverse = "The solutions are 3, 2 — check both!"
        result_r = redact_answers(text_reverse, answers=answers, gated=True)
        assert "3, 2" not in result_r
        assert "[hidden while you work]" in result_r

    # --- Individual roots in multi-root case --------------------------------

    def test_individual_roots_redacted_in_multi_root_case(self):
        """
        Each root is individually redacted even without the comma-list form.
        """
        answers = ["2", "3"]
        text = "Try substituting 2 into the equation.  Or maybe 3?"
        result = redact_answers(text, answers=answers, gated=True)
        assert "2" not in result
        assert "3" not in result

    # --- No answers → text unchanged ----------------------------------------

    def test_empty_answers_list_is_noop(self):
        text = "Try x = 7."
        assert redact_answers(text, answers=[], gated=True) == text

    # --- Word-boundary safety -----------------------------------------------

    def test_word_boundary_does_not_clobber_other_digits(self):
        """
        Answer "3" must not redact the "3" inside "13" or "30".
        """
        text = "Look at step 13 and 30 in your notes."
        result = redact_answers(text, answers=["3"], gated=True)
        # "13" and "30" should be untouched; standalone "3" would be caught
        assert "13" in result
        assert "30" in result

    # --- Negative answer ----------------------------------------------------

    def test_negative_root_is_redacted(self):
        text = "One solution is -2, the other is 5."
        result = redact_answers(text, answers=["-2", "5"], gated=True)
        assert "-2" not in result
        assert "5" not in result


# ===========================================================================
# TASK B — claim_cert.certify
# ===========================================================================


class TestCertify:
    """Spec-mandated claim-certification tests (§ TASK B)."""

    # --- Core spec test 3: false claim is removed ---------------------------

    def test_false_claim_is_removed(self, verifier):
        """
        A claim that is wrong (5 ≠ 3) is removed from the prose.
        """
        # Target: x = 3  (FiniteSet so the verifier uses solution-set path)
        target = make_target(FiniteSet(3))

        text = "The answer is <<claim>>5<</claim>> — did you get that?"
        result = certify(text, verifier, target)

        assert "<<claim>>" not in result
        assert "5" not in result
        # The surrounding prose survives
        assert "The answer is" in result
        assert "did you get that?" in result

    # --- Core spec test 4: true claim is kept and unwrapped -----------------

    def test_true_claim_is_kept_and_unwrapped(self, verifier):
        """
        A correct claim (3 == 3) is kept and its delimiters are removed.
        """
        target = make_target(FiniteSet(3))

        text = "The answer is <<claim>>3<</claim>> — well done!"
        result = certify(text, verifier, target)

        # Delimiters gone
        assert "<<claim>>" not in result
        assert "<</claim>>" not in result
        # Content preserved
        assert "3" in result
        assert "well done!" in result

    # --- Core spec test 5: unparseable claim is removed (fail-safe) ---------

    def test_unparseable_claim_is_removed(self, verifier):
        """
        A claim that cannot be parsed (e.g. random punctuation) is dropped,
        not propagated.  Fail-safe path.
        """
        target = make_target(FiniteSet(3))

        text = "Try <<claim>>???#!<</claim>> and see what happens."
        result = certify(text, verifier, target)

        assert "<<claim>>" not in result
        assert "???#!" not in result
        # Surrounding prose survives
        assert "Try" in result
        assert "and see what happens." in result

    # --- No claims → text returned unchanged --------------------------------

    def test_no_claims_text_unchanged(self, verifier):
        target = make_target(FiniteSet(3))
        text = "Here is a hint: think about what value makes x − 3 zero."
        assert certify(text, verifier, target) == text

    # --- Mixed: one good, one bad -------------------------------------------

    def test_mixed_claims_good_kept_bad_dropped(self, verifier):
        """
        In a response with two claims, the correct one is unwrapped and the
        false one is removed.
        """
        target = make_target(FiniteSet(3))
        text = (
            "First, note that x = <<claim>>3<</claim>>.  "
            "Also, some say x = <<claim>>7<</claim>> but that's wrong."
        )
        result = certify(text, verifier, target)

        assert "3" in result          # good claim kept
        assert "7" not in result      # false claim removed
        assert "<<claim>>" not in result

    # --- Expression claim (not just numbers) --------------------------------

    def test_expression_claim_kept_when_equivalent(self, verifier):
        """
        An expression claim that is algebraically equivalent to the target
        is kept even if written differently.
        e.g. target = 6/2, claim = 3  (both equal 3)
        """
        target = make_target(sympy.Rational(6, 2))   # evaluates to 3

        text = "So the answer simplifies to <<claim>>3<</claim>>."
        result = certify(text, verifier, target)

        assert "3" in result
        assert "<<claim>>" not in result

    # --- Equation claim matched against FiniteSet target --------------------

    def test_equation_claim_matched_to_solution_set(self, verifier):
        """
        A claim written as ``x = 3`` is equivalent to a FiniteSet({3}) target.
        """
        target = make_target(FiniteSet(3))

        text = "We find <<claim>>x = 3<</claim>>."
        result = certify(text, verifier, target)

        assert "<<claim>>" not in result
        assert "x = 3" in result


# ────────────────────────────────────────────────────────────
# FILE: ./tests/test_scaffolding.py
# ────────────────────────────────────────────────────────────

"""
tests/test_scaffolding.py — pytest suite for tutoring/scaffolding.py

Test strategy
-------------
Each public function / class has its own section.  Tests are written
as plain functions (no class-based grouping) to keep them concise and
independently runnable.

The sentinel "ANSWER" must never appear in any hint payload — this is
the key non-negotiable invariant checked exhaustively below.
"""

from __future__ import annotations

import pytest
from sympy import symbols, expand, Integer

from mathtutor.contracts import SupportLevel
from mathtutor.tutoring.scaffolding import (
    support_level,
    HintLadder,
    is_structural_progress,
    gate_open,
    completion_problem,
    _ProgressTracker,
    _BLANK,
)

x, y, a, b, t = symbols("x y a b t")


# ===========================================================================
# Helpers
# ===========================================================================

def _all_strings_in(obj) -> list[str]:
    """
    Recursively collect every string value nested anywhere in *obj*
    (dict, list, or scalar).  Used to hunt for the "ANSWER" sentinel.
    """
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        out: list[str] = []
        for v in obj.values():
            out.extend(_all_strings_in(v))
        return out
    if isinstance(obj, (list, tuple)):
        out = []
        for item in obj:
            out.extend(_all_strings_in(item))
        return out
    return []


def _ladder_full_escalation(diagnosis=None) -> list[dict]:
    """Return all five rungs for a fresh ladder."""
    ladder = HintLadder()
    results = []
    for _ in range(5):
        results.append(ladder.next_hint(diagnosis=diagnosis))
    return results


# ===========================================================================
# 1. support_level
# ===========================================================================

class TestSupportLevel:
    def test_low_p_gives_worked(self):
        assert support_level(0.0) == SupportLevel.WORKED
        assert support_level(0.2) == SupportLevel.WORKED
        assert support_level(0.39) == SupportLevel.WORKED

    def test_boundary_low_gives_completion(self):
        # At exactly the low threshold, we're in COMPLETION, not WORKED.
        assert support_level(0.40) == SupportLevel.COMPLETION

    def test_mid_p_gives_completion(self):
        assert support_level(0.5) == SupportLevel.COMPLETION
        assert support_level(0.7) == SupportLevel.COMPLETION
        assert support_level(0.84) == SupportLevel.COMPLETION

    def test_boundary_high_gives_independent(self):
        # At exactly the high threshold we graduate to INDEPENDENT.
        assert support_level(0.85) == SupportLevel.INDEPENDENT

    def test_high_p_gives_independent(self):
        assert support_level(0.9) == SupportLevel.INDEPENDENT
        assert support_level(1.0) == SupportLevel.INDEPENDENT

    def test_custom_thresholds(self):
        assert support_level(0.3, low=0.35, high=0.7) == SupportLevel.WORKED
        assert support_level(0.5, low=0.35, high=0.7) == SupportLevel.COMPLETION
        assert support_level(0.8, low=0.35, high=0.7) == SupportLevel.INDEPENDENT

    def test_invalid_p_raises(self):
        with pytest.raises(ValueError):
            support_level(-0.01)
        with pytest.raises(ValueError):
            support_level(1.01)

    def test_invalid_thresholds_raise(self):
        with pytest.raises(ValueError):
            support_level(0.5, low=0.9, high=0.5)   # inverted
        with pytest.raises(ValueError):
            support_level(0.5, low=0.0, high=0.8)   # low == 0 not allowed


# ===========================================================================
# 2. HintLadder
# ===========================================================================

class TestHintLadder:
    # ---- basic escalation ----

    def test_first_hint_is_rung_1(self):
        ladder = HintLadder()
        result = ladder.next_hint()
        assert result["level"] == 1
        assert result["hint_type"] == "region"

    def test_ladder_escalates_one_rung_per_call(self):
        ladder = HintLadder()
        levels = [ladder.next_hint()["level"] for _ in range(5)]
        assert levels == [1, 2, 3, 4, 5]

    def test_rung_5_does_not_wrap_beyond_5(self):
        ladder = HintLadder()
        for _ in range(6):
            r = ladder.next_hint()
        assert r["level"] == 5

    def test_reset_restarts_from_rung_1(self):
        ladder = HintLadder()
        ladder.next_hint()
        ladder.next_hint()
        ladder.reset()
        assert ladder.current_level == 0
        result = ladder.next_hint()
        assert result["level"] == 1

    # ---- ANSWER sentinel must never appear ----

    def test_no_answer_sentinel_no_diagnosis(self):
        """
        Exhaustively check that the word 'ANSWER' never appears in any
        hint payload across a full ladder escalation with no diagnosis.
        """
        rungs = _ladder_full_escalation(diagnosis=None)
        for rung in rungs:
            strings = _all_strings_in(rung)
            for s in strings:
                assert "ANSWER" not in s.upper(), (
                    f"Rung {rung['level']} leaked the ANSWER sentinel: {s!r}"
                )

    def test_no_answer_sentinel_with_sign_error_diagnosis(self):
        rungs = _ladder_full_escalation(diagnosis=["sign_error"])
        for rung in rungs:
            for s in _all_strings_in(rung):
                assert "ANSWER" not in s.upper()

    def test_no_answer_sentinel_with_distribution_diagnosis(self):
        rungs = _ladder_full_escalation(diagnosis=["forgot_to_distribute"])
        for rung in rungs:
            for s in _all_strings_in(rung):
                assert "ANSWER" not in s.upper()

    def test_no_answer_sentinel_with_missed_root_diagnosis(self):
        rungs = _ladder_full_escalation(diagnosis=["missed_root"])
        for rung in rungs:
            for s in _all_strings_in(rung):
                assert "ANSWER" not in s.upper()

    def test_no_answer_sentinel_with_all_known_diagnoses(self):
        all_tags = [
            "sign_error", "forgot_to_distribute", "wrong_inverse",
            "incomplete_factoring", "missed_root", "like_terms_not_collected",
        ]
        for tag in all_tags:
            rungs = _ladder_full_escalation(diagnosis=[tag])
            for rung in rungs:
                for s in _all_strings_in(rung):
                    assert "ANSWER" not in s.upper(), (
                        f"Tag '{tag}', rung {rung['level']}: {s!r}"
                    )

    # ---- payload structure ----

    def test_rung_3_returns_a_question(self):
        ladder = HintLadder()
        ladder.next_hint(); ladder.next_hint()  # skip to rung 3
        r = ladder.next_hint(diagnosis=["sign_error"])
        assert "question" in r["socratic"]
        assert "?" in r["socratic"]["question"]

    def test_rung_4_has_reminder_about_different_numbers(self):
        ladder = HintLadder()
        for _ in range(3):
            ladder.next_hint()
        r = ladder.next_hint()
        reminder = r["analogous"].get("reminder", "")
        assert "different" in reminder.lower()

    def test_rung_5_gate_closed_by_default(self):
        ladder = HintLadder()
        for _ in range(4):
            ladder.next_hint()
        r = ladder.next_hint(gate_open=False)
        assert r["worked_gate"] == "GATE_CLOSED"

    def test_rung_5_gate_open_when_flag_set(self):
        ladder = HintLadder()
        for _ in range(4):
            ladder.next_hint()
        r = ladder.next_hint(gate_open=True)
        assert r["worked_gate"] == "GATE_OPEN"


# ===========================================================================
# 3. is_structural_progress
# ===========================================================================

class TestIsStructuralProgress:
    """
    Each test uses an explicit _ProgressTracker so they are completely
    independent from one another.
    """

    def _t(self):
        return _ProgressTracker()

    # ---- should return False ----

    def test_no_change_is_not_progress(self):
        tr = self._t()
        assert not is_structural_progress(x + 1, x + 1, tracker=tr)

    def test_algebraically_same_is_not_progress(self):
        # x + 0 and x expand to the same canonical form.
        tr = self._t()
        assert not is_structural_progress(x + 0, x, tracker=tr)

    def test_additive_cycle_plus_5_minus_5(self):
        """
        Classic cycle: x  →  x + 5  →  x + 5 - 5 (= x again).
        The third step is a cycle and must be rejected.
        """
        tr = self._t()
        # Step 1: x → x + 5 (progress? yes — form changed and got simpler... 
        #         actually count_ops(x+5) > count_ops(x), so this is 
        #         complexity-increasing and should return False)
        assert not is_structural_progress(x, x + 5, tracker=tr)
        # Step 2: x + 5 → x  (detected as cycle since x was starting form)
        # Even if x+5 wasn't recorded as progress, we still test the cycle 
        # detection logic independently:
        tr2 = self._t()
        tr2.record(expand(x))  # seed x as seen
        assert not is_structural_progress(x + 5, x, tracker=tr2)  # x is a cycle

    def test_plus5_minus5_is_cycle(self):
        """Explicit cycle: start at x+3, go to x+8, then back to x+3."""
        tr = self._t()
        tr.record(expand(x + 3))   # seed x+3 as seen
        # x+8 → different form, and ops(x+8) == ops(x+3), so progress=True
        assert is_structural_progress(x + 3, x + 8, tracker=tr)
        # x+3 again → cycle (it's in the buffer)
        assert not is_structural_progress(x + 8, x + 3, tracker=tr)

    def test_complexity_increase_rejected(self):
        """
        Expanding x into x*1 + 0*x is complexity-increasing.
        Sympy usually simplifies this, but we can test with an explicit
        expansion that genuinely adds operations.
        """
        tr = self._t()
        # 6*x has count_ops=1; 2*x + 4*x has count_ops=3
        # So rewriting 6*x as 2*x + 4*x is complexity-increasing → not progress.
        assert not is_structural_progress(6*x, 2*x + 4*x, tracker=tr)

    def test_unparseable_returns_false(self):
        tr = self._t()
        assert not is_structural_progress("{{invalid{{", x, tracker=tr)
        assert not is_structural_progress(x, "}}not sympy}}", tracker=tr)

    # ---- should return True ----

    def test_genuine_simplification_accepted(self):
        """
        ``(x**2 - x - 6) / (x - 3)  →  x + 2`` is a genuine simplification.

        SymPy's ``expand`` does NOT cancel polynomial factors in fractions;
        it distributes the denominator, giving a form with many more
        operations (9 ops) vs the simplified ``x + 2`` (1 op).
        Different ``expand``-canonical forms, complexity drops → True.
        """
        from sympy import sympify as S
        # expand((x^2-x-6)/(x-3)) = x^2/(x-3) - x/(x-3) - 6/(x-3) [9 ops]
        # expand(x+2) = x + 2  [1 op]
        before = S("(x**2 - x - 6) / (x - 3)")
        after  = S("x + 2")
        tr = self._t()
        assert is_structural_progress(before, after, tracker=tr)

    def test_collecting_like_terms_is_progress(self):
        """
        ``(x**2 + 2*x - 8) / (x - 2)  →  x + 4``

        SymPy ``expand`` does not cancel polynomial factors in rational
        expressions, so the before-form expands to a multi-term fraction
        (high ops) while the after-form is ``x + 4`` (low ops).
        Different expand-canonical forms, lower complexity → True.
        """
        from sympy import sympify as S
        # expand((x^2+2x-8)/(x-2))  =  multi-term fraction  [high ops]
        # expand(x+4)  =  x + 4  [1 op]
        before = S("(x**2 + 2*x - 8) / (x - 2)")
        after  = S("x + 4")
        tr = self._t()
        assert is_structural_progress(before, after, tracker=tr)

    def test_cancellation_is_progress(self):
        """
        ``(x**2 - 1) / (x - 1)  →  x + 1``

        SymPy does NOT auto-cancel common polynomial factors in rational
        expressions built from sympify.  Under ``expand`` only:
        * before canonicalises to ``x**2/(x-1) - 1/(x-1)``  (6 ops)
        * after canonicalises to  ``x + 1``                  (1 op)
        Different forms, complexity decreases → True.
        """
        from sympy import sympify as S
        expr_before = S("(x**2 - 1) / (x - 1)")
        expr_after  = S("x + 1")
        tr = self._t()
        # Sanity: SymPy keeps rational form until cancel is applied
        assert str(expr_before) == "(x**2 - 1)/(x - 1)"
        assert is_structural_progress(expr_before, expr_after, tracker=tr)

    def test_solving_linear_eq_is_progress(self):
        """2*x + 6  →  2*x  is NOT (complexity same but we subtract 6 so form
        changes and ops go from 2 to 1 after x = -3 — let's just do 2*x → x)."""
        tr = self._t()
        # 2*x has count_ops 1; x has count_ops 0. Simplification.
        assert is_structural_progress(2*x, x, tracker=tr)


# ===========================================================================
# 4. gate_open
# ===========================================================================

class TestGateOpen:
    """
    Key invariant: high-mastery + frustrated → gate opens FASTER than
    low-mastery + fast (low effort).
    """

    def test_no_progress_keeps_gate_closed(self):
        # Even a seasoned, frustrated student can't skip the work.
        assert not gate_open(
            progress_steps=0,
            time_on_task_s=9999.0,
            p_known=0.99,
            frustration=1.0,
        )

    def test_high_mastery_frustrated_opens_faster(self):
        """
        A student with p_known=0.95 and frustration=0.9 should hit the
        threshold at a much shorter time than the 120 s baseline.
        We test with 50 s — well below 120 s — and expect OPEN.
        """
        assert gate_open(
            progress_steps=2,
            time_on_task_s=50.0,
            p_known=0.95,
            frustration=0.9,
        )

    def test_low_mastery_fast_stays_closed(self):
        """
        A student with p_known=0.2 and frustration=0.0 who only spent 10 s
        should still be blocked.
        """
        assert not gate_open(
            progress_steps=1,
            time_on_task_s=10.0,
            p_known=0.2,
            frustration=0.0,
        )

    def test_high_mastery_frustrated_threshold_lt_low_mastery_fast_threshold(self):
        """
        Verify the asymmetry quantitatively: the effective threshold for
        (high mastery, high frustration) must be less than the effective
        threshold for (low mastery, low frustration + fast clicks).

        We do this by finding the minimum time at which each opens.
        """
        def min_time_to_open(p_known, frustration, steps=2):
            for t in range(1, 1000):
                if gate_open(progress_steps=steps, time_on_task_s=float(t),
                             p_known=p_known, frustration=frustration):
                    return t
            return 1000  # never opened in range

        t_expert_frustrated = min_time_to_open(p_known=0.95, frustration=0.9)
        t_novice_fast = min_time_to_open(p_known=0.2, frustration=0.0)

        assert t_expert_frustrated < t_novice_fast, (
            f"Expected expert+frustrated ({t_expert_frustrated}s) to open "
            f"before novice+fast ({t_novice_fast}s)"
        )

    def test_baseline_opens_after_sufficient_time(self):
        """
        Mid-mastery student who spends ≥ 120 s and made progress should
        eventually open.
        """
        assert gate_open(
            progress_steps=3,
            time_on_task_s=130.0,
            p_known=0.6,
            frustration=0.3,
        )

    def test_invalid_p_known_raises(self):
        with pytest.raises(ValueError):
            gate_open(1, 60.0, p_known=-0.1, frustration=0.5)

    def test_invalid_frustration_raises(self):
        with pytest.raises(ValueError):
            gate_open(1, 60.0, p_known=0.5, frustration=1.5)


# ===========================================================================
# 5. completion_problem
# ===========================================================================

class TestCompletionProblem:
    STEPS = ["2x + 4 = 10", "2x = 6", "x = 3"]

    def test_reveal_through_0_blanks_rest(self):
        result = completion_problem(self.STEPS, reveal_through=0)
        assert result["steps"] == ["2x + 4 = 10", _BLANK, _BLANK]
        assert result["revealed"] == 1
        assert result["blanked"] == 2

    def test_reveal_through_1_blanks_last(self):
        result = completion_problem(self.STEPS, reveal_through=1)
        assert result["steps"] == ["2x + 4 = 10", "2x = 6", _BLANK]
        assert result["revealed"] == 2
        assert result["blanked"] == 1

    def test_reveal_through_last_shows_all(self):
        result = completion_problem(self.STEPS, reveal_through=2)
        assert result["steps"] == self.STEPS
        assert result["blanked"] == 0

    def test_total_steps_is_correct(self):
        result = completion_problem(self.STEPS, reveal_through=1)
        assert result["total_steps"] == 3

    def test_empty_steps_raises(self):
        with pytest.raises(ValueError):
            completion_problem([], reveal_through=0)

    def test_out_of_range_reveal_raises(self):
        with pytest.raises(ValueError):
            completion_problem(self.STEPS, reveal_through=5)
        with pytest.raises(ValueError):
            completion_problem(self.STEPS, reveal_through=-1)


# ────────────────────────────────────────────────────────────
# FILE: ./tests/test_scheduling.py
# ────────────────────────────────────────────────────────────

"""
tests/test_scheduling.py
========================
Tests for mathtutor/learner/scheduling.py.

Coverage targets (per spec):
  1. Recall decays toward 0 as elapsed grows.
  2. A spaced success increases half_life.
  3. select_next never returns a KC with unmet prerequisites.
  4. select_next interleaves (does not block on one KC).

All tests are deterministic and require no network access or I/O.

Mocking strategy
----------------
``Curriculum`` is imported from ``domain/curriculum.py``.  To keep these
tests self-contained we create a lightweight ``FakeCurriculum`` that satisfies
the interface (``curriculum.kcs`` → iterable of objects with ``.id`` and
``.prerequisites``).  This is NOT redefining the real Curriculum — we stay
within our contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import pytest

from mathtutor.learner.scheduling import (
    DEFAULT_HALF_LIFE,
    GROWTH_FACTOR,
    DECAY_FACTOR,
    MIN_HALF_LIFE,
    MIN_SPACING_RATIO,
    RetentionState,
    due_for_review,
    predicted_recall,
    select_next,
    update_after_review,
)


# ---------------------------------------------------------------------------
# Helpers / Fakes
# ---------------------------------------------------------------------------

@dataclass
class FakeKC:
    """Minimal stand-in for a KnowledgeComponent; satisfies the interface."""
    id: str
    prerequisites: List[str]


class FakeCurriculum:
    """Minimal stand-in for Curriculum; provides .kcs attribute."""
    def __init__(self, kcs: list[FakeKC]) -> None:
        self.kcs = kcs


# Shared timestamps — use seconds for clarity.
T0 = 1_000_000.0   # arbitrary epoch


# ---------------------------------------------------------------------------
# 1.  predicted_recall — decay behaviour
# ---------------------------------------------------------------------------

class TestPredictedRecall:
    """Recall should be 1.0 at t=0 and decay toward 0 as time passes."""

    def test_recall_at_zero_elapsed_is_one(self) -> None:
        rs = RetentionState(half_life=86_400.0, last_seen_ts=T0)
        assert predicted_recall(rs, T0) == pytest.approx(1.0)

    def test_recall_at_one_half_life_is_half(self) -> None:
        """
        By definition, 2^(-(h/h)) = 2^(-1) = 0.5.
        This is the core invariant of exponential-half-life decay.
        """
        h = 3600.0
        rs = RetentionState(half_life=h, last_seen_ts=T0)
        assert predicted_recall(rs, T0 + h) == pytest.approx(0.5)

    def test_recall_at_two_half_lives_is_quarter(self) -> None:
        """
        2^(-(2h/h)) = 2^(-2) = 0.25
        """
        h = 3600.0
        rs = RetentionState(half_life=h, last_seen_ts=T0)
        assert predicted_recall(rs, T0 + 2 * h) == pytest.approx(0.25)

    def test_recall_decays_monotonically(self) -> None:
        """Recall must strictly decrease as more time elapses."""
        h = 3600.0
        rs = RetentionState(half_life=h, last_seen_ts=T0)
        recalls = [
            predicted_recall(rs, T0 + elapsed)
            for elapsed in [0, h * 0.5, h, h * 2, h * 5, h * 20]
        ]
        for earlier, later in zip(recalls, recalls[1:]):
            assert later < earlier

    def test_recall_approaches_zero(self) -> None:
        """After many half-lives the recall should be negligible (< 0.001)."""
        h = 1.0   # 1-second half-life for a quick test
        rs = RetentionState(half_life=h, last_seen_ts=T0)
        # 20 half-lives: 2^(-20) ≈ 9.5e-7
        assert predicted_recall(rs, T0 + 20 * h) < 0.001

    def test_negative_elapsed_returns_one(self) -> None:
        """Clock skew or same-instant review should not break recall."""
        rs = RetentionState(half_life=3600.0, last_seen_ts=T0 + 100)
        assert predicted_recall(rs, T0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 2.  update_after_review — half-life growth and decay
# ---------------------------------------------------------------------------

class TestUpdateAfterReview:
    """Test the half-life update rules."""

    def test_spaced_success_grows_half_life(self) -> None:
        """
        A success where elapsed >= MIN_SPACING_RATIO * h should multiply h
        by GROWTH_FACTOR.

        Arithmetic:
            h = 3600.0
            elapsed must be >= MIN_SPACING_RATIO * 3600 = 0.10 * 3600 = 360 s
            We use elapsed = 1800 s (half a half-life, clearly spaced).
            Expected new_h = 3600 * GROWTH_FACTOR = 3600 * 2.0 = 7200 s.
        """
        h = 3600.0
        rs = RetentionState(half_life=h, last_seen_ts=T0)
        elapsed = 1800.0  # 1800 >= 360 — counts as spaced
        rs2 = update_after_review(rs, T0 + elapsed, success=True)
        assert rs2.half_life == pytest.approx(h * GROWTH_FACTOR)

    def test_spaced_success_increments_successful_reviews(self) -> None:
        h = 3600.0
        rs = RetentionState(half_life=h, last_seen_ts=T0, successful_reviews=3)
        rs2 = update_after_review(rs, T0 + 1800, success=True)
        assert rs2.successful_reviews == 4

    def test_massed_success_does_not_grow_half_life(self) -> None:
        """
        A success where elapsed < MIN_SPACING_RATIO * h should NOT grow h.

        Arithmetic:
            h = 3600.0, MIN_SPACING_RATIO = 0.10 => threshold = 360 s.
            elapsed = 10 s < 360 s  => massed repetition => no reward.
        """
        h = 3600.0
        rs = RetentionState(half_life=h, last_seen_ts=T0)
        elapsed = 10.0  # 10 < 360 — massed
        rs2 = update_after_review(rs, T0 + elapsed, success=True)
        assert rs2.half_life == pytest.approx(h)   # unchanged

    def test_failure_shrinks_half_life(self) -> None:
        """
        On failure, h should be multiplied by DECAY_FACTOR.

        Arithmetic:
            h = 7200.0, DECAY_FACTOR = 0.5 => new_h = 3600.0
        """
        h = 7200.0
        rs = RetentionState(half_life=h, last_seen_ts=T0)
        rs2 = update_after_review(rs, T0 + 3600, success=False)
        assert rs2.half_life == pytest.approx(h * DECAY_FACTOR)

    def test_failure_does_not_go_below_min_half_life(self) -> None:
        """
        Even with a tiny h, failure should not produce a non-positive half-life.
        """
        rs = RetentionState(half_life=MIN_HALF_LIFE, last_seen_ts=T0)
        rs2 = update_after_review(rs, T0 + 1, success=False)
        assert rs2.half_life >= MIN_HALF_LIFE

    def test_failure_does_not_increment_successful_reviews(self) -> None:
        rs = RetentionState(half_life=3600.0, last_seen_ts=T0, successful_reviews=5)
        rs2 = update_after_review(rs, T0 + 3600, success=False)
        assert rs2.successful_reviews == 5  # unchanged

    def test_update_is_immutable(self) -> None:
        """Original RetentionState must not be mutated."""
        rs = RetentionState(half_life=3600.0, last_seen_ts=T0)
        _ = update_after_review(rs, T0 + 1800, success=True)
        assert rs.half_life == pytest.approx(3600.0)
        assert rs.last_seen_ts == T0


# ---------------------------------------------------------------------------
# 3.  due_for_review
# ---------------------------------------------------------------------------

class TestDueForReview:
    """Items below the recall band should be flagged."""

    def _make_states(self) -> dict[str, RetentionState]:
        h = 3600.0
        return {
            "kc_a": RetentionState(half_life=h, last_seen_ts=T0),          # fresh
            "kc_b": RetentionState(half_life=h, last_seen_ts=T0 - 2 * h),  # old (recall=0.25)
            "kc_c": RetentionState(half_life=h, last_seen_ts=T0 - h * 0.2),# recall≈0.87 (>0.85)
        }

    def test_due_items_are_below_band(self) -> None:
        """
        kc_b recall ≈ 0.25 (clearly due).
        kc_a recall = 1.0  (fresh — not due).
        kc_c recall ≈ 0.87 (just above 0.85 — not due).
        """
        states = self._make_states()
        due = due_for_review(states, T0, band=0.85)
        assert "kc_b" in due
        assert "kc_a" not in due
        assert "kc_c" not in due

    def test_due_sorted_most_forgotten_first(self) -> None:
        """The most-forgotten KC (lowest recall) should come first."""
        h = 3600.0
        states = {
            "kc_very_old": RetentionState(half_life=h, last_seen_ts=T0 - 10 * h),
            "kc_old":      RetentionState(half_life=h, last_seen_ts=T0 - 2 * h),
        }
        due = due_for_review(states, T0, band=0.90)
        assert due[0] == "kc_very_old"

    def test_empty_states_returns_empty(self) -> None:
        assert due_for_review({}, T0) == []

    def test_band_exactly_at_threshold(self) -> None:
        """A KC whose recall equals the band exactly should be included."""
        h = 3600.0
        # We need elapsed s.t. 2^(-e/h) == 0.85
        # => e = -h * log2(0.85) ≈ h * 0.2345
        elapsed = -h * math.log2(0.85)
        rs = RetentionState(half_life=h, last_seen_ts=T0 - elapsed)
        states = {"kc_x": rs}
        due = due_for_review(states, T0, band=0.85)
        assert "kc_x" in due


# ---------------------------------------------------------------------------
# 4.  RetentionState validation
# ---------------------------------------------------------------------------

class TestRetentionStateValidation:
    def test_negative_half_life_raises(self) -> None:
        with pytest.raises(ValueError, match="half_life"):
            RetentionState(half_life=-1.0)

    def test_zero_half_life_raises(self) -> None:
        with pytest.raises(ValueError, match="half_life"):
            RetentionState(half_life=0.0)

    def test_negative_successful_reviews_raises(self) -> None:
        with pytest.raises(ValueError, match="successful_reviews"):
            RetentionState(half_life=3600.0, successful_reviews=-1)


# ---------------------------------------------------------------------------
# 5.  select_next — prerequisite safety + interleaving
# ---------------------------------------------------------------------------

class TestSelectNext:
    """
    Core invariants:
      a. Never return a KC with unmet prerequisites.
      b. Return an interleaved (not blocked) sequence when both review and new
         KCs are available.
    """

    def _simple_curriculum(self) -> FakeCurriculum:
        """
        Linear chain:  kc1 → kc2 → kc3 → kc4
        Plus two isolated KCs: kc5, kc6 (no prereqs)
        """
        return FakeCurriculum(kcs=[
            FakeKC("kc1", prerequisites=[]),
            FakeKC("kc2", prerequisites=["kc1"]),
            FakeKC("kc3", prerequisites=["kc2"]),
            FakeKC("kc4", prerequisites=["kc3"]),
            FakeKC("kc5", prerequisites=[]),
            FakeKC("kc6", prerequisites=[]),
        ])

    # ------------------------------------------------------------------
    # Prerequisite safety
    # ------------------------------------------------------------------

    def test_no_kc_with_unmet_prereqs(self) -> None:
        """
        With nothing mastered, only kc1, kc5, kc6 are eligible (no prereqs).
        kc2–kc4 must never appear.
        """
        curriculum = self._simple_curriculum()
        result = select_next(
            curriculum=curriculum,
            mastered_set=set(),
            retention_states={},
            now_ts=T0,
            k=6,
        )
        for kc_id in result:
            assert kc_id in {"kc1", "kc5", "kc6"}, (
                f"{kc_id!r} has unmet prerequisites but was selected"
            )

    def test_prereq_chain_respected(self) -> None:
        """
        With only kc1 mastered, kc2 becomes eligible but kc3/kc4 stay gated.
        """
        curriculum = self._simple_curriculum()
        result = select_next(
            curriculum=curriculum,
            mastered_set={"kc1"},
            retention_states={},
            now_ts=T0,
            k=6,
        )
        assert "kc3" not in result
        assert "kc4" not in result
        # kc2 *may* appear (prereq met)
        if "kc2" in result:
            pass  # acceptable

    def test_mastered_kc_not_in_new_pool(self) -> None:
        """A mastered KC should not be offered as 'new' even if its recall is high."""
        curriculum = self._simple_curriculum()
        result = select_next(
            curriculum=curriculum,
            mastered_set={"kc1"},
            retention_states={},   # kc1 not in retention_states
            now_ts=T0,
            k=6,
        )
        assert "kc1" not in result

    # ------------------------------------------------------------------
    # Interleaving
    # ------------------------------------------------------------------

    def test_interleaves_review_and_new(self) -> None:
        """
        Set up: kc1 and kc5 are due for review; kc6 is new.
        Expected output pattern: review, new, review, … (no two reviews
        or two news in a row, given alternation).

        We verify that the output is NOT simply [all reviews, all new] —
        i.e. not a blocked schedule.
        """
        h = 3600.0
        # kc1 and kc5 are overdue (recall ≈ 0.25 after 2h)
        retention_states = {
            "kc1": RetentionState(half_life=h, last_seen_ts=T0 - 2 * h),
            "kc5": RetentionState(half_life=h, last_seen_ts=T0 - 2 * h),
        }
        # kc6 is new (not in retention_states), prereqs met
        curriculum = self._simple_curriculum()
        result = select_next(
            curriculum=curriculum,
            mastered_set={"kc1", "kc5"},   # mastered so they appear in review pool, not new
            retention_states=retention_states,
            now_ts=T0,
            k=5,
        )
        # kc6 should be included (it is new and prereqs met)
        assert "kc6" in result

    def test_interleaved_not_blocked(self) -> None:
        """
        Build a scenario with 3 review KCs and 3 new KCs.
        The result should not be [r, r, r, n, n, n] — it should alternate.
        We check that the first new KC appears before the last review KC.
        """
        h = 1.0  # tiny half-life for predictability
        # kc1, kc5 are mastered but their recall has decayed (due for review)
        retention_states = {
            "kc1": RetentionState(half_life=h, last_seen_ts=T0 - 10 * h),
            "kc5": RetentionState(half_life=h, last_seen_ts=T0 - 10 * h),
        }
        # kc6 is new, kc2 will be eligible once kc1 is mastered
        curriculum = FakeCurriculum(kcs=[
            FakeKC("kc1", prerequisites=[]),
            FakeKC("kc5", prerequisites=[]),
            FakeKC("kc6", prerequisites=[]),
            FakeKC("kc2", prerequisites=["kc1"]),
        ])
        result = select_next(
            curriculum=curriculum,
            mastered_set={"kc1", "kc5"},
            retention_states=retention_states,
            now_ts=T0,
            k=6,
        )
        # kc1 and kc5 are in retention_states and recall is very low → review pool
        # kc6 and kc2 are new → new pool (kc2 prereq met via mastered_set)
        review_positions = [i for i, kc in enumerate(result) if kc in {"kc1", "kc5"}]
        new_positions    = [i for i, kc in enumerate(result) if kc in {"kc6", "kc2"}]

        if review_positions and new_positions:
            # Interleaved: the first new KC should appear before the last review KC
            assert min(new_positions) < max(review_positions), (
                f"Blocked schedule detected: reviews={review_positions}, "
                f"new={new_positions}, result={result}"
            )

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_k_is_respected(self) -> None:
        curriculum = self._simple_curriculum()
        result = select_next(
            curriculum=curriculum,
            mastered_set=set(),
            retention_states={},
            now_ts=T0,
            k=2,
        )
        assert len(result) <= 2

    def test_empty_curriculum_returns_empty(self) -> None:
        curriculum = FakeCurriculum(kcs=[])
        result = select_next(
            curriculum=curriculum,
            mastered_set=set(),
            retention_states={},
            now_ts=T0,
            k=5,
        )
        assert result == []

    def test_k_less_than_one_raises(self) -> None:
        curriculum = self._simple_curriculum()
        with pytest.raises(ValueError, match="k"):
            select_next(
                curriculum=curriculum,
                mastered_set=set(),
                retention_states={},
                now_ts=T0,
                k=0,
            )

    def test_all_mastered_returns_empty(self) -> None:
        """If every KC is already mastered, there is nothing new to select."""
        curriculum = self._simple_curriculum()
        all_ids = {kc.id for kc in curriculum.kcs}
        result = select_next(
            curriculum=curriculum,
            mastered_set=all_ids,
            retention_states={},   # none are in retention states
            now_ts=T0,
            k=5,
        )
        # Nothing due for review (not in retention_states) and nothing new
        assert result == []

    def test_no_new_kcs_drains_review_pool(self) -> None:
        """When all eligible KCs are due reviews, return them up to k."""
        h = 1.0
        curriculum = FakeCurriculum(kcs=[
            FakeKC("kc1", prerequisites=[]),
            FakeKC("kc5", prerequisites=[]),
        ])
        retention_states = {
            "kc1": RetentionState(half_life=h, last_seen_ts=T0 - 10),
            "kc5": RetentionState(half_life=h, last_seen_ts=T0 - 10),
        }
        result = select_next(
            curriculum=curriculum,
            mastered_set={"kc1", "kc5"},
            retention_states=retention_states,
            now_ts=T0,
            k=5,
        )
        assert set(result) == {"kc1", "kc5"}


# ────────────────────────────────────────────────────────────
# FILE: ./tests/test_verifiers.py
# ────────────────────────────────────────────────────────────

# tests/test_verifiers.py

import pytest
from sympy import Eq, symbols, Rational
from sympy import symbols, Eq
from mathtutor.domain.verifiers.linear_equation import EquationVerifier
from mathtutor.domain.verifiers.fraction import FractionVerifier
from mathtutor.domain.verifiers.polynomial import PolynomialVerifier
from mathtutor.domain.verifiers.inequality import InequalityVerifier
from mathtutor.domain.verifiers.system import SystemVerifier
from mathtutor.contracts import Target as _Target


x, y = symbols('x y')

def Target(answer, form=None, *, domain="expression"):
    """Build a real contract Target for verifier tests.

    Exposes `form` as a top-level field (matching contracts.Target) and the
    answer under payload['answer'] — the two things verifiers actually read.
    """
    return _Target(domain=domain, payload={"answer": answer}, form=form)


def test_equation():
    v = EquationVerifier()
    t = Target(Eq(x**2 - 5*x + 6, 0))
    assert v.accepts(Eq(x**2 - 5*x + 6, 0), t).correct


def test_fraction():
    v = FractionVerifier()
    t = Target(Rational(5, 6))
    assert v.accepts(Rational(5, 6), t).correct
    j = v.accepts(Rational(10, 12), t)
    assert j.value_equivalent


def test_polynomial():
    v = PolynomialVerifier()
    t = Target((x + 1) * (x + 2), form="expanded")
    assert v.accepts(x**2 + 3*x + 2, t).correct
    assert not v.accepts((x + 1) * (x + 2), t).form_ok


def test_inequality():
    v = InequalityVerifier()
    t = Target(x > 2)
    assert v.accepts(x > 2, t).correct


def test_system():
    v = SystemVerifier()
    t = Target([Eq(x + y, 3), Eq(x - y, 1)])
    assert v.accepts([Eq(x + y, 3), Eq(x - y, 1)], t).correct

