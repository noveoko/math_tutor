import traceback
from sympy import Eq, symbols
from mathtutor.domain.verifiers.system import SystemVerifier

x, y = symbols("x y")
v = SystemVerifier()
cand = [Eq(x + y, 3), Eq(x - y, 1)]

# Build whatever Target the test uses:
from tests.test_verifiers import Target   # reuse the test's Target
t = Target([Eq(x + y, 3), Eq(x - y, 1)])

# Find the internal parse method and call it directly, letting it raise:
for name in dir(v):
    if "parse" in name.lower():
        print("trying:", name)
        try:
            getattr(v, name)(cand)
        except Exception:
            traceback.print_exc()