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