#!/usr/bin/env bash
# combine_py.sh
# Finds all .py files in the current directory (recursively) and combines them into one file.

OUTPUT="combined.py"
SEARCH_DIR="."

# ── Argument handling ─────────────────────────────────────────────────────────
# Optional: pass a custom output filename as the first argument
if [[ -n "$1" ]]; then
  OUTPUT="$1"
fi

# Refuse to overwrite itself if the output file already exists
if [[ -f "$OUTPUT" ]]; then
  echo "⚠️  '$OUTPUT' already exists. Remove it first or pass a different filename:"
  echo "   $0 my_output.py"
  exit 1
fi

# ── Collect .py files ─────────────────────────────────────────────────────────
# -type f        → regular files only
# -name "*.py"   → ending in .py
# ! -name "$OUTPUT" → skip the output file itself (if it were a .py)
mapfile -t PY_FILES < <(find "$SEARCH_DIR" -type f -name "*.py" ! -name "$OUTPUT" ! -path "*/.venv/*" | sort)

if [[ ${#PY_FILES[@]} -eq 0 ]]; then
  echo "No Python files found in '$SEARCH_DIR'."
  exit 0
fi

echo "Found ${#PY_FILES[@]} Python file(s). Writing to '$OUTPUT'..."

# ── Write header ──────────────────────────────────────────────────────────────
{
  echo "# ============================================================"
  echo "# Combined Python file"
  echo "# Generated: $(date)"
  echo "# Source dir: $(pwd)"
  echo "# Files included: ${#PY_FILES[@]}"
  echo "# ============================================================"
  echo ""
} >> "$OUTPUT"

# ── Concatenate each file ─────────────────────────────────────────────────────
for FILE in "${PY_FILES[@]}"; do
  {
    echo ""
    echo "# ────────────────────────────────────────────────────────────"
    echo "# FILE: $FILE"
    echo "# ────────────────────────────────────────────────────────────"
    echo ""
    cat "$FILE"
    echo ""   # ensure a blank line between files
  } >> "$OUTPUT"

  echo "  ✔ $FILE"
done

echo ""
echo "✅ Done! Combined file: '$OUTPUT' ($(wc -l < "$OUTPUT") lines)"