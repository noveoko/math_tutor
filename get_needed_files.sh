#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "========================================================================"
echo "🎯 Fetching project files for test diagnosis"
echo "========================================================================"

# 1. Define known test files from the log
TEST_EVAL="src/mathtutor/tests/test_eval.py"
TEST_MISC="src/mathtutor/tests/test_misconceptions.py"

# 2. Dynamically locate implementation files containing the core logic
echo "🔍 Searching for implementation files..."
SRC_EVAL=$(grep -rl "def fit_afm" src/ 2>/dev/null || true)
SRC_MISC=$(grep -rl "def diagnose" src/ 2>/dev/null || true)

# 3. Print helper function to output file contents with clean headers
print_file() {
    local file_path="$1"
    if [ -f "$file_path" ]; then
        echo -e "\n--- START OF FILE: $file_path ---"
        cat "$file_path"
        echo -e "--- END OF FILE: $file_path ---\n"
    else
        echo "⚠️ Warning: File not found -> $file_path"
    fi
}

# --- FETCHING FILES FOR FAILURE 1: AFM Convergence Failure ---
echo "📦 [Failure 1] Gathering AFM Optimization files..."
print_file "$TEST_EVAL"
if [ -n "$SRC_EVAL" ]; then
    for f in $SRC_EVAL; do print_file "$f"; done
else
    echo "❌ Could not find implementation for 'def fit_afm'"
fi

# --- FETCHING FILES FOR FAILURE 2: Misconception Diagnosis Failure ---
echo "📦 [Failure 2] Gathering Misconception Diagnosis files..."
print_file "$TEST_MISC"
if [ -n "$SRC_MISC" ]; then
    for f in $SRC_MISC; do print_file "$f"; done
else
    echo "❌ Could not find implementation for 'def diagnose'"
fi

echo "========================================================================"
echo "✅ All available context files fetched."
echo "========================================================================"