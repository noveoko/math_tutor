#!/bin/bash

# Define the name of your final output file
OUTPUT_FILE="combined_scripts.py"

# List of files to find and concatenate
FILES=(
    "linear_equation.py"
    "fraction.py"
    "inequality.py"
    "polynomial.py"
    "test_safety.py"
)

# Initialize the output file (this clears it if it already exists)
> "$OUTPUT_FILE"

echo "Concatenating files into $OUTPUT_FILE..."

# Loop through each file in the array
for FILE in "${FILES[@]}"; do
    # Search for the file recursively from the current directory
    FOUND_PATH=$(find . -name "$FILE" -not -path "./$OUTPUT_FILE" | head -1)

    if [[ -n "$FOUND_PATH" ]]; then
        echo "Adding $FOUND_PATH..."

        echo -e "\n# ==========================================" >> "$OUTPUT_FILE"
        echo "# START OF FILE: $FOUND_PATH" >> "$OUTPUT_FILE"
        echo -e "# ==========================================\n" >> "$OUTPUT_FILE"

        cat "$FOUND_PATH" >> "$OUTPUT_FILE"
    else
        echo "Warning: '$FILE' not found anywhere under $(pwd). Skipping."
    fi
done

echo "Process complete! Your combined file is: $OUTPUT_FILE"