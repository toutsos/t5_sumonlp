#!/bin/bash

# Paths to original files
input_file="/home/angelos.toutsios.gr/workspace/sumonlp/data/combined-eng.txt-0"  # Full input file with English sentences
output_file="/home/angelos.toutsios.gr/workspace/sumonlp/data/combined-log.txt-0"  # Full output file with logical forms

# Paths to new files
new_input_file="input_sentences_500k.txt"  # New input file with last 500k English sentences
new_output_file="output_logical_500k.txt"  # New output file with last 500k logical forms

# Extract the last 500,000 lines and save them to new files
tail -n 500000 "$input_file" > "$new_input_file"
tail -n 500000 "$output_file" > "$new_output_file"

# Notify the user
echo "New files created: $new_input_file, $new_output_file"
