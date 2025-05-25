#!/bin/bash

# This script creates a markdown file containing the full context of a git repository
# by exporting the directory structure and contents of all tracked files.
# It skips binary files, large files (>500KB), and specific files like pnpm-lock.yaml.
# The output file can be used by LLMs for repository-wide analysis, evaluation and refactoring.

OUTFILE="git_ingest_output.md"
echo -e "## Directory structure:\n" > "$OUTFILE"
# tree -I 'node_modules|.git|.next|dist|build|*.log|*.png|*.jpg|*.jpeg|*.webp|*.svg|*.ico|pnpm-lock.yaml' -a -F >> "$OUTFILE"
git ls-files | grep -v -E '^\.cursor/|^\.documentation/' | tree --fromfile >> "$OUTFILE"
echo -e "\n\n## Files Content:\n" >> "$OUTFILE"

# List all Git tracked files, skipping ignored/untracked
git ls-files | while read -r file; do
  # Skip .cursor and .documentation directories
  if [[ "$file" =~ ^\.cursor/ ]] || [[ "$file" =~ ^\.documentation/ ]]; then
    echo "Skipping directory: $file" >&2
    continue
  fi

  # Skip pnpm-lock.yaml and .gitignore
  if [[ "$file" == "pnpm-lock.yaml" ]] || [[ "$file" == ".gitignore" ]]; then
    echo "Skipping file: $file" >&2
    continue
  fi

  # Skip binary files
  if [[ "$(file --mime "$file")" =~ binary ]]; then
    echo "Skipping binary file: $file" >&2
    continue
  fi

  # Skip large files (>500KB)
  filesize=$(stat -f%z "$file" 2>/dev/null || echo 0)
  if [ "$filesize" -gt 500000 ]; then
    echo "Skipping large file: $file" >&2
    continue
  fi

  echo -e "\n\n================================================" >> "$OUTFILE"
  echo "FILE: $file" >> "$OUTFILE"
  echo "================================================" >> "$OUTFILE"
  cat "$file" >> "$OUTFILE"
done

echo -e "\n\n✅ Export complete: $OUTFILE"