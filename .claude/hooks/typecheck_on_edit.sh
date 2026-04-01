#!/bin/bash
# PostToolUse hook: run py_compile (syntax) then mypy (types) on any edited Python file.
# Errors are injected back into Claude Code as additional context for self-correction.

REPO="$PWD"
input=$(cat)

# Extract file path from tool input
file_path=$(echo "$input" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(d.get('tool_input', {}).get('file_path', ''))
" 2>/dev/null)

# Only check Python files that exist
if [[ "$file_path" != *.py ]] || [[ ! -f "$file_path" ]]; then
  exit 0
fi

errors=""

# Stage 1: syntax check (fast, catches errors before mypy)
syntax_out=$(cd "$REPO" && uv run python -m py_compile "$file_path" 2>&1)
if [ $? -ne 0 ]; then
  errors="[Syntax Error]\n$syntax_out"
fi

# Stage 2: mypy type check (only if syntax is clean)
if [ -z "$errors" ]; then
  mypy_out=$(cd "$REPO" && uv run mypy "$file_path" --ignore-missing-imports --no-error-summary 2>&1)
  if [ $? -ne 0 ]; then
    errors="[Type Error]\n$mypy_out"
  fi
fi

# If any errors, send feedback to Claude Code
if [ -n "$errors" ]; then
  python3 -c "
import json, sys
errors = sys.argv[1]
file = sys.argv[2]
output = {
    'hookSpecificOutput': {
        'hookEventName': 'PostToolUse',
        'additionalContext': f'Type checker found errors in {file}:\n{errors}\nFix these errors before proceeding.'
    }
}
print(json.dumps(output))
" "$errors" "$file_path"
fi
