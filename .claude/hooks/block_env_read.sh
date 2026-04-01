#!/bin/bash
# Blocks Claude Code from reading .env files
input=$(cat)
file_path=$(echo "$input" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('tool_input',{}).get('file_path',''))" 2>/dev/null)

if echo "$file_path" | grep -qE '(^|/)\.env(\.|$|$)'; then
  echo '{"continue": false, "stopReason": "Access to .env files is blocked for security."}'
  exit 0
fi
