#!/bin/bash
# PostToolUse hook: after parse_altumint_docling.py runs,
# invoke the docling-parser-verifier agent via claude CLI,
# save the full report to a timestamped file, and inject
# only a brief "done, see file" message into the main conversation.

input=$(cat)

# Extract the bash command from tool input
command=$(echo "$input" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(d.get('tool_input', {}).get('command', ''))
" 2>/dev/null)

# Only fire when parse_altumint_docling.py is executed as a python script,
# not when referenced by mypy, grep, cat, or other tools.
if ! echo "$command" | grep -qE "(python|uv run python)[^|]*parse_documents"; then
    exit 0
fi

REPORT_DIR="$PWD/data/altumint/docling_inspection"
REPORT_PATH="$REPORT_DIR/verification_report_$(date +%Y%m%d_%H%M%S).md"
mkdir -p "$REPORT_DIR"

# Run the verifier agent in non-interactive mode, save full output to file.
# --allowed-tools restricts what the agent can do (Read, Bash for fitz rendering, Glob, Grep).
claude -p "Verify the latest Docling parsing output for the Altumint dataset. Check all 7 documents (DC001, DC002, DC004, DC005, TM001, TM002, wiring). Produce the full structured verification report." \
    --agent docling-parser-verifier \
    --allowed-tools "Bash,Read,Glob,Grep" \
    --output-format text \
    > "$REPORT_PATH" 2>&1

EXIT_CODE=$?

# Inject a brief status message back into the main conversation — no full report
python3 -c "
import json, sys
report_path = sys.argv[1]
exit_code = int(sys.argv[2])

if exit_code == 0:
    msg = f'Docling verification complete. Full report saved to: {report_path}\nRead that file to review findings before proceeding to QA generation.'
else:
    msg = f'Docling verifier exited with code {exit_code}. Partial report (if any) at: {report_path}'

output = {
    'hookSpecificOutput': {
        'hookEventName': 'PostToolUse',
        'additionalContext': msg
    }
}
print(json.dumps(output))
" "$REPORT_PATH" "$EXIT_CODE"
