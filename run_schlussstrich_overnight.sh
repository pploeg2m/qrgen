#!/bin/bash
# Overnight resilient Schlussstrich batch: re-run the gate-respecting full pass
# until every non-empty page has schlussstrich.json (retries transient FSLab errors).
cd /Users/paul/scan2e2x
source .venv/bin/activate

remaining() {
python - <<'PY'
import glob, json
from pathlib import Path
r=0
for p in glob.glob('data4/runs/WiMa_SS26_*/*/oriented.png'):
    l=Path(p).parent
    eg=l/'empty_gate.json'
    empty = json.loads(eg.read_text()).get('is_empty') if eg.exists() else False
    if not empty and not (l/'schlussstrich.json').exists():
        r+=1
print(r)
PY
}

for i in $(seq 1 60); do
  echo "===== PASS $i  $(date '+%F %T') ====="
  python -u projects/qrgen/schlussstrich_all.py
  rem=$(remaining)
  echo "remaining non-empty todo after pass $i: $rem"
  if [ "$rem" -eq 0 ]; then echo "ALL NON-EMPTY DONE $(date '+%F %T')"; break; fi
  sleep 20
done
echo "OVERNIGHT DRIVER EXIT $(date '+%F %T')"
