#!/bin/bash
# Translates the dumped shaders and runs spirv-val on each; prints error kinds.
cd /d/keylime || exit 1
V=/d/keylime-tools/spirv-build/tools/spirv-val.exe
rm -f logs/shaders/spv/*.spv
./build/keylime.exe shader logs/shaders --spirv-only > logs/val.txt 2>&1
ok=0; bad=0; > logs/val-errors.txt
for f in logs/shaders/spv/*.spv; do
  if out=$($V --target-env vulkan1.3 "$f" 2>&1); then ok=$((ok+1)); else bad=$((bad+1)); echo "$out" | head -1 >> logs/val-errors.txt; fi
done
echo "spirv-val: $ok valid, $bad invalid"
sed -E 's/[0-9]+/N/g; s/%[A-Za-z_0-9]+/%x/g' logs/val-errors.txt | sort | uniq -c | sort -rn | head -${1:-8}
