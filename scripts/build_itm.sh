#!/usr/bin/env bash
# Build NTIA's Irregular Terrain Model (Longley-Rice) as a shared library for sp/itm.py.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d itm-src ] || git clone --depth 1 https://github.com/NTIA/itm.git itm-src
sed -i.bak 's#\.\.\\include\\#../include/#g' itm-src/src/*.cpp itm-src/include/*.h 2>/dev/null || true
mkdir -p itm-src/build
EXT=so; [ "$(uname)" = Darwin ] && EXT=dylib
g++ -O2 -fPIC -shared -std=c++11 '-D__declspec(x)=' -Iitm-src/include itm-src/src/*.cpp -o itm-src/build/libitm.$EXT
echo "built itm-src/build/libitm.$EXT"
