#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

OUTPUT_DIR="$PROJECT_ROOT/app/adapters/proto_generated"
mkdir -p "$OUTPUT_DIR"

python -m grpc_tools.protoc \
  -I="$PROJECT_ROOT/proto" \
  --python_out="$OUTPUT_DIR" \
  "$PROJECT_ROOT/proto/analysis.proto"

touch "$OUTPUT_DIR/__init__.py"

echo "Proto compiled successfully -> $OUTPUT_DIR/analysis_pb2.py"
