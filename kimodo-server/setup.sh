#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Kimodo Motion Generation API Setup ==="
echo ""

# ── 1. Check prerequisites ──────────────────────────────────────────
echo "[1/5] Checking prerequisites..."

if ! command -v docker &>/dev/null; then
    echo "ERROR: Docker not found. Install Docker with GPU support first."
    exit 1
fi

if ! docker compose version &>/dev/null; then
    echo "ERROR: Docker Compose v2 not found."
    exit 1
fi

if ! docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &>/dev/null; then
    echo "ERROR: NVIDIA Container Toolkit not working. Ensure GPU drivers and nvidia-container-toolkit are installed."
    exit 1
fi

echo "  Docker, Compose, and GPU support OK."

# ── 2. Check HF token ───────────────────────────────────────────────
echo ""
echo "[2/5] Checking Hugging Face token..."

HF_TOKEN_PATH="${HOME}/.cache/huggingface/token"
if [ ! -f "$HF_TOKEN_PATH" ]; then
    echo ""
    echo "  No HF token found at $HF_TOKEN_PATH"
    echo "  You need a token with access to meta-llama/Meta-Llama-3-8B-Instruct."
    echo ""
    echo "  1. Create account: https://huggingface.co/join"
    echo "  2. Accept Llama 3 license: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct"
    echo "  3. Create Read token: https://huggingface.co/settings/tokens"
    echo ""
    read -sp "  Paste your HF token (hf_...): " HF_TOKEN
    echo ""
    mkdir -p "$(dirname "$HF_TOKEN_PATH")"
    echo "$HF_TOKEN" > "$HF_TOKEN_PATH"
    echo "  Token saved to $HF_TOKEN_PATH"
else
    echo "  Token found at $HF_TOKEN_PATH"
fi

# ── 3. Clone Kimodo ─────────────────────────────────────────────────
echo ""
echo "[3/5] Cloning Kimodo repository..."

if [ ! -d "kimodo" ]; then
    git clone https://github.com/nv-tlabs/kimodo.git
    cd kimodo
    git clone https://github.com/nv-tlabs/kimodo-viser.git
    cd ..
    echo "  Kimodo cloned."
else
    echo "  Kimodo directory already exists, skipping clone."
fi

# ── 4. Build Docker images ──────────────────────────────────────────
echo ""
echo "[4/5] Building Docker images (this takes 5-10 minutes on first run)..."

# Build the base Kimodo image
cd kimodo
docker compose build
cd ..

# Build the API image on top
docker build -f Dockerfile.api -t kimodo-api:latest .

echo "  Images built successfully."

# ── 5. Start services ───────────────────────────────────────────────
echo ""
echo "[5/5] Starting services..."

docker compose up -d text-encoder
echo "  Text encoder starting (first run downloads Llama 3 weights, ~16GB)..."
echo "  Waiting for text encoder to be healthy..."

# Wait for healthcheck
TIMEOUT=300
ELAPSED=0
while [ $ELAPSED -lt $TIMEOUT ]; do
    STATUS=$(docker inspect --format='{{.State.Health.Status}}' text-encoder 2>/dev/null || echo "not_running")
    if [ "$STATUS" = "healthy" ]; then
        echo "  Text encoder is healthy."
        break
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    echo "  Waiting... (${ELAPSED}s / ${TIMEOUT}s, status: ${STATUS})"
done

if [ "$STATUS" != "healthy" ]; then
    echo "  WARNING: Text encoder did not become healthy within ${TIMEOUT}s."
    echo "  Check logs: docker compose logs text-encoder"
    echo "  It may still be downloading model weights. You can start the API manually later."
    exit 1
fi

# Start API
docker compose up -d api
echo "  API starting..."

# Wait for API
sleep 10
if curl -s http://localhost:8420/health > /dev/null 2>&1; then
    echo ""
    echo "=== Setup complete! ==="
    echo ""
    echo "API is running at http://localhost:8420"
    echo ""
    echo "Endpoints:"
    echo "  GET  /health        - status check"
    echo "  POST /generate      - JSON qpos trajectory"
    echo "  POST /generate/csv  - CSV file download"
    echo "  POST /generate/pt   - ProtoMotions .pt MotionLib"
    echo ""
    echo "Test:"
    echo "  curl -s -X POST http://localhost:8420/generate \\"
    echo "    -H 'Content-Type: application/json' \\"
    echo "    -d '{\"prompt\":\"A person walks forward\",\"duration\":2.0,\"diffusion_steps\":50}'"
    echo ""
    echo "To expose publicly: ngrok http 8420"
    echo "To start the visual demo: docker compose --profile demo up -d demo"
    echo "To stop everything: docker compose --profile demo down"
else
    echo "  API not responding yet. Check logs: docker compose logs api"
fi
