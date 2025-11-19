#!/bin/bash
# Backend Management Script - Manages FastAPI Server

set -e

# Kill any existing API server processes
echo "🔍 Checking for existing API server processes..."
if pgrep -f "api/server.py" > /dev/null; then
    echo "🛑 Stopping existing API server processes..."
    pkill -f "api/server.py" || true
    sleep 1
fi

# Navigate to dashboards/api directory
cd "$(dirname "$0")/api"

# Start FastAPI server
echo "🚀 Starting Timepoint Dashboard API on http://localhost:8000"
echo "📖 API docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop"

python3.10 server.py
