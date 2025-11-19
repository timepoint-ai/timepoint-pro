#!/bin/bash
# Frontend Management Script - Manages Quarto Preview Server

set -e

# Kill any existing Quarto processes
echo "🔍 Checking for existing Quarto processes..."
if pgrep -f "quarto preview" > /dev/null; then
    echo "🛑 Stopping existing Quarto processes..."
    pkill -f "quarto preview" || true
    sleep 1
fi

# Navigate to dashboards directory
cd "$(dirname "$0")"

# Start Quarto preview
echo "🚀 Starting Quarto preview server on http://localhost:8888"
echo "📖 Dashboard: http://localhost:8888/index.html"
echo "🔍 Browse Runs: http://localhost:8888/runs.html"
echo "📊 Analytics: http://localhost:8888/analytics.html"
echo ""
echo "Press Ctrl+C to stop"

quarto preview --port 8888 --no-browser
