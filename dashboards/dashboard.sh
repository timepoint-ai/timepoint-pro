#!/bin/bash
# Dashboard Launcher - Starts both Backend API and Frontend

set -e

# Navigate to dashboards directory
cd "$(dirname "$0")"

# Kill existing processes
echo "🔍 Checking for existing processes..."
pkill -f "api/server.py" 2>/dev/null || true
pkill -f "quarto preview" 2>/dev/null || true

# Force kill any processes on the ports
lsof -ti:8888 2>/dev/null | xargs kill -9 2>/dev/null || true
lsof -ti:8000 2>/dev/null | xargs kill -9 2>/dev/null || true
sleep 2

# Start backend in background with error logging
echo "🚀 Starting Backend API on http://localhost:8000"
cd api
python3.10 server.py > /tmp/dashboard_backend.log 2>&1 &
BACKEND_PID=$!
cd ..

# Wait for backend to start with retries
echo "⏳ Waiting for backend to start..."
MAX_RETRIES=10
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    sleep 1
    if curl -s http://localhost:8000/ > /dev/null 2>&1; then
        echo "✅ Backend API running (PID: $BACKEND_PID)"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
        echo "❌ Backend failed to start after ${MAX_RETRIES} seconds"
        echo "📋 Check logs: tail /tmp/dashboard_backend.log"
        tail -20 /tmp/dashboard_backend.log
        exit 1
    fi
done

# Start frontend
echo "🚀 Starting Frontend on http://localhost:8888"
echo ""
echo "📊 Dashboard System Ready:"
echo "   • Frontend: http://localhost:8888"
echo "   • Backend:  http://localhost:8000"
echo "   • API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Trap to kill backend when script exits
trap "echo ''; echo '🛑 Stopping servers...'; kill $BACKEND_PID 2>/dev/null; pkill -f 'quarto preview' 2>/dev/null; echo '✅ Stopped'; exit 0" EXIT INT TERM

quarto preview --port 8888 --no-browser
