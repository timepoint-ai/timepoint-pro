#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
PROJECT_DIR="claude-sandbox"
IMAGE_NAME="claude-code-dev"
COLIMA_CPU=4
# FIX: Colima expects the memory flag to be a number (GiB), not a string with 'G' suffix.
COLIMA_MEMORY="6" 

echo "=========================================================="
echo " Starting Claude Code Sandbox Setup (Colima/Docker/Node)"
echo " Project Directory: $PROJECT_DIR"
echo " Docker Image: $IMAGE_NAME"
# Note: Updated output to reflect the corrected memory value for Colima
echo " VM Specs: ${COLIMA_CPU} CPU, ${COLIMA_MEMORY} GiB RAM"
echo "=========================================================="

# 1. Check for Homebrew (The macOS package manager)
if ! command -v brew &> /dev/null
then
    echo "🚨 Homebrew not found!"
    echo "Please install Homebrew first: /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    exit 1
fi
echo "✅ Homebrew found."

# 2. Install Docker CLI and Colima
echo "⚙️ Installing docker and colima via Homebrew..."
# Using 'brew install' will automatically handle upgrades if they are outdated.
brew install docker colima

# 3. Start Colima (Docker Desktop alternative)
echo "🚀 Starting Colima with specified resources..."
# Note: --memory now uses the corrected variable COLIMA_MEMORY="6"
colima start --cpu "$COLIMA_CPU" --memory "$COLIMA_MEMORY" --disk 100 --runtime docker

# Check if Docker is running and configured correctly
if docker info &> /dev/null
then
    echo "✅ Colima and Docker daemon running successfully."
else
    echo "❌ Failed to connect to Docker daemon. Please check Colima status."
    echo "Run 'colima status' for details."
    exit 1
fi

# 4. Create the Project Directory and Dev Files
echo "📁 Creating project directory: $PROJECT_DIR"
# The -p flag ensures that if the directory exists, it does not throw an error.
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Create the Dockerfile
echo "🏗️ Writing Dockerfile..."
cat << EOF > Dockerfile
# Use a Node.js image as the base, as claude code is an npm package
FROM node:20-alpine

# FIX: Install bash to provide a robust POSIX shell environment for the Claude CLI
RUN apk update && apk add --no-cache bash

# Set the working directory inside the container
WORKDIR /code

# Install the official Anthropic Claude Code CLI globally
# The 'claude' executable will be available in the container's PATH
RUN npm install -g @anthropic-ai/claude-code

# Set a non-root user for security (optional but recommended)
# Set /bin/bash as the default shell for the new user
RUN adduser -D appuser -s /bin/bash
USER appuser

# Set the entrypoint to the claude CLI
ENTRYPOINT ["claude"]
EOF

# Define the content of run.sh
RUN_SH_CONTENT=$(cat << EOF
#!/bin/bash
# Starts the Docker container, mounts the current directory's parent (the project root) 
# to /code inside the container, and executes the 'claude' command.

# 1. Clean up any previous stopped container with the same name
# This prevents 'container name already in use' errors. Output is suppressed.
docker rm -f claude-sandbox >/dev/null 2>&1

# 2. Check for the 'yolo' argument to enable the dangerous, permission-skipping mode.
# This allows Claude to execute commands without explicitly asking for permission.
CLAUDE_ARGS=()
if [[ "\$1" == "yolo" ]]; then
  echo "⚠️ WARN: Running Claude in 'YOLO Mode' (--dangerously-skip-permissions). Be cautious! 









[Image of a warning sign]




"
  CLAUDE_ARGS+=("--dangerously-skip-permissions")
  shift # Remove 'yolo' from the arguments passed to claude
fi

# 3. Run the container
# Ensure the container runs interactively (-it), deletes itself on exit (--rm),
# and mounts the local code directory.
# Mount the parent directory (..) to /code inside the container.
# This allows Claude to see the entire project root when run from inside the sandbox folder.
# Pass the collected claude arguments and the rest of the user's arguments.
docker run \\
  --name claude-sandbox \\
  -it \\
  --rm \\
  -v "\$(pwd)/..:/code" \\
  $IMAGE_NAME "\${CLAUDE_ARGS[@]}" "\$@"
EOF
)

# 5. Full Purge: Clean up old image and container before building
echo "🧹 Fully purging previous Docker image and container..."
# Stop and remove the old container instance if it exists
docker rm -f claude-sandbox >/dev/null 2>&1 || true
# Remove the old image if it exists. The -f flag forces removal even if it's currently used (which it shouldn't be).
docker rmi -f "$IMAGE_NAME" >/dev/null 2>&1 || true 

# 6. Build the Docker Image
echo "🛠️ Building Docker image: $IMAGE_NAME..."
# set -e ensures that the script stops here if the build fails.
docker build -t "$IMAGE_NAME" .

# 7. Write and permission run.sh only if the build succeeded (due to 'set -e')
echo "▶️ Writing run.sh execution script..."
echo "$RUN_SH_CONTENT" > run.sh
chmod +x run.sh

echo "=========================================================="
echo "             🎉 Setup Complete! 🎉"
echo "=========================================================="
echo "Next Steps:"
echo "1. Navigate to the project directory (which contains the sandbox folder):"
echo "   cd .. && cd $PROJECT_DIR"
echo "2. Run the Claude Code Sandbox in standard mode (requires confirmation for actions):"
echo "   ./run.sh"
echo "3. Run the Claude Code Sandbox in 'YOLO Mode' (no confirmation, high autonomy):"
echo "   ./run.sh yolo"
echo ""
echo "When you run './run.sh', the container's /code directory will be mapped to the parent directory where your main project files are located."
echo "You may need to provide your Anthropic API Key the first time you run 'claude'."

cd ..
