#!/bin/bash
# Starts the Docker container, mounts the current directory's parent (the project root) 
# to /code inside the container, and executes the 'claude' command.

# 1. Clean up any previous stopped container with the same name
# This prevents 'container name already in use' errors. Output is suppressed.
docker rm -f claude-sandbox >/dev/null 2>&1

# 2. Check for the 'yolo' argument to enable the dangerous, permission-skipping mode.
# This allows Claude to execute commands without explicitly asking for permission.
CLAUDE_ARGS=()
if [[ "$1" == "yolo" ]]; then
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
docker run \
  --name claude-sandbox \
  -it \
  --rm \
  -v "$(pwd)/..:/code" \
  claude-code-dev "${CLAUDE_ARGS[@]}" "$@"
