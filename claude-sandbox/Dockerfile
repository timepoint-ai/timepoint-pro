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
