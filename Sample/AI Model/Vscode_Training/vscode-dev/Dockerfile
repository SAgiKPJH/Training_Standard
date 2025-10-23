FROM mirero/daq:train-develop-sam2

# Install code-server
RUN curl -fsSL https://code-server.dev/install.sh | sh

# Install MinIO client
RUN wget https://dl.min.io/client/mc/release/linux-amd64/mc -O /usr/local/bin/mc && \
    chmod +x /usr/local/bin/mc

# Create workspace directory
RUN mkdir -p /workspace

# Set working directory
WORKDIR /workspace

# Install VSCode extensions (Python, Jupyter, etc.)
RUN code-server --install-extension ms-python.python \
    && code-server --install-extension ms-toolsai.jupyter \
    && code-server --install-extension ms-toolsai.jupyter-keymap \
    && code-server --install-extension ms-toolsai.jupyter-renderers

# Create training-develop directory
RUN mkdir -p /training-develop

# Install s3fs-fuse for MinIO bucket mounting
RUN apt-get update && apt-get install -y s3fs && rm -rf /var/lib/apt/lists/*

# Create directories for MinIO mount
RUN mkdir -p /workspace/traindata

# Set environment variables
ENV PASSWORD=system
ENV SUDO_PASSWORD=system

# Copy entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose port
EXPOSE 8080

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]
