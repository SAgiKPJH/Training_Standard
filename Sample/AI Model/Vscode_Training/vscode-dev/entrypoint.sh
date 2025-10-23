#!/bin/bash

# Wait for MinIO to be ready
echo "Waiting for MinIO to be ready..."
until curl -f http://192.168.70.101:9000/minio/health/ready; do
  echo "MinIO is not ready yet. Waiting..."
  sleep 2
done

# Configure MinIO client
echo "Setting up MinIO client connection..."
if mc alias set myminio http://192.168.70.101:9000 mirero mirero2816! 2>/dev/null; then
  echo "MinIO connection successful"

    # Create traindata bucket if it doesn't exist
    mc mb myminio/traindata --ignore-existing || echo "Bucket creation failed"

    # Mount MinIO bucket using s3fs (if not already mounted)
    if ! mountpoint -q /workspace/myminio; then
      echo "mirero:mirero2816!" > /tmp/.passwd-s3fs
      chmod 600 /tmp/.passwd-s3fs
      s3fs traindata /workspace/myminio -o passwd_file=/tmp/.passwd-s3fs -o url=http://192.168.70.101:9000 -o use_path_request_style -o allow_other -o nonempty -o uid=1000 -o gid=1000 || echo "MinIO mount failed"
    fi
  else
    echo "MinIO connection failed - continuing without MinIO integration"
  fi

# Create code-server config
mkdir -p ~/.config/code-server
cat > ~/.config/code-server/config.yaml << EOF
bind-addr: 0.0.0.0:8080
auth: password
password: system
cert: false
user-data-dir: /workspace/.vscode
EOF

# Copy pyproject.toml to workspace for reference
# cp /pyproject.toml /workspace/ 2>/dev/null || echo "# pyproject.toml not found in base image"

# Generate package list documentation
echo "# Installed Python Packages" > /workspace/INSTALLED_PACKAGES.md
echo "" >> /workspace/INSTALLED_PACKAGES.md
echo "## Conda Environment: daq" >> /workspace/INSTALLED_PACKAGES.md
echo "" >> /workspace/INSTALLED_PACKAGES.md
echo "\`\`\`bash" >> /workspace/INSTALLED_PACKAGES.md
conda run -n daq pip freeze >> /workspace/INSTALLED_PACKAGES.md 2>/dev/null || echo "# pip freeze failed" >> /workspace/INSTALLED_PACKAGES.md
echo "\`\`\`" >> /workspace/INSTALLED_PACKAGES.md
echo "" >> /workspace/INSTALLED_PACKAGES.md
echo "## GPU Information" >> /workspace/INSTALLED_PACKAGES.md
echo "" >> /workspace/INSTALLED_PACKAGES.md
echo "\`\`\`bash" >> /workspace/INSTALLED_PACKAGES.md
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits >> /workspace/INSTALLED_PACKAGES.md 2>/dev/null || echo "# GPU info not available" >> /workspace/INSTALLED_PACKAGES.md
echo "\`\`\`" >> /workspace/INSTALLED_PACKAGES.md
echo "" >> /workspace/INSTALLED_PACKAGES.md
echo "*Generated at container startup*" >> /workspace/INSTALLED_PACKAGES.md

# Start code-server with workspace directory
exec code-server --host 0.0.0.0 --port 8080 --auth password --user-data-dir /workspace/.vscode /workspace
