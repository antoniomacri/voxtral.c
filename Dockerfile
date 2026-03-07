# Stage 1: Build
FROM debian:bookworm-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy source code
COPY . .

# Build the application using OpenBLAS backend
RUN make blas

# Stage 2: Runtime
FROM debian:bookworm-slim

# Install runtime dependencies
# libopenblas0: BLAS math library
# curl, ca-certificates: model auto-download
# ffmpeg: audio format conversion (used by --server mode)
RUN apt-get update && apt-get install -y \
    libopenblas0 \
    curl \
    ca-certificates \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the binary and the download script from the builder stage
COPY --from=builder /app/voxtral .
COPY --from=builder /app/download_model.sh .

# Ensure the download script is executable
RUN chmod +x download_model.sh

# Set the entrypoint to the application
ENTRYPOINT ["./voxtral"]

# Default command (prints help)
CMD ["--help"]
