# Stage 1: Build
FROM debian:bookworm-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only the source files needed for building
COPY Makefile *.c *.h ./

# Build with AVX-512 BF16 backend (AMD Zen 4+/Intel SPR+, no OpenBLAS needed)
RUN make avx512

# Stage 2: Runtime
FROM debian:bookworm-slim

# Install runtime dependencies
# curl, ca-certificates: model auto-download
# ffmpeg: audio format conversion (used by --server mode)
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy the binary and the download script from the builder stage
COPY --from=builder /app/voxtral /usr/local/bin/voxtral

# Set the entrypoint to the application
ENTRYPOINT ["voxtral"]

# Default command (prints help)
CMD ["--help"]
