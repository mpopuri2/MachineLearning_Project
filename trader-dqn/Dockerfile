# ============================
# 1️⃣ Base Image
# ============================
FROM python:3.10-slim

# Prevent interactive prompts during package installs
ENV DEBIAN_FRONTEND=noninteractive

# ============================
# 2️⃣ System Dependencies
# ============================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    ca-certificates \
    libffi-dev \
    libssl-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# ============================
# 3️⃣ Working Directory
# ============================
WORKDIR /app

# ============================
# 4️⃣ Copy project files
# ============================
COPY . /app

# ============================
# 5️⃣ Install Python packages
# ============================
# Install TA-Lib dependencies first
RUN pip install --upgrade pip setuptools wheel \
    && pip install numpy pandas matplotlib yfinance torch ta PyYAML scikit-learn

# ============================
# 6️⃣ Environment variables
# ============================
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ============================
# 7️⃣ Default command
# ============================
# You can change this to `make train` or `make eval`
CMD ["python", "-m", "src.train"]