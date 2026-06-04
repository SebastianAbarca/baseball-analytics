FROM python:3.12-slim

WORKDIR /app

# System deps for pybaseball / pandas / pyarrow
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY baseball_construction/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY baseball_construction/ ./baseball_construction/

# Reference data (small — Chadwick crosswalk for debut years)
# data/raw parquets are excluded — portraits served from Supabase Storage

WORKDIR /app/baseball_construction

# Pre-create cache directories (populated from Supabase Storage at runtime)
RUN mkdir -p data/processed/portraits data/processed

EXPOSE 8050

# 1 worker — Dash apps use shared in-process state
CMD ["gunicorn", "-w", "1", "--timeout", "120", "--bind", "0.0.0.0:8050", \
     "--worker-class", "sync", "--log-level", "info", \
     "dashboard.app:server"]
