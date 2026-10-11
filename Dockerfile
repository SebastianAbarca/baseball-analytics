FROM python:3.12-slim

WORKDIR /app

# Build deps for pandas / pyarrow wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY baseball_construction/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY baseball_construction/ ./baseball_construction/

# No data in the image. The dashboard reads everything it shows from the
# serving tables in Supabase (dashboard/store.py), with SUPABASE_URL and
# SUPABASE_ANON_KEY from the runtime environment. Nothing is baked in and
# nothing is fetched at build time, so the image is the same whatever was
# published when it was built.

WORKDIR /app/baseball_construction

EXPOSE 8050

# 1 process — store.py keeps a small in-process cache (portraits, league
# seasons, the tag catalogue), and a second process would duplicate rather
# than share it.
#
# Threaded, not sync. A page load fires ~20 callbacks at once and each can wait
# on a database round trip; a sync worker answered them one at a time and the
# first page took ~30s. store.py is thread-safe (one lock per cache key).
CMD ["gunicorn", "-w", "1", "--threads", "8", "--worker-class", "gthread", \
     "--timeout", "120", "--bind", "0.0.0.0:8050", "--log-level", "info", \
     "dashboard.app:server"]
