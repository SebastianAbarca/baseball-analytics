FROM python:3.12-slim

WORKDIR /app

# Build deps for pandas / pyarrow wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY baseball_construction/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY baseball_construction/ ./baseball_construction/

# Bake the portraits in.
#
# portraits/ and league_identity_*.json are gitignored, so a build from a clean
# clone has neither, and the COPY above ships an empty portrait directory. That
# would leave every request making a Storage round trip — and a dead dashboard
# whenever the free-tier Supabase project is paused, which it will be.
#
# So fetch them at build time. The running container then reads from local
# disk: fast, and independent of Supabase being up. The runtime fallback in
# callbacks._storage_download still covers anything uploaded after this build,
# so new data does not need a redeploy.
#
# Reads with SUPABASE_ANON_KEY against an anon-readable bucket, so there is no
# build secret to manage. NEVER fails the build — if Storage is unreachable the
# image is still correct, just without the local copy.
ARG SUPABASE_URL=""
ARG SUPABASE_ANON_KEY=""
RUN SUPABASE_URL="$SUPABASE_URL" SUPABASE_ANON_KEY="$SUPABASE_ANON_KEY" \
    python baseball_construction/scripts/fetch_artifacts.py || true

WORKDIR /app/baseball_construction

EXPOSE 8050

# 1 worker — Dash keeps in-process state (the portrait LRU, the league-average
# memos), and a second worker would duplicate rather than share it.
CMD ["gunicorn", "-w", "1", "--timeout", "120", "--bind", "0.0.0.0:8050", \
     "--worker-class", "sync", "--log-level", "info", \
     "dashboard.app:server"]
