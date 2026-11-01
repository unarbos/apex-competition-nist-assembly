# syntax=docker/dockerfile:1
# nist_assembly REFEREE image: MuJoCo simulation + scoring.
# Vendored gym_v1 (do NOT build FROM apex-referee-base). Build context = repo root:
#   docker build -f referee/Dockerfile -t nist-assembly-referee .
# Release builds pass a build secret:  --secret id=heldout_seed,env=NIST_HELDOUT_SEED
FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app /data && chown -R 1000:1000 /app /data
WORKDIR /app

COPY referee/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY referee/gym_v1/ /app/gym_v1/
COPY referee/nist_sim/ /app/nist_sim/
COPY referee/referee.py /app/referee.py

RUN --mount=type=secret,id=heldout_seed,required=false \
    if [ -s /run/secrets/heldout_seed ]; then \
      python -m nist_sim.reserve "$(cat /run/secrets/heldout_seed)" > /app/nist_sim/reserved.json; \
    fi

USER 1000
ENV OMP_NUM_THREADS=1 MUJOCO_GL=disable NIST_PROFILE=v06
CMD ["python", "/app/referee.py"]
