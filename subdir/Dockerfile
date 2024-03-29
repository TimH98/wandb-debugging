# ----- stage 1: build TIMH DEBUG 2 -----
FROM python:3.11 as build

# requirements section depends on pip vs conda, and presence of buildx
ENV PIP_PROGRESS_BAR off

RUN python -m venv /env
# make sure we install into the env
ENV PATH="/env/bin:$PATH"

COPY src/requirements.txt ./
RUN --mount=type=cache,mode=0777,target=/root/.cache/pip pip install -r requirements.txt


# ----- stage 2: base -----

FROM python:3.11-buster as base


COPY --from=build /env /env
ENV PATH="/env/bin:$PATH"

ENV SHELL /bin/bash

# some resources (eg sagemaker) must run on root

RUN useradd     --create-home     --no-log-init     --shell /bin/bash     --gid 0     --uid 1000     timh || echo ""

USER timh

WORKDIR /home/timh
RUN chown -R 1000 /home/timh

# make artifacts cache dir unrelated to build
RUN mkdir -p /home/timh/.cache && chown -R 1000 /home/timh/.cache

# copy code/etc
COPY --chown=1000 src/ /home/timh

ENV PYTHONUNBUFFERED=1

RUN echo "hello from Dockerfile in SUBDIR"

ENTRYPOINT ["python3.11", "train.py"]