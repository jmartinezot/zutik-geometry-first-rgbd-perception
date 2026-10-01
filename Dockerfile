FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Open3D's registration path is CPU-oriented, but its wheel still imports a
# small set of system OpenGL/runtime libraries. No GUI or CUDA stack is added.
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        libegl1 \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libusb-1.0-0 \
        libsm6 \
        libx11-6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/zutik-geometry

# Install dependencies before copying documentation/tests so ordinary source
# edits keep the expensive Open3D wheel layer cached.
COPY pyproject.toml README.md ./
COPY zutik_geometry ./zutik_geometry
RUN python -m pip install --upgrade pip \
    && python -m pip install '.[test,bop]'

COPY . .

# The image intentionally contains no benchmark data or model checkpoints.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin zutik \
    && chown -R zutik:zutik /opt/zutik-geometry
USER zutik

ENTRYPOINT ["python", "-m", "zutik_geometry"]
CMD ["--help"]
