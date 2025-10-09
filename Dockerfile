# Dockerfile for pyOMA2
# Compatible with x86_64 and ARM64 (Apple Silicon)

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libqt5gui5 libqt5widgets5 libqt5core5a libqt5x11extras5 \
    qt5-qmake qtbase5-dev \
    libx11-6 libxext6 libxrender1 libxrandr2 libxinerama1 libxcursor1 libxi6 libxtst6 \
    libglib2.0-0 libsm6 libxkbcommon-x11-0 \
    libgl1 libglx-mesa0 libgl1-mesa-dev libglu1-mesa libgomp1 libosmesa6 \
    libxcb-xinerama0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
    libxcb-render-util0 libxcb-shape0 libxcb-xfixes0 libxcb-xkb1 \
    python3-pyqt5 python3-pyqt5.qtsvg python3-pyqt5.qtopengl pyqt5-dev-tools \
    gcc g++ make git python3-tk tk-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md LICENSE CHANGELOG.md ./
COPY src/ ./src/
COPY Examples/ ./Examples/
COPY docs/ ./docs/

RUN pip install --upgrade pip setuptools wheel

RUN pip install --no-cache-dir \
    numpy>=1.25 pandas>=2.0.3 scipy>=1.9.3 pydantic>=2.5.1 tqdm>=4.66.1 \
    matplotlib>=3.7.4 openpyxl>=3.1.3 scikit-learn>=1.3.2 \
    "vtk>=9.3" pyvista pyvistaqt \
    trame trame-vuetify trame-vtk

RUN pip install --no-deps -e .

# Set Qt platform plugin path for PyQt5 (auto-detect architecture)
RUN if [ "$(uname -m)" = "aarch64" ]; then \
        echo 'export QT_QPA_PLATFORM_PLUGIN_PATH=/usr/lib/aarch64-linux-gnu/qt5/plugins' >> /etc/environment; \
    else \
        echo 'export QT_QPA_PLATFORM_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/qt5/plugins' >> /etc/environment; \
    fi

RUN useradd -m -u 1000 pyoma && chown -R pyoma:pyoma /app
USER pyoma

ENV DISPLAY=:0
WORKDIR /app/Examples

CMD ["python3"]
