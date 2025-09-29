FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

# Prevent __pycache__ creation
ENV PYTHONDONTWRITEBYTECODE=1

# Define the service home directory
ARG service_home="/home/EasyOCR"

# Configure apt and install packages
RUN apt-get update -y && \
    apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-dev \
    git \
    # cleanup
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists

# Create the service directory
RUN mkdir -p "$service_home"

# Copy the local project (your custom EasyOCR) into the container
COPY . "$service_home"

# Create a dummy README.md to avoid FileNotFoundError in setup.py
RUN touch "$service_home/README.md"

# Install dependencies and build
RUN cd "$service_home" \
    && python -m pip install --no-cache-dir -r requirementsDocker.txt \
    && python -m pip install --no-cache-dir ultralytics "fastapi[all]" python-dotenv requests \
    && python -m pip install --no-cache-dir --upgrade numpy \
    && python setup.py build_ext --inplace -j 4 \
    && python -m pip install -e .

# Expose the port for the FastAPI app
EXPOSE 8000

# Set the working directory to the trainer subdirectory
WORKDIR "$service_home/trainer"

# Run the FastAPI app from App.py (adjusted for new WORKDIR)
CMD ["uvicorn", "App:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]