FROM nvcr.io/nvidia/pytorch:22.05-py3
# Set the working directory inside the container
WORKDIR /TaBuddy

# Install system dependencies and clean up to reduce image size
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libldap2-dev \
    libsasl2-dev \
    libssl-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first (to take advantage of Docker cache)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY ./entrypoint.sh /entrypoint.sh
