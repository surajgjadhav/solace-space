FROM python:3.10-slim

WORKDIR /app

# Set home to the user's home directory
ENV GRADIO_ALLOW_FLAGGING=never \
    GRADIO_NUM_PORTS=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_THEME=huggingface

# Install system dependencies for OpenBLAS
RUN apt-get update && apt-get install -y \
    gcc g++ make cmake git libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

COPY . .

RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user
# Switch to the "user" user
USER user

# Set the working directory to the user's home directory
# WORKDIR $HOME/app

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
# COPY --chown=user . $HOME/app
RUN ls

CMD ["python", "app.py"]