FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive

# Install git and libgl1 (Required for cv2)
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone repo
RUN git clone https://github.com/zengxianyu/PPD-examples.git .

# Install dependencies
RUN pip install -r requirements.txt
RUN pip install git+https://github.com/zengxianyu/structured-noise

# Install opencv specifically (headless is better for servers/docker)
RUN pip install opencv-python-headless scikit-learn

CMD ["/bin/bash"]