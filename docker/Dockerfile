# Use use previous versions, modify these variables
# ARG PYTORCH="1.9.0"
# ARG CUDA="11.1"

ARG PYTORCH="2.0.1"
ARG CUDA="11.7"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# setup timezone
RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -s /usr/share/zoneinfo/Asia/Singapore /etc/localtime && \
    apt-get update && \
    apt-get install -q -y --no-install-recommends tzdata && \
    rm -rf /var/lib/apt/lists/*

##############################################
# You should modify this to match your GPU compute capability
# ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6"
##############################################

ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

# Install dependencies
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update && apt-get install -y --no-install-recommends \
    git \
    ninja-build \
    cmake \
    build-essential \
    libopenblas-dev \
    xterm \
    xauth \
    openssh-server \
    tmux \
    wget \
    vim \
    sudo \
    mate-desktop-environment-core \
    pip \
    && rm -rf /var/lib/apt/lists/*

# For faster build, use more jobs.
ENV MAX_JOBS=4
RUN git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine" && cd MinkowskiEngine; python setup.py install --force_cuda --blas=openblas


ARG UID=1000

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --uid ${UID} --shell /bin/bash user
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user
ENV HOME=/home/user
RUN chmod 777 /home/user
WORKDIR /home/user

# Install zsh related
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.1/zsh-in-docker.sh)" -- \
    -t ys \
    -p git \
    -p https://github.com/zsh-users/zsh-autosuggestions \
    -p https://github.com/zsh-users/zsh-completions \
    -p https://github.com/zsh-users/zsh-history-substring-search \
    -p https://github.com/zsh-users/zsh-syntax-highlighting
RUN sudo chsh -s /bin/zsh

RUN sudo echo "set-option -g default-shell /bin/zsh" >> ~/.tmux.conf


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY entrypoint.sh /
RUN sudo chmod +x /entrypoint.sh

RUN echo "/opt/conda/lib/python3.10/site-packages/torch/lib" | sudo tee /etc/ld.so.conf.d/torch.conf > /dev/null && sudo ldconfig



ENTRYPOINT ["/entrypoint.sh"]
CMD [ "/bin/zsh" ]