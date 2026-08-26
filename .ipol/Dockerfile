# use one of the images from this repository: https://github.com/centreborelli/ipol-docker-images/
FROM registry.ipol.im/ipol:v2-py3.11-pytorch

# Install additional Debian packages needed by the project.
COPY .ipol/packages.txt packages.txt
RUN apt-get update && apt-get install -y $(cat packages.txt) && rm -rf /var/lib/apt/lists/* && rm packages.txt

# Install Python dependencies.
COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt && rm requirements.txt

# Copy the project source tree under /workdir/bin.
ENV bin /workdir/bin/
RUN mkdir -p $bin
WORKDIR $bin

# Download the pretrained demo assets into the same layout used by the repo and
# the run script. These files are expected to be hosted under demoextras/ on your
# IPOL server. We prefer the CPU-safe MANIQA checkpoint so the demo runs in the
# CPU-only IPOL environment. The run script also keeps a fallback for CUDA-serialized
# checkpoints when they are available.
RUN mkdir -p $bin/maniqa $bin/torch/hub/checkpoints && \
    wget -q -O $bin/maniqa/ckpt_koniq10k_cpu.pt \
      "https://dev.ipol.im:4443/~lraad/a_survey_of_metrics_used_for_image_colorization_assessment_tmp/demoextras/maniqa/ckpt_koniq10k_cpu.pt" && \
    wget -q -O $bin/torch/hub/checkpoints/alexnet-owt-7be5be79.pth \
      "https://dev.ipol.im:4443/~lraad/a_survey_of_metrics_used_for_image_colorization_assessment_tmp/demoextras/torch/hub/checkpoints/alexnet-owt-7be5be79.pth" && \
    wget -q -O $bin/torch/hub/checkpoints/pt_inception-2015-12-05-6726825d.pth \
      "https://dev.ipol.im:4443/~lraad/a_survey_of_metrics_used_for_image_colorization_assessment_tmp/demoextras/torch/hub/checkpoints/pt_inception-2015-12-05-6726825d.pth" && \
    wget -q -O $bin/torch/hub/checkpoints/squeezenet1_1-b8a52dc0.pth \
      "https://dev.ipol.im:4443/~lraad/a_survey_of_metrics_used_for_image_colorization_assessment_tmp/demoextras/torch/hub/checkpoints/squeezenet1_1-b8a52dc0.pth" && \
    wget -q -O $bin/torch/hub/checkpoints/vgg16-397923af.pth \
      "https://dev.ipol.im:4443/~lraad/a_survey_of_metrics_used_for_image_colorization_assessment_tmp/demoextras/torch/hub/checkpoints/vgg16-397923af.pth"

COPY . .

# The execution will happen in /workdir/exec, created by IPOL.
ENV PYTHONDONTWRITEBYTECODE 1
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION python
ENV PATH $bin:$PATH

# Make $HOME writable by the ipol user, and leave a writable cache area for PyTorch.
ENV HOME /home/ipol
RUN groupadd -g 1000 ipol || true && \
    useradd -m -u 1000 -g 1000 ipol -d $HOME 2>/dev/null || true && \
    mkdir -p /home/ipol/.cache /home/ipol/.cache/torch /home/ipol/.cache/maniqa && \
    chmod -R 777 $HOME && \
    chmod -R 777 $bin
USER ipol
