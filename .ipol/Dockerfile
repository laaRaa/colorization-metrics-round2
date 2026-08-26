# use one of the images from this repository: https://github.com/centreborelli/ipol-docker-images/
FROM registry.ipol.im/ipol:v2-py3.11-pytorch

# install additional debian packages
COPY .ipol/packages.txt packages.txt
RUN apt-get update && apt-get install -y $(cat packages.txt) && rm -rf /var/lib/apt/lists/* && rm packages.txt

# copy the requirements.txt and install python packages
COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt && rm requirements.txt

# copy the code to $bin
ENV bin /workdir/bin/
RUN mkdir -p $bin
WORKDIR $bin
RUN wget "https://dev.ipol.im:4443/~lraad/a_survey_of_metrics_used_for_image_colorization_assessment_tmp/demoextras/maniqa/ckpt_koniq10k.pt"
RUN wget "https://dev.ipol.im:4443/~lraad/a_survey_of_metrics_used_for_image_colorization_assessment_tmp/demoextras/torch/hub/checkpoints/alexnet-owt-7be5be79.pth"
RUN wget "https://dev.ipol.im:4443/~lraad/a_survey_of_metrics_used_for_image_colorization_assessment_tmp/demoextras/torch/hub/checkpoints/pt_inception-2015-12-05-6726825d.pth"
RUN wget "https://dev.ipol.im:4443/~lraad/a_survey_of_metrics_used_for_image_colorization_assessment_tmp/demoextras/torch/hub/checkpoints/squeezenet1_1-b8a52dc0.pth"
RUN wget "https://dev.ipol.im:4443/~lraad/a_survey_of_metrics_used_for_image_colorization_assessment_tmp/demoextras/torch/hub/checkpoints/vgg16-397923af.pth"
COPY . .

# the execution will happen in the folder /workdir/exec
# it will be created by IPOL

# some QoL tweaks
ENV PYTHONDONTWRITEBYTECODE 1
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION python
ENV PATH $bin:$PATH

# $HOME is writable by the user `ipol`
ENV HOME /home/ipol
# chmod 777 so that any user can use the HOME, in case the docker is run with -u 1001:1001
RUN groupadd -g 1000 ipol && useradd -m -u 1000 -g 1000 ipol -d $HOME && chmod -R 777 $HOME
USER ipol
