FROM node:12
MAINTAINER Cumberland Group <apps@cumberlandgroupit.com>

# Set node env to production
ENV NODE_ENV production

# Setup Apt
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# Install packages
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
  git cmake build-essential g++ vim curl wget librtmp-dev libglib2.0* \
  libssl-dev libcurl4-openssl-dev liblog4cplus-1.1-9 liblog4cplus-dev \
  gstreamer1.0-libav \
  libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
  gstreamer1.0-plugins-base-apps gstreamer1.0-x \
  gstreamer1.0-plugins-bad gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-ugly gstreamer1.0-tools \
  && apt-get -y autoremove && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/* /tmp/*

# Download AWK KVS SDK, compile kvssink plugin, and install
RUN cd /opt && \
  git clone --branch 2.1.0 --single-branch --depth 1 https://github.com/awslabs/amazon-kinesis-video-streams-producer-sdk-cpp.git && \
  cd /opt/amazon-kinesis-video-streams-producer-sdk-cpp/kinesis-video-native-build && \
  ./min-install-script && \
  cp libgstkvssink.so /usr/lib/x86_64-linux-gnu/gstreamer-1.0/
