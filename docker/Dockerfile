FROM gaseooo/maskrcnn-benchmark:0.2

# install PyTorch Detection
ARG FORCE_CUDA="1"
ENV FORCE_CUDA=${FORCE_CUDA}
RUN git clone https://github.com/ZhongYingMatrix/maskrcnn-benchmark.git \
 && cd maskrcnn-benchmark \
 && python setup.py build develop

WORKDIR /maskrcnn-benchmark
