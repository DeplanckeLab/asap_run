FROM r-base:4.3.1

# Set up Debian Bullseye and update
RUN echo 'deb http://deb.debian.org/debian bullseye main' > /etc/apt/sources.list \
    && apt-get update -y \
    && apt-get dist-upgrade -y

# Install system dependencies
#RUN apt-get install -y --no-install-recommends \
#    libxml2-dev python3 python3-pip python3-dev git \
#    libcairo2-dev libxt-dev xorg openbox software-properties-common \
#    openssl libcurl4-openssl-dev libssl-dev wget net-tools \
#    default-jre default-jdk time curl gcc g++ \
#    libtool bison flex build-essential cmake

# Install additional Python3 packages
#RUN apt-get install -y --no-install-recommends \
#    python3-numpy python3-scipy python3-h5py \
#    python3-numba python3-pandas python3-matplotlib

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxml2-dev python3 python3-pip python3-dev git \
    libcairo2-dev libxt-dev xorg openbox software-properties-common \
    openssl libcurl4-openssl-dev libssl-dev wget net-tools \
    default-jre default-jdk time curl gcc g++ build-essential \
    libtool bison flex python3-venv

# Set up Python virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip, setuptools, and wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install cmake via pip if needed
RUN pip install --upgrade cmake==3.14.3

# Verify cmake installation
RUN cmake --version

# Install Python packages with debugging and custom CMake args
#RUN CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release" pip install --no-cache-dir leidenalg umap-learn scikit-learn MulticoreTSNE
RUN python3 -m pip install --break-system-packages leidenalg umap-learn scikit-learn MulticoreTSNE


# Upgrade pip, setuptools, and wheel to avoid legacy issues
#RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install required Python libraries
#RUN python3 -m pip install --no-cache-dir leidenalg umap-learn scikit-learn MulticoreTSNE

#RUN apt-get update && apt-get install -y python3-venv

#Create and activate a virtual environment:

#RUN python3 -m venv /opt/venv
#ENV PATH="/opt/venv/bin:$PATH"

#Install Python packages inside the virtual environment:

#RUN pip install --no-cache-dir --upgrade pip setuptools wheel
#RUN pip install --no-cache-dir leidenalg umap-learn scikit-learn MulticoreTSNE

#From r-base:4.3.1

#RUN echo 'deb http://deb.debian.org/debian bullseye main' > /etc/apt/sources.list
#RUN apt-get -y update && apt-get dist-upgrade -y

## if adding this, we can gather the 2 apt-get lines but the size of the image goes from 3.65Go to 4.2Go
#&& apt-get dist-upgrade -y
#RUN apt-get -y install libxml2-dev python3-pip python-dev git libcairo2-dev libxt-dev xorg openbox software-properties-common openssl libcurl4-openssl-dev libssl-dev wget net-tools default-jre default-jdk time curl python3-pip gcc
#RUN apt-get -y install libtool bison flex



#install Python3 packages
#RUN apt-get -y install python3-numpy python3-scipy python3-h5py  python3-numba python3-pandas python3-matplotlib 
#RUN python3 -m pip install --break-system-packages 'cmake==3.14.3'
#RUN python3 -m pip install --break-system-packages leidenalg umap-learn scikit-learn MulticoreTSNE

#install HDF5

COPY lib/hdf5-1.10.6-linux-centos7-x86_64-shared.tar.gz hdf5.tar.gz 
RUN tar -zxf hdf5.tar.gz
#RUN wget -O hdf5.tar.gz http://gecftools.epfl.ch/share/fab/hdf5-1.10.6-linux-centos7-x86_64-shared.tar.gz; tar -zxf hdf5.tar.gz
#RUN tar -zxf hdf5.tar.gz
ENV LD_LIBRARY_PATH=/hdf5/lib/
ENV PATH=$PATH:/hdf5/bin
RUN cd /hdf5/bin && ./h5redeploy -force && cd / && rm hdf5.tar.gz

#Install R packages

RUN Rscript -e "install.packages(c('BiocManager', 'Matrix', 'reticulate', 'KernSmooth', 'nlme', 'MASS', 'sanon'), repos='http://stat.ethz.ch/CRAN/'); \
BiocManager::install(c('multtest'))"
RUN Rscript -e "install.packages(c('R6', 'bit64', 'Rtsne', 'jsonlite', 'data.table', 'devtools', 'flexmix', 'rPython', 'statmod', 'plotly', 'future.apply', 'Seurat'), repos='http://stat.ethz.ch/CRAN/');"
#RUN apt-get -y install libharfbuzz-dev libfribidi-dev libfreetype6 libfreetype6-dev libpng-dev libtiff5-dev libjpeg-dev

# Install the required packages
RUN apt-get install -y libharfbuzz-dev libfribidi-dev libpng-dev libtiff5-dev libjpeg-dev
RUN apt-get install -y libfreetype6 libfreetype-dev
#RUN apt-get install -y libfreetype-dev=2.13.0+dfsg-1

# Clear broken packages and resolve dependencies
RUN dpkg --configure -a
RUN apt-get install -f

# Update and upgrade packages
RUN apt-get update
RUN apt-get upgrade -y


RUN Rscript -e "install.packages(c('devtools'), repos='http://stat.ethz.ch/CRAN/');"
RUN Rscript -e "install.packages(c('MAST'), repos='http://stat.ethz.ch/CRAN/');"

RUN Rscript -e "devtools::install_github(c('grimbough/rhdf5','tallulandrews/M3Drop'))"
RUN Rscript -e "BiocManager::install(c('sva', 'DESeq2', 'limma', 'cluster', 'Cairo', 'scater', 'SC3', 'XVector', 'scran'));"

#RUN python3 -m pip uninstall umap-learn; python3 -m pip install 'umap-learn==0.3.7'
#RUN pip install numpy scipy h5py leidenalg 'numba==0.42.0' scikit-learn umap-learn
#uninstall
#RUN python3 -m pip uninstall -y  leidenalg 'numba==0.42.0' umap-learn MulticoreTSNE
ENV RETICULATE_PYTHON=/usr/bin/python3

#RUN Rscript -e "install.packages(c('Rcpp'), repos='http://stat.ethz.ch/CRAN/');"
#RUN Rscript -e "install.packages(c('Eigen'), repos='http://stat.ethz.ch/CRAN/');"
#RUN Rscript -e "install.packages(c('RcppEigen'), repos='http://stat.ethz.ch/CRAN/');"
#RUN Rscript -e "install.packages(c('Seurat'), repos='http://stat.ethz.ch/CRAN/');"

#RUN python3 -m pip install 'loompy'
RUN apt-get -y install python3-loompy

RUN Rscript -e "install.packages(c('sanon'))"

RUN Rscript -e "BiocManager::install(c('batchelor'))"

#RUN python3 -m pip install 'cmake==3.13.3'; python3 -m pip install numpy scipy h5py leidenalg numba umap-learn scikit-learn pandas matplotlib MulticoreTSNE
#RUN python3 -m pip install --upgrade pip; python3 -m pip install 'cmake==3.16.6'; python3 -m pip install python-igraph 'umap-learn==0.5.1' 'numba==0.53.0' 'numpy==1.20.3'
#RUN python3 -m pip install --upgrade pip;  python3 -m pip install cmake 'setuptools==56.0.0' 'numpy==1.20.3' 'numba==0.53.1' 'scipy==1.6.3' 'h5py==3.1.0' 'leidenalg==0.8.7' 'louvain==0.7.0' 'umap-learn==0.5.1' 'scikit-learn==0.24.2' 'pandas==1.2.4' 'matplotlib==3.3.2' 'MulticoreTSNE==0.1' 'loompy==3.0.6'

RUN apt-get -y install libhdf5-dev
RUN Rscript -e "install.packages('hdf5r')"

RUN Rscript -e "BiocManager::install(c('LoomExperiment', 'SingleCellExperiment')); \
devtools::install_github('cellgeni/sceasy')"

#RUN python3 -m pip install 'anndata'
#RUN python3 -m pip install 'scanpy'
RUN apt-get -y install python3-anndata 
RUN python3 -m pip install --break-system-packages scanpy

RUN Rscript -e "install.packages(c('SeuratObject'))"
RUN Rscript -e "devtools::install_github(repo = 'mojaveazure/loomR', ref = 'develop')"

ENV USER=rvmuser USER_ID=1006 USER_GID=1006
# now creating user NO NEED IN LAST INSTALL
RUN groupadd --gid "${USER_GID}" "${USER}" && \
    useradd \
      --uid ${USER_ID} \
      --gid ${USER_GID} \
      --create-home \
      --shell /bin/bash \
${USER}

USER ${USER}

LABEL maintainer="Fabrice P.A. David <fabrice.david@epfl.ch>"
ARG DUMMY=unknown
RUN DUMMY=${DUMMY}

COPY R /srv
COPY python /srv
COPY java /srv
COPY bin /srv

WORKDIR /srv
