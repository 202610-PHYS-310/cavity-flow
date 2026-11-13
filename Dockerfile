# One image for a laptop and for the cluster.
# Laptop:  docker run --rm -v "$PWD":/work phys310-mpi mpirun -n 4 python wave_mpi.py
FROM python:3.12-slim-bookworm

RUN apt-get update \
 && apt-get install -y --no-install-recommends openmpi-bin libopenmpi-dev gcc \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir numpy mpi4py

# OpenMPI refuses to run as root; give it an ordinary user.
RUN useradd -m mpi
USER mpi
WORKDIR /work

# A laptop has fewer cores than students will ask for, and containers
# have no ssh between "nodes" -- both are fine for a single machine.
ENV OMPI_MCA_rmaps_base_oversubscribe=1 \
    OMPI_MCA_plm=isolated \
    OMPI_MCA_btl_vader_single_copy_mechanism=none
