FROM nvcr.io/nvidia/jax:24.10-py3

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
