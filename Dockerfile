FROM python:3.12-slim
# Use a 'large' base container to show-case how to load pytorch and use the GPU (when enabled)

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED=1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"

COPY --chown=user:user requirements.txt /opt/app/

RUN python -m pip install --upgrade pip \
 && python -m pip install --cache-dir=/root/.cache/pip -r requirements.txt
    
COPY --chown=user:user inference.py /opt/app/
COPY --chown=user:user Model.py /opt/app/
COPY --chown=user:user Transform.py /opt/app/
COPY --chown=user:user Uncertainty.py /opt/app/
COPY --chown=user:user resources/Prediction_TS.yml /opt/app/resources/Prediction_TS.yml
COPY --chown=user:user resources/Model/M291.pt /opt/app/resources/Model/M291.pt
COPY --chown=user:user resources/Prediction.yml /opt/app/resources/Prediction.yml
COPY --chown=user:user resources/Model/FT_0.pt /opt/app/resources/Model/FT_0.pt
COPY --chown=user:user resources/Model/FT_1.pt /opt/app/resources/Model/FT_1.pt
COPY --chown=user:user resources/Model/FT_2.pt /opt/app/resources/Model/FT_2.pt
COPY --chown=user:user resources/Model/FT_3.pt /opt/app/resources/Model/FT_3.pt
COPY --chown=user:user resources/Model/FT_4.pt /opt/app/resources/Model/FT_4.pt

COPY --chown=user:user residual.py /usr/local/lib/python3.12/site-packages/dynamic_network_architectures/building_blocks/

ENTRYPOINT ["python", "inference.py"]
