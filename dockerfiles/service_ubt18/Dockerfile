ARG INPUT_TAG
FROM $INPUT_TAG

# Install test requirements
RUN . /opt/miniconda/etc/profile.d/conda.sh && \
    conda activate openxrlab && \
    pip install -r https://raw.githubusercontent.com/openxrlab/xrmocap/main/requirements/service.txt && \
    pip cache purge
