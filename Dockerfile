FROM tensorflow/tensorflow:2.16.1-gpu

# Install extra Python packages
RUN pip install seaborn matplotlib pandas scikit-learn ipython ipykernel

# (Optional) system packages
# RUN apt update && apt install -y ffmpeg

WORKDIR /workspace