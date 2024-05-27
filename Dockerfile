FROM python:3.10.0

RUN pip install --upgrade pip

RUN mkdir yoloNasSkeletonDetection
RUN cd yoloNasSkeletonDetection

RUN mkdir video
RUN mkdir video-detected

COPY requirements.txt /yoloSkeletonDetection/requirements.txt
COPY video /yoloSkeletonDetection/video
COPY video-detected /yoloSkeletonDetection/video-detected
COPY yolo-nas_video_inference.py /yoloSkeletonDetection/yolo-nas_video_inference.py

WORKDIR /yoloSkeletonDetection

RUN pip3.10 install -r /yoloSkeletonDetection/requirements.txt

RUN python3.10 yolo-nas_video_inference.py video video-detected