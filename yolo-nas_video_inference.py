import os
import sys
import time
import torch
import pathlib

from concurrent.futures import ThreadPoolExecutor

from super_gradients.training import models
from super_gradients.common.object_names import Models

def get_filenames(directory):
    """
    Collect in list filenames of videos

    Parametres:
    directory (str): Source folder of images

    Returns: List of filenames
    """

    filenames = []
    try:
        entries = os.listdir(directory)
        
        for entry in entries:
            full_path = os.path.join(directory, entry)
            if os.path.isfile(full_path):
                filenames.append(entry)
    except Exception as e:
        print(f"[ERROR] Error occured in get_filenames(): {e}, {type(e)=}")
    
    return filenames


def inference_video(model, video_src_folder, file_name, video_dest_folder, confidence = 0.6):
    """
    Inferences video

    Parametres:
    model: Yolo-NAS Pose model
    video_src_folder (str): Source folder of video
    file_name (str): Name of video file
    video_dest_folder (str): Destination folder of proccessed video
    confidence (float) [OPTIONAL]: Confidence threshold
    """
    try:
        
        input_file = os.path.join(video_src_folder, file_name)

        output_file = pathlib.Path(file_name).stem + "-detections" + pathlib.Path(file_name).suffix
        output_file = os.path.join(video_dest_folder, output_file)

        start_time = time.time()
        model.predict(input_file, conf=confidence).save(output_file)
        end_time = time.time()

        print(f"[INFO] Succeed in video {file_name} inference and saved to {output_file}")

        inference_time = end_time - start_time
        inf_time_minutes = inference_time // 60
        inf_time_sec = inference_time % 60
        print(f"Video {file_name} inference time: {inf_time_minutes} m {inf_time_sec} s")

    except Exception as err:
        print(f"[ERROR] Error occured in inference_video() {err=}, {type(err)=}")
        raise

def process_videos(file_names, model, video_src_folder, video_dest_folder, confidence = 0.6):
    """
    Start inference videos

    Parameters:
    file_names (List): List of files' names
    model: Yolo-NAS Pose model
    video_src_folder (str): Source folder of video
    video_dest_folder (str): Destination folder of proccessed video
    confidence (float) [OPTIONAL]: Confidence threshold
    
    """
    try: 
        # Check if path exist, if not - create
        os.makedirs(video_dest_folder, exist_ok=True)

        print("[INFO] Start inferencing videos")

        # Using ThreadPoolExecutor for parallel copying files
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(inference_video, model, video_src_folder, file_name, video_dest_folder, confidence) for file_name in file_names]
            
            # Wait until each task finished
            for future in futures:
                future.result()
    except Exception as err:
        print(f"[ERROR] Error occured in process_videos() {err=}, {type(err)=}")
        raise

def main():
 
    if len(sys.argv) < 3:
        raise ValueError('Please provide paths to source folder and destination folder')
    
    source_folder = sys.argv[1]
    dest_folder = sys.argv[2]

    print("[INFO] Getting model:")
    model = models.get("yolo_nas_pose_l", pretrained_weights="coco_pose")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    videos_filenames = get_filenames(source_folder)

    process_videos(videos_filenames, model, source_folder, dest_folder)


if __name__ == "__main__":
    main()