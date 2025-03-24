from agents.inference import load_model, load_image, process2, annotate, load_model_with_lora
import cv2
import imageio
import os
import logging

import torch
import loralib as lora
import argparse
import pathlib
import re
import numpy as np

def numerical_sort(value):
    numbers = re.findall(r'\d+', value)
    return int(numbers[0]) if numbers else value

def create_video_from_frames(annotated_frames_bgr, output_path, fps=30):
    """
    Create a video from annotated frames using imageio.
    
    :param annotated_frames: List of frames as NumPy arrays.
    :param output_path: Path to save the output video file.
    :param fps: Frames per second of the output video.
    """
    annotated_frames = [frame[..., ::-1] for frame in annotated_frames_bgr]
    writer = imageio.get_writer(output_path, fps=fps)
    for frame in annotated_frames:
        writer.append_data(frame)
    writer.close()
        
    # Calculate the size of the video file
    file_size = os.path.getsize(output_path)
    print(f"Video saved to {output_path}")
    print(f"File size: {file_size / (1024 * 1024):.2f} MB")

def create_video_from_png_files(dirpath, output_path, fps=30):
    """
    Create a video from PNG files in a directory, converting from BGR to RGB using imageio.
    
    :param dirpath: Directory path containing PNG files.
    :param output_path: Path to save the output video file.
    :param fps: Frames per second of the output video.
    """
    # Get list of PNG files in the directory
    png_files = [os.path.join(dirpath, file) for file in os.listdir(dirpath) if file.endswith('.png')]
    
    # Sort files by filename (assuming numerical order if file names are numbered)
    png_files.sort()
    
    # Read PNG files, convert BGR to RGB, and store in a list
    annotated_frames_bgr = [cv2.imread(file) for file in png_files]
    annotated_frames_rgb = [frame[..., ::-1] for frame in annotated_frames_bgr]  # Convert BGR to RGB
    
    # Write RGB frames to a video using imageio
    with imageio.get_writer(output_path, fps=fps, codec='libx264') as writer:
        for frame in annotated_frames_rgb:
            writer.append_data(frame)
    
    # Calculate the size of the video file
    file_size = os.path.getsize(output_path)
    print(f"Video saved to {output_path}")
    print(f"File size: {file_size / (1024 * 1024):.2f} MB")

def annotation(args):
    cur_dir = pathlib.Path(__file__).parent.resolve()
    main_dir = pathlib.Path(__file__).parent.parent.resolve()

    BOX_TRESHOLD = args.boxtreshold
    TEXT_TRESHOLD = args.texttreshold
    ogcpy = main_dir.joinpath("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    ogcpth = main_dir.joinpath("weights/dino_weights/groundingdino_swint_ogc.pth")

    objeto = args.object
    # FOLDER_PATH = cur_dir.joinpath(f"images/my_images_{objeto}")
    FOLDER_PATH = cur_dir.joinpath(f"images/my_images_cow")
    TEXT_PROMPT = f"{objeto} ."

    if args.notune:
        # ------------------ Without LoRA --------------------
        model = load_model(ogcpy, ogcpth)
        laterpath = f'{objeto}_nofinetune_b{int(BOX_TRESHOLD*100)}'
        pre_path = ""
    else:  
        # ------------------- With LoRA  ---------------------
        print("\nTraining with lora")
        lora_r = args.lora_rank
        lora_a = args.lora_alpha
        ckpt_path = cur_dir.joinpath(args.path)
        pre_path = ckpt_path.parent.parent.name + "/" + ckpt_path.stem + "/"

        model = load_model_with_lora(ogcpy, ogcpth, rank=lora_r, lora_alpha=lora_a)
        lora_checkpoint = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(lora_checkpoint, strict=False)
        laterpath = f'{objeto}_loraAFTER_alpha{lora_a}_rank{lora_r}_b{int(BOX_TRESHOLD*100)}'     # <------------------------ Change name of output ------------------------------
        # ----------------------------------------------------
    model.eval()

    OUTPUT_PATH = "images/" + pre_path + "ann_" + laterpath
    IMAGES_OUTPUT_PATH = "images/" + pre_path + "ann_" + laterpath + '/images'

    if not os.path.exists(IMAGES_OUTPUT_PATH):
        os.makedirs(IMAGES_OUTPUT_PATH)

    log_file = f"{OUTPUT_PATH}/output.log"
    logging.basicConfig(
        filename=log_file,
        format="[%(asctime)s] [%(levelname)8s] --- %(message)s (%(filename)s:%(lineno)s)", datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        filemode='w'
    )
    if not args.notune:
        logging.info(f"From checkpoint: {ckpt_path}")

    # Prepare batches
    batch_size = 40
    image_files = [f for f in os.listdir(FOLDER_PATH) if f.endswith(('.png', '.jpg', '.jpeg'))]
    image_files = sorted(image_files, key=numerical_sort)

    annotated_frames = []
    total_detections = 0
    total_logit_sum = 0

    # Process in batches
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i + batch_size]
        batch_images = []
        batch_sources = []

        for image_file in batch_files:
            IMAGE_PATH = os.path.join(FOLDER_PATH, image_file)
            image_source, image = load_image(IMAGE_PATH)
            batch_sources.append(image_source)
            batch_images.append(image)

        # Convert batch_images to a tensor if necessary
        batch_images = torch.stack(batch_images)

        # Process the batch
        batch_results = process2(
            model=model,
            images=batch_images,  # Modify `process` to accept batched input
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )

        # Parse results for each image in the batch
        for ind, (boxes, logits, phrases) in enumerate(batch_results):
            valid_logits = [logit for logit in logits if logit > BOX_TRESHOLD]
            valid_phrases = [phrase for phrase, logit in zip(phrases, logits) if logit > BOX_TRESHOLD]

            total_detections += len(valid_logits)
            total_logit_sum += sum(valid_logits)

            annotated_frame = annotate(image_source=batch_sources[ind], boxes=boxes, logits=logits, phrases=phrases)
            annotated_frames.append(annotated_frame)

            output_path = os.path.join(IMAGES_OUTPUT_PATH, f"{laterpath}_{i + ind}.png")
            cv2.imwrite(output_path, annotated_frame)

            logging.info(f"Image{ i + ind }: {', '.join(f'{phrase} {logit:.2f}' for phrase, logit in zip(phrases, logits))}")
            print(f"Image{ i + ind }:", ", ".join(f"{phrase} {logit:.2f}" for phrase, logit in zip(phrases, logits)))

    # Calculate average logit if there are valid detections
    if total_detections > 0:
        average_logit = total_logit_sum / total_detections
    else:
        average_logit = 0.0

    print(f"Total detections: {total_detections}")
    print(f"Average logit of detections: {average_logit:.2f}")
    logging.info(f"Total detections: {total_detections}")
    logging.info(f"Average logit of detections: {average_logit:.2f}")

    create_video_from_frames(annotated_frames, os.path.join(OUTPUT_PATH, "video.mp4"), 10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--boxtreshold", type=float, default=0.60, help="box treshold")
    parser.add_argument("-t", "--texttreshold", type=float, default=0.25, help="text treshold")
    parser.add_argument("-la","--lora_alpha", type=int, default=8, help="lora alpha")
    parser.add_argument("-lr","--lora_rank", type=int, default=64, help="lora rank")
    parser.add_argument("-o", "--object", type=str, required=True, help="object for detection")
    parser.add_argument("-p", "--path", type=str, required=True, help="ckpt path")
    parser.add_argument("--save", action="store_true", help="save annotated images")
    parser.add_argument('--notune', action='store_true', help="no fine tune")
    
    args = parser.parse_args()

    # print arguments
    print("Arguments:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    annotation(args)
