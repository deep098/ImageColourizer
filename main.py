
import argparse
import matplotlib.pyplot as plt
import os
from glob import glob
import cv2
from colorizers import *

def extract_images(path):
    vidcap = cv2.VideoCapture(path)
    success,image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(f"vidd_out/{str(count).zfill(6)}.jpg", image)     # save frame as JPEG file      
        success,image = vidcap.read()
        count += 1


def colour(input_path):
    colorizer_siggraph17 = siggraph17(pretrained=True).eval()
    dir = os.path.join(os.getcwd(), input_path, "*.jpg")
    output_dir = os.path.join(os.getcwd(), "imgg_out")
    images = glob(dir)

    for cnt, image in enumerate(images):
        if cnt % 100 == 0:
            print("Image ", cnt)
        img = load_img(image)
        (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))

        img_bw = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())
        output_loc = os.path.join(output_dir, f"{cnt+1:04d}.jpg")
        plt.imsave(output_loc, img_bw)
    
    return output_dir

def convert(input_folder):
    path = os.path.join(os.getcwd(), input_folder)

    out_video_full_path = os.path.join(os.getcwd(), 'coloured_video.mp4')
    img = glob(os.path.join(path,"*.jpg"))

    cv2_fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    frame = cv2.imread(img[0])
    size = list(frame.shape)
    del size[2]
    size.reverse()

    video = cv2.VideoWriter(out_video_full_path, cv2_fourcc, 24, size)

    for image in img: 
        video.write(cv2.imread(image))
        
    video.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--i','--vid_path', type=str)
    parser.add_argument('--d','--vid_dir', type=str)
    parser.add_argument('-o','--save_prefix', type=str, default='saved')
    opt = parser.parse_args()

    extract_images(opt.i)
    output_dir = colour(opt.d)
    convert(output_dir)

