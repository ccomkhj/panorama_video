import cv2
import numpy as np
from tqdm import tqdm
import argparse
import signal
import sys


def save_on_kill(signal, frame):
    global panorama, output_path
    cv2.imwrite(output_path, panorama)
    print(f"Panorama saved to {output_path} upon receiving kill signal.")
    sys.exit(0)


signal.signal(signal.SIGTERM, save_on_kill)
signal.signal(signal.SIGINT, save_on_kill)


def create_panorama(video_path, output_path):
    global panorama
    try:
        cap = cv2.VideoCapture(video_path)
        orb = cv2.ORB_create(nfeatures=500)

        _, prev_frame = cap.read()
        if prev_frame is None:
            print("Cannot read the first frame.")
            return

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        panorama = prev_frame
        sliding_window_width = int(prev_frame.shape[1] * 0.2)
        _, initial_descriptors = orb.detectAndCompute(prev_gray, None)
        initial_keypoints = prev_gray.shape[1] - sliding_window_width

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = tqdm(total=frame_count, unit="frame", desc="Creating Panorama")

        while True:
            ret, current_frame = cap.read()
            if not ret:
                break

            current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            kp2, des2 = orb.detectAndCompute(current_gray, None)

            if (
                initial_descriptors is not None
                and des2 is not None
                and panorama.shape[1] > sliding_window_width
            ):
                overlap_region = panorama[:, -sliding_window_width:]
                kp1, des1 = orb.detectAndCompute(overlap_region, None)

                bf = cv2.BFMatcher(cv2.NORM_HAMMING)
                matches = bf.knnMatch(des1, des2, k=2)

                good_matches = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

                if good_matches:
                    dx = int(
                        np.mean(
                            [
                                kp1[m.queryIdx].pt[0] - kp2[m.trainIdx].pt[0]
                                for m in good_matches
                            ]
                        )
                    )
                    dy = 0
                    panorama = stitch_images(panorama, current_frame, dx)
                    initial_descriptors = des1

            prev_gray = current_gray
            progress_bar.update(1)
        cap.release()
        progress_bar.close()

        cv2.imwrite(output_path, panorama)
        print(f"Panorama saved to {output_path}")
    except KeyboardInterrupt:
        pass
    finally:
        if "panorama" in globals():
            cv2.imwrite(output_path, panorama)
            print(f"Panorama saved to {output_path} (intermediate)")


def stitch_images(img1, img2, dx):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    height = max(h1, h2)
    width = w1 + w2 - dx

    stitched_image = np.zeros((height, width, 3), dtype=np.uint8)
    stitched_image[:h1, :w1] = img1
    stitched_image[:h2, w1 - dx : w1 - dx + w2] = img2

    return stitched_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a panorama image from a video."
    )
    parser.add_argument("input_video_path", help="Path to the input video file.")
    parser.add_argument(
        "-o",
        "--output_panorama_path",
        default="output_panorama.jpg",
        help="Path to save the output panorama image (optional).",
    )

    args = parser.parse_args()

    # Set global output_path for the save_on_kill function to access
    output_path = args.output_panorama_path

    create_panorama(args.input_video_path, output_path)
