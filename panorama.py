import cv2
import numpy as np
from tqdm import tqdm
import sys
import signal


# Function to handle saves on kill signals
def save_on_kill(signal, frame):
    global panorama, output_path
    cv2.imwrite(output_path, panorama)
    print(f"Panorama saved to {output_path} upon receiving kill signal.")
    sys.exit(0)


# Register signal handlers
signal.signal(signal.SIGTERM, save_on_kill)
signal.signal(signal.SIGINT, save_on_kill)


def create_panorama(video_path, output_path):
    global panorama  # Make panorama accessible globally
    try:
        cap = cv2.VideoCapture(video_path)
        orb = cv2.ORB_create(nfeatures=500)

        _, prev_frame = cap.read()
        if prev_frame is None:
            print("Cannot read the first frame.")
            return

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        panorama = prev_frame

        # Initialize sliding window size (e.g., the expected maximum translation plus some buffer)
        sliding_window_width = int(prev_frame.shape[1] * 0.2)

        # Initial features in the overlapping region
        _, initial_descriptors = orb.detectAndCompute(prev_gray, None)
        initial_keypoints = prev_gray.shape[1] - sliding_window_width

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = tqdm(total=frame_count, unit="frame", desc="Creating Panorama")

        while True:
            ret, current_frame = cap.read()
            if not ret:
                break

            current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

            # Detect keypoints and compute descriptors in the current frame
            kp2, des2 = orb.detectAndCompute(current_gray, None)

            # Ensure that we have descriptors to match and that the panorama has enough width
            if (
                initial_descriptors is not None
                and des2 is not None
                and panorama.shape[1] > sliding_window_width
            ):
                # Extract the overlap region from the panorama for feature matching
                overlap_region = panorama[:, -sliding_window_width:]
                kp1, des1 = orb.detectAndCompute(overlap_region, None)

                # Match descriptors using KNN with k=2 (to apply ratio test later)
                bf = cv2.BFMatcher(cv2.NORM_HAMMING)
                matches = bf.knnMatch(des1, des2, k=2)

                # Apply ratio test
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

                # Compute the net translation between the two images if good matches are found
                if good_matches:
                    dx = int(
                        np.mean(
                            [
                                kp1[m.queryIdx].pt[0] - kp2[m.trainIdx].pt[0]
                                for m in good_matches
                            ]
                        )
                    )
                    dy = 0  # Assuming only horizontal movement

                    panorama = stitch_images(panorama, current_frame, dx)
                    initial_descriptors = (
                        des1  # Update the descriptors for the overlap region
                    )

            prev_gray = current_gray  # Update the previous image
            progress_bar.update(1)  # Update the progress bar
        cap.release()
        progress_bar.close()

        cv2.imwrite(output_path, panorama)  # Save the final stitched image
        print(f"Panorama saved to {output_path}")
    except KeyboardInterrupt:
        # Handle any cleanup here if needed
        pass
    finally:
        if "panorama" in globals():
            cv2.imwrite(output_path, panorama)
            print(f"Panorama saved to {output_path} (intermediate)")


def stitch_images(img1, img2, dx):
    # Assuming img2 is to the right of img1
    # We need to create a new image with sufficient width to hold both images
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Determine dimensions of the resulting stitched image
    height = max(h1, h2)
    width = w1 + w2 - dx  # Subtract overlap (dx is negative for left shifts)

    # Create a new canvas with determined dimensions
    stitched_image = np.zeros((height, width, 3), dtype=np.uint8)
    stitched_image[:h1, :w1] = img1  # Place the first image on the left

    # Overwrite the pixels from img2 onto the stitched_image at the right location
    stitched_image[:h2, w1 - dx : w1 - dx + w2] = img2

    return stitched_image


if __name__ == "__main__":
    input_video_path = "EBoss.MOV"
    output_panorama_path = "output_panorama.jpg"
    output_path = output_panorama_path  # Keep track globally for save_on_kill
    create_panorama(input_video_path, output_panorama_path)
