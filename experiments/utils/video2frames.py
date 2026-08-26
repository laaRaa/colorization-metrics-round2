"""This module provides functionality to create videos from images.

Function:
- create_video_from_images: Generates a video from a directory of images.
"""

import os

import av
from skimage.io import imread
from skimage.util import img_as_ubyte

# ------------------------------------------------------------
# PUBLIC FUNCTIONS
# ------------------------------------------------------------


def create_video_from_images(
    image_dir: str, output_video: str, frame_rate: int = 30
) -> None:
    """Create a video from images in a directory using PyAV.

    This function reads images from the specified directory, sorts them,
        and writes them into a video file with the specified frame rate.
        The video is encoded using the FFV1 codec.

    Args:
        image_dir: Directory containing key frame images.
        output_video: Output video filename.
        frame_rate: Frames per second for the output video.

    Raises:
        FileNotFoundError: If the image directory is empty or does not exist.
    """
    # Verify that the directory exists and contains images
    images = sorted(
        [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.endswith(".png")
        ]
    )
    if not images:
        raise FileNotFoundError(f"No images found in directory: {image_dir}")

    # Open a PyAV container for writing
    container = av.open(output_video, mode="w")
    stream = container.add_stream("ffv1", rate=frame_rate)
    stream.height, stream.width, _ = imread(images[0]).shape
    stream.pix_fmt = "yuv420p"  # Pixel format

    for img_path in images:
        # Open the image and convert it to the correct format
        img = img_as_ubyte(imread(img_path))
        frame = av.VideoFrame.from_ndarray(img)
        packet = stream.encode(frame)
        container.mux(packet)

    # Finalize the video
    # Flush any remaining frames in the stream
    for packet in stream.encode():
        container.mux(packet)

    container.close()
    print(f"Video created successfully: {output_video}")


# ------------------------------------------------------------
# PRIVATE FUNCTIONS
# ------------------------------------------------------------


def _extract_middle_frames(
    video_path: str, output_dir: str, num_frames: int = 64
) -> None:
    """Extract and save the middle frames of a video.

    This function calculates the middle section of the video and extracts the specified
        number of frames from that section. The frames are saved as PNG images
        in the specified output directory.

    Args:
        video_path: Path to the input video file.
        output_dir: Directory to save the extracted frames.
        num_frames: Number of frames to extract.

    Raises:
        ValueError: If the video contains fewer frames
            than the requested number of frames.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Open the video
    container = av.open(video_path)
    total_frames = int(container.streams.video[0].frames)

    if total_frames < num_frames:
        raise ValueError(
            f"The video contains only {total_frames} frames, "
            f"which is less than the requested {num_frames} frames."
        )

    # Calculate middle frame indices
    start_idx = (total_frames - num_frames) // 2
    end_idx = start_idx + num_frames

    # Extract and save frames
    for frame_idx, frame in enumerate(container.decode(video=0)):
        if start_idx <= frame_idx < end_idx:
            output_path = os.path.join(output_dir, f"{(frame_idx - start_idx):03d}.png")
            frame.to_image().save(output_path)
        if frame_idx >= end_idx:
            break


# ------------------------------------------------------------
# SCRIPT
# ------------------------------------------------------------

if __name__ == "__main__":
    VIDEO_NAME = "train"
    _extract_middle_frames(f"data/videos/{VIDEO_NAME}.mp4", f"data/videos/{VIDEO_NAME}")
    create_video_from_images(
        f"data/videos/{VIDEO_NAME}", f"data/videos/{VIDEO_NAME}.mkv"
    )
