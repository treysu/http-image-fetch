# Additional import for parsing boolean arguments
import argparse
import asyncio
import os
import time  # Import for timing
from datetime import datetime
from io import BytesIO
from urllib.parse import urlparse

import aiohttp
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim


def get_folder_name(url: str) -> str:
    """
    Generate a safe folder name based on the IP/port or domain.
    """
    parsed_url = urlparse(url)
    folder_name = parsed_url.netloc.replace('.', '_').replace(':', '-')
    return folder_name


def get_daily_folder(base_dir: str, url: str) -> str:
    """
    Create a directory structure: base_dir/<url_folder>/<YYYY-MM-DD>.
    """
    url_folder = get_folder_name(url)
    date_folder = datetime.now().strftime("%Y-%m-%d")
    full_path = os.path.join(base_dir, date_folder, url_folder)
    os.makedirs(full_path, exist_ok=True)
    return full_path


async def fetch_image(session: aiohttp.ClientSession, url: str) -> bytes | None:
    """
    Fetch an image from a URL asynchronously and log the time taken.
    """
    start_time = time.time()
    try:
        async with session.get(url, timeout=10) as response:
            if response.status == 200:
                duration = time.time() - start_time
                print(f"Image fetched in {duration:.4f} seconds")
                return await response.read()
            else:
                print(f"Failed to fetch image: HTTP {response.status}")
    except aiohttp.ClientConnectorError as e:
        print(f"Connection error for {url}: {e}")
    except asyncio.TimeoutError:
        print(f"Connection to {url} timed out.")
    except Exception as e:
        print(f"Unexpected error fetching {url}: {e}")
    return None


def compare_ssim(img1: Image.Image, img2: Image.Image) -> float:
    """
    Compare two images using Structural Similarity Index (SSIM).
    """
    start_time = time.time()
    try:
        img1_array = np.array(img1)
        img2_array = np.array(img2)
        score, _ = ssim(img1_array, img2_array, full=True)
        duration = time.time() - start_time
        print(f"SSIM comparison completed in {duration:.4f} seconds")
        return score
    except ValueError as e:
        print(f"Error comparing images: {e}")
        return -1.0  # Default score for comparison failure


async def process_image(
    img_data: bytes, base_dir: str, url: str, previous_image_gray: Image.Image | None, ssim_threshold: float, disable_ssim: bool
) -> Image.Image:
    """
    Process the fetched image: compare for motion detection, save if necessary, and log timing details.
    """
    overall_start_time = time.time()

    # Convert image to PIL format
    start_time = time.time()
    current_image = Image.open(BytesIO(img_data))
    duration = time.time() - start_time
    print(f"Image loaded in {duration:.4f} seconds")

    # Convert to grayscale
    start_time = time.time()
    current_image_gray = current_image.convert('L')
    duration = time.time() - start_time
    print(f"Grayscale conversion completed in {duration:.4f} seconds")

    # Determine if the image should be saved
    save_image = False
    if not disable_ssim:
        if previous_image_gray is not None:
            similarity = compare_ssim(previous_image_gray, current_image_gray)
            print(f"SSIM Similarity: {similarity:.4f}")
            if similarity < ssim_threshold:
                save_image = True
        else:
            save_image = True  # Always save the first frame
    else:
        save_image = True  # Save all images if SSIM is disabled

    # Save the image if necessary
    if save_image:
        save_start_time = time.time()
        save_dir = get_daily_folder(base_dir, url)
        timestamp = datetime.now().strftime("%H%M%S_%f")  # Only time for filename
        filename = os.path.join(save_dir, f"{timestamp}.jpg")
        current_image.save(filename)
        save_duration = time.time() - save_start_time
        print(f"Image saved in {save_duration:.4f} seconds to {filename}")

    overall_duration = time.time() - overall_start_time
    print(f"Image processing completed in {overall_duration:.4f} seconds")

    return current_image_gray


async def main():
    """
    Main function to continuously fetch and process images from a URL.
    """
    parser = argparse.ArgumentParser(description="Fetch images from a URL and save them.")
    parser.add_argument("--url", type=str, help="The URL to fetch images from")
    parser.add_argument("--output", type=str, help="Base directory to save images (default: ../images)")
    parser.add_argument("--interval", type=float, help="Interval (seconds) between fetches (default: 10)")
    parser.add_argument("--ssim-threshold", type=float, help="SSIM threshold for saving (default: 0.95)")
    parser.add_argument("--disable-ssim", action="store_true", help="Disable SSIM comparisons and save all images")
    args = parser.parse_args()

    # Use environment variables as fallback
    url = args.url or os.getenv("IMAGE_FETCH_URL")
    base_dir = os.path.abspath(args.output or os.getenv("IMAGE_FETCH_OUTPUT", "images"))
    interval = args.interval or float(os.getenv("IMAGE_FETCH_INTERVAL", 10))
    ssim_threshold = args.ssim_threshold or float(os.getenv("IMAGE_FETCH_SSIM_THRESHOLD", 0.95))
    disable_ssim = args.disable_ssim or os.getenv("IMAGE_FETCH_DISABLE_SSIM", "false").lower() in ("1", "true", "yes")

    if not url:
        raise ValueError("A URL must be provided either via --url or IMAGE_FETCH_URL environment variable.")

    os.makedirs(base_dir, exist_ok=True)
    print(f"Base directory: {base_dir}")
    print(f"SSIM comparisons {'disabled' if disable_ssim else 'enabled'}.")

    previous_image_gray = None
    async with aiohttp.ClientSession() as session:
        while True:
            iteration_start_time = time.time()  # Start of the iteration

            fetch_start_time = time.time()
            img_data = await fetch_image(session, url)
            fetch_duration = time.time() - fetch_start_time
            print(f"Fetch operation took {fetch_duration:.4f} seconds")

            if img_data:
                process_start_time = time.time()
                previous_image_gray = await process_image(
                    img_data, base_dir, url, previous_image_gray, ssim_threshold, disable_ssim
                )
                process_duration = time.time() - process_start_time
                print(f"Processing operation took {process_duration:.4f} seconds")

            sleep_start_time = time.time()
            await asyncio.sleep(interval)  # Sleep for the specified interval
            sleep_duration = time.time() - sleep_start_time

            iteration_duration = time.time() - iteration_start_time
            print(f"Iteration completed in {iteration_duration:.4f} seconds (including {sleep_duration:.4f} seconds of sleep)")



if __name__ == "__main__":
    asyncio.run(main())
