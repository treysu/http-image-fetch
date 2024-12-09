import argparse
import asyncio
import os
from datetime import datetime
from urllib.parse import urlparse
from io import BytesIO

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import aiohttp


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
    full_path = os.path.join(base_dir, url_folder, date_folder)
    os.makedirs(full_path, exist_ok=True)
    return full_path


async def fetch_image(session: aiohttp.ClientSession, url: str) -> bytes | None:
    """
    Fetch an image from a URL asynchronously.
    """
    try:
        async with session.get(url, timeout=10) as response:
            if response.status == 200:
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
    try:
        img1_array = np.array(img1)
        img2_array = np.array(img2)
        score, _ = ssim(img1_array, img2_array, full=True)
        return score
    except ValueError as e:
        print(f"Error comparing images: {e}")
        return -1.0  # Default score for comparison failure


async def process_image(
    img_data: bytes, base_dir: str, url: str, previous_image_gray: Image.Image | None, ssim_threshold: float
) -> Image.Image:
    """
    Process the fetched image: compare for motion detection and save if necessary.
    """
    current_image = Image.open(BytesIO(img_data))
    current_image_gray = current_image.convert('L')  # Convert to grayscale for comparison

    save_image = False
    if previous_image_gray is not None:
        similarity = compare_ssim(previous_image_gray, current_image_gray)
        print(f"SSIM Similarity: {similarity:.4f}")
        if similarity < ssim_threshold:
            save_image = True
    else:
        save_image = True  # Always save the first frame

    if save_image:
        # Get the folder for the current date
        save_dir = get_daily_folder(base_dir, url)
        timestamp = datetime.now().strftime("%H%M%S_%f")  # Only time for filename
        filename = os.path.join(save_dir, f"{timestamp}.jpg")
        current_image.save(filename)
        print(f"Saved: {filename}")

    return current_image_gray


async def main():
    """
    Main function to continuously fetch and process images from a URL.
    """
    parser = argparse.ArgumentParser(description="Fetch images from a URL and save them.")
    parser.add_argument("--url", type=str, required=True, help="The URL to fetch images from")
    parser.add_argument(
        "--output",
        type=str,
        default="../images",
        help="Base directory to save images (default: ../images)"
    )
    parser.add_argument("--interval", type=float, default=10, help="Interval (seconds) between fetches (default: 0.1)")
    parser.add_argument("--ssim-threshold", type=float, default=0.95, help="SSIM threshold for saving (default: 0.95)")
    args = parser.parse_args()

    # Set up directories
    base_dir = os.path.abspath(args.output)
    os.makedirs(base_dir, exist_ok=True)
    print(f"Base directory: {base_dir}")

    previous_image_gray = None
    async with aiohttp.ClientSession() as session:
        while True:
            img_data = await fetch_image(session, args.url)
            if img_data:
                previous_image_gray = await process_image(
                    img_data, base_dir, args.url, previous_image_gray, args.ssim_threshold
                )
            await asyncio.sleep(args.interval)


if __name__ == "__main__":
    asyncio.run(main())
