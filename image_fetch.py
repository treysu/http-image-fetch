import argparse
import asyncio
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from io import BytesIO
from urllib.parse import urlparse

import aiohttp
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# Tracks {url: (next_retry_time, current_backoff_seconds)}
backoff_tracker = defaultdict(lambda: (0, 0))  # (next_allowed_time, backoff_seconds)



executor = ThreadPoolExecutor()

def get_folder_name(url: str) -> str:
    parsed_url = urlparse(url)
    return parsed_url.netloc.replace('.', '_').replace(':', '-')

def get_daily_folder(base_dir: str, url: str) -> str:
    url_folder = get_folder_name(url)
    date_folder = datetime.now().strftime("%Y-%m-%d")
    full_path = os.path.join(base_dir, date_folder, url_folder)
    os.makedirs(full_path, exist_ok=True)
    return full_path

async def fetch_image(session: aiohttp.ClientSession, url: str) -> bytes | None:
    try:
        async with session.get(url, timeout=10) as response:
            if response.status == 200:
                return await response.read()
            else:
                print(f"[{url}] Failed to fetch: HTTP {response.status}")
    except Exception as e:
        print(f"[{url}] Fetch error: {e}")
    return None

def downscale(image: Image.Image, size=(320, 240)) -> Image.Image:
    return image.resize(size, Image.BILINEAR)

def compare_ssim(img1: Image.Image, img2: Image.Image) -> float:
    try:
        img1_array = np.array(img1)
        img2_array = np.array(img2)
        score, _ = ssim(img1_array, img2_array, full=True)
        return score
    except Exception as e:
        print(f"SSIM error: {e}")
        return -1.0

async def decode_image(img_data: bytes) -> tuple[Image.Image, Image.Image]:
    loop = asyncio.get_event_loop()
    def load_both():
        img = Image.open(BytesIO(img_data))
        return img, img.convert('L')  # color and grayscale
    return await loop.run_in_executor(executor, load_both)

async def save_image(image: Image.Image, base_dir: str, url: str):
    loop = asyncio.get_event_loop()
    def _save():
        folder = get_daily_folder(base_dir, url)
        timestamp = datetime.now().strftime("%H%M%S_%f")
        filename = os.path.join(folder, f"{timestamp}.jpg")
        image.save(filename)
        print(f"[{url}] Image saved: {filename}")
    await loop.run_in_executor(executor, _save)

async def handle_stream(url, session, base_dir, interval, ssim_threshold, disable_ssim):
    print(f"Started stream: {url}")
    previous_image = None
    global backoff_tracker

    while True:
        now = time.time()
        next_time, backoff = backoff_tracker[url]

        if now < next_time:
            await asyncio.sleep(1)  # wait 1s and check again
            continue

        img_data = await fetch_image(session, url)
        if img_data:
            color_image, gray_image = await decode_image(img_data)
            save = False
            if disable_ssim or previous_image is None:
                save = True
            else:
                sim = compare_ssim(downscale(previous_image), downscale(gray_image))
                if sim < ssim_threshold:
                    save = True
                print(f"[{url}] SSIM: {sim:.4f}")

            if save:
                await save_image(color_image, base_dir, url)

            previous_image = gray_image
            backoff_tracker[url] = (0, 0)  # reset backoff on success

            await asyncio.sleep(interval)
        else:
            # Increase backoff on failure
            _, prev_backoff = backoff_tracker[url]
            new_backoff = min(prev_backoff * 2 + 5, 120)  # start small, cap at 2 min
            backoff_tracker[url] = (time.time() + new_backoff, new_backoff)
            print(f"[{url}] Backing off for {new_backoff:.0f}s due to error")

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, help="Comma-separated list of URLs to fetch from")
    parser.add_argument("--output", type=str, default="images")
    parser.add_argument("--interval", type=float)
    parser.add_argument("--ssim-threshold", type=float)
    parser.add_argument("--disable-ssim", action="store_true")
    args = parser.parse_args()

    urls = args.url or os.getenv("IMAGE_FETCH_URLS")
    if not urls:
        raise ValueError("Provide URLs via --url or IMAGE_FETCH_URLS")
    url_list = [u.strip() for u in urls.split(",")]

    base_dir = os.path.abspath(args.output or os.getenv("IMAGE_FETCH_OUTPUT", "images"))
    interval = args.interval or float(os.getenv("IMAGE_FETCH_INTERVAL", 10))
    ssim_threshold = args.ssim_threshold or float(os.getenv("IMAGE_FETCH_SSIM_THRESHOLD", 0.95))
    disable_ssim = args.disable_ssim or os.getenv("IMAGE_FETCH_DISABLE_SSIM", "false").lower() in ("1", "true", "yes")

    print("=== Configuration ===")
    print(f"Base Directory     : {base_dir}")
    print(f"Interval (seconds) : {interval}")
    print(f"SSIM Threshold     : {ssim_threshold}")
    print(f"SSIM Disabled      : {disable_ssim}")
    print(f"Camera URLs        :")
    for url in url_list:
        print(f"  - {url}")
    print("======================")

    os.makedirs(base_dir, exist_ok=True)
    connector = aiohttp.TCPConnector(limit=10)
    async with aiohttp.ClientSession(connector=connector) as session:
        await asyncio.gather(*[
            handle_stream(url, session, base_dir, interval, ssim_threshold, disable_ssim)
            for url in url_list
        ])

if __name__ == "__main__":
    asyncio.run(main())
