# HTTP Image Fetch Script (with Motion Detection)

This script continuously fetches images from a specified URL, compares them using the Structural Similarity Index (SSIM) for motion detection, and saves the images to a structured folder hierarchy if a change is detected. It is useful for applications like monitoring or time-lapse photography.

## Features

- Fetch images asynchronously from a given URL.
- Save images to a daily folder structure based on the URL and date.
- Compare images using SSIM to detect motion or significant changes.
- Option to disable SSIM comparisons and save all images.
- Customizable fetch interval and SSIM threshold.

## Requirements

- Python 3.8 or higher
- Libraries: `aiohttp`, `numpy`, `Pillow`, `scikit-image`

Install the required libraries with:


```bash
pip install -r requirements.txt
```
or 
```bash
pip install aiohttp numpy Pillow scikit-image
```

## Usage

### Command-Line Arguments

| Argument            | Description                                      | Default Value                  |
|---------------------|--------------------------------------------------|--------------------------------|
| `--url`             | The URL to fetch images from.                   | None (required)               |
| `--output`          | Base directory to save images.                  | `./images`                    |
| `--interval`        | Interval (in seconds) between fetches.          | 10                             |
| `--ssim-threshold`  | SSIM threshold for saving images.               | 0.95                          |
| `--disable-ssim`    | Disable SSIM comparisons and save all images.   | False                         |

### Environment Variables

If command-line arguments are not provided, environment variables can be used:

- `IMAGE_FETCH_URL`: The URL to fetch images from.
- `IMAGE_FETCH_OUTPUT`: Base directory to save images.
- `IMAGE_FETCH_INTERVAL`: Interval between fetches.
- `IMAGE_FETCH_SSIM_THRESHOLD`: SSIM threshold for saving images.
- `IMAGE_FETCH_DISABLE_SSIM`: Disable SSIM comparisons (`true` or `false`).

### Examples

Fetch images every 5 seconds and save only when significant changes are detected:

```bash
python fetch_images.py --url "http://example.com/image.jpg" --interval 5 --output "./images" --ssim-threshold 0.9
```

Fetch images and save all without SSIM comparison:

```bash
python fetch_images.py --url "http://example.com/image.jpg" --disable-ssim
```

## Folder Structure

Images are saved in the following structure:

```
<output_dir>/<url_folder>/<YYYY-MM-DD>/<HHMMSS_microsecond>.jpg
```

For example:

```
images/example_com/2024-12-09/123456_789123.jpg
```

## Notes

- The first image is always saved, even if SSIM comparisons are enabled.
- The script supports graceful handling of timeouts, connection errors, and invalid image data.

## Contributing

Feel free to submit issues or pull requests for improvements or bug fixes.



