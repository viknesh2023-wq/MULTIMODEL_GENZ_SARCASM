"""Google Drive direct download helper supporting large file virus scan confirmations."""
import os
import re
import requests
from typing import Tuple


def download_file_from_google_drive(file_id: str, destination: str) -> Tuple[bool, str]:
    """Downloads a public file from Google Drive, handling large file virus warnings."""
    URL = f"https://drive.google.com/uc?export=download&id={file_id}"
    session = requests.Session()
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    response = session.get(URL, headers=headers, stream=True)

    content_type = response.headers.get('Content-Type', '')
    if 'text/html' in content_type:
        html = response.text
        # Parse download form action and hidden inputs
        action_match = re.search(r'<form[^>]*id=["\']download-form["\'][^>]*action=["\']([^"\']+)["\']', html)
        if not action_match:
            action_match = re.search(r'<form[^>]*action=["\']([^"\']+)["\']', html)

        if action_match:
            action_url = action_match.group(1)
            # Find all input name/value pairs
            input_pattern = re.compile(r'<input[^>]+name=["\']([^"\']+)["\'][^>]+value=["\']([^"\']*)["\']')
            params = {}
            for name, val in input_pattern.findall(html):
                params[name] = val

            # Also check alternative input attribute order
            input_pattern_rev = re.compile(r'<input[^>]+value=["\']([^"\']*)["\'][^>]+name=["\']([^"\']+)["\']')
            for val, name in input_pattern_rev.findall(html):
                params[name] = val

            response = session.get(action_url, params=params, headers=headers, stream=True)
        else:
            return False, "Google Drive download form could not be parsed."

    if response.status_code != 200:
        return False, f"Failed with HTTP status {response.status_code}"

    # Verify response is not an error page
    if 'text/html' in response.headers.get('Content-Type', ''):
        return False, f"Received HTML instead of binary stream: {response.text[:200]}"

    os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
    chunk_size = 65536
    downloaded = 0

    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)

    return True, f"Successfully downloaded {downloaded} bytes to {destination}"


if __name__ == "__main__":
    import sys
    fid = sys.argv[1] if len(sys.argv) > 1 else "18m3KdDCXgkAlvTbjNhfftvU9LdhyDUTt"
    dest = sys.argv[2] if len(sys.argv) > 2 else "data/raw/sarcnet/sarcnet_bundle.zip"
    print(f"Downloading {fid} to {dest}...")
    success, msg = download_file_from_google_drive(fid, dest)
    print("Success:", success)
    print("Message:", msg)
