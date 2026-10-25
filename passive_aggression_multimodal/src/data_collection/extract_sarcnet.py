"""Extracts SarcNet zip bundle into data/raw/sarcnet and data/images/sarcnet."""
import os
import zipfile

zip_path = "data/raw/sarcnet/sarcnet_bundle.zip"
raw_out = "data/raw/sarcnet"
img_out = "data/images/sarcnet"

os.makedirs(raw_out, exist_ok=True)
os.makedirs(img_out, exist_ok=True)

print(f"Opening {zip_path}...")
with zipfile.ZipFile(zip_path, "r") as z:
    for member in z.namelist():
        filename = os.path.basename(member)
        if not filename:
            continue
        if member.endswith(".csv"):
            target_path = os.path.join(raw_out, filename)
            with open(target_path, "wb") as f_out, z.open(member) as f_in:
                f_out.write(f_in.read())
            print(f"Extracted CSV: {target_path}")
        elif member.startswith("SarcNet Image-Text/Image/") and (member.endswith(".jpg") or member.endswith(".png")):
            target_path = os.path.join(img_out, filename)
            with open(target_path, "wb") as f_out, z.open(member) as f_in:
                f_out.write(f_in.read())

print("SarcNet extraction complete.")
print(f"Total extracted images: {len(os.listdir(img_out))}")
