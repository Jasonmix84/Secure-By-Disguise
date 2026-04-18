import sys,os
from PIL import Image
from collections import Counter
import argparse

def scan_all_subdirectories(root_dir):
    size_counts = Counter()
    extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
    total_files = 0

    # os.walk yields (current_path, directories, files)
    for subdir, dirs, files in os.walk(root_dir):
        for filename in files:
            if filename.lower().endswith(extensions):
                total_files += 1
                path = os.path.join(subdir, filename)
                try:
                    with Image.open(path) as img:
                        # Store as (width, height)
                        size_counts[img.size] += 1
                except Exception as e:
                    print(f"Skipping {filename}: {e}", file=sys.stderr)

    print(f"Total images found: {total_files}", file=sys.stderr)
    print(f"{'Resolution (W x H)':<20} | {'Count':<8} | {'Aspect Ratio'}", file= sys.stderr)
    print("-" * 50, file=sys.stderr)

    # Sort by most common
    max_w = 0
    max_h = 0
    for size, count in size_counts.most_common():
        w, h = size
        if w > max_w:
            max_w = w
        if h > max_h:
            max_h = h
        ratio = f"{w/h:.2f}:1"
        print(f"{str(size):<20} | {count:<8} | {ratio}", file=sys.stderr)
    print("max w, h:" + str(max_w) + " " + str(max_h), file=sys.stderr)
    return max_w, max_h


def resize_recursive(source_dir, output_root, target_size=(224, 224)):
    extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')

    for subdir, dirs, files in os.walk(source_dir):
        for filename in files:
            if filename.lower().endswith(extensions):
                # 1. Construct the full input path
                img_path = os.path.join(subdir, filename)

                # 2. Construct the matching output path
                relative_path = os.path.relpath(subdir, source_dir)
                save_dir = os.path.join(output_root, relative_path)

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                save_path = os.path.join(save_dir, filename)

                # 3. Process the image
                try:
                    with Image.open(img_path) as img:
                        # Convert to RGB to handle PNG transparency/RGBA issues
                        if img.mode in ("RGBA", "P"):
                            img = img.convert("RGB")

                        resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
                        resized_img.save(save_path)
                        print(f"Resized: {relative_path}/{filename}")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")


def main():
    # example run
    # python3 /home/jason/experiments/Secure-By-Disguise/encryption/img-size.py --input /path/to/images --output /path/to/resized_images

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type =str)
    parser.add_argument('--output', type = str, default = "")
    args = parser.parse_args()
    if args.output == "":
        scan_all_subdirectories(args.input)
    else:
        w, h = scan_all_subdirectories(args.input)
        resize_recursive(args.input, args.output, target_size=(w, h))
        print(f"Resized all images to {w}x{h} and saved to {args.output}")

if __name__ == "__main__":
    main()
