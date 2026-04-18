import os
import shutil
from PIL import Image
import imagehash
import sys

def copy_unique_images(target_dir, output_dir, output_log="duplicate_report.txt", copy_labels_flag=0):
    valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    
    seen_hashes = {}
    skipped_duplicates = []
    copied_files = 0
    copied_labels = 0

    labels_dir = None
    if int(copy_labels_flag) == 1:
        parent_dir = os.path.dirname(os.path.normpath(target_dir))
        labels_dir = os.path.join(parent_dir, "labels")
        print(f"Label copy enabled. Looking for labels in '{labels_dir}'")
        if not os.path.isdir(labels_dir):
            print(f"Warning: labels directory not found at '{labels_dir}'. Label copying will be skipped.")
            labels_dir = None

    print(f"Scanning '{target_dir}' for visually identical images...")
    print(f"Unique images will be safely copied to '{output_dir}'\n")

    # Step 1: Gather all images and sort them by file size (largest first).
    image_files = []
    for root, _, files in os.walk(target_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_extensions:
                filepath = os.path.join(root, file)
                image_files.append((filepath, os.path.getsize(filepath)))
    
    # Sort descending by size
    image_files.sort(key=lambda x: x[1], reverse=True)

    # Step 2: Process images
    for filepath, _ in image_files:
        try:
            # Open the image and calculate its perceptual hash
            with Image.open(filepath) as img:
                img_hash = str(imagehash.phash(img))
            
            if img_hash in seen_hashes:
                # Duplicate found -> skip copying, add to log
                original_kept = seen_hashes[img_hash]
                skipped_duplicates.append(f"{filepath} (Duplicate of {original_kept})")
                print(f"Skipped duplicate: {filepath}")
            else:
                # Unique visual structure -> add to dictionary and copy
                seen_hashes[img_hash] = filepath
                
                # Calculate the relative path from the target directory
                rel_path = os.path.relpath(filepath, target_dir)
                
                # Construct the new destination path
                dest_path = os.path.join(output_dir, rel_path)
                
                # Ensure the subdirectories exist in the output folder
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                
                # Copy the file (shutil.copy2 preserves file metadata like creation dates)
                shutil.copy2(filepath, dest_path)
                copied_files += 1

                # Optionally copy corresponding label files from sibling /labels directory
                if labels_dir is not None:
                    rel_no_ext = os.path.splitext(rel_path)[0]
                    labels_parent = os.path.join(labels_dir, os.path.dirname(rel_no_ext))
                    labels_stem = os.path.basename(rel_no_ext)

                    if os.path.isdir(labels_parent):
                        for label_name in os.listdir(labels_parent):
                            label_path = os.path.join(labels_parent, label_name)
                            if os.path.isfile(label_path) and os.path.splitext(label_name)[0] == labels_stem:
                                label_rel_path = os.path.relpath(label_path, labels_dir)
                                label_dest_path = os.path.join(output_dir, "labels", label_rel_path)
                                os.makedirs(os.path.dirname(label_dest_path), exist_ok=True)
                                shutil.copy2(label_path, label_dest_path)
                                copied_labels += 1
                
        except Exception as e:
            print(f"Error processing {filepath}. It might be corrupted: {e}")

    # Write the skipped file paths to the output log
    total_duplicates = len(skipped_duplicates)
    with open(output_log, 'w', encoding='utf-8') as log_file:
        log_file.write(f"Total duplicates filtered out: {total_duplicates}\n")
        log_file.write(f"Total unique files copied: {copied_files}\n")
        log_file.write(f"Total labels copied: {copied_labels}\n")
        log_file.write("-" * 60 + "\n")
        if skipped_duplicates:
            for entry in skipped_duplicates:
                log_file.write(f"{entry}\n")
        else:
            log_file.write("No visual duplicate images found.\n")

    # Print final summary
    print("\n" + "=" * 40)
    print("Process Complete.")
    print(f"Unique images safely copied: {copied_files}")
    print(f"Labels copied: {copied_labels}")
    print(f"Duplicates filtered out: {total_duplicates}")
    print(f"A report of the skipped duplicates was saved to: {os.path.abspath(output_log)}")

# --- Execution ---
if __name__ == "__main__":
    # example run for classification dataset
    # python find-dupes.py /path/to/target /path/to/output/ duplicate_report.txt 0

    # example run for segmentation datasets
    # python find-dupes.py /path/to/target/images /path/to/output/images duplicate_report.txt 1

    
    # Take the first and second command-line arguments as the target and output directories
    TARGET_DIRECTORY = sys.argv[1] if len(sys.argv) > 1 else input("Enter the path to the target directory containing images: ")
    OUTPUT_DIRECTORY = sys.argv[2] if len(sys.argv) > 2 else input("Enter the path to the output directory where unique images will be copied: ")
    log_file = sys.argv[3] if len(sys.argv) > 3 else "duplicate_report.txt"
    copy_labels_flag = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    

    copy_unique_images(TARGET_DIRECTORY, OUTPUT_DIRECTORY, log_file, copy_labels_flag)