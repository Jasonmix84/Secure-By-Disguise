import sys, random, os
import argparse
import pickle
import imageio.v2 as imageio
import numpy as np
from cryp import RMT, AES
from Neuracrypt import NeuraCrypt
from PIL import Image
from tqdm import tqdm
from torchvision import datasets
import time


def save_image_from_array(img_array, save_path):
    """
    Save a numpy array as an image using PIL.

    Args:
        img_array (numpy.ndarray): The image data in numpy array format.
        save_path (str): The path to save the image.
    """
    img = Image.fromarray(img_array.astype('uint8'))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure the directory exists
    img.save(save_path, 'JPEG')

def load_image_to_array(image_path):
    """
    Load an image from a file path and return it as a numpy array.

    Args:
        image_path (str): The path to the image file.

    Returns:
        numpy.ndarray: The image data in numpy array format.
    """
    with Image.open(image_path) as img:
        img_array = np.array(img, dtype=np.float32)
    return img_array

class ImageDisguisingApp:
    def __init__(self, method, block_size, noise_level, dataset_directory, output_directory, shuffle=False):
        self.method = method
        self.block_size = block_size
        self.noise_level = noise_level
        self.dataset_directory = dataset_directory
        self.output_directory = output_directory
        self.shuffle = shuffle

        if self.method == 'RMT':
            self.encoder = RMT
        elif self.method == 'AES':
            self.encoder = AES
            self.encoder_instance = None
        elif self.method == 'NeuraCrypt':
            self.encoder = NeuraCrypt
        else:
            raise ValueError("Method must be either 'RMT', 'AES', or 'NeuraCrypt'")

    def encrypt_images(self):
        self.image_files = []
        self.image_paths = []

        for root, _, files in os.walk(self.dataset_directory):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.bmp')):
                    self.image_files.append(file)
                    self.image_paths.append(os.path.join(root, file))

        encrypted_images = []
        original_images = []

        for image_path in tqdm(self.image_paths, desc="Encrypting Images"):
            image = load_image_to_array(image_path)

            adjusted_row = (image.shape[0] + self.block_size - 1) // self.block_size * self.block_size
            adjusted_col = (image.shape[1] + self.block_size - 1) // self.block_size * self.block_size

            pad_row = adjusted_row - image.shape[0]
            pad_col = adjusted_col - image.shape[1]

            if len(image.shape) == 3:
                image_padded = np.pad(image, ((0, pad_row), (0, pad_col), (0, 0)), mode='edge')
            else:
                image_padded = np.pad(image, ((0, pad_row), (0, pad_col)), mode='edge')

            original_images.append(image_padded)

            if self.method == 'RMT':
                encoder_instance = self.encoder(
                    image_size=(image_padded.shape[0], image_padded.shape[1], image_padded.shape[2]) if len(image.shape) == 3 else (image_padded.shape[0], image_padded.shape[1]),
                    block_size=self.block_size,
                    Shuffle=self.shuffle
                )
                noise = self.noise_level != 0
                encrypted_img_array = encoder_instance.Encode(image_padded, noise=noise, noise_level=self.noise_level)
            elif self.method == 'AES':
                # Create encoder ONCE on first image, reuse for all others with SAME key
                if self.encoder_instance is None:
                    self.encoder_instance = self.encoder(
                        image_size=(image_padded.shape[0], image_padded.shape[1], image_padded.shape[2]) if len(image.shape) == 3 else (image_padded.shape[0], image_padded.shape[1]),
                        block_size=(self.block_size, self.block_size),
                        One_cipher=True,
                        Shuffle=self.shuffle
                    )
                    
                noise = self.noise_level != 0
                encrypted_img_array = self.encoder_instance.Encode(image_padded, noise=noise, noise_level=self.noise_level)
            elif self.method == 'NeuraCrypt':
                encoder_instance = self.encoder(
                    image_size=(image_padded.shape[0], image_padded.shape[1], image_padded.shape[2]) if len(image.shape) == 3 else (image_padded.shape[0], image_padded.shape[1]),
                    patch_size=self.block_size
                )
                encrypted_img_array = encoder_instance.forward(image_padded).detach().numpy()

            encrypted_images.append(encrypted_img_array)

            relative_path = os.path.relpath(image_path, self.dataset_directory)
            os.makedirs(self.dataset_directory, exist_ok=True)
            encrypted_image_path = os.path.join(self.output_directory, relative_path)
            save_image_from_array(encrypted_img_array, encrypted_image_path)

        print("Encryption done for all images in the directory!")
        self.original_images = original_images
        self.encrypted_images = encrypted_images


    def attack_images(self, known_pairs, original_dataset_dir=None, encrypted_dataset_dir=None, output_dir='recovered'):
        
        # If user supplied dataset directories, load images from them (matching relative paths)
        if original_dataset_dir and encrypted_dataset_dir:
            if not os.path.isdir(original_dataset_dir) or not os.path.isdir(encrypted_dataset_dir):
                raise ValueError("Both original_dataset_dir and encrypted_dataset_dir must be valid directories.")

           
            def collect_files(root_dir, strip_prefix=None):
                files = []
                for root, _, filenames in os.walk(root_dir):
                    for f in filenames:
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                            full = os.path.join(root, f)
                            rel = os.path.relpath(full, root_dir).replace("\\", "/")
                            # if requested, also add an alternate key with prefix stripped from filename
                            if strip_prefix and os.path.basename(rel).startswith(strip_prefix):
                                dirname = os.path.dirname(rel)
                                stripped_name = os.path.basename(rel)[len(strip_prefix):]
                                alt_rel = os.path.join(dirname, stripped_name).replace("\\", "/")
                                files.append((alt_rel, full))
                            files.append((rel, full))
                return dict(files)
    
            orig_map = collect_files(original_dataset_dir)
            # strip the "encrypted_" prefix from filenames in the encrypted dir so keys match originals
            enc_map = collect_files(encrypted_dataset_dir) # strip_prefix='encrypted_'


            # intersect relative paths to ensure same structure
            common_rels = sorted(set(orig_map.keys()).intersection(set(enc_map.keys())))
            if len(common_rels) == 0:
                raise ValueError("No matching files found between provided directories.")
            if len(common_rels) < known_pairs:
                raise ValueError(f"Not enough matching files ({len(common_rels)}) for known_pairs={known_pairs}.")

            # load arrays in deterministic order
            original_images = []
            encrypted_images = []
            self.image_files = []
            self.image_paths = []
            for rel in common_rels:
                orig_path = orig_map[rel]
                enc_path = enc_map[rel]
                original_images.append(load_image_to_array(orig_path))
                encrypted_images.append(load_image_to_array(enc_path))
                self.image_files.append(os.path.basename(rel))
                self.image_paths.append(rel)

            self.original_images = original_images
            self.encrypted_images = encrypted_images

        # if not present, ensure we have images from previous encrypt_images()
        if not hasattr(self, 'original_images') or not hasattr(self, 'encrypted_images'):
            raise ValueError("Images must be available. Call encrypt_images() first or provide dataset dirs to attack_images().")

        print("Starting attack...")
        print(len(self.original_images), len(self.encrypted_images))
        if known_pairs <= 0 or known_pairs > len(self.original_images):
            raise ValueError("known_pairs must be >0 and <= number of available images.")

        index = random.sample(range(len(self.original_images)), known_pairs)
        rec = []

        # create an encoder instance to call instance methods (Estimate, Recover, normalize)
        sample_img = self.original_images[0]
        if len(sample_img.shape) == 3:
            image_size = (sample_img.shape[0], sample_img.shape[1], sample_img.shape[2])
        else:
            image_size = (sample_img.shape[0], sample_img.shape[1])


        if self.method == 'RMT':
            encoder_instance = self.encoder(
                image_size=image_size,
                block_size=self.block_size,
                Shuffle=self.shuffle
            )

            # call instance Estimate (bound) with selected known pairs
            if len(sample_img.shape) == 3:
                RMT_Mat = encoder_instance.Estimate(np.array(self.original_images)[index, :, :, :], np.array(self.encrypted_images)[index, :, :, :])
            else:
                RMT_Mat = encoder_instance.Estimate(np.array(self.original_images)[index, :, :], np.array(self.encrypted_images)[index, :, :])


            # Determine the output directory (handle both absolute and relative paths)
            if os.path.isabs(output_dir):
                recovered_base = output_dir
            else:
                # If relative, make it relative to the dataset directory
                base_dataset = original_dataset_dir if original_dataset_dir else self.dataset_directory
                recovered_base = os.path.join(base_dataset, output_dir)
            
            os.makedirs(recovered_base, exist_ok=True)

            for i in tqdm(range(len(self.encrypted_images)), desc="Attacking Images"):
                encoded_img = self.encrypted_images[i]
                recover = encoder_instance.Recover(encoded_img, RMT_Mat)
                rec.append(recover)

                # convert recovered image to uint8 for saving
                r = recover.copy()
                if np.nanmax(r) <= 1.0:
                    r = (r * 255.0)
                r = np.clip(r, 0, 255).astype(np.uint8)

                # Determine the output path preserving directory structure
                if hasattr(self, "image_paths") and i < len(self.image_paths):
                    img_path = self.image_paths[i]
                    # Get relative path from source directory
                    if os.path.isabs(img_path):
                        base_dataset = original_dataset_dir if original_dataset_dir else self.dataset_directory
                        relative_path = os.path.relpath(img_path, base_dataset)
                    else:
                        relative_path = img_path
                    # Join with output directory to preserve structure
                    recovered_path = os.path.join(recovered_base, relative_path)
                else:
                    # Fallback if no image paths available
                    recovered_path = os.path.join(recovered_base, f"recovered_{i}.jpg")

                save_image_from_array(r, recovered_path)
                print(f"Saved recovered image: {recovered_path}")

            print("Attack done!")
            return rec

        elif self.method == 'AES':
            # AES Codebook Attack
            def build_codebook(pairs):
                codebook = {}
                for e_bytes, o_bytes in pairs:
                    for i in range(0, len(e_bytes), 16):
                        e_block = e_bytes[i:i+16]
                        o_block = o_bytes[i:i+16]
                        codebook[e_block] = o_block
                return codebook

            def codebook_attack(codebook, encrypted_bytes):
                reconstructed = bytearray()
                hits = 0
                total_checks = 0
                for i in range(0, len(encrypted_bytes), 16):
                    block = encrypted_bytes[i:i+16]
                    total_checks += 1
                    if block in codebook:
                        reconstructed.extend(codebook[block])
                        hits += 1
                    else:
                        reconstructed.extend(b"\x00" * 16)
                return bytes(reconstructed), hits, total_checks

            # Build codebook from known pairs
            codebook_pairs = []
            for idx in index:
                enc_img = self.encrypted_images[idx]
                orig_img = self.original_images[idx]
                enc_bytes = enc_img.astype(np.uint8).tobytes()
                orig_bytes = orig_img.astype(np.uint8).tobytes()
                codebook_pairs.append((enc_bytes, orig_bytes))
            codebook = build_codebook(codebook_pairs)

            # Determine the output directory (handle both absolute and relative paths)
            if os.path.isabs(output_dir):
                recovered_base = output_dir
            else:
                # If relative, make it relative to the dataset directory
                base_dataset = original_dataset_dir if original_dataset_dir else self.dataset_directory
                recovered_base = os.path.join(base_dataset, output_dir)
            
            os.makedirs(recovered_base, exist_ok=True)

            total_hits = 0
            total_checks = 0

            for i in tqdm(range(len(self.encrypted_images)), desc="Attacking Images (AES Codebook)"):
                enc_img = self.encrypted_images[i]
                enc_bytes = enc_img.astype(np.uint8).tobytes()
                rec_bytes, hits, checks = codebook_attack(codebook, enc_bytes)
                total_hits += hits
                total_checks += checks
                
                recover = np.frombuffer(rec_bytes, dtype=np.uint8).reshape(enc_img.shape)
                rec.append(recover)

                # convert recovered image to uint8 for saving
                r = recover.copy()
                if np.nanmax(r) <= 1.0:
                    r = (r * 255.0)
                r = np.clip(r, 0, 255).astype(np.uint8)

                # Determine the output path preserving directory structure
                if hasattr(self, "image_paths") and i < len(self.image_paths):
                    img_path = self.image_paths[i]
                    # Get relative path from source directory
                    if os.path.isabs(img_path):
                        base_dataset = original_dataset_dir if original_dataset_dir else self.dataset_directory
                        relative_path = os.path.relpath(img_path, base_dataset)
                    else:
                        relative_path = img_path
                    # Join with output directory to preserve structure
                    recovered_path = os.path.join(recovered_base, relative_path)
                else:
                    # Fallback if no image paths available
                    recovered_path = os.path.join(recovered_base, f"recovered_{i}.jpg")

                save_image_from_array(r, recovered_path)
                print(f"Saved recovered image: {recovered_path}")

            overall_hit_rate = (total_hits / total_checks * 100) if total_checks > 0 else 0
            print(f"\nOverall Codebook Hit Rate: {overall_hit_rate:.2f}% ({total_hits}/{total_checks})")
            
            # Write hit rate to file in current directory
            stats_file = "attack_stats.txt"
            with open(stats_file, "a") as f:
                f.write(f"Output Directory: {recovered_base}\n")
                f.write(f"Overall Codebook Hit Rate: {overall_hit_rate:.2f}%\n")
                f.write(f"Total Hits: {total_hits}\n")
                f.write(f"Total Checks: {total_checks}\n")
            print(f"Attack statistics saved to {stats_file}")
            
            print("AES Codebook Attack done!")
            return rec
        else:
            raise NotImplementedError("Attack is implemented for RMT/AES only.")
        

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type = str) #input directory
    parser.add_argument('--output', type = str) #output directory
    parser.add_argument('--method', type =str)
    parser.add_argument('--block_size', type = int, default = 4) # n x n
    parser.add_argument('--noise_level', type = int, default = 0)
    parser.add_argument('--shuffle', type=bool, default=False)
    args = parser.parse_args()
    
    # example using command line arguments
    # app = ImageDisguisingApp(method=args.method, block_size=args.block_size, noise_level=args.noise_level, dataset_directory=args.input, output_directory=args.output, shuffle=args.shuffle)
    # app.encrypt_images()
    # app.attack_images(known_pairs=10, original_dataset_dir=args.input, encrypted_dataset_dir=args.output, output_dir=os.path.join(args.output, 'recovered'))
    

    # example using hardcoded parameters
    # method = 'RMT'  # or 'AES'
    # block_size = 4  # Example block size
    # noise_level = 0  # Example noise level
    # dataset_directory = './40X_all'  # Directory containing the dataset
    # output_directory = './Breast/rmt-4-0'  # Directory to save encrypted images
    # app = ImageDisguisingApp(method, block_size, noise_level, dataset_directory, output_directory)
    # app.encrypt_images()
    # app.attack_images(known_pairs=10, original_dataset_dir='./40X_all', encrypted_dataset_dir='./Breast/rmt-4-0', output_dir='./Breast/rmt-4-0/recovered')
