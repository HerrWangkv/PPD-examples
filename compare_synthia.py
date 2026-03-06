import os
import shutil
import argparse

def collect_synthia_files(target_filename, source_root="data", output_dir="collected_results"):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    count = 0
    # Walk through the source_root
    synthia_dirs = [
        d for d in os.listdir(source_root)
        if d.startswith("synthia") and os.path.isdir(os.path.join(source_root, d))
    ]
    for dir in synthia_dirs:
        if dir.startswith("synthia"):
            assert "RGB" in os.listdir(os.path.join(source_root, dir)), f"'RGB' folder not found in {os.path.join(source_root, dir)}"
            rgb_folder = os.path.join(source_root, dir, "RGB")
            if target_filename in os.listdir(rgb_folder):
                source_file = os.path.join(rgb_folder, target_filename)
                
                # Use the folder name as part of the new filename to avoid overwriting
                # Example: synthia_v1/0007943.png -> collected_results/synthia_v1_0007943.png
                new_filename = dir + ".png"
                dest_file = os.path.join(output_dir, new_filename)

                shutil.copy2(source_file, dest_file)
                print(f"Copied: {source_file} -> {dest_file}")
                count += 1

    print(f"\nFinished! Collected {count} files into '{output_dir}'.")

def arg_parser():
    parser = argparse.ArgumentParser(description="Collect specific files from Synthia dataset.")
    parser.add_argument("index", type=str, help="The file index.")
    parser.add_argument("--source_root", type=str, default="/mrtstorage/users/kwang/synthia_sim2real", help="Root directory of the Synthia dataset.")
    parser.add_argument("--output_dir", type=str, default="synthia_comparison", help="Directory to save collected files.")
    return parser

if __name__ == "__main__":
    args = arg_parser().parse_args()
    filename = f"{int(args.index):07d}.png"
    os.makedirs(args.output_dir, exist_ok=True)
    collect_synthia_files(filename, source_root=args.source_root, output_dir=args.output_dir)