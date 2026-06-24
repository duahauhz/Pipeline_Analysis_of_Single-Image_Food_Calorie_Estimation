import shutil
import os

base = "ECUSTFD-calorie-estimation-using-food-image"
out_base = "datasets/ECUSTFD"
img_dir = os.path.join(base, "JPEGImages")
lbl_dir = os.path.join(base, "labels")

# Faulty files to exclude as per plan
exclude_files = ["mix002T(2)", "mix005S(4)"]

splits = ['train', 'val', 'test']

for split in splits:
    os.makedirs(f"{out_base}/images/{split}", exist_ok=True)
    os.makedirs(f"{out_base}/labels/{split}", exist_ok=True)
    
    split_file = f"{base}/ImageSets/Main/{split}.txt"
    if not os.path.exists(split_file):
        print(f"Warning: Split file {split_file} not found.")
        continue
        
    with open(split_file) as f:
        names = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"Processing {split} split with {len(names)} entries...")
    count = 0
    for name in names:
        if name in exclude_files:
            print(f"Excluding faulty file: {name}")
            continue
            
        # Copy image
        img_src = os.path.join(img_dir, f"{name}.JPG")
        if os.path.exists(img_src):
            shutil.copy2(img_src, f"{out_base}/images/{split}/{name}.JPG")
        else:
            # Check for lowercase extension just in case
            img_src_alt = os.path.join(img_dir, f"{name}.jpg")
            if os.path.exists(img_src_alt):
                shutil.copy2(img_src_alt, f"{out_base}/images/{split}/{name}.jpg")
        
        # Copy label
        lbl_src = os.path.join(lbl_dir, f"{name}.txt")
        if os.path.exists(lbl_src):
            shutil.copy2(lbl_src, f"{out_base}/labels/{split}/{name}.txt")
        
        count += 1

    print(f"Finished {split} split. Copied {count} files.")

print("Dataset organization complete!")
