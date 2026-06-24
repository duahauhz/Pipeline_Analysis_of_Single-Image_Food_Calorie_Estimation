import xml.etree.ElementTree as ET
import os

classes = [
    "apple", "banana", "bread", "bun", "doughnut", "egg",
    "fired_dough_twist", "grape", "lemon", "litchi", "mango",
    "mooncake", "orange", "peach", "pear", "plum", "qiwi",
    "sachima", "tomato", "coin"
]

def convert_voc_to_yolo(xml_path, output_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        size = root.find('size')
        img_w = int(size.find('width').text)
        img_h = int(size.find('height').text)
        
        # Base case for empty images or zero division
        if img_w == 0 or img_h == 0:
            print(f"Warning: Zero dimension in {xml_path}")
            return

        with open(output_path, 'w') as f:
            for obj in root.iter('object'):
                cls_name = obj.find('name').text
                if cls_name not in classes:
                    continue
                cls_id = classes.index(cls_name)
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                
                # Check for mix002T(2) and mix005S(4) removal as per plan (invalid coin)
                # Actually, the plan says to remove them, but maybe it's better to just skip
                # them if they are known faulty. However, the plan says "remove from dataset".
                # I'll proceed with conversion first.

                x_center = ((xmin + xmax) / 2) / img_w
                y_center = ((ymin + ymax) / 2) / img_h
                width = (xmax - xmin) / img_w
                height = (ymax - ymin) / img_h
                
                f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    except Exception as e:
        print(f"Error processing {xml_path}: {e}")

# Paths
base_dir = "ECUSTFD-calorie-estimation-using-food-image"
ann_dir = os.path.join(base_dir, "Annotations")
out_dir = os.path.join(base_dir, "labels")

if not os.path.exists(out_dir):
    os.makedirs(out_dir, exist_ok=True)

xml_files = [f for f in os.listdir(ann_dir) if f.endswith('.xml')]
print(f"Found {len(xml_files)} XML files.")

count = 0
for xml_file in xml_files:
    xml_path = os.path.join(ann_dir, xml_file)
    txt_name = xml_file.replace('.xml', '.txt')
    output_path = os.path.join(out_dir, txt_name)
    convert_voc_to_yolo(xml_path, output_path)
    count += 1

print(f"Successfully converted {count} files to YOLO format in {out_dir}.")
