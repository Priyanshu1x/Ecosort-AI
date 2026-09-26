from PIL import Image
import os
import imagehash
if os.path.exists("/content"):
    base_folder = "/content/dataset"
else:
    base_folder=""
folders = ("Dry", "ewaste", "Recyclable", "Wet")
image_extensions = (".jpg", ".jpeg", ".png", ".jfif")


def find_corrupted_images():
    corrupted_images = []
    total_images = 0

    for folder in folders:
        folder_path = os.path.join(base_folder, folder)

        if not os.path.isdir(folder_path):
            print(f"Folder not found: {folder_path}")
            continue

        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if not file.lower().endswith(image_extensions):
                    continue

                image_path = os.path.join(root, file)
                total_images += 1

                try:
                    with Image.open(image_path) as image:
                        image.verify()
                except Exception:
                    corrupted_images.append(image_path)

    return total_images, corrupted_images


total_images, corrupted_images = find_corrupted_images()

print("Total images checked:", total_images)
print("Corrupted images found:", len(corrupted_images))

for image_path in corrupted_images:
    os.remove(image_path)


# find duplicate images

def find_duplicates():

    seen_hash= set()
    counter=0
    foldr_path= base_folder
    for root,fold,files in os.walk(foldr_path):
        for img in files:
            img_path=os.path.join(str(root),str(img))
            opened_img=Image.open(img_path)
            img_hash=str(imagehash.average_hash(opened_img))
            opened_img.close()
            if img_hash not in seen_hash:
                seen_hash.add(img_hash)

            else:
                counter+=1
                os.remove(img_path)

    return counter

total_removed=find_duplicates()
print(f"{total_removed} images found out to be duplicate")