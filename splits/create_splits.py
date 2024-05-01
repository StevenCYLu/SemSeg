import os

split_dir = "ade20k"
training_prefix = "/scratch/gilbreth/lu842/data/ADEChallengeData2016/images/validation"
annotation_prefix = "/scratch/gilbreth/lu842/data/ADEChallengeData2016/annotations/validation"

with open("val.txt", "w") as f:
    for img in os.listdir(training_prefix):
        ann = os.path.join(annotation_prefix, img.replace(".jpg", ".png"))
        f.write(os.path.join(training_prefix, img) + " " + ann + "\n")


# for split in os.listdir(split_dir):
#     new_line = []
#     split_path = os.path.join(split_dir, split)
#     with open(split_path, "r") as f:
#         lines = f.readlines()
#         for line in lines:
#             ann = os.path.join(annotation_prefix, line.split("/")[-1].replace(".jpg", ".png").strip())
#             new_line.append(line.strip() + " " + ann + "\n")
#     with open(split_path, "w") as f:
#         f.writelines(new_line)
        
