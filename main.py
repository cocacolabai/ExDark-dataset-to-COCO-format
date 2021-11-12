from pathlib import Path

from create_annotations import (
    create_image_annotation,
    create_annotation_from_yolo_format,
    coco_format,
)
import cv2
import argparse
import json
import numpy as np

#################################################
# Change the classes depend on your own dataset.#
# Don't change the list name 'Classes'          #
#################################################

YOLO_DARKNET_SUB_DIR = "YOLO_darknet"
label_dict = {"Bicycle":0, "Boat":1, "Bottle":2, "Bus":3, "Car":4, "Cat":5, \
              "Chair":6, "Cup":7, "Dog":8, "Motorbike":9, "People":10, "Table":11}# when the code is too long, use "\" at the bottom of the code and then add enter

classes = [
    "Bicycle",
    "Boat",
    "Botle"
    "Bus",
    "Car",
    "Cat",
    "Chair",
    "Cup",
    "Dog",
    "Motorbike",
    "People",
    "Table"
]

color_lists = [(255, 0, 0),(0, 255, 0),(0, 0, 225),(255, 255, 0),(0, 255, 255),(255, 0, 255),\
             (100, 0, 0),(0, 100, 0),(0, 0, 100),(100, 100, 0),(100, 0, 100),(0, 100, 100),(100, 100, 100)]

def get_images_info_and_annotations(opt):
    path = Path(opt.path)  #train.txt's path
    annotations = []
    images_annotations = []
    print(path.is_dir())   #is_dir mean not the final file, maybe a folder ,while is_file mean the final file. both two functions need absolute path
    if path.is_dir():
        file_paths = sorted(path.rglob("*.jpg"))  #look for all the files
    else:
        with open(path, "r") as fp:
            read_lines = fp.readlines()
        file_paths = [Path(line.replace("\n", "")) for line in read_lines]

    image_id = 0
    annotation_id = 1  # In COCO dataset format, you must start annotation id with '1'

    print(image_id)
    for file_path in file_paths:
        # Check how many items have progressed
        if image_id % 1000 == 0:
            print("Processing " + str(image_id) + " ...")


        img_file = cv2.imread(str(file_path))
        h, w, _ = img_file.shape
        image_annotation = create_image_annotation(
            file_path=file_path, width=w, height=h, image_id=image_id
        )
        images_annotations.append(image_annotation)

        label_file_name = f"{file_path.stem}.jpg.txt"  #     file_path.stem:acquire the file's name without the format.
                                                                             #     file_path.name:acquire the file's name with the format.
                                                                             #     f{} : formate the print
        if opt.yolo_subdir:
            annotations_path = file_path.parent / YOLO_DARKNET_SUB_DIR / label_file_name
        else:
            annotations_path = file_path.parent / label_file_name



        if not annotations_path.exists():
            continue  # The image may not have any applicable annotation txt file.

        with open(str(annotations_path), "r") as label_file:
            label_read_line = label_file.readlines()

        # yolo format - (class_id, x_center, y_center, width, height)
        # coco format - (annotation_id, x_upper_left, y_upper_left, width, height)
        for line1 in label_read_line[1:]:
            label_line = line1
            key = label_line.split()[0]
            category_id = (
                int(label_dict[key]) + 1
            )  # you start with annotation id with '1'
            min_x = int(label_line.split()[1])
            min_y = int(label_line.split()[2])
            width = int(label_line.split()[3])
            height = int(label_line.split()[4])

            min_x = int(img_file.shape[1] * min_x)
            min_y = int(img_file.shape[0] * min_y)
            int_width = int(img_file.shape[1] * width)
            int_height = int(img_file.shape[0] * height)



            annotation = create_annotation_from_yolo_format(
                min_x,
                min_y,
                width,
                height,
                image_id,
                category_id,
                annotation_id,
                segmentation=opt.box2seg,
            )
            annotations.append(annotation)
            annotation_id += 1

        image_id += 1  # if you finished annotation work, updates the image id.
        print(image_id)

    return images_annotations, annotations


def debug(opt):
    path = opt.path
    color_list = np.random.randint(low=0, high=256, size=(len(classes), 3)).tolist()

    # read the file
    file = open(path, "r")
    read_lines = file.readlines()
    file.close()

    for line in read_lines:
        print("Image Path : ", line)
        # read image file
        img_file = cv2.imread(line[:-1])

        # read .txt file
        label_path = line[:-4] + "jpg." + "txt"
        label_file = open(label_path, "r")
        label_read_line = label_file.readlines()[1:]
        label_file.close()

        for line1 in label_read_line:
            label_line = line1

            category_id = int(label_dict[label_line.split()[0]]) + 1
            min_x = int(label_line.split()[1])
            min_y = int(label_line.split()[2])
            width = int(label_line.split()[3])
            height = int(label_line.split()[4])

            x1 = min_x
            y1 = min_y
            x2 = min_x + width
            y2 = min_y + height
            
            cv2.rectangle(
                img_file,
                (x1, y1), (x2, y2),
                color_lists[category_id],
                3,
            )

        cv2.imshow('1', img_file)
        delay = cv2.waitKeyEx()

        # If you press ESC, exit
        if delay == 27 or delay == 113:
            break

        cv2.destroyAllWindows()


def get_args():
    parser = argparse.ArgumentParser("Yolo format annotations to COCO dataset format")
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default='train.txt',
        help="Absolute path for 'train.txt' or 'test.txt', or the root dir for images.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=True,
        help="Visualize bounding box and print annotation information",
    )
    parser.add_argument(
        "--output",
        default="train_coco.json",
        type=str,
        help="Name the output json file",
    )
    parser.add_argument(
        "--yolo-subdir",
        action="store_true",
        help="Annotations are stored in a subdir not side by side with images.",
    )
    parser.add_argument(
        "--box2seg",
        action="store_true",
        help="Coco segmentation will be populated with a polygon "
        "that matches replicates the bounding box data.",
    )
    args = parser.parse_args()
    return args


def main(opt):
    output_name = opt.output
    output_path = "output/" + output_name

    print("Start!")

    if opt.debug is True:
        debug(opt)
        print("Debug Finished!")
    else:
        (
            coco_format["images"],
            coco_format["annotations"],
        ) = get_images_info_and_annotations(opt)

        for index, label in enumerate(classes):
            categories = {
                "supercategory": "Defect",
                "id": index + 1,  # ID starts with '1' .
                "name": label,
            }
            coco_format["categories"].append(categories)

        with open(output_path, "w") as outfile:
            json.dump(coco_format, outfile, indent=4)

        print("Finished!")


if __name__ == "__main__":
    options = get_args()
    main(options)
