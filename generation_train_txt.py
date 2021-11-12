import glob
with open('./tutorial/train.txt', 'w') as f:
    for i in glob.glob(r"/home/shu-usv002/baiyunfer_file/Yolo-to-COCO-format-converter-master/Yolo-to-COCO-format-converter-master/tutorial/train/*.jpg"):
        f.write(i + '\n')
