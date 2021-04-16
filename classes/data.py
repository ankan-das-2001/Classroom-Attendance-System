import os
from copy import deepcopy

import cv2
from tqdm import tqdm


class DataOps:
    # Load HAAR face classifier
    __face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    def __init__(self, count, split):
        self.__count = count
        self.__split = split
        self.__cap = cv2.VideoCapture(0)

    def __del__(self):
        self.__cap.release()
        cv2.destroyAllWindows()

    def __face_extractor(self, img):
        # Function detects faces and returns all cropped faces
        # If no face detected, returns None
        faces = self.__face_classifier.detectMultiScale(img, 1.3, 5)
        if faces is ():
            return None
        # Crop all faces found
        cropped_faces = []
        for (x, y, w, h) in faces:
            x = x - 10
            y = y - 10
            cropped_faces.append(img[y : y + h + 50, x : x + w + 50])
        return cropped_faces

    def __write_show_img(self, name, count, img):
        # Writes image to train or val directories
        if count <= self.__count * self.__split:
            file_name_path = "Datasets/train/{}/".format(name) + str(count) + ".jpg"
        else:
            file_name_path = "Datasets/val/{}/".format(name) + str(count) + ".jpg"
        cv2.imwrite(file_name_path, img)

    def __check_make_dir(self, name, parent):
        # Checks if required directory exists
        # If not, make directory
        if not os.path.exists("Datasets/{}/{}".format(parent, name)):
            path = os.path.join(os.getcwd(), "Datasets", parent, name)
            os.makedirs(path)

    def update_roll(self, roll):
        # Update images for single roll entry
        prog = tqdm(total=self.__count)
        self.__check_make_dir(roll, "train")
        self.__check_make_dir(roll, "val")
        count = 0
        while count < self.__count:
            _, frame = self.__cap.read()
            faces = self.__face_extractor(frame)
            if (faces != None and roll != "blank") or (
                faces == None and roll == "blank"
            ):
                if faces == None and roll == "blank":
                    faces = [deepcopy(frame)]
                count += 1
                img = cv2.resize(faces[0], (400, 400))
                self.__write_show_img(roll, count, img)
                print(end="\r")
            else:
                print("Insufficient data", end="\r")
            if cv2.waitKey(1) == 13:
                break
            prog.update(1)
        prog.close()

    def create_dataset(self):
        while True:
            name = input("Enter candidate name:")
            self.update_roll(name)
            if name == "blank":
                break
        print("Dataset created successfully")
