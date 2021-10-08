import os
from skimage import io
from helper.face_detection import detect_faces
from PIL import Image


def load_image_folder(number):
    training_path = "../data/raw/Training/{}"
    training_file_path = "../data/raw/Training/{}/{}"
    observation_folder_path = training_path.format(number)
    filenames = os.listdir(observation_folder_path)
    images_path_inside_folder = [
        training_file_path.format(number, filename) for filename in
        filenames]
    images = [io.imread(image_path) for image_path in images_path_inside_folder]
    faces_detected = [(n, detect_faces(img)) for n, img in enumerate(images)]
    extracted_face_detected = []

    for item in faces_detected:
        (img_index, face_rect) = item
        if len(face_rect) > 1:
            for face in face_rect:
                extracted_face_detected.append((img_index, face))
        else:
            extracted_face_detected.append((img_index, face_rect[0]))

    list_face_image = []
    for n, (img_index, face_rect) in enumerate(extracted_face_detected):
        face = Image.fromarray(images[img_index]).crop(face_rect)
        list_face_image.append(face)

    return list_face_image
