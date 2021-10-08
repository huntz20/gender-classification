from typing import List
from PIL import Image
import numpy as np
import face_recognition
from sklearn.cluster import DBSCAN
from scipy import stats


def image_clustering(list_face_image: List):
    list_encoded_face = [face_recognition.face_encodings(np.asarray(img)) for img in list_face_image]
    list_encoded_face_filtered = [filtered[0] if len(filtered) else np.random.rand(128) for filtered in
                                  list_encoded_face]
    dbscan_cluster = DBSCAN(min_samples=2)
    dbscan_cluster.fit(list_encoded_face_filtered)
    mode_label = stats.mode( np.array(dbscan_cluster.labels_)[dbscan_cluster.labels_ != -1])[0][0]
    filtered_label = [label == mode_label and mode_label != -1 for label in dbscan_cluster.labels_]
    return np.array(list_face_image)[filtered_label]


