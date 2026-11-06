import numpy as np
from sklearn.neighbors import KDTree
import json

from src.utils.input import read_obj
from src.utils.output import write_obj

if __name__ == '__main__':
    # input_file = '<REDACTED>'
    # label_file = '<REDACTED>'
    input_file = '<REDACTED>'
    label_file = '<REDACTED>'

    with open(label_file, 'r') as fp:
        dic = json.load(fp)

    # id_patient = dic['id_patient']
    # jaw = dic['jaw']
    labels = np.array(dic['labels'])
    verts, faces, normals = read_obj(input_file, True)
    kdtree = KDTree(verts)

    crop_results = np.zeros((verts.shape[0], ), dtype=np.int32)
    nn_indices = kdtree.query_radius(verts[labels > 0], 3)
    for indices in nn_indices:
        crop_results[indices] = 1

    vid_map = np.ones((verts.shape[0], ), dtype=np.int32) * -1
    real_id = 0
    for i in range(len(crop_results)):
        if crop_results[i] > 0:
            vid_map[i] = real_id
            real_id += 1

    verts = verts[crop_results > 0]

    final_faces = []
    for face in faces:
        cnt = 0
        for vid in face:
            if crop_results[vid] > 0:
                cnt += 1
        if cnt == 3:
            final_faces.append(vid_map[face])

    faces = np.array(final_faces)
    normals = normals[crop_results > 0]
    labels = np.expand_dims(labels[crop_results > 0], -1)
    labels = np.repeat(labels, 3, -1)

    verts = np.concatenate((verts, labels), axis=-1)

    # write_obj('<REDACTED>', verts, faces, None)
    # np.savetxt('<REDACTED>', np.asarray(labels[:, 0], dtype=np.int32))
    write_obj('<REDACTED>', verts, faces, None)
    np.savetxt('<REDACTED>', np.asarray(labels[:, 0], dtype=np.int32))
