
from model.FaceMap import FaceMap


if __name__ == '__main__':

    face_map = FaceMap()
    face_map.adjust_transition_prob()
    face_map.face_cluster()
