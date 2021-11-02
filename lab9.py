import dlib
import os
from skimage import io
from scipy.spatial import distance
import cv2

train_dirpath = f'{os.getcwd()}/training_dataset/'
test_dirpath = f'{os.getcwd()}/test_dataset/'
sp = dlib.shape_predictor(
    f"{os.getcwd()}/shape_predictor_68_face_landmarks.dat")
face_rec = dlib.face_recognition_model_v1(
    f'{os.getcwd()}/dlib_face_recognition_resnet_model_v1.dat')
detector = dlib.get_frontal_face_detector()

BATCH = 3
VALID_DISTANCE = 0.65
EPOCH = 5
STEP = 0.05
ACCURACY = 0


def get_class(files):
    class_counter = dict()

    for file in files:
        counter = 0
        file_name = file.split('/')[len(file.split('/')) - 1].split('-')[0]
        for s_file in files:
            if file_name in s_file:
                counter += 1
        class_counter[file_name] = counter

    return list(class_counter.keys())[list(class_counter.values()).index(max(class_counter.values()))]


def get_data(files):
    mas = []
    for i in range(0, len(files), BATCH):
        mas.append(files[i])
    return mas


def getfilelist(dirpath):
    global vsego
    mas = []
    for root, dirs, files in os.walk(dirpath):
        for name in files:
            fullname = os.path.join(root, name)
            if '.jpg' in fullname:
                mas.append(fullname)
    vsego = str(len(mas))
    print(f'Count of files: {vsego}')
    return mas


def get_face_descriptors(filename):
    facemas = []
    img = io.imread(filename)
    detected_faces = detector(img, 1)

    dir(cv2.ADAPTIVE_THRESH_MEAN_C)
    for k, d in enumerate(detected_faces):
        shape = sp(img, d)
        try:
            face_descriptor = face_rec.compute_face_descriptor(img, shape)
            if face_descriptor:
                facemas.append(face_descriptor)
        except Exception as ex:
            print(str(ex))
    return facemas


def validate_distance(test_file, find_neighbors):
    global VALID_DISTANCE

    test_file_class = test_file.split('/')[len(test_file.split('/'))-1].split(".")[0]
    counter = 0
    for neighbor in find_neighbors:
        if test_file_class in neighbor:
            counter += 1
    accuracy = counter/len(find_neighbors) * 100

    print()
    print("--------------------------------------")
    print("Accuracy:", accuracy)
    print("Distance:", VALID_DISTANCE)
    print("--------------------------------------")
    print()

    if accuracy < 90:
        VALID_DISTANCE -= STEP
    else:
        print()
        print("FIND VALID DISTANCE:", VALID_DISTANCE)
        print()
        return accuracy


def find_class(test_file, f1, epoch):
    global ACCURACY
    print(f"Epoch: {epoch}")
    files = get_data(getfilelist(train_dirpath))
    flag = 0
    find_neighbors = []
    for f in files:
        flag = flag + 1
        print('Count of file: ' + f + ' - ' + str(flag) + ' from ' + str(len(files)))
        if os.path.exists(f):
            findfaces = get_face_descriptors(f)
            print('Count of faces: ' + str(len(findfaces)))
            for f2 in findfaces:
                if f2:
                    euc_distance = distance.euclidean(f1, f2)
                    print("Distance:", euc_distance)
                    if euc_distance < VALID_DISTANCE:
                        print('Faces: ' + f)
                        find_neighbors.append(f)
    print('find_photos:', find_neighbors)

    accuracy = validate_distance(test_file, find_neighbors)

    if accuracy:
        ACCURACY = accuracy
        return find_neighbors


def train_model():
    print("Prepare for test: \n")
    test_files = getfilelist(test_dirpath)
    for test_file in test_files:
        print("---------------------------------------")
        print("File: ", test_file)
        print("---------------------------------------")

        for epoch in range(EPOCH):
            f1 = get_face_descriptors(test_file)[0]
            find_neighbors = find_class(test_file, f1, epoch)

            if find_neighbors:
                print(f"Class for {test_file}: ", get_class(find_neighbors))

                print("--------------------------------------------------------------------------")
                break

''' 
    Для полноценной установки,пожалуйста, прочитайте README.md файл и выполните инструкции 
'''
if __name__ == "__main__":
    train_model()
