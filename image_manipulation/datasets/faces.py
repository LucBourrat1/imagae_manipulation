from torch_snippets import *
import os

from image_manipulation.datasets.random_warp import get_training_data


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def crop_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        for x, y, w, h in faces:
            img2 = img[y : (y + h), x : (x + w), :]
        img2 = cv2.resize(img2, (256, 256))
        return img2, True
    else:
        return img, False


def crop_images(folder):
    images = Glob(folder + "/*.jpg")
    for i in range(len(images)):
        img = read(images[i], 1)
        img2, face_detected = crop_face(img)
        if face_detected == False:
            continue
        else:
            cv2.imwrite(
                "cropped_faces_" + folder + "/" + str(i) + ".jpg",
                cv2.cvtColor(img2, cv2.COLOR_RGB2BGR),
            )


def prepare_data():
    if not os.path.exists("personA") or not os.path.exists("personB"):
        os.system("./faces_download.sh")
    new_paths = ["cropped_faces_personA", "cropped_faces_personB"]
    for p in new_paths:
        if not os.path.exists(p):
            os.mkdir(p)
    crop_images("personA")
    crop_images("personB")


class FacesDataset(Dataset):
    def __init__(self, items_A, items_B):
        self.items_A = np.concatenate([read(f, 1)[None] for f in items_A]) / 255.0
        self.items_B = np.concatenate([read(f, 1)[None] for f in items_B]) / 255.0
        self.items_A += self.items_B.mean(axis=(0, 1, 2)) - self.items_A.mean(
            axis=(0, 1, 2)
        )

    def __len__(self):
        return min(len(self.items_A), len(self.items_B))

    def __getitem__(self, ix):
        a, b = choose(self.items_A), choose(self.items_B)
        return a, b

    def collate_fn(self, batch):
        imsA, imsB = list(zip(*batch))
        imsA, targetA = get_training_data(imsA, len(imsA))
        imsB, targetB = get_training_data(imsB, len(imsB))
        imsA, imsB, targetA, targetB = [
            torch.Tensor(i).permute(0, 3, 1, 2).to(device)
            for i in [imsA, imsB, targetA, targetB]
        ]
        return imsA, imsB, targetA, targetB


if __name__ == "__main__":
    prepare_data()
