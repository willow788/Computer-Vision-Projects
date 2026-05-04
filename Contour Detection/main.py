import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image


def load_image(path):
    image = cv2.imread(path)

    if image is not None:
        return image

    pil_image = Image.open(path).convert('RGB')
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

#upload the image
img = load_image('istockphoto-2082112210-612x612.jpg')

#creating a copy of the image
img_copy = img.copy()

# DEBUG TEST: synthetic markers to validate watershed behavior and mapping
DEBUG_TEST = True
if DEBUG_TEST:
    print('Running synthetic watershed test')
    test_marker = np.zeros(img.shape[:2], dtype=np.int32)
    # two seeded markers
    cv2.circle(test_marker, (50, 50), 40, 1, -1)
    cv2.circle(test_marker, (200, 200), 60, 2, -1)

    before = np.unique(test_marker)
    print('marker labels before watershed:', before)

    test_copy = test_marker.copy()
    cv2.watershed(img, test_copy)

    after = np.unique(test_copy)
    print('marker labels after watershed:', after)

    # build segments using same mapping logic
    test_segments = np.zeros(img.shape, dtype=np.uint8)
    for label in after:
        if label == -1:
            test_segments[test_copy == label] = (255, 255, 255)
            continue
        if label == 0:
            continue
        color = (int(cm.tab10(int(label) % 10)[0]*255),
                 int(cm.tab10(int(label) % 10)[1]*255),
                 int(cm.tab10(int(label) % 10)[2]*255))
        test_segments[test_copy == label] = color

    cv2.imwrite('segments_test.png', test_segments)
    print('Wrote segments_test.png')

#creating two black img of the same size as the original image
marker_img = np.zeros(img.shape[:2], dtype=np.int32)

segments = np.zeros(img.shape, dtype=np.uint8)  

#function to create tuple of random colors
def create_rgb(i):

    x = np.array(cm.tab10(i)[:3]) * 255
    return tuple(x.astype(int))

#storing the contours in a list
colors = []

#one color for a single digit
for i in range(10):
    colors.append(create_rgb(i))

#global vars
no_markers = 10

#current markers
current_markers = 1

#flag
marks_updated = False

#callback function for mouse events

def mouse_callback(event, x, y, flags, params):
    global marks_updated

    if event == cv2.EVENT_LBUTTONDOWN:

        #tracking for the markers
        cv2.circle(img_copy, (x, y), 10, (0, 255, 0), -1)

        #display on the user img
        cv2.circle(marker_img, (x, y), 10, current_markers, -1)

        marks_updated = True

#creating a window and bind the function to window
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', mouse_callback)

while True:

    #show the two windows
    cv2.imshow('Image', img_copy)
    cv2.imshow('Segments', segments)

    #close everything when the user presses the 'q' key
    k = cv2.waitKey(1)
    if k == 27:
        break

    elif k > 0 and chr(k).isdigit():
        current_markers = int(chr(k))

    if marks_updated:
        marker_img_copy = marker_img.copy()

        #apply watershed
        cv2.watershed(img, marker_img_copy)

        segments = np.zeros(img.shape, dtype=np.uint8)

        # map each label produced by watershed to a color
        labels = np.unique(marker_img_copy)
        for label in labels:
            if label == -1:
                # boundary (optional: mark boundaries white)
                segments[marker_img_copy == label] = (255, 255, 255)
                continue
            if label == 0:
                # background, leave black
                continue

            # choose a color for this label (wrap if necessary)
            color = colors[int(label) % len(colors)]
            segments[marker_img_copy == label] = color

        marks_updated = False

#destroy all windows
cv2.destroyAllWindows() 

