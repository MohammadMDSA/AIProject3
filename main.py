import numpy as np
import cv2 as cv
import random
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input', type=str, default='', required=False)
parser.add_argument('-c', '--camera', default=False, action='store_true', required=False)
args = parser.parse_args()

print(args)
print(args.camera)

mode = ''
if args.camera:
    mode = 'cam'
else:
    mode = 'file'

FORGROUND_THRESHOLD = 100

snow_img = cv.imread("snow.png")
# snow_img.resize(100, 100)
snow_img = cv.resize(snow_img, (10, 10), interpolation=cv.INTER_AREA)
# print(snow_img.shape)

print('mode', mode)

cap = cv.VideoCapture(0) if mode is 'cam' else cv.VideoCapture(args.input)
# cap = cv.VideoCapture('test1.mp4')
lenm = -1 if mode == 'cam' else int(cap.get(cv.CAP_PROP_FRAME_COUNT))
# lenm = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
print(lenm)

fgbg = cv.createBackgroundSubtractorKNN(history=30)

_, frame = cap.read()

vid_size = frame.shape

fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter("out.avi", fourcc, 30.0, (vid_size[1], vid_size[0]))

print(vid_size)

snows = []

windowName = "Live Camera Input"  # window name
windowNameFG = "Foreground Objects"  # window name

if mode == 'cam':
    cv.namedWindow(windowName, cv.WINDOW_NORMAL)
    cv.namedWindow(windowNameFG, cv.WINDOW_NORMAL)


def paste_image(background, forground, loc):
    for i in range(forground.shape[1]):
        for j in range(forground.shape[0]):
            if not (forground[i][j][2] == 0 and forground[i][j][1] == 0 and forground[i][j][0] == 0):
                # print(i, j, loc)
                try:
                    background[j + loc[0] - 1][i + loc[1] - 1] = forground[j, i]
                except:
                    pass

    return background

def put_point(img, pos, color):
    try:
        img[pos[1], pos[0]] = color
    except:
        pass

def draw_contour(img, contours):
    for contour in contours:
        for [point] in contour:
            # print(point)
            put_point(img, point, 255)


class Snow:
    speed = 15
    valid = True

    def __init__(self, position):
        self.position = (0, position)

    def update(self, mask):
        if not self._check_below(mask):
            return
        # print(self.position, self.position[1] + self.speed)
        self.position = (self.position[0] + self.speed, self.position[1])
        if self.position[0] + snow_img.shape[0] > vid_size[0]:
            self.valid = False

    def _check_below(self, mask):
        res = True
        for i in range(self.speed):
            for j in range(snow_img.shape[1]):
                try:

                    if FORGROUND_THRESHOLD < mask[self.position[0] + snow_img.shape[0] + i][self.position[1] + j]:
                        return False

                except:
                    pass
        return True


def generate_snow():
    for i in range(random.choice(range(0, 5))):
        position = random.choice(range(vid_size[1] - snow_img.shape[1]))
        snows.append(Snow(position))


def remove_out_of_bound():
    for snow in snows:
        if not snow.valid:
            snows.remove(snow)


def reduce_light(brg):
    hsv = cv.cvtColor(brg, cv.COLOR_BGR2HSV)
    hsv[:, :, 2] = 255
    res = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return res


def compute_bg():
    bg = cv.imread('bg.png')
    if bg is not None:
        return bg

    frame_num = 0
    while 1:
        frame_num = frame_num + 1
        ret, frame = cap.read()
        fgmask = fgbg.apply(frame)
        print('Computing background:', frame_num + 1, '/', lenm)
        if frame_num >= lenm - 1 and lenm > -1:
            break
    bg = fgbg.getBackgroundImage()
    bg = cv.GaussianBlur(bg, (7, 7), 0)
    cv.imwrite('bg.png', bg)
    return bg


frame_num = 0

bg = compute_bg()

# cap = cv.VideoCapture(0) if mode is 'cam' else cv.VideoCapture(args.input)

while (1):
    frame_num = frame_num + 1
    generate_snow()
    remove_out_of_bound()

    ret, frame = cap.read()

    low_ligght = reduce_light(frame)

    fgmask = fgbg.apply(low_ligght)

    _, thresh = cv.threshold(fgmask, 127, 255, 0)
    im2, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    blank = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)

    draw_contour(blank, contours)
    blank = cv.GaussianBlur(blank, (3, 3), 0)

    # im = fgmask
    # im = low_ligght
    im = frame

    for snow in snows:
        im = paste_image(im, snow_img, snow.position)

    if mode == 'cam':
        # gg = cv.threshold(im - bg, 190, 255, cv.THRESH_TOZERO)[1]
        #
        # print('here1')
        cv.imshow(windowName, im)
        cv.imshow(windowNameFG, blank)
        k = cv.waitKey(1) & 0xff
        if k == 27:
            break
    else:
        out.write(im)
        print(frame_num + 1, '/', lenm)
    for snow in snows:
        snow.update(blank)
    if frame_num >= lenm - 1 and lenm > -1:
        break
# cv.imshow(windowName, fgbg.getBackgroundImage())
# cv.waitKey(10000)

out.release()
cap.release()
cv.destroyAllWindows()
