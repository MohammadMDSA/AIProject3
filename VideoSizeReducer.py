import cv2 as cv

cap = cv.VideoCapture('test.mp4')
# cap = cv.VideoCapture('test1.mp4')
lenm = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

_, frame = cap.read()

vid_size = frame.shape

fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter("test_red.avi", fourcc, 30.0, (vid_size[1], vid_size[0]))

frame_count = 0
while(1):
    frame_count = frame_count + 1

    ret, frame = cap.read()
    if not ret:
        break;

    red = cv.resize(frame.copy(), (10, 10), interpolation=cv.INTER_AREA)
    out.write(red)
    print(frame_count, '/', lenm)

out.release()
cap.release()