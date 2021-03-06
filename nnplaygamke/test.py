from nnplaygamke.grabscreen import grab_screen
import cv2
import time
from directkeys import PressKey, ReleaseKey, W, A, S, D
from nnplaygamke.network import ksnet4
from nnplaygamke.getkeys import key_check



WIDTH = 160
HEIGHT = 120
LR = 1e-3
EPOCHS = 20
MODEL_NAME = 'self-car-fast-{}-{}-{}-epochs-4.9k-data.model'.format(LR, 'ksnet4',EPOCHS)

t_time = 0.025


def straight():
    ##    if random.randrange(4) == 2:
    ##        ReleaseKey(W)
    ##    else:
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)


def left():
    PressKey(W)
    PressKey(A)
    # ReleaseKey(W)
    ReleaseKey(D)
    # ReleaseKey(A)
    time.sleep(t_time)
    ReleaseKey(A)


def right():
    PressKey(W)
    PressKey(D)
    ReleaseKey(A)
    # ReleaseKey(W)
    # ReleaseKey(D)
    time.sleep(t_time)
    ReleaseKey(D)

def brake():
    PressKey(S)
    ReleaseKey(W)
    ReleaseKey(D)
    ReleaseKey(A)

model = ksnet4(WIDTH, HEIGHT, LR)
model.load(MODEL_NAME)


def main():
    last_time = time.time()
    for i in list(range(4))[::-1]:
        print(i + 1)
        time.sleep(1)

    paused = False
    while (True):

        if not paused:
            # 800x600 windowed mode
            # screen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))
            screen = grab_screen(region=(0, 40, 800, 640))
            print('loop took {} seconds'.format(time.time() - last_time))
            last_time = time.time()
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (160, 120))

            prediction = model.predict([screen.reshape(160, 120, 1)])[0]
            print(prediction)

            turn_thresh = .60
            fwd_thresh = 0.70
            #brake_thresh = 0.60

            if prediction[1] > fwd_thresh:
                straight()
            elif prediction[0] > turn_thresh:
                left()
            elif prediction[2] > turn_thresh:
                right()
            #elif prediction[3] > brake_thresh & prediction[3] > fwd_thresh:
                #brake()
            else:
                straight()

        keys = key_check()

        # p pauses game and can get annoying.
        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                time.sleep(1)


main()


