import numpy as np
from nnplaygamke.network import ksnet4

WIDTH = 160
HEIGHT = 120
LR = 1e-3
EPOCHS = 20
MODEL_NAME = 'self-car-fast-{}-{}-{}-epochs-4.9k-data.model'.format(LR, 'ksnet4',EPOCHS)

model = ksnet4(WIDTH, HEIGHT,LR)

hm_data = 2
for i in range(EPOCHS):
    for i in range(1,hm_data+1):
        train_data = np.load('training_data_v1_fps.npy',allow_pickle=True)

        train = train_data[:-100]
        test = train_data[-100:]

        X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
        Y = [i[1] for i in train]

        test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
        test_y = [i[1] for i in test]


        model.fit({'input': X}, {'targets': Y}, n_epoch=1, validation_set=({'input': test_x}, {'targets': test_y}),
            snapshot_step=500, show_metric=True, run_id=MODEL_NAME )

        model.save(MODEL_NAME)

##tensorboard --logdir=log4 --host localhost
