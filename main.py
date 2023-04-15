# import libraries

# 1. DataPreparation
# 2. DataPreprocessing
# 3. ModelBuilding
# 4. ModelEvaluation

import glob
from DataPreparation import Data
from DataPreprocessing import CassavaDataset
import paddle
import predictions
from visualdl import LogWriter
import numpy as np
import os
# import paddle.fluid as fluid
# paddle.static import InputSpec
from cv2.gapi.core import fluid
from paddle.vision.models import mobilenet_v1

data = Data()
data.show_img()

# Test defined dataset
train_dataset = CassavaDataset()
val_dataset = CassavaDataset(is_train=False)
num = 0
print('=============train dataset=============')
for data, label in train_dataset:
    num += 1
    print(data, label)
    break
print(num)
num = 0

print('=============evaluation dataset=============')
for data, label in val_dataset:
    num += 1
    print(data, label)
    break
print(num)

# Initialize the model
# To use the built-in model, select the network mobilenet_v1
Cassava = mobilenet_v1()
# create LeNet log file
writer = LogWriter("./work/log")

# look mobilenet_v1 model structure
# model structure
model = paddle.Model(Cassava)
print('Paddle frame built-in model：', paddle.vision.models.__all__)
# paddle.summary easily print network infrastructure and parameter information
# paddle.summary((-1, 3, 600, 800))
model.summary((-1, 3, 448, 448))

# define data reader
paddle.set_device('gpu')
print(f"paddle devide {paddle.get_device()}")
train_loader = paddle.io.DataLoader(train_dataset, places=paddle.CUDAPlace(0), batch_size=32, shuffle=True)
# begin train
Cassava.train()
# set epoch number
epochs = 5
step = 0
# define optimizer
opt = paddle.optimizer.Adam(learning_rate=0.001, parameters=Cassava.parameters())
# define loss function
loss_fn = paddle.nn.CrossEntropyLoss()

epoch: int
for epoch in range(epochs):
    for batch_id, data in enumerate(train_loader()):
        # img label
        x_data = data[0]
        y_data = data[1]
        # predict result
        predict = Cassava(x_data)
        # import loss function
        loss = loss_fn(predict, y_data)
        # loss_sum += loss.numpy().sum()
        # acc
        acc = paddle.metric.accuracy(predict, y_data)
        # acc_sum += acc.numpy().sum()
        # backward
        loss.backward()
        # print out
        if batch_id % 20 == 0:
            print("epoch:{}, batch:{}, loss:{}, acc:{}".format(epoch, batch_id, loss.numpy(), acc.numpy()))

        # VDL log
        step += 1
        if step % 20 == 0:
            # add acc
            writer.add_scalar(tag="train/acc", step=step, value=float(acc.numpy()))
            # add loss
            writer.add_scalar(tag="train/loss", step=step, value=float(loss.numpy()))

            # first image
            img = np.reshape(np.array(data[0][0].numpy()), [600, 800, 3])
            writer.add_image(tag="train/input", step=step, img=img)

            # plot P-R curve
            for i in range(5):
                labels = np.array(data, dtype='int32')[0, 1]
                prediction = np.where(np.squeez(np.array(predict[0, 1])) == 1)
                writer.add_pr_curve(tag='train/class_{}_pr_curve'.format(i),
                                    labels=labels,
                                    predictions=prediction,
                                    step=step,
                                    num_thresholds=20)

        # update step
        opt.step()
        # clear step
        opt.clear_grad()

    # Save model parameters and optimizer parameters
    paddle.save(Cassava.state_dict(), os.path.join("MODEL/save_model", str(epoch), str(epoch) + ".pdparams"))
    paddle.save(opt.state_dict(), os.path.join("MODEL/save_model", str(epoch), str(epoch) + ".pdopt"))

    exe = fluid.Executor(fluid.GPUPlace())
    exe.run(fluid.default_startup_program())

    # save model
    fluid.io.save_inference_model(dirname=os.path.join("MODEL/save_model", str(epoch)), feeded_var_names=['img'],
                                  target_vars=[predictions], executor=exe)

# Model loading
Cassava_state_dict = paddle.load("MODEL/save_model/4/4.pdparams")
opt_state_dict = paddle.load("MODEL/AMLS_LeafDiseaseClassification/save_model/4/4.pdopt")
Cassava.set_state_dict(Cassava_state_dict)
opt.set_state_dict(opt_state_dict)

# load validation dataset
val_loader = paddle.io.DataLoader(val_dataset, places=paddle.CUDAPlace(0), batch_size=32, shuffle=True)
loss_fn = paddle.nn.CrossEntropyLoss()

Cassava.eval()

for batch_id, data in enumerate(val_loader()):
    x_data = data[0]  # data
    y_data = data[1]  # label
    predicts = Cassava(x_data)  # predict result

    # Computing Loss and Accuracy
    loss = loss_fn(predicts, y_data)
    acc = paddle.metric.accuracy(predicts, y_data)

    # print information
    if (batch_id + 1) % 20 == 0:
        print("batch_id: {}, loss is: {}, acc is: {}".format(batch_id, loss.numpy(), acc.numpy()))
