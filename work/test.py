import paddle
from paddle.vision.models import mobilenet_v1

# build model
model = mobilenet_v1()

# build model and load imagenet pretrained weight
# model = mobilenet_v1(pretrained=True)

# build mobilenet v1 with scale=0.5
model_scale = mobilenet_v1(scale=0.5)

x = paddle.rand([1, 3, 224, 224])
out = model(x)

print(out.shape)
# [1, 1000]
