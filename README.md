# CNN Predictor

Takes in individual calcium image (./Calcium/input.txt) and outputs prediction label.

Performance is around 0.02-0.03 ms per image.

Implementation based on following PyTorch model:

---

class Simplenet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 6, 3, 1)
        self.bn = nn.BatchNorm2d(6)
        self.relu = nn.ReLU(inplace=False)
        self.fc1 = nn.Linear(150, 23)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

---

Baseline model (located in ./FP_model) attains 36.18% Hit 1 and 78.42% Hit 3 accuracy.

Compressed model has 8-bit quantization & 60% sparsity (located in ./compressed_model) attains 39.02% Hit 1 and 73.42% Hit 3 accuracy.
