# CNN Predictor

Takes in individual calcium image (./Calcium/input.txt) and outputs prediction label.

Performance is around 0.02-0.03 ms per image.

Implementation based on following PyTorch model:

---

class Simplenet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 6, 3, 1)
        self.relu = nn.ReLU(inplace=False)
        self.fc1 = nn.Linear(150, 23)

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

---

Baseline model (located in ./FP_model) attains 37.20% Hit 1 and 76.09% Hit 3 accuracy.

Compressed model with 8-bit quantization & 60% sparsity (located in ./compressed_model) is outdated and needs updating.
