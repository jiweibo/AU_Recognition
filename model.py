import os
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch

__all__ = ['AlexNet', 'alexnet']

# You need to download the model
model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

model_name = r'alexnet-owt-4df8aa71.pth'


class AlexNet(nn.Module):

    def __init__(self, num_classes=256 * 6 * 6):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # self.classifier = nn.Sequential(
        #     nn.Dropout(),
        #     nn.Linear(256 * 6 * 6, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(4096, num_classes),
        # )

        # <editor-fold desc="10 basic classifiers">
        self.classifier1 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(num_classes, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.classifier2 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(num_classes, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.classifier3 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(num_classes, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.classifier4 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(num_classes, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.classifier5 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(num_classes, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.classifier6 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(num_classes, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.classifier7 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(num_classes, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.classifier8 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(num_classes, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.classifier9 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(num_classes, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.classifier10 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(num_classes, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        # </editor-fold>

        # self.classifier1 = nn.Sequential(
        #     nn.Dropout(),
        #     nn.Linear(256 * 6 * 6, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(256, 1),
        #     nn.Sigmoid()
        # )
        # self.classifier2 = nn.Sequential(
        #     nn.Dropout(),
        #     nn.Linear(256 * 6 * 6, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(256, 1),
        #     nn.Sigmoid()
        # )
        # self.classifier3 = nn.Sequential(
        #     nn.Dropout(),
        #     nn.Linear(256 * 6 * 6, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(256, 1),
        #     nn.Sigmoid()
        # )
        # self.classifier4 = nn.Sequential(
        #     nn.Dropout(),
        #     nn.Linear(256 * 6 * 6, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(256, 1),
        #     nn.Sigmoid()
        # )
        # self.classifier5 = nn.Sequential(
        #     nn.Dropout(),
        #     nn.Linear(256 * 6 * 6, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(256, 1),
        #     nn.Sigmoid()
        # )
        # self.classifier6 = nn.Sequential(
        #     nn.Dropout(),
        #     nn.Linear(256 * 6 * 6, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(256, 1),
        #     nn.Sigmoid()
        # )
        # self.classifier7 = nn.Sequential(
        #     nn.Dropout(),
        #     nn.Linear(256 * 6 * 6, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(256, 1),
        #     nn.Sigmoid()
        # )
        # self.classifier8 = nn.Sequential(
        #     nn.Dropout(),
        #     nn.Linear(256 * 6 * 6, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(256, 1),
        #     nn.Sigmoid()
        # )
        # self.classifier9 = nn.Sequential(
        #     nn.Dropout(),
        #     nn.Linear(256 * 6 * 6, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(256, 1),
        #     nn.Sigmoid()
        # )
        # self.classifier10 = nn.Sequential(
        #     nn.Dropout(),
        #     nn.Linear(256 * 6 * 6, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(256, 1),
        #     nn.Sigmoid()
        # )


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        # x = self.classifier(x)
        x1 = self.classifier1(x)
        x2 = self.classifier2(x)
        x3 = self.classifier3(x)
        x4 = self.classifier4(x)
        x5 = self.classifier5(x)
        x6 = self.classifier6(x)
        x7 = self.classifier7(x)
        x8 = self.classifier8(x)
        x9 = self.classifier9(x)
        x10 = self.classifier10(x)
        return torch.cat((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10), 1)


def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
        torch_home = os.path.expanduser(os.getenv('TORCH_HOME', '~/.torch'))
        model_dir = os.getenv('TORCH_MODEL_ZOO', os.path.join(torch_home, 'models'))
        model_path = os.path.join(model_dir, model_name)
        model_param = torch.load(model_path)
        for name, value in model_param.items():
            if name.startswith('features') and name in model.state_dict():
                model.state_dict()[name].copy_(value.data)
            # if name.startswith('classifier') and name in model.state_dict():
            #     model.state_dict()[name].copy_(value.data)

    return model
