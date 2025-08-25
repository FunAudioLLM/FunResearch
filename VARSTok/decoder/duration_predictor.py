import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm
from collections import OrderedDict

import pdb


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded

    
class LengthRegulator(nn.Module):
    """Length Regulator"""

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(x.device)

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len=None):
        x = x.transpose(1, 2)
        # pdb.set_trace()
        output, mel_len = self.LR(x, duration, max_len)
        # pdb.set_trace()
        output = output.transpose(1, 2)
        # pdb.set_trace()
        return output, mel_len


class DurationPredictor(nn.Module):
    """Duration, Pitch and Energy Predictor"""

    def __init__(self, input_dim=512, filter_size=512, kernel=3, dropout=0.5, S=4):
        super(DurationPredictor, self).__init__()

        self.input_size = input_dim
        self.filter_size = filter_size
        self.kernel = kernel
        self.conv_output_size = filter_size
        self.dropout = dropout
        self.num_classes = S 

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=1,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.conv_output_size, self.num_classes)

    def forward(self, encoder_output, mask=None):
        """
        Args:
            x: [B, D, T']  — input token features (e.g., quantized or after projection)
            mask: [B, T']  — optional, True for padding
        Returns:
            durations: [B, T'] — predicted durations (float)
        """
        encoder_output = encoder_output.transpose(1, 2) # -> [B, T, D]
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)


        # Apply softmax to get class probabilities
        class_probs = F.softmax(out, dim=-1)  # [B, T, S]

        if mask is not None:
            class_probs = class_probs.masked_fill(~mask.unsqueeze(-1), 0.0)

        return class_probs

    def inference(self, encoder_output, mask=None):
        """
        Inference function to get predicted durations from the model.

        Args:
            encoder_output: Tensor of shape [B, D, T'] — input token features.
            mask: Optional, Tensor of shape [B, T'] — True for padding positions.

        Returns:
            predicted_durations: Tensor of shape [B, T'] — predicted durations.
        """
        self.eval()  # Set the model to evaluation mode

        with torch.no_grad():  # Disable gradient calculation for inference
            # Perform the forward pass to get class probabilities
            class_probs = self(encoder_output, mask)  # Output: [B, T', S]
            
            # Get the predicted class indices (the class with the highest probability)
            predicted_classes = torch.argmax(class_probs, dim=-1)  # [B, T']
            
            # Optional: Map the predicted classes to the actual duration values (if you have a mapping)
            # Here, we'll assume a simple direct mapping where class 0 maps to 1, class 1 maps to 2, and so on.
            # You should adjust this mapping according to your specific task.
            class_to_duration = torch.tensor([1, 2, 3, 4]).to(class_probs.device)  # Example: class 0 -> 1, class 1 -> 2, etc.
            predicted_durations = class_to_duration[predicted_classes]  # [B, T']
            
            return class_probs, predicted_durations


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        w_init="linear",
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x


class DurationScheduler:
    def __init__(self, start_step: int, end_step: int, max_step: int, steepness: float = 6.0):
        self.start = start_step
        self.end = end_step
        self.max_step = max_step
        self.steepness = steepness

    def get_p_pred(self, step: int) -> float:
        if step < self.start:
            return 0.0
        elif step > self.end:
            return 1.0
        else:
            relative = (step - self.start) / (self.end - self.start)
            x = (relative - 0.5) * self.steepness
            return float(torch.sigmoid(torch.tensor(x)))

    def sample_duration(self, step: int, gt_durations: torch.Tensor, pred_durations: torch.Tensor) -> torch.Tensor:
        p_pred = self.get_p_pred(step)
        use_gt = torch.rand(1).item() > p_pred
        return (gt_durations, 1.0, p_pred) if use_gt else (pred_durations, 0.0, p_pred)
