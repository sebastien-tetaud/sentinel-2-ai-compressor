import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.nn.functional as F

def define_model(
    name,
    encoder_name,
    out_channels=3,
    in_channel=3,
    encoder_weights=None,
    activation=None,

):
    # Get the model class dynamically based on name
    try:
        # Get the model class from segmentation_models_pytorch
        ModelClass = getattr(smp, name)


        # Create the model
        model = ModelClass(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channel,
            classes=out_channels,
            activation=None,

        )

        # Add ReLU activation after the model
        if activation == "relu":
            model = nn.Sequential(
                model,
                nn.ReLU()
            )
        if activation == "sigmoid":
            model = nn.Sequential(
                model,
                nn.Sigmoid()
            )



        return model


    except AttributeError:
        # If the model name is not found in the library
        raise ValueError(f"Model '{name}' not found in segmentation_models_pytorch. Available models: {dir(smp)}")


class AutoEncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, depth=5, bottleneck_factor=0.5):
        """
        Configurable AutoEncoder with variable depth

        Args:
            in_channels: Input channels (e.g., 3 for RGB)
            base_channels: Starting number of channels
            depth: Number of downsampling/upsampling layers
            bottleneck_factor: Factor to reduce channels at bottleneck (0.5 = half channels)
        """
        super(AutoEncoder, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels  # Output channels same as input

        # Calculate channel progression
        channels = [in_channels]
        for i in range(depth):
            channels.append(min(base_channels * (2 ** i), 1024))

        # Bottleneck channels
        bottleneck_channels = int(channels[-1] * bottleneck_factor)

        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        for i in range(depth):
            self.encoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(channels[i], channels[i+1], kernel_size=3, padding=1),
                    nn.BatchNorm2d(channels[i+1]),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels[i+1], channels[i+1], kernel_size=3, padding=1),
                    nn.BatchNorm2d(channels[i+1]),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2)  # Downsample by 2
                )
            )

        # Bottleneck layer (optional compression)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(channels[-1], bottleneck_channels, kernel_size=1),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, channels[-1], kernel_size=1),
            nn.BatchNorm2d(channels[-1]),
            nn.ReLU(inplace=True)
        )

        # Decoder layers (reverse of encoder)
        self.decoder_layers = nn.ModuleList()
        for i in range(depth-1, -1, -1):
            self.decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(channels[i+1], channels[i+1], kernel_size=2, stride=2),  # Upsample
                    nn.Conv2d(channels[i+1], channels[i], kernel_size=3, padding=1),
                    nn.BatchNorm2d(channels[i]),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels[i], channels[i], kernel_size=3, padding=1),
                    nn.BatchNorm2d(channels[i]),
                    nn.ReLU(inplace=True)
                )
            )

        # Final output layer
        self.last_layer = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.output_layer = nn.Sigmoid()
        self.channel_progression = channels
        self.bottleneck_channels = bottleneck_channels

    def forward(self, x):
        # Store input for size reference
        input_size = x.shape[-2:]

        # Encoder
        features = [x]
        for layer in self.encoder_layers:
            x = layer(features[-1])
            features.append(x)

        # Bottleneck
        self.bottleneck_tensor = self.bottleneck(x)

        # Decoder
        x =  self.bottleneck_tensor
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x)
            # Ensure proper size matching
            if x.shape[-2:] != input_size and i == len(self.decoder_layers)-1:
                x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)

        # Final output
        x = self.last_layer(x)
        x = self.output_layer(x)  # Apply ReLU activation if needed

        return x

    def get_bottleneck_tensor(self):
        """Get the bottleneck tensor for analysis"""
        return self.bottleneck_tensor

    def count_parameters(self):
        """Count total and trainable parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params

    def get_compression_info(self, input_tensor, bottleneck_tensor):
        """Calculate compression statistics"""
        input_size = input_tensor.numel()
        bottleneck_size = bottleneck_tensor.numel()
        compression_ratio = input_size / bottleneck_size
        reduction_percent = (1 - bottleneck_size / input_size) * 100

        return {
            'input_shape': input_tensor.shape,
            'bottleneck_shape': bottleneck_tensor.shape,
            'compression_ratio': compression_ratio,
            'reduction_percent': reduction_percent,
            'input_elements': input_size,
            'bottleneck_elements': bottleneck_size
        }