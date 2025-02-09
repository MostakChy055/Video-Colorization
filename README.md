# Adaptive Multi-Scale Binning
## Binning
Binning is the process of dividing continuous values into discrete intervals (bins). In deep learning, histogram-based binning is often used to extract distribution-based features from data.
* Instead of using raw pixel values, we categorize them into different bins.
* The frequency of values in each bin provides insights into the data distribution.
In the context of your AdaptiveMultiScaleBinning module, each channel's pixel values are grouped into multiple bin sizes, capturing different levels of granularity.

```python
class AdaptiveMultiScaleBinning(nn.Module):
    def __init__(self, num_bins=[32, 64, 128]):
        super().__init__()
        self.num_bins = num_bins # Number of bins

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        x_flatten = x.view(batch_size, channels, -1).to(device)
        histograms = [torch.histc(x_flatten, bins=b, min=0, max=1).to(device) for b in self.num_bins]
        histograms = [h / h.sum(dim=-1, keepdim=True) for h in histograms] # Each histogram is normalized by dividing by its sum, ensuring it represents a probability distribution.
        return torch.cat(histograms, dim=-1)  # # Concatenating multi-scale histograms
```

At first, it might seem logical that simply increasing the number of bins would give finer details. However, multi-scale binning offers distinct advantages:
# Why Multi-Scale Binning Instead of Just Using More Bins?
**Handling Varying Granularity:**
* Small bin sizes (e.g., 128 bins) capture fine-grained details.
* Large bin sizes (e.g., 32 bins) capture broader distributions, helping with generalization.
If we only use a high number of bins, we might overfit to minor intensity variations in pixel values, causing instability across frames.
**Robustness to Illumination & Noise:**
* Fine bins are sensitive to small variations in pixel values, making them vulnerable to noise.
* Coarse bins smooth out small fluctuations, preserving meaningful patterns across frames.
**Better Generalization in Temporal Learning:**
* Video frames change dynamically;
* multi-scale histograms prevent dependency on an exact bin count, helping the model adapt to different lighting conditions and motion speeds.

# Why histogram?
**Why Does Flickering Occur in Video Colorization?**

In video colorization, we process frames individually but must ensure color consistency across time. If the model predicts different colors for the same object across consecutive frames, we get temporal
flickering—a common artifact where colors jump or change suddenly between frames.

**How Does a CNN-Based Pixel-Wise Model Cause Flickering?**
Traditional CNN-based colorization models predict colors based on pixel-wise features, meaning:
* Color predictions depend on local spatial features → If an object moves slightly between frames, the extracted features might change.
* Lack of Temporal Awareness → CNNs treat each frame independently and don’t consider how colors should remain stable across time.
* Unstable Color Predictions → Even minor changes in grayscale values can lead to different colorizations, especially when lighting conditions shift.

**Histograms do not depend on the exact position of pixels but instead capture the distribution of colors in the frame.**

- Aggregating Colors Regardless of Position → Since histograms summarize how often each color appears, they don’t get affected if an object moves slightly.-
- Reducing the Effect of Small Motion Variations → Unlike CNN features, histograms remain stable even if objects shift slightly between frames.
- Ensuring Temporal Stability in Colorization → By using histogram-based color guidance, we ensure that the same objects get the same color treatment across different frames.
