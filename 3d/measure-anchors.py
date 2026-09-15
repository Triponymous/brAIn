"""Read-only reference metrics; no browser automation or image transformation."""
import json
import sys
from collections import Counter
import numpy as np
from PIL import Image

result = {}
for path in sys.argv[1:]:
    pixels = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32)
    lum = pixels @ np.array([.2126, .7152, .0722])
    p5, p50, p95 = np.percentile(lum, [5, 50, 95], method='nearest')
    colors = (pixels.astype(np.uint16) >> 3)
    bins = (colors[:, :, 0] << 10) | (colors[:, :, 1] << 5) | colors[:, :, 2]
    counts = np.array(list(Counter(bins.flatten()).values()))
    probabilities = counts / bins.size
    gx = -lum[:-2, :-2] - 2*lum[1:-1, :-2] - lum[2:, :-2] + lum[:-2, 2:] + 2*lum[1:-1, 2:] + lum[2:, 2:]
    gy = -lum[:-2, :-2] - 2*lum[:-2, 1:-1] - lum[:-2, 2:] + lum[2:, :-2] + 2*lum[2:, 1:-1] + lum[2:, 2:]
    result[path] = dict(width=pixels.shape[1], height=pixels.shape[0], mask='full',
        luminance=dict(p5=round(float(p5), 1), p50=round(float(p50), 1), p95=round(float(p95), 1)),
        dynamicRange=round(float(p95-p5), 1), clippedPct=round(float(np.all(pixels >= 254, axis=2).mean()*100), 3),
        colorEntropyBits=round(float(-(probabilities*np.log2(probabilities)).sum()), 2),
        dominantFraction=round(float(probabilities.max()), 4),
        edgeDensity=round(float((np.hypot(gx, gy) > 60).mean()), 4))
print(json.dumps(result, indent=2))
