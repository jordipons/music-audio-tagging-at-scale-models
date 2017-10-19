## End-to-end learning for music audio tagging at scale

The lack of data tends to limit the outcomes of deep learning research - specially, when dealing with end-to-end learning stacks processing raw features such as waveforms. In this study we make use of musical labels annotated for 1.5 million tracks. This large amount of data allows us to explore different front-end paradigms: from assumption-free models - using waveforms as input with very small convolutional filters; to models that rely on domain knowledge - log-MEL spectrograms with a convolutional neural network designed to learn temporal and timbral features. Results suggest that while spectrogram-based models surpass their waveform-based counterparts, the difference in performance shrinks as more data is employed.

## Models
The following models were used in our study, and their tensorflow implementation is available in `models.py`. We study models based on two conceptually very different design principles. The first is based on a waveform front-end, and no decisions based on domain knowledge inspired its design. Note that the assumptions of this model are reduced to its minimum expression: raw audio is set as input, and the used CNN does minimal assumptions over the structure of the data - since it is using a stack of very small filters. But for the second model, with a spectrogram front-end, it was key to make use of domain knowledge to propose efficient and intuitive improvements to our model.

### Waveform front-end	

It is based on the sample-level front-end proposed by ```Lee, et al. "Sample-level Deep Convolutional Neural Networks for Music Auto-tagging Using Raw Waveforms." arXiv preprint arXiv:1703.01789 (2017)```.

<p align="center"><img src="waveform.png"></p>

Each layer has 64, 64, 64, 128, 128, 128 and 256 filters respectively. Via hierarchically combining small-context representations and making use of max pooling, the sample-level front-end delivers a feature map of an audio segment.

### Spectrogram front-end

The proposed front-end is a single-layer CNN with many filter shapes that are grouped into two branches: (i) top branch - timbral features; and (ii) lower branch - temporal features.

<p align="center"><img src="spectrogram.png" height="290"></p>

The top branch is designed to capture pitch-invariant timbral features that are occurring at different time-frequency scales in the spectrogram. Pitch invariance is enforced via enabling CNN filters to convolve through the frequency domain, and via max-pooling the feature map vertical axis. 

The lower branch is meant to learn temporal features, designed to efficiently capture different time-scale representations by using several filter shapes. But note that CNN filters operate over an energy envelope (not directly over the spectrogram) obtained via mean-pooling the frequency-axis of the spectrogram.

### Back-end
In order to allow a fair comparison among models, the previous front-ends share this same back-end.

<p align="center"><img src="backend.png" height="190"></p>

It is conformed by three CNN layers (with 512 filters each and two of those having residual connections), some pooling layers and a dense layer. We found this filter shapes setup to be (i) computationally very efficient and (ii) capable of taking into consideration all the extracted features at once (observe the M'-axis of the CNN filters’ shape) for a reasonable amount of temporal context (observe the 7-axis of the CNN filters’ shape). We also make a drastic use of temporal pooling: firstly, via down-sapling x2 the temporal dimensionality of the CNNs feature map; and secondly, by making use of a global pooling layer. The assumption behind the global pooling strategy is that for music auto-tagging it is not relevant when a music characteristic is happening, but if it is happening. Finally, a dense layer connects the pooled features to the output.
