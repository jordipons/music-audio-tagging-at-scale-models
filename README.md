# End-to-end learning for music audio tagging at scale

## Summary
It is commonly argued among music/audio engineering researchers that the lack of data is severely limiting the outcomes of our deep learning research - specially, when dealing with end-to-end learning stacks processing waveforms. Fortunately, for this study we make use of the Music Genome Project dataset of Pandora; and given the big bulk of data available for our study, we are able to unrestrictedly explore different front-end paradigms: from assumption-free models - using raw waveforms as input with very small convolutional filters; to models that heavily rely on domain knowledge - log-MEL spectrograms with a convolutional neural network designed to learn temporal and timbral features. Can waveform-based models achieve better performance than spectrogram-based models given that a sizable dataset is provided?

We study two conceptually very different design principles. The first is based on a waveform front-end, and no decisions based on domain knowledge inspired its design. Note that the assumptions of this model are reduced to its minimum expression: raw audio is set as input, and the used CNN does minimal assumptions over the structure of the data - since it is using a stack of very small filters. But for the second model, with a spectrogram front-end, it was key to make use of domain knowledge to propose efficient and intuitive improvements to our model. The proposed models are capable of outperforming the current music audio tagging system in production at Pandora, and our results confirm that spectrogram-based architectures are still superior to waveform-based models. However, the gap between these models has been reduced via training with more data, and using the waveform (sample-level) front-end proposed by Lee et al.

## Models
In the following, we present the several architectures we used for our study that are implemented in tensorflow at `models.py`:

### Waveform front-end:	

<center><img src="waveform.png"></center>

### Spectrogram front-end:

<img src="spectrogram.png" height="290" align="middle">

### Back-end
In order to allow a fair comparison among front-ends, the previous front-ends share this same back-end.

<img src="backend.png" height="190" align="middle">
