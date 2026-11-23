# ImageIdentify
Paper ("Optimizing AI image detection using a Convolutional Neural Network model with Fast Fourier Transform"): https://doi.org/10.59720/24-270

_Abstract_: Recent progress in generative artificial intelligence (AI) technology has made it increasingly difficult to tell whether an image is AI-generated or real. Current AI detection methods rely solely on training a model through convolutional neural networks (CNNs) or using the U-net architecture with intersection-over-union as an evaluation metric. However, these approaches are not accurate enough to obtain sufficient detail from spatial (traditional) and frequency domain features, missing discrete patterns in edges and textures. Our study aimed to determine if a tool incorporating CNN and Fast Fourier Transform (FFT) can improve detection of AI-generated images. We hypothesized that using a CNN model and frequency domain features that leverage FFT would lead to improvements in AI-generated image detection. Our study presented a machine learning model that utilizes FFT to transform images from the spatial to the frequency domain, where subtle features can be extracted. This process involved training the model on a dataset comprised of 100,000 training images and 20,000 testing images. The model preprocesses images through FFT and merges the images into CNN logic to achieve 93.30% accuracy while handling both colored and grayscale inputs. Our results demonstrated that the integrated approach performed better than the CNN-only approach, showing 10.06% lower loss, 3.27% higher accuracy, 7.79% higher true positive rate, and 3.48% higher F1-score. This research offers an improved method for distinguishing between AI-generated and genuine images, which will be highly impactful as generative Al produces more malicious fake news and video attempts.

_Dataset_: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images


--------
By: Pranav Gupta
