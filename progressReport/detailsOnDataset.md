## Details About The 'UBIRIS_V2' Dataset
The UBIRIS.v2 dataset, developed at the University of Beira Interior, is designed to advance iris recognition under real-world, unconstrained conditions. This dataset contains over 11,000 images from 261 subjects, captured in visible light with challenges like occlusions, varying lighting, reflections, and movement, making it uniquely suitable for training models in realistic, non-ideal settings. Images are labeled by general iris pigmentation levels (light, medium, heavy), a feature that could aid in understanding color variability.

## Project Goals and Dataset Utility
Our project aims to develop a machine learning algorithm that can recognize and alter the eye color in images, such as transforming a brown eye into a green eye. The UBIRIS.v2 dataset offers a valuable foundation since it provides varied iris data under diverse conditions, allowing us to train a model that can generalize across lighting and occlusion challenges.

Once we are able train the model on the .v2 dataset it will hopefully produce binarized values of color so that we can then upload data/ images of eyes and it will be able to transform an eye that is for example, blue and make it red. 

Some ***fun applications*** towards the end that we migt explore could be making an image of an eye that takes into account all eye colors and makes one (perfect) eye color that has all of the eyecolors in relation to how common certain colors are. We could also create Dreamlike Transformations: Use eye images to generate surreal, abstract representations, like visual "dreamscapes" inspired by the texture and color of the iris. Imagine an eye morphing into fluid, organic patterns, evoking landscapes or celestial scenes. Biomimicry and hybridization which would be inspired by Sofia Crespo’s work with AI and nature, the algorithm could blend eye patterns with natural textures—like scales, leaves, or feathers—to mimic how different species might "see." Or Emotional reflection: Another direction could involve manipulating the color and shape of the eye to reflect various emotions or states. For instance, calm might translate to cooler hues, while excitement might bring vibrant, warm patterns.

## Technical Challenges and Solutions
Key challenges we may face:

**Eye Color Labeling:**  Since the dataset only provides broad pigmentation labels, we may need to develop a process to assign finer-grained color labels or use clustering algorithms to categorize eye color.
**Image Processing Under Noise:** The dataset's noise factors (reflections, varied lighting) can complicate model training. We'll need effective preprocessing, such as noise reduction and normalization, to ensure consistent color changes.
**Transformation Accuracy:** Altering eye color while maintaining natural texture and shading requires precise image generation, likely through CNNs, HAC, PCA or similar models, which we can train on this dataset's broad pigmentation range.
Using UBIRIS.v2, we can create a robust model that recognizes and transforms eye color effectively, enhancing its resilience to real-world variations in image quality and lighting.









## References
http://iris.di.ubi.pt/ubiris2.html
