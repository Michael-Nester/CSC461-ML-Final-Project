# Eye Color Prediction and Transformation

## Project Overview
This project explores the use of machine learning to analyze, reconstruct, and manipulate eye features in images. Our primary goals were to classify eye colors and enable transformations to a user-defined target.

## Final Model
The final model is a Convolutional Neural Network (CNN)-based autoencoder, which serves two main purposes:
- **Reconstruction**: Deconstructs and reconstructs eye images while preserving spatial features, such as the iris, pupil, and surrounding details.
- **Transformation**: Modifies the eye color by manipulating the latent space representation, enabling targeted adjustments while maintaining realistic details.

## Key Features
- **Latent Space Representation**: Encodes the eye image into a compressed latent vector, capturing its most salient features.
- **Dimensionality Reduction**: Uses UMAP and t-SNE for visualizing the latent space and understanding clustering behavior.
- **CNN Architecture**: Optimized for preserving spatial relationships, outperforming Multi-Layer Perceptron (MLP) approaches in reconstruction quality.

## Dataset
The UBIRIS.v2 dataset, containing over 11,000 images, was utilized to train and evaluate the models. Images were labeled manually for eye color categories (e.g., blue, brown, green, hazel, and gray).

## Results
- High-quality reconstructions of eye images, preserving key spatial features.
- Latent space visualizations revealed clustering challenges, with broader features (e.g., face shape) being prioritized over subtle differences like eye color.
- Insights gained into the tradeoff between latent vector size, reconstruction accuracy, and feature separability.

## Future Directions
- Integrating supervised components to improve clustering based on eye color.
- Incorporating advanced architectures like Variational Autoencoders (VAEs) for better feature separability.
- Applying statistical analyses to rigorously evaluate clustering and reconstruction performance.

## Team Members
- Aeyva Rebelo
- Michael Nester
- Esra Bequir

## Repository Structure
- `/code`: Python scripts for training, evaluation, and visualizations.
- `/data`: Preprocessed and raw datasets (linked as necessary).
- `/final_results`: Outputs, visualizations, and metrics from experiments.
  - To run the final results of this project:
    - Download FINAL_NOTEBOOK.ipynb from `/final_results`
    - Download testData, trainData, test_labels.csv and train_labels.csv from `/data`
    - download any number of the autoencoder, encoder, and decoder folders from `/code/models`
      - NOTE: One autoencoder, encoder, and decoder are needed for running
      - NOTE: Make sure that when you change the path to load the models in the notebook, you are loading the same three model versions
      - I.E. autoencoder_model_f16 goes with encoder_model_f16 and decoder_model_f16
- For any questions, feel free to contact us:
  - { emails would go here, not currently listed for privacy reasons }
## Acknowledgment
This project utilized insights and support from OpenAI's ChatGPT and Anthropic's Claude 3.5 Sonnet for drafting, structuring, and refining analyses.
