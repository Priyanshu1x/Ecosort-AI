# AI Usage and Verification Log

## Overview

During the development of **EcoSort-AI**, ChatGPT and Gemini were used as engineering and learning assistants. Their role was to support understanding, debugging, planning, and documentation. The project code, experiments, and final technical decisions were reviewed and carried out by the developer.

## Areas of AI Assistance

1. **Environment and Dataset Debugging**
   AI assisted in investigating a `n_samples=0` error during the Train/Validation/Test split. It suggested that the dataset directory structure might not match the expected path after extraction in Google Colab. This was checked manually using directory-listing commands, and the dataset path was corrected.

2. **Data Pipeline and Model Concepts**
   AI helped explain the purpose of dataset splitting, DataLoaders, image transformations, tensor shapes, transfer learning, and classifier layers. These explanations were used to support implementation and understanding of the project.

3. **Model Architecture and Experiment Planning**
   AI assisted in planning the comparison between a custom Baseline CNN, ResNet18, and EfficientNet-B0. It helped explain the differences between a custom model and pretrained feature extractors, and supported the interpretation of evaluation metrics. Reported model performance is based on the project’s actual evaluation outputs, not AI-generated estimates.

4. **Training and Evaluation Troubleshooting**
   AI helped investigate issues involving training configuration, checkpoint paths, imports, and running the project in Google Colab. Suggested changes were tested in the relevant environment before being accepted.

5. **Deployment Troubleshooting**
   AI assisted with investigating file-path and environment differences when moving from Google Colab training to local Windows development and Streamlit deployment.

6. **Documentation and Testing Support**
   AI assisted with the structure and wording of the README, project documentation, and test-planning materials. Any generated or suggested content was reviewed and adapted to match the project.

## Verification Process

AI suggestions were treated as guidance rather than automatically accepted answers.

* Dataset paths and image counts were checked using terminal or Colab commands.
* Model input/output shapes and training behavior were checked by running the code.
* Evaluation metrics were taken from the project’s actual PyTorch evaluation outputs.
* Suggested code and debugging changes were reviewed and tested in the relevant environment.
* Documentation was checked against the implemented project features.

## Limitations and Responsibility

AI-generated suggestions may contain mistakes or assumptions. The developer is responsible for reviewing the code, verifying experimental results, documenting limitations, and ensuring that the final project accurately represents the work completed.

AI was not treated as a source of experimental evidence. Model metrics, comparisons, and conclusions must be supported by actual project runs and recorded outputs.
