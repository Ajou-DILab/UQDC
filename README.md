# EDRR

Evidential Document Re-Ranking

The quality of the two-step Retrieval Augmented Generation (RAG) in question-answering (QA) relies partially on the accuracy of the re-ranking phase to select the most relevant context for answer generation. 
Text re-ranking models are often based on classification models that use predicted probabilities as relevance scores. However, typical deep neural network classification models that directly minimize the prediction loss to find point estimates are often poorly calibrated. 
We propose the Evidential Document Re-Ranking (EDRR) model that incorporates Evidential deep learning (EDL) for better calibration of the prediction probabilities and estimation of the uncertainties of model predictions. The proposed EDRR utilizes the calibrated prediction probabilities and uncertainties to generate more reliable relevance criteria for re-ranking. 
Furthermore, estimated uncertainty values can be utilized as active learning criteria to select more diverse training samples.   
We compare our EDRR to a regular cross-encoder structure under on Wikipedia-NQ dataset and show that our model outperforms the regular cross-encoder model with up to $10\%$ increases in mean average precision (mAP@10) at the top $10$.
 
## Key Features
- **Uncertainty-Based Learning**: Active learning functionality through `AL_CE.py` and `AL_ENN.py`.
- **Calibration Tools**: Evaluate model calibration performance using `Calibration.py`.
- **Support Functions**: Mathematical functions for uncertainty handling in `edl_function.py`.

## File Structure
- **`AL_CE.py`**: Active learning based on cross-entropy loss.
- **`AL_ENN.py`**: Active learning using Ensemble Neural Networks (ENN).
- **`Calibration.py`**: Tools for evaluating calibration performance.
- **`edl_function.py`**: Key mathematical functions for uncertainty processing.
- **`LICENSE`**: Project license information.
- **`Project/`**: Additional project-related files.

## Installation and Usage

### 1. Prerequisites
This project requires Python 3.8 or higher and the following libraries:
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `torchmetrics`
- `transformers`
- `datasets`

Install the dependencies using:
```bash
pip install -r requirements.txt

