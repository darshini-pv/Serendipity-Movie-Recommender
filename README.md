

\# Serendipity: A Novelty-Aware Personalized Movie Recommendation System



\## 1. Project Overview



Serendipity is a personalized movie recommendation system designed to recommend movies based on users' preferences while also introducing less obvious and potentially interesting choices.



Traditional recommendation systems often prioritize popular movies or items similar to a user's previous preferences. This project investigates how collaborative filtering and novelty-aware ranking can balance recommendation relevance and discovery.



This is an academic machine learning prototype developed using the MovieLens 1M dataset.



\## 2. Objectives



\- Build a personalized movie recommendation system.

\- Implement popularity-based and collaborative filtering approaches.

\- Improve collaborative filtering using mean-centered cosine similarity.

\- Investigate novelty-aware recommendation ranking.

\- Evaluate recommendation quality using ranking metrics.

\- Explore content-based and hybrid recommendation approaches.

\- Analyze model limitations and recommendation diversity.



\## 3. Dataset



\*\*MovieLens 1M Dataset\*\*



The dataset contains approximately:

\- 1,000,209 movie ratings

\- 6,040 users

\- 3,883 movies



Files used:

\- `ratings.csv`

\- `movies.csv`

\- `users.csv`



The original dataset is not included in this repository. Download MovieLens 1M separately and place the required CSV files inside a folder named `data`.



\## 4. Techniques and Algorithms



The project investigates the following approaches:



1\. Popularity-based recommendation

2\. User-based collaborative filtering

3\. Cosine similarity

4\. Mean-centered cosine similarity

5\. Regularized similarity

6\. Novelty-aware recommendation reranking

7\. Content-based filtering using movie genres

8\. Hybrid recommendation experiments



The final selected collaborative filtering configuration from the model-selection experiments uses:

\- Mean-centered cosine similarity

\- Regularization parameter: 10

\- Number of neighbors: 50

\- Novelty weight: 0.005

\- Recommendation list size: 10



\## 5. Evaluation Metrics



The recommendation system is evaluated using:



\- Precision@10

\- Recall@10

\- NDCG@10

\- Novelty

\- Catalog coverage

\- Intra-list diversity

\- Unexpectedness

\- Serendipity proxy



Prediction accuracy is also investigated using MAE and RMSE.



\## 6. Key Experimental Results



\### Selected Model — Stage 16



On the sampled final-test evaluation:



| Metric | Result |

|---|---:|

| Precision@10 | 0.09560 |

| Recall@10 | 0.11888 |

| NDCG@10 | 0.14028 |

| Novelty | 0.15934 |

| Catalog Coverage | 7.63% |

| Distinct Movies Recommended | 281 |



\### Serendipity Analysis — Stage 17



| Metric | Result |

|---|---:|

| Unexpectedness | 0.53212 |

| Serendipity Proxy | 0.04440 |

| Intra-list Diversity | 0.70921 |



The serendipity value is an experimental proxy based on recommendation relevance and unexpectedness. It is not a measure validated through actual user feedback.



\### Hybrid Experiment — Stage 22



Content-based genre recommendations were combined with collaborative filtering using different weights.



The validation experiment selected pure collaborative filtering (CF weight = 1.0, content weight = 0.0). Adding the current genre-based content score did not improve NDCG, precision, or recall.



This is an important experimental finding: the tested hybrid approach did not outperform the collaborative filtering model.



\## 7. Project Development Stages



The repository contains scripts and generated results from Stages 1–22, covering:



\- Dataset loading and exploration

\- Baseline recommendation development

\- Collaborative filtering

\- Novelty-aware ranking

\- Model evaluation and tuning

\- Prediction diagnostics

\- Mean-centered similarity comparison

\- Model selection and verification

\- Robustness and serendipity analysis

\- Temporal evaluation investigation

\- Content-based recommendation

\- Hybrid recommendation experiments



\## 8. Limitations



\- MovieLens 1M is a historical dataset containing ratings from 2000–2003.

\- Random train/test evaluation does not fully represent real-world future recommendations.

\- The system has not been tested with live users.

\- The serendipity measure is a proxy rather than a user-validated measure.

\- Cold-start users and movies remain challenging.

\- The current content-based model did not improve the selected collaborative filtering model.

\- The project is an academic prototype, not a production-ready recommendation service.



\## 9. Future Improvements



\- Perform more comprehensive temporal evaluation.

\- Investigate cold-start recommendation strategies.

\- Improve content-based feature representations.

\- Calibrate collaborative and content-based scores before hybrid ranking.

\- Conduct user studies to validate perceived serendipity.

\- Develop an interactive recommendation interface.



\## 10. Technologies Used



\- Python

\- Pandas

\- NumPy

\- SciPy

\- Scikit-learn

\- Matplotlib

\- Spyder IDE

\- Git and GitHub



\## 11. Running the Project



1\. Clone this repository.

2\. Download the MovieLens 1M dataset separately.

3\. Place the required CSV files inside the `data` directory.

4\. Install the required Python libraries:



```bash

pip install pandas numpy scipy scikit-learn matplotlib

```



5\. Open the desired stage script in Spyder.

6\. Update the dataset path if necessary and execute the script.



\## 12. Author



\*\*Darshini P. V.\*\*



B.Tech — Data Science and Artificial Intelligence



The Apollo University



GitHub: \[darshini-pv](https://github.com/darshini-pv)



\---



\*This project is developed for academic learning and experimentation in personalized recommendation systems.\*



