# tf_ddrl_plus_plus
It utilizes a Sparse Graph-Transformer (SGT) for dependency modeling, Knowledge Distillation (KD) for efficient Edge deployment, and a Hybrid Feedback (HF) loop for adaptive policy refinement.
Project Structure Overview

The code is organized to align with the thesis chapters:

src/arch/: Neural network definitions including the SGT (graph_transformer.py).

src/core/: Core DRL logic, notably PPO (ppo_agent.py) and PER (per_buffer.py).

src/env/: The custom simulation environment (ioht_scheduler_env.py).

src/distillation/: Knowledge transfer logic (knowledge_distiller.py, hybrid_feedback.py).

Getting Started

Prerequisites

You need Python 3.9+ installed.

Installation

Clone the repository:

git clone https://github.com/your-username/tf_ddrl_plus_plus.git
cd tf_ddrl_plus_plus


Install dependencies using the provided requirements.txt:

pip install -r requirements.txt


Review and adjust hyperparameters in the config.yaml file.

Execution Workflow

The project supports two main entry points corresponding to the Cloud training and the Edge deployment phase.

1. Training the Cloud Teacher Agent (Chapter 5)

This script trains the large Teacher Policy, which generates the high-quality policy used for distillation.

python main_cloud_trainer.py


Outputs logs to the runs/ directory for viewing with TensorBoard.

2. Deploying the Edge Student Agent (Chapter 6 & 8)

This script loads a pre-trained Student Agent and runs it in an inference-only loop to measure its real-time performance and scalability.

python main_edge_deployer.py


Running Thesis Experiments (Chapters 8 & 9)

The experiments/ directory contains scripts used to generate the final results for your thesis:

Script

Purpose (Thesis Chapter)

run_ablation_study.py

Runs Ablation Studies (Chapter 9) by toggling SGT/PER features.

run_scalability_test.py

Runs Scalability Tests (Chapter 8) to compare Edge latency under increasing load.

run_baselines.py

Runs comparative baselines (FCFS, DRL-MLP) for Chapter 9 results.

License

This project is licensed under the MIT License. See the LICENSE file for details.
