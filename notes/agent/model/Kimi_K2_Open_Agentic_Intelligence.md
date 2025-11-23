# Kimi K2: Open Agentic Intelligence

**Author**: Kimi Team

**Publish Date**: 2025

**Add Date**: 2025.11.23

**Journal/Meeting**: 

**Star**: 🌟🌟🌟🌟🌟

**PDF**: [Kimi K2: Open Agentic Intelligence](original_files/Team_2025_Kimi_K2_Open_Agentic_Intelligence.pdf)

## 1 Introduction

**Agentic Intelligence**: actively learn through interactions. Agentic intelligence is rapidly emerging as a defining capability for the next generation of foundation models.

**Challenges**

- **Pre-training**: endow models with broad general-purpose priors under *constraints of limited high-quality data*, elevating *token efficiency*.

- **Post-training**: transform those priors into actionable behaviors, yet agentic capabilities such as multi-step reasoning, long-term planning, and tool use are *rare in natural data* and costly to scale.

Scalable synthesis of structured, high-quality agentic trajectories, combined with general reinforcement learning (RL) techniques that incorporate preferences and self-critique, are essential to bridge this gap.

**Contributions**

- **MuonClip**: a novel optimizer that integrates the *token-efficient ***Muon*** algorithm* with a *stability-enhancing mechanism called* ***QK-Clip***.

- **Large-scale agentic data synthesis pipeline**: systematically generates tool-use demonstrations via simulated and real-world environments.

- **General reinforcement learning framework**: combines verifiable rewards (RLVR) with a selfcritique rubric reward mechanism.

## 2 Pre-training

### 2.1 MuonClip: Stable Training with Weight Clipping

### 2.2 Pre-training Data: Improving Token Utility with Rephrasing

### 2.3 Model Architecture

### 2.4 Training Infrastructure

#### 2.4.1 Compute Cluster

#### 2.4.2 Parallelism for Model Scaling

#### 2.4.3 Activation Reduction

### 2.5 Training recipe

## 3 Post-Training  

### 3.1 Supervised Fine-Tuning

#### 3.1.1 Large-Scale Agentic Data Synthesis for Tool Use Learning

### 3.2 Reinforcement Learning

#### 3.2.1 Verifiable Rewards Gym

#### 3.2.2 Beyond Verification: Self-Critique Rubric Reward

#### 3.2.3 RL Algorithm

### 3.3 RL Infrastructure

#### 3.3.1 Colocated Architecture

#### 3.3.2 Efficient Engine Switching

#### 3.3.3 Efficient System Startup

#### 3.3.4 Agentic Rollout

## 4 Evaluations

### 4.1 Post-training Evaluations

#### 4.1.1 Evaluation Settings

#### 4.1.2 Evaluation Results

### 4.2 Pre-training Evaluations

#### 4.2.1 Evaluation Settings

#### 4.2.2 Evaluation Results

### 4.3 Safety Evaluation

#### 4.3.1 Experiment Settings

#### 4.3.2 Safety Evaluation Results

## 5 Limitations

## 6 Conclusions
