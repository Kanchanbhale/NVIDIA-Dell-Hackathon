# NVIDIA-Dell-Hackathon
An NVIDIA  and Dell-optimized Physical AI based vision &amp; agentic reasoning with continuous knowledge-graph updating system for autonomous elderly fall and health-surveillance.

My goal is to reimagine elder care with autonomous vision and agentic intelligence - delivering fall prediction, posture monitoring, and real-time health insights that make caregiving proactive, precise, and scalable. The mission is a global AI platform that safeguards elders through continuous, intelligent health surveillance, enabling longer, safer, independent living.

The system integrates a multi-stage perception pipeline - YOLOv8-Nano for person and fall-state detection, RTMPose for pose estimation, RAFT Lite for optical-flow analysis, and an LSTM-based temporal classifier for trajectory modeling. These outputs are processed by an NVIDIA LLM powered agentic reasoning module that interprets posture transitions, evaluates temporal context, and autonomously escalates safety decisions. This includes early warnings, family-level SOS notifications, and automated 911 escalation when high-risk conditions persist.

For training and evaluation, I used a multimodal dataset stack combining 20,000+ labeled images, 500+ fall and daily-activity videos, and IoT/sensor-based motion datasets (accelerometer and environmental activity streams). All modalities were standardized into a unified 30-FPS pipeline with optical-flow extraction, skeleton-keypoint mapping, and weighted temporal sampling to reduce false positives in varied environments.

CareMatrix AI demonstrates how multimodal perception combined with LLM-driven agentic intelligence can create dependable, safety-critical autonomous systems. At scale, technologies like this have the potential to significantly reduce injury severity and prevent large numbers of fall-related deaths.
I look forward to continuing work in computer vision, agentic AI architectures, multimodal intelligence, and real-time autonomous decision systems.


