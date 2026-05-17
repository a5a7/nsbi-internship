# Tasks
Data (toy) acquired!
Tasks:
- Perform file analysis: understand format, packaging, and parsing
- Explore patterns: plot kinematics against each other, read Tae's slides for any processing we can do
- Integration: find how to integrate data with EveNet, this will take Prof. Hsu or Yulei's mentorship for sure

Allen:
Finished a implementation of NSBI in EveNet Lite:
- Two Carl Models
- Parameters:
  Classification head: 1e-4
  ObjectEncoder: 5e-5
  PET + GlobalEmbedding: 1e-5
  
  Regularization:
  Weight decay: 0.01
  Gradient clip: 1.0
  
  Warmup:
  Epochs: 1
  Ratio: 0.1
  Start factor: 0.1
  Loss:
  
  Focal gamma: 0.0 (standard BCE)
  
  Training:
  Epochs: 20
  Batch size: 512
  Sampler: weighted
  Monitor: val_loss
  Save top k: 1
  Data (from logs):

  Train size: 4,610,116
  Val size: 1,152,530
  Steps per epoch: 9,005
