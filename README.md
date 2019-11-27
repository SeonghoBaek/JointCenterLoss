# JointCenterLoss

Fine Face Discrimination Network

- JointCenterLoss.py
  - Custom Intra Class Discrimination Network
  - Joint Loss: Cross Entropy Loss with Class Center Loss & Center Distance Loss
  - Network
    - ResDense Block: ResNet + DenseNet
 
- FDN.py
  - Effective Unknown Class Generation Network
  - Learn Marginal Decision Boundary between Target Class & Very Similar(Generated) Class
  - Adversarial Auto Encoder with Self Attention
 
- SR.py
  - Super Resolution Network
  - Transform Fake Face created by FDN.py to better face for discrimination with JointCenterLoss.py

TBD
