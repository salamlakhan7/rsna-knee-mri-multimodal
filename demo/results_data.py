import pandas as pd
TEXT = pd.DataFrame({
    "Method": ["Rules v2", "TF-IDF + LR", "DistilBERT (multilingual)"],
    "F1 all 58": [0.587, 0.547, 0.571], "F1 covered (n=46)": [0.695, 0.629, 0.616],
    "F1 other languages (n=12)": [0.000, 0.066, 0.299], "Macro AUC": [0.698, 0.713, 0.713]})
TEXT_F1 = pd.DataFrame({
    "Rules v2": [.79, .61, .72, .78, .48, .57, .36, .64, .49, .55, .55, .52],
    "TF-IDF + LR": [.70, .46, .67, .65, .41, .48, .46, .59, .51, .54, .55, .55],
    "DistilBERT": [.68, .40, .55, .67, .56, .52, .47, .72, .57, .67, .58, .48]},
    index=["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
           "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"])
IMAGE = pd.DataFrame({
    "Model": ["Small CNN (scratch)", "ResNet18, mean pooling", "ResNet18, attention pooling"],
    "Gold macro AUC": [0.631, 0.672, 0.692], "95% CI low": [0.568, 0.614, 0.642], "95% CI high": [0.690, 0.725, 0.742]})
IMAGE_AUC = pd.DataFrame({
    "CNN scratch": [.63, .55, .61, .66, .66, .79, .57, .72, .63, .72, .49, .54],
    "ResNet18 mean": [.67, .63, .56, .71, .73, .83, .70, .64, .57, .79, .61, .62],
    "ResNet18 attention": [.69, .63, .60, .71, .76, .85, .66, .70, .61, .78, .67, .63]}, index=TEXT_F1.index)
