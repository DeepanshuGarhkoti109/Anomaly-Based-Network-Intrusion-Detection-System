# Technical Artifact: Intrusion Detection Model Evaluation Details

## 1. Mathematical Formulation & Metrics

### Accuracy
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

### Precision
$$\text{Precision} = \frac{TP}{TP + FP}$$

### Recall (True Positive Rate)
$$\text{Recall} = \frac{TP}{TP + FN}$$

### F1-Score
$$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### ROC-AUC
$$\text{ROC-AUC} = \int_{0}^{1} \text{TPR}(FPR^{-1}(t)) \, dt$$

## 2. Top Selected Features (Mutual Information Ranking)

1. **sbytes** - Source-to-destination transaction bytes
2. **dbytes** - Destination-to-source transaction bytes
3. **sload** - Source bits per second
4. **dload** - Destination bits per second
5. **spkts** - Source-to-destination packet count
6. **dpkts** - Destination-to-source packet count
7. **dur** - Record total duration
8. **ct_srv_src** - Connections containing same service and source address
9. **ct_dst_sport_ltm** - Connections to same destination and source port in 100 records
10. **ct_src_dport_ltm** - Connections from same source to destination port in 100 records

## 3. Directory Layout

All generated reports and metric tables are consolidated in `final_reports/`.
