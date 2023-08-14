#%%

import matplotlib.pyplot as plt

results = {"efficientnet_b0": {"train_mean_metrics": {"loss": 2.309359774679843, "f1": 0.874454768575925, "acc": 0.8714413798620777, "precision": 0.9050067728474133, "recall": 0.8714413798620779, "BPR_F1": 0.8983359743232716, "FPR_F1": 0.8255625343968169, "Healthy_F1": 0.8657320336588276, "WBD_F1": 0.8932600299235048}, "val_mean_metrics": {"loss": 3.1552665684467707, "f1": 0.6904469582445772, "acc": 0.6922262922262921, "precision": 0.7339669820919821, "recall": 0.6922262922262921, "BPR_F1": 0.7298321123321123, "FPR_F1": 0.5659615384615384, "Healthy_F1": 0.7100213675213676, "WBD_F1": 0.7490820454055748}, "train_se_metrics": {"loss": 0.21553292558684073, "f1": 0.03254619980435627, "acc": 0.032704878967734455, "precision": 0.026583551787776816, "recall": 0.03270487896773447, "BPR_F1": 0.02446216620992823, "FPR_F1": 0.04387299092708682, "Healthy_F1": 0.04403322466659507, "WBD_F1": 0.02371684244615807}, "val_se_metrics": {"loss": 0.22691300144705767, "f1": 0.027148792121150084, "acc": 0.027073213844450248, "precision": 0.030036624712527153, "recall": 0.027073213844450248, "BPR_F1": 0.039272658230405696, "FPR_F1": 0.047996552480018044, "Healthy_F1": 0.03975489741262587, "WBD_F1": 0.041435304331877636}},
            "efficientnetV2-s": {"train_mean_metrics": {"loss": 2.2575084927122484, "f1": 0.8902356324754619, "acc": 0.8860672266756267, "precision": 0.9181074667708968, "recall": 0.8860672266756268, "BPR_F1": 0.9030008704802871, "FPR_F1": 0.8346702910490986, "Healthy_F1": 0.8928380073762916, "WBD_F1": 0.9107896195502233}, "val_mean_metrics": {"loss": 3.323391774627898, "f1": 0.7059293803639042, "acc": 0.707051282051282, "precision": 0.7724171463457178, "recall": 0.7070512820512821, "BPR_F1": 0.745501867413632, "FPR_F1": 0.6164085914085914, "Healthy_F1": 0.7201700423759247, "WBD_F1": 0.729126925705873}, "train_se_metrics": {"loss": 0.17869478862703936, "f1": 0.02130521369064472, "acc": 0.022004135989800742, "precision": 0.01610404268938686, "recall": 0.02200413598980075, "BPR_F1": 0.022108629734222483, "FPR_F1": 0.03337910158163515, "Healthy_F1": 0.01831247942299695, "WBD_F1": 0.020013214503844967}, "val_se_metrics": {"loss": 0.2633412571483362, "f1": 0.021582871491258704, "acc": 0.020588088785288894, "precision": 0.02193612766773713, "recall": 0.020588088785288894, "BPR_F1": 0.04272956735146777, "FPR_F1": 0.03974750277570198, "Healthy_F1": 0.037968999551482696, "WBD_F1": 0.03361057446206518}},
           "resnet18": {"train_mean_metrics": {"loss": 1.795058268391226, "f1": 0.6172005746237982, "acc": 0.6228601225619609, "precision": 0.6764403341393723, "recall": 0.622860122561961, "BPR_F1": 0.7217006134796367, "FPR_F1": 0.45293909215655725, "Healthy_F1": 0.5771030207250306, "WBD_F1": 0.6981829463527867}, "val_mean_metrics": {"loss": 1.9390275902600371, "f1": 0.6485550831347355, "acc": 0.6532153032153032, "precision": 0.7101909994767137, "recall": 0.6532153032153032, "BPR_F1": 0.7356065503124326, "FPR_F1": 0.5760139860139859, "Healthy_F1": 0.5604739704739703, "WBD_F1": 0.7087978619247969}, "train_se_metrics": {"loss": 0.18667081037172442, "f1": 0.021661584022085536, "acc": 0.019136368412923432, "precision": 0.024007093581767017, "recall": 0.019136368412923422, "BPR_F1": 0.01729918788361364, "FPR_F1": 0.03254700542377541, "Healthy_F1": 0.02317963311917654, "WBD_F1": 0.022614218346873636}, "val_se_metrics": {"loss": 0.14386486198012813, "f1": 0.01639641870579217, "acc": 0.014826507269436911, "precision": 0.02110055555696848, "recall": 0.014826507269436923, "BPR_F1": 0.0332269778172585, "FPR_F1": 0.04725257900397079, "Healthy_F1": 0.022457150752444643, "WBD_F1": 0.0215325750355198}},
           "DisNet-pico": {"train_mean_metrics": {"loss": 55.262336772098124, "f1": 0.49337088009137675, "acc": 0.5122729707675322, "precision": 0.5630378293040966, "recall": 0.5122729707675322, "BPR_F1": 0.514436831778878, "FPR_F1": 0.4498208558525979, "Healthy_F1": 0.45401139350956543, "WBD_F1": 0.610070395018259}, "val_mean_metrics": {"loss": 55.459377086672006, "f1": 0.5229636844377741, "acc": 0.5342897842897842, "precision": 0.58439269724984, "recall": 0.5342897842897842, "BPR_F1": 0.6131045751633986, "FPR_F1": 0.35469139556096074, "Healthy_F1": 0.43233211233211233, "WBD_F1": 0.641187773354956}, "train_se_metrics": {"loss": 0.06480841702738396, "f1": 0.03691453311787006, "acc": 0.034349208195977345, "precision": 0.03688126395911415, "recall": 0.03434920819597735, "BPR_F1": 0.04100612731099118, "FPR_F1": 0.0379263300067851, "Healthy_F1": 0.04032770867062077, "WBD_F1": 0.02597923019921057}, "val_se_metrics": {"loss": 0.04104453091860221, "f1": 0.02662272055879515, "acc": 0.02585129494514215, "precision": 0.02441633432921078, "recall": 0.025851294945142156, "BPR_F1": 0.054705891137816424, "FPR_F1": 0.04084918117285004, "Healthy_F1": 0.056947936233977765, "WBD_F1": 0.03726090098795309}}
           }



# Extract mean and SE metrics
models = list(results.keys())

metrics_to_plot = ['loss', 'f1', 'acc', 'precision', 'recall', 'BPR_F1', 'FPR_F1', 'Healthy_F1', 'WBD_F1']

fig, axs = plt.subplots(len(metrics_to_plot), 1, figsize=(10, 15))

for idx, metric in enumerate(metrics_to_plot):
    train_mean = [results[model]['train_mean_metrics'][metric] for model in models]
    val_mean = [results[model]['val_mean_metrics'][metric] for model in models]
    train_se = [results[model]['train_se_metrics'][metric] for model in models]
    val_se = [results[model]['val_se_metrics'][metric] for model in models]

    # Plotting
    bar_width = 0.35
    index = range(len(models))

    bar1 = axs[idx].bar(index, train_mean, bar_width, yerr=train_se, label='Train', alpha=0.8, capsize=10)
    bar2 = axs[idx].bar([i+bar_width for i in index], val_mean, bar_width, yerr=val_se, label='Validation', alpha=0.8, capsize=10)

    axs[idx].set_ylabel(metric)
    axs[idx].set_xticks([i+bar_width/2 for i in index])
    axs[idx].set_xticklabels(models)

    # Remove top and right spines
    axs[idx].spines['right'].set_visible(False)
    axs[idx].spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    axs[idx].yaxis.set_ticks_position('left')
    axs[idx].xaxis.set_ticks_position('bottom')

# Add legend only to the last subplot
axs[-1].legend()

plt.tight_layout()

# save plot
plt.savefig('/users/jrs596/scripts/CocoaReader/CocoaNet/IR-RGB/cross-validation/results.png', dpi=300, bbox_inches='tight')
plt.show()

#%%