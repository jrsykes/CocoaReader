#%%

import matplotlib.pyplot as plt
import numpy as np

#%%
# results = {
#             # "DisNet": {"train_mean_metrics": {"Loss": 6.309433592388681, "F1": 0.5095193567766932, "Acc": 0.5198748331973311, "Precision": 0.590993457949595, "Recall": 0.5198748331973312, "BPR_F1": 0.5664311738657404, "FPR_F1": 0.40530019929759353, "Healthy_F1": 0.47405515525735453, "WBD_F1": 0.6026920655442204}, "val_mean_metrics": {"Loss": 6.529871252561227, "F1": 0.5273524284098514, "Acc": 0.5340048840048841, "Precision": 0.5738358092524759, "Recall": 0.534004884004884, "BPR_F1": 0.6106349206349206, "FPR_F1": 0.4086711001416884, "Healthy_F1": 0.43610500610500613, "WBD_F1": 0.6235103690599046}, "train_se_metrics": {"Loss": 0.0439430660230937, "F1": 0.030196094841224534, "Acc": 0.026338167800415414, "Precision": 0.027362395800434203, "Recall": 0.02633816780041542, "BPR_F1": 0.034867149804626366, "FPR_F1": 0.037771852256538965, "Healthy_F1": 0.032584083378942544, "WBD_F1": 0.020526361897890977}, "val_se_metrics": {"Loss": 0.06343326335431318, "F1": 0.02045175008967055, "Acc": 0.021580967750193046, "Precision": 0.018251361931118502, "Recall": 0.021580967750193046, "BPR_F1": 0.048246683869559055, "FPR_F1": 0.04904771478428722, "Healthy_F1": 0.041491644644347826, "WBD_F1": 0.023600872434701395}},
#            "DisResNet": {"train_mean_metrics": {"Loss": 1.4825807547801386, "F1": 0.5482191262216578, "Acc": 0.5691779260468166, "Precision": 0.5901000052422195, "Recall": 0.5691779260468166, "BPR_F1": 0.7202813973297811, "FPR_F1": 0.3300081012133579, "Healthy_F1": 0.4171326177962225, "WBD_F1": 0.7096212908867781}, "val_mean_metrics": {"Loss": 1.6188842267816903, "F1": 0.5851207588707588, "Acc": 0.6063695563695563, "Precision": 0.625000690447119, "Recall": 0.6063695563695564, "BPR_F1": 0.6977705859671803, "FPR_F1": 0.39900099900099895, "Healthy_F1": 0.5489743589743589, "WBD_F1": 0.694819494231259}, "train_se_metrics": {"Loss": 0.08244988491556662, "F1": 0.017038047850870288, "Acc": 0.016613607421906023, "Precision": 0.019119766741262298, "Recall": 0.01661360742190602, "BPR_F1": 0.02004100815898391, "FPR_F1": 0.03441004651281485, "Healthy_F1": 0.03756870955354669, "WBD_F1": 0.014715165915305273}, "val_se_metrics": {"Loss": 0.07633226545889607, "F1": 0.015312852059871472, "Acc": 0.013881661633135511, "Precision": 0.018908747819505457, "Recall": 0.013881661633135528, "BPR_F1": 0.03690677963604218, "FPR_F1": 0.032776063004960464, "Healthy_F1": 0.04656893118531147, "WBD_F1": 0.02735595704508639}},
#             "ResNet18": {"train_mean_metrics": {"Loss": 1.795058268391226, "F1": 0.6172005746237982, "Acc": 0.6228601225619609, "Precision": 0.6764403341393723, "Recall": 0.622860122561961, "BPR_F1": 0.7217006134796367, "FPR_F1": 0.45293909215655725, "Healthy_F1": 0.5771030207250306, "WBD_F1": 0.6981829463527867}, "val_mean_metrics": {"Loss": 1.9390275902600371, "F1": 0.6485550831347355, "Acc": 0.6532153032153032, "Precision": 0.7101909994767137, "Recall": 0.6532153032153032, "BPR_F1": 0.7356065503124326, "FPR_F1": 0.5760139860139859, "Healthy_F1": 0.5604739704739703, "WBD_F1": 0.7087978619247969}, "train_se_metrics": {"Loss": 0.18667081037172442, "F1": 0.021661584022085536, "Acc": 0.019136368412923432, "Precision": 0.024007093581767017, "Recall": 0.019136368412923422, "BPR_F1": 0.01729918788361364, "FPR_F1": 0.03254700542377541, "Healthy_F1": 0.02317963311917654, "WBD_F1": 0.022614218346873636}, "val_se_metrics": {"Loss": 0.14386486198012813, "F1": 0.01639641870579217, "Acc": 0.014826507269436911, "Precision": 0.02110055555696848, "Recall": 0.014826507269436923, "BPR_F1": 0.0332269778172585, "FPR_F1": 0.04725257900397079, "Healthy_F1": 0.022457150752444643, "WBD_F1": 0.0215325750355198}},
#             # "efficientnet_b0": {"train_mean_metrics": {"Loss": 2.309359774679843, "F1": 0.874454768575925, "Acc": 0.8714413798620777, "Precision": 0.9050067728474133, "Recall": 0.8714413798620779, "BPR_F1": 0.8983359743232716, "FPR_F1": 0.8255625343968169, "Healthy_F1": 0.8657320336588276, "WBD_F1": 0.8932600299235048}, "val_mean_metrics": {"Loss": 3.1552665684467707, "F1": 0.6904469582445772, "Acc": 0.6922262922262921, "Precision": 0.7339669820919821, "Recall": 0.6922262922262921, "BPR_F1": 0.7298321123321123, "FPR_F1": 0.5659615384615384, "Healthy_F1": 0.7100213675213676, "WBD_F1": 0.7490820454055748}, "train_se_metrics": {"Loss": 0.21553292558684073, "F1": 0.03254619980435627, "Acc": 0.032704878967734455, "Precision": 0.026583551787776816, "Recall": 0.03270487896773447, "BPR_F1": 0.02446216620992823, "FPR_F1": 0.04387299092708682, "Healthy_F1": 0.04403322466659507, "WBD_F1": 0.02371684244615807}, "val_se_metrics": {"Loss": 0.22691300144705767, "F1": 0.027148792121150084, "Acc": 0.027073213844450248, "Precision": 0.030036624712527153, "Recall": 0.027073213844450248, "BPR_F1": 0.039272658230405696, "FPR_F1": 0.047996552480018044, "Healthy_F1": 0.03975489741262587, "WBD_F1": 0.041435304331877636}},
#             # "efficientnetV2-s": {"train_mean_metrics": {"Loss": 2.2575084927122484, "F1": 0.8902356324754619, "Acc": 0.8860672266756267, "Precision": 0.9181074667708968, "Recall": 0.8860672266756268, "BPR_F1": 0.9030008704802871, "FPR_F1": 0.8346702910490986, "Healthy_F1": 0.8928380073762916, "WBD_F1": 0.9107896195502233}, "val_mean_metrics": {"Loss": 3.323391774627898, "F1": 0.7059293803639042, "Acc": 0.707051282051282, "Precision": 0.7724171463457178, "Recall": 0.7070512820512821, "BPR_F1": 0.745501867413632, "FPR_F1": 0.6164085914085914, "Healthy_F1": 0.7201700423759247, "WBD_F1": 0.729126925705873}, "train_se_metrics": {"Loss": 0.17869478862703936, "F1": 0.02130521369064472, "Acc": 0.022004135989800742, "Precision": 0.01610404268938686, "Recall": 0.02200413598980075, "BPR_F1": 0.022108629734222483, "FPR_F1": 0.03337910158163515, "Healthy_F1": 0.01831247942299695, "WBD_F1": 0.020013214503844967}, "val_se_metrics": {"Loss": 0.2633412571483362, "F1": 0.021582871491258704, "Acc": 0.020588088785288894, "Precision": 0.02193612766773713, "Recall": 0.020588088785288894, "BPR_F1": 0.04272956735146777, "FPR_F1": 0.03974750277570198, "Healthy_F1": 0.037968999551482696, "WBD_F1": 0.03361057446206518}},
#            }

results = {
            "DisNet" : {"train_mean_metrics": {"Loss": 5.751327107954907, "F1": 0.5778870488498523, "Acc": 0.5891042096673547, "Precision": 0.6555103869427568, "Recall": 0.5891042096673547, "BPR_F1": 0.6283444389855397, "FPR_F1": 0.5128225913306519, "Healthy_F1": 0.5648154501580868, "WBD_F1": 0.6386317184655708}, 
                         "val_mean_metrics": {"Loss": 6.0806663032270905, "F1": 0.5487692190386425, "Acc": 0.551984126984127, "Precision": 0.600510179855418, "Recall": 0.551984126984127, "BPR_F1": 0.6616135531135531, "FPR_F1": 0.4392507492507492, "Healthy_F1": 0.5093164515223338, "WBD_F1": 0.5587587850745745}, 
                         "train_se_metrics": {"Loss": 0.05284516768793222, "F1": 0.03459359004957016, "Acc": 0.03148602063532409, "Precision": 0.031009032165963856, "Recall": 0.03148602063532408, "BPR_F1": 0.035209504558448755, "FPR_F1": 0.03478881107332921, "Healthy_F1": 0.036149664873384973, "WBD_F1": 0.02849833577183997}, 
                         "val_se_metrics": {"Loss": 0.06626393861925348, "F1": 0.02430925917055632, "Acc": 0.02265348256027852, "Precision": 0.018025793210340323, "Recall": 0.02265348256027853, "BPR_F1": 0.0422703676317821, "FPR_F1": 0.028855992803753925, "Healthy_F1": 0.054584917678838794, "WBD_F1": 0.029962658594638446}},
           
           "DisResNet": {"train_mean_metrics": {"Loss": 1.4825807547801386, "F1": 0.5482191262216578, "Acc": 0.5691779260468166, "Precision": 0.5901000052422195, "Recall": 0.5691779260468166, "BPR_F1": 0.7202813973297811, "FPR_F1": 0.3300081012133579, "Healthy_F1": 0.4171326177962225, "WBD_F1": 0.7096212908867781}, 
                         "val_mean_metrics": {"Loss": 1.6188842267816903, "F1": 0.5851207588707588, "Acc": 0.6063695563695563, "Precision": 0.625000690447119, "Recall": 0.6063695563695564, "BPR_F1": 0.6977705859671803, "FPR_F1": 0.39900099900099895, "Healthy_F1": 0.5489743589743589, "WBD_F1": 0.694819494231259}, 
                         "train_se_metrics": {"Loss": 0.08244988491556662, "F1": 0.017038047850870288, "Acc": 0.016613607421906023, "Precision": 0.019119766741262298, "Recall": 0.01661360742190602, "BPR_F1": 0.02004100815898391, "FPR_F1": 0.03441004651281485, "Healthy_F1": 0.03756870955354669, "WBD_F1": 0.014715165915305273}, 
                         "val_se_metrics": {"Loss": 0.07633226545889607, "F1": 0.015312852059871472, "Acc": 0.013881661633135511, "Precision": 0.018908747819505457, "Recall": 0.013881661633135528, "BPR_F1": 0.03690677963604218, "FPR_F1": 0.032776063004960464, "Healthy_F1": 0.04656893118531147, "WBD_F1": 0.02735595704508639}},
            "ResNet18": {"train_mean_metrics": {"Loss": 1.795058268391226, "F1": 0.6172005746237982, "Acc": 0.6228601225619609, "Precision": 0.6764403341393723, "Recall": 0.622860122561961, "BPR_F1": 0.7217006134796367, "FPR_F1": 0.45293909215655725, "Healthy_F1": 0.5771030207250306, "WBD_F1": 0.6981829463527867}, 
                         "val_mean_metrics": {"Loss": 1.9390275902600371, "F1": 0.6485550831347355, "Acc": 0.6532153032153032, "Precision": 0.7101909994767137, "Recall": 0.6532153032153032, "BPR_F1": 0.7356065503124326, "FPR_F1": 0.5760139860139859, "Healthy_F1": 0.5604739704739703, "WBD_F1": 0.7087978619247969}, 
                         "train_se_metrics": {"Loss": 0.18667081037172442, "F1": 0.021661584022085536, "Acc": 0.019136368412923432, "Precision": 0.024007093581767017, "Recall": 0.019136368412923422, "BPR_F1": 0.01729918788361364, "FPR_F1": 0.03254700542377541, "Healthy_F1": 0.02317963311917654, "WBD_F1": 0.022614218346873636}, 
                         "val_se_metrics": {"Loss": 0.14386486198012813, "F1": 0.01639641870579217, "Acc": 0.014826507269436911, "Precision": 0.02110055555696848, "Recall": 0.014826507269436923, "BPR_F1": 0.0332269778172585, "FPR_F1": 0.04725257900397079, "Healthy_F1": 0.022457150752444643, "WBD_F1": 0.0215325750355198}},
           }


# Extract mean and SE metrics
models = list(results.keys())

metrics_to_plot = ['Loss', 'F1', 'Acc', 'Precision', 'Recall', 'BPR_F1', 'FPR_F1', 'Healthy_F1', 'WBD_F1']

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

converted_results = {
    "train": {
        "DisNet": {"Loss": 5.751327107954907, "F1": 0.5778870488498523, "Acc": 0.5891042096673547, "Precision": 0.6555103869427568, "Recall": 0.5891042096673547, "BPR_F1": 0.6283444389855397, "FPR_F1": 0.5128225913306519, "Healthy_F1": 0.5648154501580868, "WBD_F1": 0.6386317184655708},
        "DisResNet": {"Loss": 1.4825807547801386, "F1": 0.5482191262216578, "Acc": 0.5691779260468166, "Precision": 0.5901000052422195, "Recall": 0.5691779260468166, "BPR_F1": 0.7202813973297811, "FPR_F1": 0.3300081012133579, "Healthy_F1": 0.4171326177962225, "WBD_F1": 0.7096212908867781},
        "ResNet18": {"Loss": 1.795058268391226, "F1": 0.6172005746237982, "Acc": 0.6228601225619609, "Precision": 0.6764403341393723, "Recall": 0.622860122561961, "BPR_F1": 0.7217006134796367, "FPR_F1": 0.45293909215655725, "Healthy_F1": 0.5771030207250306, "WBD_F1": 0.6981829463527867},
        "DisNet_se": {"Loss": 0.05284516768793222, "F1": 0.03459359004957016, "Acc": 0.03148602063532409, "Precision": 0.031009032165963856, "Recall": 0.03148602063532408, "BPR_F1": 0.035209504558448755, "FPR_F1": 0.03478881107332921, "Healthy_F1": 0.036149664873384973, "WBD_F1": 0.02849833577183997},
        "DisResNet_se": {"Loss": 0.08244988491556662, "F1": 0.017038047850870288, "Acc": 0.016613607421906023, "Precision": 0.019119766741262298, "Recall": 0.01661360742190602, "BPR_F1": 0.02004100815898391, "FPR_F1": 0.03441004651281485, "Healthy_F1": 0.03756870955354669, "WBD_F1": 0.014715165915305273},
        "ResNet18_se": {"Loss": 0.18667081037172442, "F1": 0.021661584022085536, "Acc": 0.019136368412923432, "Precision": 0.024007093581767017, "Recall": 0.019136368412923422, "BPR_F1": 0.01729918788361364, "FPR_F1": 0.03254700542377541, "Healthy_F1": 0.02317963311917654, "WBD_F1": 0.022614218346873636},
    },
    "val": {
        "DisNet": {"Loss": 6.0806663032270905, "F1": 0.5487692190386425, "Acc": 0.551984126984127, "Precision": 0.600510179855418, "Recall": 0.551984126984127, "BPR_F1": 0.6616135531135531, "FPR_F1": 0.4392507492507492, "Healthy_F1": 0.5093164515223338, "WBD_F1": 0.5587587850745745},
        "DisResNet": {"Loss": 1.6188842267816903, "F1": 0.5851207588707588, "Acc": 0.6063695563695563, "Precision": 0.625000690447119, "Recall": 0.6063695563695564, "BPR_F1": 0.6977705859671803, "FPR_F1": 0.39900099900099895, "Healthy_F1": 0.5489743589743589, "WBD_F1": 0.694819494231259},
        "ResNet18": {"Loss": 1.9390275902600371, "F1": 0.6485550831347355, "Acc": 0.6532153032153032, "Precision": 0.7101909994767137, "Recall": 0.6532153032153032, "BPR_F1": 0.7356065503124326, "FPR_F1": 0.5760139860139859, "Healthy_F1": 0.5604739704739703, "WBD_F1": 0.7087978619247969},
        "DisNet_se": {"Loss": 0.06626393861925348, "F1": 0.02430925917055632, "Acc": 0.02265348256027852, "Precision": 0.018025793210340323, "Recall": 0.02265348256027853, "BPR_F1": 0.0422703676317821, "FPR_F1": 0.028855992803753925, "Healthy_F1": 0.054584917678838794, "WBD_F1": 0.029962658594638446},
        "DisResNet_se": {"Loss": 0.07633226545889607, "F1": 0.015312852059871472, "Acc": 0.013881661633135511, "Precision": 0.018908747819505457, "Recall": 0.013881661633135528, "BPR_F1": 0.03690677963604218, "FPR_F1": 0.032776063004960464, "Healthy_F1": 0.04656893118531147, "WBD_F1": 0.02735595704508639},
        "ResNet18_se": {"Loss": 0.14386486198012813, "F1": 0.01639641870579217, "Acc": 0.014826507269436911, "Precision": 0.02110055555696848, "Recall": 0.014826507269436923, "BPR_F1": 0.0332269778172585, "FPR_F1": 0.04725257900397079, "Healthy_F1": 0.022457150752444643, "WBD_F1": 0.0215325750355198},
    },
}

# Metrics to plot
metrics = list(converted_results["train"]["DisResNet"].keys())
#drop loss
metrics.remove("Loss")

# Create subplots
# fig, axes = plt.subplots(1, 2, figsize=(15, 6))
def plot_data(key):
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Extract data
    disnet_data = [converted_results[key]["DisNet"][metric] for metric in metrics]
    disresnet_data = [converted_results[key]["DisResNet"][metric] for metric in metrics]
    ResNet18_data = [converted_results[key]["ResNet18"][metric] for metric in metrics]
    disnet_se = [converted_results[key]["DisNet_se"][metric] for metric in metrics]
    disresnet_se = [converted_results[key]["DisResNet_se"][metric] for metric in metrics]
    ResNet18_se = [converted_results[key]["ResNet18_se"][metric] for metric in metrics]
    
    # X-axis positions
    x = np.arange(len(metrics))
    
    # Bar width
    width = 0.2
    
    # Calculate the positions for each bar
    disnet_positions = x - width
    disresnet_positions = x
    ResNet18_positions = x + width
    
    # Plot bars
    ax.bar(disnet_positions, disnet_data, width, yerr=disnet_se, label='DisNet', color='black', capsize=10)  # Solid black color
    ax.bar(disresnet_positions, disresnet_data, width, yerr=disresnet_se, label='DisResNet', color='white', edgecolor='black', hatch='//', capsize=10)
    ax.bar(ResNet18_positions, ResNet18_data, width, yerr=ResNet18_se, label='ResNet18', color='white', edgecolor='black', capsize=10)
    
    # Labels
    ax.set_ylabel('Value ±1 SE')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45)
    
    # Legend with larger fontsize
    ax.legend(fontsize=18)

    # Remove top and right spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # Add multiple evenly spaced horizontal lines
    ymin, ymax = ax.get_ylim()
    for y in np.linspace(ymin, ymax, 9):  # 9 evenly spaced lines
        ax.axhline(y, color='gray', linestyle='--', linewidth=0.5)

    # Save plot
    plt.tight_layout()
    plt.savefig("/users/jrs596/scratch/" + f"{key}_metrics_comparison.png")
    
    # Show plot
    plt.tight_layout()
    plt.show()

# Plot train data
plot_data("train")
# Plot val data
plot_data("val")



#%%