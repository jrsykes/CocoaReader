#%%

import matplotlib.pyplot as plt
import numpy as np



results = {
            "DisNet": {"train_mean_metrics": {"Loss": 0.9281933787730269, "F1": 0.600684234443453, "Acc": 0.6161512679440967, "Precision": 0.6525210785485165, "Recall": 0.6161512679440967, "BPR_F1": 0.7618139273760405, "FPR_F1": 0.5011373373080215, "Healthy_F1": 0.5473526625665663, "WBD_F1": 0.6306011498251838}, 
                          "val_mean_metrics": {"Loss": 1.1659505471587182, "F1": 0.6084482646982647, "Acc": 0.6128306878306877, "Precision": 0.6392422524565381, "Recall": 0.6128306878306877, "BPR_F1": 0.717784421460892, "FPR_F1": 0.5109915084915085, "Healthy_F1": 0.5425869383454522, "WBD_F1": 0.653048004626952}, 
                          "train_se_metrics": {"Loss": 0.05621062229527728, "F1": 0.031620601310700186, "Acc": 0.028339884283216106, "Precision": 0.03172675959326852, "Recall": 0.028339884283216103, "BPR_F1": 0.023365469653792804, "FPR_F1": 0.031033986257765506, "Healthy_F1": 0.0554719380259939, "WBD_F1": 0.022211053687192877}, 
                          "val_se_metrics": {"Loss": 0.049918904249324464, "F1": 0.02538295701181918, "Acc": 0.02523166784483935, "Precision": 0.025335282690242544, "Recall": 0.02523166784483936, "BPR_F1": 0.04443952913902344, "FPR_F1": 0.03204368320839922, "Healthy_F1": 0.03798668929722697, "WBD_F1": 0.04985587355941246}},
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
plt.savefig('/home/userfs/j/jrs596/scripts/CocoaReader/CocoaNet/IR-RGB/cross-validation/resultsV1.2.png', dpi=300, bbox_inches='tight')
plt.show()

#%%

converted_results = {
    "Train": {
        "ResNet18_RGB": {
        "Loss": 2.0079982290665312, 
        "F1": 0.6367080329700715, 
        "Acc": 0.6468253968253967, 
        "Precision": 0.7039135005206434, 
        "Recall": 0.6468253968253969, 
        "BPR_F1": 0.5797487462459381, 
        "FPR_F1": 0.5328822248627441, 
        "Healthy_F1": 0.6483699156552666, 
        "WBD_F1": 0.7891818332712817
    },
        "ResNet18": {"Loss": 1.795058268391226, "F1": 0.6172005746237982, "Acc": 0.6228601225619609, "Precision": 0.6764403341393723, "Recall": 0.622860122561961, "BPR_F1": 0.7217006134796367, "FPR_F1": 0.45293909215655725, "Healthy_F1": 0.5771030207250306, "WBD_F1": 0.6981829463527867},
        "DisNet": {"Loss": 0.9281933787730269, "F1": 0.600684234443453, "Acc": 0.6161512679440967, "Precision": 0.6525210785485165, "Recall": 0.6161512679440967, "BPR_F1": 0.7618139273760405, "FPR_F1": 0.5011373373080215, "Healthy_F1": 0.5473526625665663, "WBD_F1": 0.6306011498251838},
        "ResNet18_RGB_se": {
            "Loss": 0.08562690483893776, 
            "F1": 0.0330616770519115, 
            "Acc": 0.030558132235203896, 
            "Precision": 0.032070979986040915, 
            "Recall": 0.030558132235203896, 
            "BPR_F1": 0.03647485904812824, 
            "FPR_F1": 0.04580776133143093, 
            "Healthy_F1": 0.0323160122178069, 
            "WBD_F1": 0.03111317120090422
        },
        "ResNet18_se": {"Loss": 0.18667081037172442, "F1": 0.021661584022085536, "Acc": 0.019136368412923432, "Precision": 0.024007093581767017, "Recall": 0.019136368412923422, "BPR_F1": 0.01729918788361364, "FPR_F1": 0.03254700542377541, "Healthy_F1": 0.02317963311917654, "WBD_F1": 0.022614218346873636},
        "DisNet_se": {"Loss": 0.05621062229527728, "F1": 0.031620601310700186, "Acc": 0.028339884283216106, "Precision": 0.03172675959326852, "Recall": 0.028339884283216103, "BPR_F1": 0.023365469653792804, "FPR_F1": 0.031033986257765506, "Healthy_F1": 0.0554719380259939, "WBD_F1": 0.022211053687192877},
    },
    "Val": {
        "ResNet18_RGB": {
        "Loss": 2.3661134123802183, 
        "F1": 0.6002210686139258, 
        "Acc": 0.6035714285714284, 
        "Precision": 0.6573299319727892, 
        "Recall": 0.6035714285714285, 
        "BPR_F1": 0.579387306879567, 
        "FPR_F1": 0.4368614718614718, 
        "Healthy_F1": 0.5813322135380959, 
        "WBD_F1": 0.7555106658047834
        },
        "ResNet18": {"Loss": 1.9390275902600371, "F1": 0.6485550831347355, "Acc": 0.6532153032153032, "Precision": 0.7101909994767137, "Recall": 0.6532153032153032, "BPR_F1": 0.7356065503124326, "FPR_F1": 0.5760139860139859, "Healthy_F1": 0.5604739704739703, "WBD_F1": 0.7087978619247969},
        "DisNet": {"Loss": 1.1659505471587182, "F1": 0.6084482646982647, "Acc": 0.6128306878306877, "Precision": 0.6392422524565381, "Recall": 0.6128306878306877, "BPR_F1": 0.717784421460892, "FPR_F1": 0.5109915084915085, "Healthy_F1": 0.5425869383454522, "WBD_F1": 0.653048004626952},
        "ResNet18_RGB_se": {
            "Loss": 0.049121769845597976, 
            "F1": 0.02100376514830497, 
            "Acc": 0.02284039565045639, 
            "Precision": 0.02594223952848782, 
            "Recall": 0.022840395650456407, 
            "BPR_F1": 0.028511910854116206, 
            "FPR_F1": 0.06615513292000817, 
            "Healthy_F1": 0.047415669823311524, 
            "WBD_F1": 0.03213875230096476
        },
        "ResNet18_se": {"Loss": 0.14386486198012813, "F1": 0.01639641870579217, "Acc": 0.014826507269436911, "Precision": 0.02110055555696848, "Recall": 0.014826507269436923, "BPR_F1": 0.0332269778172585, "FPR_F1": 0.04725257900397079, "Healthy_F1": 0.022457150752444643, "WBD_F1": 0.0215325750355198},
        "DisNet_se": {"Loss": 0.049918904249324464, "F1": 0.02538295701181918, "Acc": 0.02523166784483935, "Precision": 0.025335282690242544, "Recall": 0.02523166784483936, "BPR_F1": 0.04443952913902344, "FPR_F1": 0.03204368320839922, "Healthy_F1": 0.03798668929722697, "WBD_F1": 0.04985587355941246},
    },
}


#%%

# Metrics to plot
metrics = list(converted_results["Train"]["DisNet"].keys())
#drop Loss
metrics.remove("Loss")

def plot_data(key):
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Extract data
    disnet_data = [converted_results[key]["DisNet"][metric] for metric in metrics]
    ResNet18_data = [converted_results[key]["ResNet18"][metric] for metric in metrics]
    ResNet18_RGB_data = [converted_results[key]["ResNet18_RGB"][metric] for metric in metrics]
    disnet_se = [converted_results[key]["DisNet_se"][metric] for metric in metrics]
    ResNet18_se = [converted_results[key]["ResNet18_se"][metric] for metric in metrics]
    ResNet18_RGB_se = [converted_results[key]["ResNet18_RGB_se"][metric] for metric in metrics]
    
    # X-axis positions
    x = np.arange(len(metrics))
    
    # Bar width (Reduced width for closer bars)
    width = 0.3
    
    # Define the additional gap between groups of bars
    group_gap = 0.4  # Adjust this value to increase/decrease the gap between groups
    
    # Adjust the x positions of each group by adding multiples of group_gap
    adjusted_x = x + np.arange(len(x)) * group_gap
    
    # Calculate the positions for each bar with the adjusted x positions
    disnetV1_2_positions = adjusted_x - width * 0.5
    ResNet18_positions = adjusted_x + width * 0.5
    ResNet18_RGB_positions = adjusted_x + width * 1.5
    
    
    # Plot bars
    ax.bar(disnetV1_2_positions, disnet_data, width, yerr=disnet_se, label='DisNet IR', color='gray', capsize=8)
    ax.bar(ResNet18_positions, ResNet18_data, width, yerr=ResNet18_se, label='ResNet18 IR', color='white', edgecolor='black', capsize=10, hatch='/')  # Added hatch pattern
    ax.bar(ResNet18_RGB_positions, ResNet18_RGB_data, width, yerr=ResNet18_RGB_se, label='ResNet18 RGB', color='black', capsize=8)
    # Labels
    ax.set_ylabel(key +' metric value ±1 SE', fontsize=16)
    ax.set_xticks(adjusted_x)
    ax.set_xticklabels(metrics, rotation=45)
    
    left_limit = disnetV1_2_positions[0] - (group_gap / 2)
    right_limit = ResNet18_RGB_positions[-1] + (group_gap / 2)
    ax.set_xlim(left_limit, right_limit)
    
    # Legend with larger fontsize
    ax.legend(fontsize=18, loc='upper left', bbox_to_anchor=(0.03, 1.15))

    # Remove top and right spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # Add multiple evenly spaced horizontal lines
    ymin, ymax = ax.get_ylim()
    for y in np.linspace(ymin, 0.8, 9):
        ax.axhline(y, color='gray', linestyle='--', linewidth=0.5)

    # Save plot
    plt.tight_layout()
    plt.savefig("/home/userfs/j/jrs596/scripts/CocoaReader/CocoaNet/IR-RGB/cross-validation/" + f"{key}_metrics_comparison_V1.2.png")
    
    # Show plot
    plt.tight_layout()
    plt.show()

# Plot train data
plot_data("Train")
# Plot val data
plot_data("Val")



#%%