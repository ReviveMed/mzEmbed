import matplotlib.pyplot as plt

def plot_metrics(metrics_df, metric_list):
    """
    Plots the training metrics over epochs with a separate axis for loss.
    
    Parameters:
    - metrics_df (pd.DataFrame): DataFrame containing metrics per epoch.
    """
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Plot loss on the primary y-axis
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Loss', color='tab:red')
    ax1.plot(metrics_df['Epoch'], metrics_df['Validation Loss'], color='tab:red', label='Validation Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.grid(True)

    # Create a secondary y-axis for the other metrics
    ax2 = ax1.twinx()  
    ax2.set_ylabel('Metric Value', color='tab:blue')

    # Plot other metrics on the secondary y-axis
    for metric in metric_list:
        ax2.plot(metrics_df['Epoch'], metrics_df[metric], label=metric)

    ax2.tick_params(axis='y', labelcolor='tab:blue')

    # Add a combined legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='center left', bbox_to_anchor=(1.15, 0.5))

    plt.title('Metrics and Validation Loss over Epochs')
    fig.tight_layout()  
    plt.show()
