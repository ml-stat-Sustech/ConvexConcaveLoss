


import matplotlib.pyplot as plt
import seaborn as sns

def plot_acc_auc(df_ce_ls, loss_type):
    if df_ce_ls.loc[0,'Train Acc'] > 1:
        df_ce_ls[["Train Acc","Test Acc"]] = df_ce_ls[["Train Acc","Test Acc"]]/100
    
    plt.figure(figsize=(10, 6))

    sns.lineplot(x='Hyper parameter', y='Train Acc', data=df_ce_ls, label='Train Acc')
    sns.lineplot(x='Hyper parameter', y='Test Acc', data=df_ce_ls, label='Test Acc')
    sns.lineplot(x='Hyper parameter', y='modified entropy test auc', data=df_ce_ls, label='modified entropy test auc')

    plt.xlabel(f'{loss_type}: Hyper parameter')
    plt.ylabel('Acc and AUC')
    plt.title('Train Acc and Test Acc and AUC vs. Hyper parameter')
    plt.legend()
    plt.grid(True)
    plt.show()
    



def plot_distribution_subplots(df, losstype , oject):
    df.sort_values(by='Hyper parameter', inplace=True)
    
    plt.figure(figsize=(16, 6))  # Increase the figure width to fit two subplots side by side
    # First subplot
    

    plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
    sns.lineplot(x='Hyper parameter', y=f'{oject}_train_mean', data=df, label=f'{oject}_train_mean')
    sns.lineplot(x='Hyper parameter', y=f'{oject}_test_mean', data=df, label=f'{oject}_test_mean')
    plt.xlabel(f'{losstype}: Hyper parameter')
    plt.ylabel(f'{oject} Mean')
    plt.title(f'{oject} mean vs. Hyper parameter')
    plt.legend()
    plt.grid(True)
    # Second subplot
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
    sns.lineplot(x='Hyper parameter', y=f'{oject}_train_variance', data=df, label=f'{oject}_train_variance')
    sns.lineplot(x='Hyper parameter', y=f'{oject}_test_variance', data=df, label=f'{oject}_test_variance')
    plt.xlabel(f'{losstype}: Hyper parameter')
    plt.ylabel(f'{oject} variance')
    plt.title(f'{oject} variance vs. Hyper parameter')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()  # To avoid overlapping labels and titles
    plt.show()

def plot_scatter_with_lines(dataframes, labels):
    """
    Plot a scatter plot with connecting lines for multiple DataFrames.

    Parameters:
    dataframes (list of pandas DataFrames): A list of DataFrames to plot.
    labels (list of str): Labels for each DataFrame.

    Returns:
    None (displays the plot).
    """
    if len(dataframes) != len(labels):
        raise ValueError("The number of DataFrames should be equal to the number of labels.")

    # Sort each DataFrame by 'modified entropy test auc'
    sorted_dataframes = [df.sort_values(by='modified entropy test auc') for df in dataframes]

    # Plotting the data points
    plt.figure(figsize=(8, 6))
    for i, df in enumerate(sorted_dataframes):
        label = labels[i]
        plt.plot(df['modified entropy test auc'], df['Test Acc'], marker='^', linestyle='-', label=label)

        

    plt.xlabel('Modified Entropy Test AUC')
    plt.ylabel('Test Accuracy')
    plt.title('Scatter Plot with Connecting Lines: Test Accuracy vs. Modified Entropy Test AUC')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    