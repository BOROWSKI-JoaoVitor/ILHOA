import os
import tifffile as tiff
import pandas as pd
import tkinter as tk
from tkinter import filedialog, simpledialog
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import csv
from scipy import stats
import statsmodels.api as sm

def scale_widths(df, scale_factor):
    """
    Apply a scale factor to width values in the dataframe.

    Parameters:
    df (pd.DataFrame): The dataframe containing width values in columns labeled with 'consec_count'.
    scale_factor (float): The scaling factor to be applied.

    Returns:
    pd.DataFrame: The dataframe with scaled width values.
    """
    for column in df.columns:
        if 'consec_count' in column:
            df[column] /= scale_factor ** 2
    return df

def calculate_area_variation(folder):
    """
    Calculate the area variation for each sample in the dataset.

    Parameters:
    folder (str): The folder path where the 'All_areas.csv' file is located.

    Saves:
    CSV file: 'All_areas_variation.csv' containing the area variations for each sample over time.
    """
    all_areas_path = os.path.join(folder, 'All_areas.csv')
    all_areas_df = pd.read_csv(all_areas_path)
    first_day = all_areas_df['time'].astype(int).min()
    all_areas_variation_df = pd.DataFrame(columns=['sample', 'time', 'area_variation'])

    rows = []
    for sample in all_areas_df['sample'].unique():
        sample_df = all_areas_df[all_areas_df['sample'] == sample]
        area0 = sample_df[sample_df['time'] == first_day]['total_area'].values[0]
        for index, row in sample_df.iterrows():
            area_variation = row['total_area'] - area0
            rows.append({'sample': row['sample'], 'time': row['time'], 'area_variation': area_variation})

    all_areas_variation_df = pd.DataFrame(rows)
    all_areas_variation_path = os.path.join(folder, 'All_areas_variation.csv')
    all_areas_variation_df.to_csv(all_areas_variation_path, index=False)

# Initialize Tkinter for folder and input dialogs
root = tk.Tk()
root.withdraw()

# Ask for scale factor
scale_factor = simpledialog.askfloat("Input", "Enter a scale factor:", parent=root)
folder = filedialog.askdirectory(title='Select the folder where the images are')

# Loop to process TIFF files
for filename in os.listdir(folder):
    if filename.endswith('.tif') or filename.endswith('.tiff'):
        img = tiff.imread(os.path.join(folder, filename))
        img_2d = img.reshape(-1, img.shape[-1])
        df = pd.DataFrame(img_2d)

        count_df = pd.DataFrame()
        for i, col in enumerate(df.columns):
            consec_count = 0
            col_counts = []
            for j, val in enumerate(df[col]):
                if val > 0:
                    consec_count += 1
                if val == 0 or j == len(df[col]) - 1:
                    if consec_count > 0:
                        col_counts.append(consec_count)
                    consec_count = 0

            if col_counts:
                for count_index, count in enumerate(col_counts):
                    count_df.loc[i, f'consec_count_{count_index}'] = count

        count_df.insert(0, 'column', count_df.index + 1)
        output_filename = os.path.splitext(filename)[0] + '_counts.csv'
        output_path = os.path.join(folder, output_filename)
        count_df.to_csv(output_path, index=False)

        count_df = pd.read_csv(output_path)
        scaled_count_df = scale_widths(count_df, scale_factor)
        scaled_output_path = os.path.join(folder, os.path.splitext(filename)[0] + '_counts_scaled.csv')
        scaled_count_df.to_csv(scaled_output_path, index=False)

def process_scaled_csv_files(folder):
    """
    Process scaled CSV files to generate area and width summaries.

    Parameters:
    folder (str): The folder path containing scaled CSV files.

    Saves:
    Two CSV files: 'All_areas.csv' and 'All_widths.csv' with processed data.
    """
    all_areas = pd.DataFrame(columns=['sample', 'time', 'total_area'])
    all_widths = pd.DataFrame(columns=['sample', 'time', 'width'])

    for filename in os.listdir(folder):
        if filename.endswith('_scaled.csv'):
            parts = filename.split('_')
            sample = parts[0]
            time = parts[1]

            df = pd.read_csv(os.path.join(folder, filename))
            total_area = df.filter(like='consec_count').sum().sum()
            all_areas = pd.concat([all_areas, pd.DataFrame({'sample': [sample], 'time': [time], 'total_area': [total_area]})], ignore_index=True)

            for index, row in df.iterrows():
                widths_df = pd.DataFrame({
                    'sample': [sample] * len(row.filter(like='consec_count').dropna()),
                    'time': [time] * len(row.filter(like='consec_count').dropna()),
                    'width': row.filter(like='consec_count').dropna().tolist()
                })
                all_widths = pd.concat([all_widths, widths_df], ignore_index=True)

    all_areas.to_csv(os.path.join(folder, 'All_areas.csv'), index=False)
    all_widths.to_csv(os.path.join(folder, 'All_widths.csv'), index=False)

process_scaled_csv_files(folder)
calculate_area_variation(folder)

def multiply_width_column():
    """
    Multiply the 'width' column by 1000 in the 'All_widths.csv' file.

    Saves:
    CSV file: 'All_widths_µm.csv' with the modified width values.
    """
    csv_file_path = os.path.join(folder, 'All_widths.csv')
    data = pd.read_csv(csv_file_path)
    data['width'] = data['width'] * 1000
    output_dir = os.path.dirname(csv_file_path)
    output_csv_path = os.path.join(output_dir, 'All_widths_µm.csv')
    data.to_csv(output_csv_path, index=False)

multiply_width_column()

# Determine first and last days from All_areas.csv
all_areas_df = pd.read_csv(os.path.join(folder, 'All_areas.csv'))
first_day = all_areas_df['time'].astype(int).min()
last_day = all_areas_df['time'].astype(int).max()

def create_and_save_violin_plot(csv_file_path, output_image_path, first_day, last_day):
    """
    Create and save a violin plot showing crack width distribution.

    Parameters:
    csv_file_path (str): Path to the CSV file containing width data.
    output_image_path (str): Path to save the violin plot image.
    first_day (int): The first time point to filter data.
    last_day (int): The last time point to filter data.

    Returns:
    sns.axisgrid.FacetGrid: The violin plot object.
    """
    data = pd.read_csv(csv_file_path)
    data = data[data['time'].isin([first_day, last_day])]
    column_of_interest = 'width'
    category_variable = 'time'
    x_variable = 'sample'

    sns.set_style('whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    violin = sns.violinplot(data=data, x=x_variable, y=column_of_interest, hue=category_variable, split=True, inner='quartile', ax=ax, cut=0)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Crack Width (µm)')
    ax.set_title('Crack Width Distribution')
    plt.savefig(output_image_path, format='jpeg', dpi=1200)
    plt.close()
    return violin

csv_file_path = f'{folder}/All_widths_µm.csv'
output_image_path = f'{folder}/Crack_Width_Distribution.jpeg'
violin_plot = create_and_save_violin_plot(csv_file_path, output_image_path, first_day, last_day)

def read_datasets_from_csv(filename):
    """
    Read datasets from a CSV file and group them by dataset names.

    Parameters:
    filename (str): Path to the CSV file.

    Returns:
    tuple: A tuple containing lists of datasets and dataset names.
    """
    datasets = []
    dataset_names = []
    current_dataset_name = None

    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header row

        for row in csv_reader:
            dataset_name = row[0] + row[1]
            data = float(row[2])

            if dataset_name != current_dataset_name:
                current_dataset_name = dataset_name
                current_dataset = []
                datasets.append(current_dataset)
                dataset_names.append(current_dataset_name)

            current_dataset.append(data)

    return datasets, dataset_names

def perform_mann_whitney_test(datasets):
    """
    Perform Mann-Whitney U tests on pairs of datasets.

    Parameters:
    datasets (list): List of datasets to compare.

    Returns:
    list: A list of tuples containing the test results, including statistics and effect sizes.
    """
    results = []

    for i in range(0, len(datasets), 2):
        dataset1 = datasets[i]
        dataset2 = datasets[i + 1]
        dataset1_name = dataset_names[i]
        dataset2_name = dataset_names[i + 1]

        statistic, p_value = stats.mannwhitneyu(dataset1, dataset2, alternative='two-sided')
        n1 = len(dataset1)
        n2 = len(dataset2)
        u1 = statistic
        cles = u1 / (n1 * n2)
        pooled_std = np.sqrt((np.var(dataset1) * (n1 - 1) + np.var(dataset2) * (n2 - 1)) / (n1 + n2 - 2))
        cohens_d = (np.mean(dataset1) - np.mean(dataset2)) / pooled_std
        results.append((dataset1_name, dataset2_name, statistic, p_value, cles, cohens_d))

    return results

def save_results_to_file(filename, results):
    """
    Save the statistical analysis results to a file.

    Parameters:
    filename (str): The path to the output file.
    results (list): The list of statistical results to save.
    """
    with open(filename, 'w') as file:
        file.write("Statistical Analysis\n\n")
        for i, result in enumerate(results, start=1):
            dataset1_name, dataset2_name, statistic, p_value, cles, cohens_d = result
            file.write(f"Comparison {i}:\n")
            file.write(f"Dataset 1: {dataset1_name}\n")
            file.write(f"Dataset 2: {dataset2_name}\n")
            file.write(f"Mann-Whitney U statistic: {statistic}\n")
            file.write(f"P-value: {p_value}\n")
            file.write(f"CLES: {cles}\n")
            file.write(f"Cohen's d: {cohens_d}\n\n")

if csv_file_path:
    datasets, dataset_names = read_datasets_from_csv(csv_file_path)
    results = perform_mann_whitney_test(datasets)
    directory = os.path.dirname(csv_file_path)
    output_filename = os.path.join(directory, "Statistical Analysis " + os.path.basename(csv_file_path).replace(".csv", ".txt"))
    save_results_to_file(output_filename, results)
else:
    print("No file selected.")

def plot_area_data(csv_file_path, first_day, last_day):
    """
    Plot area data for line and scatter plots, grouping samples based on numeration.

    Parameters:
    csv_file_path (str): Path to the CSV file containing area data.
    first_day (int): First day time point.
    last_day (int): Last day time point.

    Saves:
    JPEG images of line and scatter plots showing data analysis.
    """
    data = pd.read_csv(csv_file_path)
    directory = os.path.dirname(csv_file_path)

    def group_sample(sample):
        parts = sample.split('-')
        return parts[0] if parts[1].isdigit() else sample

    data['group'] = data['sample'].apply(group_sample)
    data['time'] = pd.Categorical(data['time'], ordered=True)

    plt.figure(figsize=(10, 6))
    line_data = data.groupby(['time', 'group']).agg(mean_area=('total_area', 'mean'), se_area=('total_area', 'sem')).reset_index()

    sns.lineplot(
        data=line_data,
        x='time',
        y='mean_area',
        hue='group',
        style='group',
        markers=True,
        markersize=8,
        dashes=False,
        err_style='bars',
        errorbar=None
    )

    for name, group in line_data.groupby('group'):
        plt.errorbar(
            group['time'],
            group['mean_area'],
            yerr=group['se_area'],
            fmt='none',
            capsize=5,
            color=sns.color_palette("deep")[line_data['group'].unique().tolist().index(name)]
        )

    plt.xlabel('Days')
    plt.ylabel('Self-healing (%)')
    plt.title('Self-healing')
    plt.legend(title='Group')
    plt.grid(True)
    output = os.path.join(folder, 'Line Plot.jpeg')
    plt.savefig(output, dpi=1200, bbox_inches='tight')
    plt.close()

    filtered_data = data[(data['time'] == first_day) | (data['time'] == last_day)]
    pivot_data = filtered_data.pivot(index='sample', columns='time', values='total_area').dropna()

    unique_groups = pivot_data.reset_index()['sample'].apply(lambda x: x.split('-')[0]).unique()
    palette = sns.color_palette("deep", len(unique_groups))
    color_map = dict(zip(unique_groups, palette))

    plt.figure(figsize=(10, 6))
    for group, color in color_map.items():
        group_data = pivot_data[pivot_data.index.str.startswith(group)]
        sns.scatterplot(x=group_data[first_day], y=group_data[last_day], label=group, color=color)

    max_val = max(pivot_data[first_day].max(), pivot_data[last_day].max()) * 1.05
    plt.plot([0, max_val], [0, max_val], 'r-', label='ΔArea = 0')

    ref_data = pivot_data[pivot_data.index.str.contains('REF')]
    if not ref_data.empty:
        X_ref = sm.add_constant(ref_data[first_day])
        model_ref = sm.OLS(ref_data[last_day], X_ref).fit()
        predictions_ref = model_ref.get_prediction(X_ref).summary_frame(alpha=0.05)
        plt.plot(ref_data[first_day], predictions_ref['obs_ci_lower'], 'r--', label='REF Lower 95% CI')

    plt.xlabel('Initial Area')
    plt.ylabel('Final Area')
    plt.title('Type of Self-healing')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    output = os.path.join(folder, 'Scatter Plot.jpeg')
    plt.savefig(output, dpi=1200, bbox_inches='tight')
    plt.close()

plot_area_data(os.path.join(folder, 'All_areas.csv'), first_day, last_day)

def plot_area_variation_data(csv_file_path, first_day, last_day):
    """
    Plot area variation data with line plots, grouping samples based on numeration.

    Parameters:
    csv_file_path (str): Path to the CSV file containing area variation data.
    first_day (int): First day time point.
    last_day (int): Last day time point.

    Saves:
    JPEG images of the line plot showing area variation analysis.
    """
    data = pd.read_csv(csv_file_path)
    directory = os.path.dirname(csv_file_path)

    def group_sample(sample):
        parts = sample.split('-')
        return parts[0] if parts[1].isdigit() else sample

    data['group'] = data['sample'].apply(group_sample)
    data['time'] = pd.Categorical(data['time'], ordered=True)

    plt.figure(figsize=(10, 6))
    line_data = data.groupby(['time', 'group']).agg(mean_area=('area_variation', 'mean'), se_area=('area_variation', 'sem')).reset_index()

    sns.lineplot(
        data=line_data,
        x='time',
        y='mean_area',
        hue='group',
        style='group',
        markers=True,
        markersize=8,
        dashes=False,
        err_style='bars',
        errorbar=None
    )

    for name, group in line_data.groupby('group'):
        plt.errorbar(
            group['time'],
            group['mean_area'],
            yerr=group['se_area'],
            fmt='none',
            capsize=5,
            color=sns.color_palette("deep")[line_data['group'].unique().tolist().index(name)]
        )

    plt.xlabel('Days')
    plt.ylabel('Crack Area Variation (mm²)')
    plt.title('Area Variation')
    plt.legend(title='Group')
    plt.grid(True)
    output = os.path.join(folder, 'Line Plot Variation.jpeg')
    plt.savefig(output, dpi=1200, bbox_inches='tight')
    plt.close()

plot_area_variation_data(os.path.join(folder, 'All_areas_variation.csv'), first_day, last_day)

def plot_bar_graph(csv_file_path):
    """
    Create a bar graph comparing initial and final crack areas.

    Parameters:
    csv_file_path (str): Path to the CSV file containing area data.

    Saves:
    JPEG image of the bar graph comparing initial and final areas.
    """
    data = pd.read_csv(csv_file_path)

    first_day = data['time'].astype(int).min()
    last_day = data['time'].astype(int).max()

    first_day_data = data[data['time'] == first_day].set_index('sample')
    last_day_data = data[data['time'] == last_day].set_index('sample')

    merged_data = first_day_data[['total_area']].join(last_day_data[['total_area']], lsuffix='_initial', rsuffix='_final')

    merged_data.reset_index(inplace=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.35
    x = range(len(merged_data))

    colors = sns.color_palette('deep', 2)
    ax.bar(x, merged_data['total_area_initial'], width=bar_width, label=str(first_day), color=colors[0])
    ax.bar([p + bar_width for p in x], merged_data['total_area_final'], width=bar_width, label=str(last_day), color=colors[1])

    ax.set_xlabel('Sample')
    ax.set_ylabel('Area (mm²)')
    ax.set_title('Crack Area - Initial vs Final')
    ax.set_xticks([p + bar_width / 2 for p in x])
    ax.set_xticklabels(merged_data['sample'], rotation=45, ha='right')
    ax.legend()

    output_path = os.path.join(os.path.dirname(csv_file_path), 'Bar Graph - Initial vs Final.jpeg')
    plt.savefig(output_path, format='jpeg', dpi=1200)
    plt.close()

plot_bar_graph(os.path.join(folder, 'All_areas.csv'))
