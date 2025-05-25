from fire import Fire
import matplotlib.pyplot as plt

from dataset import get_dataset
from plot_results import print_results, plot_results


def main(
        # Results dir
        results_dir,  # where you store the trackers' results
        # Choice
        to_print_results=False,  # whether to print trackers' results
        to_plot_results=False,  # whether to plot the trackers' results with matplotlib
        # Plot parameters
        plot_types=('success', 'prec', 'norm_prec'),
        # Save options
        save_result_plot=False,  # whether to save the result plot
        save_result_plot_dir=''  # where to save the result plot

):
    plt.rcParams['figure.figsize'] = [12, 12]
    dataset = get_dataset('vasttrack_test')

    if to_plot_results:
        plot_results(results_dir, dataset, 'VastTrack', plot_types=plot_types, skip_missing_seq=False,
                     force_evaluation=True, plot_bin_gap=0.05,
                     save_result_plot=save_result_plot, save_result_plot_dir=save_result_plot_dir)
    if to_print_results:
        print_results(results_dir, dataset, merge_results=False, plot_types=plot_types)


if __name__ == '__main__':
    Fire(main)

