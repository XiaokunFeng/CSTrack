# Python Evaluation Toolkit for VastTrack
This is the official Python evaluation toolkit for [VastTrack: Vast Category Visual Object Tracking](https://arxiv.org/abs/2403.03493).
## News
**[Mar. 28, 2024]**
- Python evaluation toolkit for VastTrack is now available, which you can use to conveniently test your tracker on VastTrack.

## Configuration
First, make sure you have downloaded our VastTrack dataset and the test set list we provided.
Then, as shown below, modify line 7 in dataset.py to the path of the VastTrack dataset and the path of the test set list.
```
dataset_dict = dict(
    vasttrack=DatasetInfo(module="vasttrack", class_name="VastTrackDataset", kwargs=dict(base_path="/path/to/vasttrack", txt_file_path="/path/to/vasttrack_test_set_list.txt"))
)
```
## Trackers Results Preparation
Format the trackers' results. They should look like this:
   ```
    -- trackers_results
        -- SeqTrack
            |-- Aardwolf-1.txt
            |-- Aardwolf-2.txt
            ...
        -- OSTrack
            |-- Aardwolf-1.txt
            |-- Aardwolf-2.txt
            ...
        -- STARK
            |-- Aardwolf-1.txt
            |-- Aardwolf-2.txt
            ...
        -- ARTrack
            |-- Aardwolf-1.txt
            |-- Aardwolf-2.txt
            ...
        ...
   ```
## Usage
Our toolkit supports two functions: 

1) Simply print the trackers' Success rate, Precision, Normalized Precision on VastTrack.
```
python analysis_results.py --results_dir /path/to/trackers/results --to_print_results True
```
2) Genarate plots of trackers' Success rate, Precision, Normalized Precision on VastTrack.
```
python analysis_results.py --results_dir /path/to/trackers/results --to_plot_results True --save_result_plot True --save_result_plot_dir ./
```

## Acknowledgments
* Thanks for the [PyTracking](https://github.com/visionml/pytracking) library, which helps us to build our toolkit.


