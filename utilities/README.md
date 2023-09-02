# What is this?
These are utilities that are used to clean up and examine files.
If a script is not described here, you will have to read through it yourself to understand what it does, or it may be documented in README found one level up (`../README.md`) or by the code that uses it.

## fix_duplicate_texts (.py, .bat, .sh)
If there are duplicate texts for a dataset, merge the texts into one singular answer.
If there are conflicts in this merge (the same annotator labeled the same text in multiple different ways), then pick one of the ratings at random.

## double_dataset.py and double_SBIC.bat
Have a dataset like SBIC that decided that using halves like 0.5 or 1.5 was okay?
Use these scripts to double those datasets so that all of the ratings have integer values.
`double_SBIC.bat` just uses `double_dataset.py` on the SBIC dataset.

## not_too_sparse.py
Makes sure that an annotation file has data for every row and column. If it doesn't have data for a column, it removes this column.
It doesn't fix row stuff because none of our datasets have that issue anymore after being cleaned, and it would require editing the text file as well as the annotations file.

## test_same_files.py
Checks to see if two npy files contain the exact same data

## check_duplicate_texts (.py and .sh)
Checks to see if there are texts that are duplicated in a dataset. If there are, you can use `fix_duplicate_texts` (see above).

## remove_cleaned_suffix
After using `fix_duplicate_texts` in `utilities` you get a lot of files with the `_cleaned` suffix. This helps with removing those.

## npy_to_csv
Converts Numpy files to CSV files so that they're human-readable.

## get_dataset_shape.py
Quick check to see the shape of a npy dataset.

## display_realtime_file.py
Displays a file in real time. Useful when running scripts using `nohup` on Linux, where all the output is sent to a file, and you want to see the progress of the script in real time.