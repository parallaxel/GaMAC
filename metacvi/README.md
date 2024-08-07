# CVI predictor based on meta-learning

### How to use it
1) Find dataset **\<data-name>** with numeric features in csv format, put it inside */data/<data-name>/orig.csv*
2) Run [generator.py](generator.py) specifying **\<data-name>**. 
It will generate many partitions, but choose only **PARTITIONS_TO_ESTIMATE** candidates for further procedures.
This step produces subfolders *data/<data-name>/\<reducer>* with following contents:
   - **gen.csv** - reduced 2D representation of **orig.csv**, obtained by applying **\<reducer>**
   - **img-\<index>.png** - scatter plot of partition **\<index>**
   - **partitions.csv** - **\<index>**'th row defines **\<index>**'th partition labels
   - **producers.json** - **\<index>**'th element describes clustering algo, which produces **\<index>**'th partition
3) After you generate partitions, you can launch [assessment gui](gui.py) to compare partitions. 
Make sure, to assign your unique **ACCESSOR_IDX** before start (should be non-negative integer).
To estimate single 2D representation partitions, assessment flow goes the following way:
    - There are already sorted list of partitions scatter plots, which is initialised by img-0.png
    - Comparison is pairwise between current candidate (red frame) and candidate from sorted list (blue frame)
    - To switch between current and sorted candidates, click **TAB**
    - To treat current candidate as better than candidate from sorted list, click **ENTER**.
    After that, current candidate is inserted into sorted list and new current candidate is rendered.
    - To treat current candidate as worse than candidate from sorted list, click **BACKSPACE**.
    After that, the next worse candidate from sorted list will be rendered
    - Comparisons repeat until all **PARTITIONS_TO_ESTIMATE** candidates will be inserted into sorted list.
    After that you will see green frame, while the next assessment iteration is loading

    This step produces *data/\<data-name>/\<reducer>/accessor-\<ACCESSOR_IDX>.txt*
4) Run [dataset.py](dataset.py) to generate pre-meta-dataframe with distances between measures and accessors orderings.
This dataframe contains references to data only, without numerical meta-features.
Note, that you can integrate arbitrary amount of internal measures here without re-estimating by accessors.
This step produces *pre-meta-dataset.json* for all accessor-estimated data. 
5) Run [features.py](features.py) to build meta-features for each 2D data representation.
Note, that you can build another meta-description here without re-estimating by accessors.
This step produces *data/\<data-name>/\<reducer>/features.txt* for all data.
6) Run [predictor.py](predictor.py) to build meta-classifier (meta-regressor) for meta-data obtained in step 4, 5
