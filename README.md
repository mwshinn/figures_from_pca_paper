# Code for [Phantom oscillations in principal component analysis](https://www.pnas.org/doi/10.1073/pnas.2311420120)

Generate the figures in my paper [Phantom oscillations in principal component
analysis](https://www.pnas.org/doi/10.1073/pnas.2311420120).

## Requirements

Standard scientific Python stack:

    pip install numpy scipy scikit-learn matplotlib seaborn statsmodels scikit-image imageio

[CanD](https://github.com/mwshinn/CanD) for pretty scientific figures:

    pip install cand

[Colorednoise](https://github.com/felixpatzelt/colorednoise) for easy 1/f noise simulations:

    pip install colorednoise
    
[Networkx](https://networkx.org/) for the branching manifold:

    pip install networkx
    
[Wbplot](https://github.com/jbburt/wbplot) for rendering the MRI images:

    pip install wbplot

I used Python 3.6 but it should work on a more modern version.

## Running

Run each Python script from the terminal.  After running all scripts, edit the
"REVERSE = False" line to be "REVERSE = True" to generate the remaining figures.

Missing data files can in theory be rendered by uncommenting the commented-out
sections of each file, but that requires installing more software (specifically
[this for fMRI](https://github.com/murraylab/spatial_and_temporal_paper) and
[this for NHP
electrophysiology](https://github.com/mwshinn/figures_from_dip_paper)) so I
uploaded the pre-generated data.

[Data from Roitman and Shadlen (2002) is available from the authors](https://shadlenlab.columbia.edu/resources/RoitmanDataCode.html).

## Copyright

All code copyright 2024 Max Shinn, available under the GNU GPLv3.

ames-task.png is copyright Ames and Churchland 2019 from [their eLife
paper](https://elifesciences.org/articles/46159).
