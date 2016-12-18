# cfd_results_python
Script for collecting CFD results from output files to dataframes and plotting them

While doing Computational Fluid Dynamics (CFD) analyses it is a typical problem to collect results from output files and compare them for the various geometries and load cases that were run.
This script demonstrates how this can be automated using Python. The output files from cfd results are organized to a directory structure (v00, v01, etc.). The script then:
  reads the output files with the results
  collects them to Pandas dataframes
  creates plots
  writes the data to an Excel sheet
The example shows some typical data extracted from underhood flow simulations (heat exchanger data, part temperatures).
