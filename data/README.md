# Data

Beijing Multi-Site Air Quality data set (UCI Machine Learning Repository, https://doi.org/10.24432/C5RK5G):
hourly measurements of twelve stations, 2013-03-01 – 2017-02-28, 420,768 records. The data set is not redistributed.

Download the archive into this folder:

    https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip

or set the environment variable SRATTELM_DATA to the archive or to a folder with the PRSA_Data_*.csv files.
`srattelm/protocol.py` verifies the SHA-256 of the loaded (35064, 12, 4) array of PM2.5, PM10, SO2 and NO2.
