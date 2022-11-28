# Geohack2022-geouser02-TriloBYTES

Dataset: [Project Penobscot](https://terranubis.com/datainfo/Penobscot) 

## Data Extraction

Data is extracted using ```segysak```, read iline at byte 21, xline at 9 and so on. ```segysak``` is preferred over ```segyio``` as it is able to
1. infill missing offset gathers with ```np.nan```
2. return data in non-flatten format: (iline, xline, twt, offset)

```python
import segysak.segy as sg

data = sg.segy_loader(seis_file, iline=21, xline=9, cdpx=73, cdpy=77, offset=37)
data.data.shape
```

```
(40, 482, 1501, 61)

```

## Data Visualization
Visualizing IL1081, first offset of the seismic cross section:

```python
vmax = np.percentile(P_i1081, 97)
plt.imshow(P_i1081[0, :, :, 0].T, aspect=0.4, cmap='seismic_r', vmin=-vmax, vmax=vmax)
plt.show()
```


## Modelling


