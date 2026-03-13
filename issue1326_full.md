title:	Feature request: Vectorize search_lightcurve API 
state:	OPEN
author:	jkrick
labels:	:athletic_shoe: sprint-worthy, :heavy_plus_sign: enhancement
comments:	2
assignees:	
projects:	
milestone:	
number:	1326
--
<!-- Fill in the information below before opening an issue. -->

#### Problem description
<!-- Provide a clear and concise description of the issue. -->
I am running a code which searches for light curves of many objects, currently I am testing with 30, but would like to run this on a much larger sample.  My code would be faster if it didn't have to wrap those search_lightcurve calls in a for loop.  Is it possible to vectorize this API call?

#### Example
<!-- Provide a link or minimal code snippet that demonstrates the issue. -->
```python
import lightkurve as lk

# Build a list of skycoords from target ra and dec
ra_list = [0.5, 23.9, 49.2]
dec_list = [88.2, 1.5, 37.9]
coords_list = [
    SkyCoord(ra, dec, frame='icrs', unit='deg')
    for ra, dec in zip(ra_list, dec_list)
]
# it would be nice if this was the case
search_result = lk.search_lightcurve(coords_list, radius = 1)
```

#### Expected behavior
<!-- Describe the behavior you expected and how it differs from the behavior observed in the example. -->
This would return the light curves for all objects
#### Environment

-  platform (e.g. Linux, OSX, Windows):
-  lightkurve version (e.g. 1.0b6):
-  installation method (e.g. pip, conda, source):

