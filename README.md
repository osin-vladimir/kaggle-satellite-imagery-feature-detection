## Description 

In this competition, DSTL provides 1km x 1km satellite images in both 3-band and 16-band formats. Main goal is to detect and classify the types of objects found in these regions. 

### Object types
In a satellite image, you will find lots of different objects like roads, buildings, vehicles, farms, trees, water ways, etc:

1. Buildings - large building, residential, non-residential, fuel storage facility, fortified building
2. Misc. Manmade structures 
3. Road 
4. Track - poor/dirt/cart track, footpath/trail
5. Trees - woodland, hedgerows, groups of trees, standalone trees
6. Crops - contour ploughing/cropland, grain (wheat) crops, row (potatoes, turnips) crops
7. Waterway 
8. Standing water
9. Vehicle Large - large vehicle (e.g. lorry, truck,bus), logistics vehicle
10. Vehicle Small - small vehicle (car, van), motorbike

## Main ideas

* Panchromatic sharpennig 
* Reflectance indices
* Generative Adversarial Networks
* State-of-the-art CNN for Image Segmentation

## [Panchromatic Sharpening](https://www.kaggle.com/resolut/panchromatic-sharpening)  

Pansharpening is a process of merging high-resolution panchromatic and lower resolution multispectral imagery to create a single high-resolution color image. Google Maps and nearly every map creating company use this technique to increase image quality. Pansharpening produces a high-resolution color image from three, four or more low-resolution multispectral satellite bands plus a corresponding high-resolution panchromatic band.

<img src="https://raw.githubusercontent.com/osin-vladimir/kaggle-satellite-imagery-feature-detection/master/images/sharpening.png?token=AHHppqngwGtJtdx0isqddg3I9RvKc3iRks5ZSRBOwA%3D%3D">

##  [Reflectance Index](https://www.kaggle.com/resolut/panchromatic-sharpening)

The reflectance is used to calculate different reflectance indices, which sum up the large amount of information contained in a reflectance spectrum. Some of them are related to plant biomass, photosynthetic size and radiation use efficiency. Other parameters are related to the physiological status, e.g. water content.
 
<img src="https://www.kaggle.io/svf/946335/41cdd3f508e0edbce109f475ecc67d1a/__results___files/__results___7_0.png"> 

## Neural networks


## Credits and Inspiration
1. [Towards Adversarial Retinal Image Synthesis](https://arxiv.org/pdf/1701.08974.pdf) [(code)](https://github.com/costapt/vess2ret)
2. [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004.pdf)
3. [Vegetation indices](http://web.pdx.edu/~nauna/resources/8-2012_lecture1-vegetationindicies.pdf)

## Slack for new Kagglers
* [Open Data Science](http://ods.ai/)
