Project 2:
----------
###### Updated on Mar.16th

To run this project:
~~~ Bash
conda env create -f environment.yaml
source activate cv-project2
Python Project2.py
~~~
You may load the full description from **Project2.html**
The result image is saved under directory ./results 
Since the Project is primarly completed with Jupytor Notebook. For better interactive result, you may use Jupytor Notebook to render this project.
~~~ Bash
jupyter Project2.ipynb
~~~

#### Stereo Matching Result
- With a few tests of gaussian sigma peremeter, the best result is achieved by 0.75, with RMS distance of 12.7824466742

- With a few tests of bilateral peremeter,the best result is achieved by d=10, sigmaColor=3, sigmaSpace=3: 13.1280188273

- After left-right consistency check, the RMS value is 9.6487828026 


Note: The RMS value in left-right consistency check does not include the occluded points.

#### Panorama stitching using homographies Result

- The best homography matrix is:

[[  9.50102773e-01  -1.08936452e-02   3.62685882e+02]

 [  6.79549185e-02   9.87803781e-01   2.40705622e+01]
 
 [ -1.42737010e-15   2.68156690e-16   1.00000000e+00]]
