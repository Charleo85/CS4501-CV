Project 1:
----------
###### Update on Feb.16th

To run this project:
~~~ Bash
Python3 Project1.py
~~~
You may include additional image file in directory ./image

Since the Project is primarly completed with Jupytor Notebook. For better interactive result, you may use Jupytor Notebook to render this project.
~~~ Bash
conda env create -f environment.yaml
source activate cv-project1
jupyter Project1.ipynb
~~~

The default parameter here are optimized for the flower.jpg

To get the best rendering result, you may use different parameter.

_Note: Some edges after thinning are not one pixel in width, the reason is because I intentionally use the edge with < instead of <=, so if two neighbor edges has equal gradient, neither of them will be eliminated_
