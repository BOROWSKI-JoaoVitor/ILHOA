![Logo ILHOA](logo_ILHOA.png)

# **ILHOA - A Complete Pipeline for Concrete Crack Microscopy Analysis**

João Vitor B. Borowski<sup>a</sup>, Augusto Schrank<sup>a</sup>, Marilene Vainstein<sup>a</sup>  

<sup>a</sup>Departamento de Biologia Molecular e Biotecnologia, Instituto de Biociências and Programa de Pós-Graduação em Biologia Celular e Molecular, Centro de Biotecnologia, Universidade Federal do Rio Grande do Sul – UFRGS, Porto Alegre, RS 91500-970, Brazil.

---

### **Contents**
- [Description](#description)
- [Recommended Implementation](#recommended-implementation)
- [Example](#example)
- [Description of This GitHub Repository](#description-of-this-github-repository)
- [Additional Information](#additional-information)
- [Citation](#citation)

---

## **Description** 
ILHOA (InstantLy Have Overall Analysis) is a robust and automated pipeline designed to analyse crack healing in self-healing concrete materials. Composed of Fiji and Python codes, it takes microscopy images as input and returns ready-to-publish graphics and statistical analysis of the whole crack extension. It offers precise and scalable tools for data acquisition, processing, and visualization, aiming to enhance research workflows by replacing manual assessments with automated and reproducible methodologies. 

## **Recommended Implementation**
1- Install [Fiji (ImageJ)](https://imagej.net/software/fiji/downloads) and all necessary Macros 

2- Install [Python 3.8.16](https://www.python.org/downloads/release/python-3816/) or [later versions](https://www.python.org/downloads/), and all necessary packages and their dependencies. All python code was run and tested with version 3.8.16

3- To run the Python code, we reccomend utilizing an IDE such as [PyCharm](https://www.jetbrains.com/pycharm/download/?section=windows)

## **Important**
> ⚠️ **Please Note**: Before using ILHOA, ensure that all microscopy image files have the right format, and are correctly named according to the guidelines provided in the [Example](#example) section. Incompatible format or improperly formatted data may cause the pipeline to fail. 
---
4- Open Fiji, drag the macro <i>ILHOA_merge_photos.ijm</i> file into the program window and run the code

5- After creating all merged microscopy images, drag the macro <i>ILHOA_crack_profile.ijm</i> file into the program window and run the code

6- After processing all images, open it with Fiji and manually exclude any remaining artifacts

7- Open the <i>ILHOA_data_extract.py</i> with the IDE of your choice. Run the code

8- Congratulations! You now have all the data ready to be interpreted! Good luck with your research 

## **Example**
Before running any codes, you should have the following directories:
- With microscopy images;
 
- To save the merged microscopy images.
---
The images in the directory must be named according to this format:
#### **Treat-#_Time##**
Where 
- **<i>Treat</i>** is the treatment condition used for that sample;
- **<i>#</i>** is the sample identifier number;
- **<i>Time</i>** is the lapsed time from the beginning of the analysis when the photo was captured (in days);
- **<i>##</i>** is the image identifier number for sequential merge.

Here's an example of a directory with the images ready for analysis:
![Directory Example](directory_example.png)

---
Executing the steps _1-4_ from the [Recommended Implementation](#recommended-implementation), you should get the following results:
![Fiji Run Example](fijirun_example.png)

Executing the steps _5 and 6_, you should get the following results:
![Crack Profile Example](crackprofile_example.png)

After executing the step _7_, you will get the following graphics and results:
![Crack Profile Example](crackprofile_example.png)
![Crack Profile Example](crackprofile_example.png)
![Crack Profile Example](crackprofile_example.png)
![Crack Profile Example](crackprofile_example.png)
![Crack Profile Example](crackprofile_example.png)
![Crack Profile Example](crackprofile_example.png)


## **Description of This GitHub Repository**
An overview of the repository’s structure, with explanations of the key directories and files. This section clarifies where to find scripts, datasets, documentation, and other essential components of the project.

## **Additional Information**
The name Ilhoa was selected prior to creating its acronym. As a pipeline integrating Fiji (named after a real island) and Python (named after a snake), it was inspired by a Brazilian island known for its unique snake population. Ilhoa is located in Rio de Janeiro, Brazil, home to the critically endangered Jararaca-ilhoa, or Golden Lancehead pit viper. This name choice is intended to raise awareness of endangered Brazilian species, spotlighting the importance of conservation efforts.

## **Citation**
J.V.B. Borowski, A. Schrank, M.H. Vainstein (2024) Ilhoa: A Novel Automated Pipeline for Enhanced Crack Analysis. Automation in Construction.
