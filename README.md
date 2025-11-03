# Investigation of Charge and Ion Drag Force Dynamics in Complex Plasma Experiments with Neon and Argon under Microgravity  

---

## Overview  

In complex (dusty) plasmas, micron-sized particles immersed in ionized gas acquire a significant negative charge and experience several competing forces. Among these, the **ion drag force** plays a key role in determining particle dynamics.  

The **PK-4 facility** aboard the *International Space Station (ISS)* enables precise investigations of such systems under **microgravity conditions**. During campaigns **#17 and #18**, we studied the ion drag force acting on **3.4 µm melamine-formaldehyde microparticles** in **argon** and **neon** plasmas across a pressure range of **10–120 Pa**.  

Our work combines **experimental measurements**, **machine learning**, and **ion drag force modeling**, allowing us to evaluate and refine theoretical approaches — from **analytical models** for weak/intermediate ion coupling to **hybrid kinetic models**.  

The results reveal a **pronounced enhancement of the ion drag force** at low pressures (below 30 Pa), especially in **neon**, where ion-neutral collisions and drift velocities strongly influence particle motion. These findings underscore the role of **gas composition** and **ion-neutral interactions** in dust–plasma coupling and highlight the need for **gas-specific model refinements**.  

📄 **Publication:** [https://doi.org/10.1088/1367-2630/ae140b](https://doi.org/10.1088/1367-2630/ae140b)

---

## Particle Reconstruction and Plasma Parameter Optimization  

Particle reconstruction is optimized using a **machine learning–based Self-Organizing Map (SOM)** approach.  

- **Particle positions** are extracted from images using a **trained U-Net** model.  
- **Plasma parameters** are derived from model equations and **optimized via Bayesian optimization**, validated against experimental data.  
- Refer to **Figure 2** in the publication for a visual summary of this workflow.  

All **final data and analysis results** are available within this repository.  

---

## Self-Organizing Map (SOM) for Particle Tracking  

This is a reworked and improved Version of a Self Organizing Map for particle tracking in complex plasmas.
See Reworked SOM.ipynb for an example of use.

Using a trained U-Net particle positions are read out from images. The particle positions of two images are read into the SOM. In this step the particles are numbered by adding a third row to each array ((x-coord, y-coord, number)), whereby no numbers are duplicated. Using the matching feature of the SOM the numbering of the particles of the second image, that match with a particle in the first image, are modified to be the same. The numbering of all unmatched particles remains unchanged. For particle matching in large images (size 2048x2048) another function that splits the image up into 16 smaller blocks and matches particles in there seperately for better performance is used. For particle tracing in image sequences, multiprocessing is used for speed up. The matched coordinates are appended in a specific way so that a tracing algorithm connects all matched particles over the image series. By doing that particle traces are extracted. For a quick analysis a quiver plot feature is implemented, that plots a quiver plot using particles of only 2 images. 
 
### Run  
main.py ```bash
