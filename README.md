# Implicit Neural Representation INR for 3D Mesh Reconstruction and Compression using SIREN

<img width="1025" height="446" alt="header" src="https://github.com/user-attachments/assets/66d6d3f0-b068-4c2e-a6a0-8a28a6a42fd7" />

This project explores the application of Implicit Neural Representations (INRs), specifically using a Sinusoidal Representation Network (SIREN), for 3D mesh reconstruction and compression. 
The methodology involves generating a Signed Distance Function (SDF) dataset by sampling points both on and in the vicinity of the mesh surface (Stanford Bunny), followed by normalization.
A SIREN model is then trained to implicitly learn this SDF. 
After training, the model's ability to represent the 3D geometry is evaluated by performing inference on a dense grid and reconstructing the mesh using the Marching Cubes algorithm.
The reconstructed mesh is visualized and saved as an OBJ file.
A key finding is the significant data compression achieved: the SIREN model, even in full precision, is considerably smaller than the original explicit mesh, with further compression realized through half-precision storage, 
demonstrating the efficiency of INRs for 3D asset representation.
