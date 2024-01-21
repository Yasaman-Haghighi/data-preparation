# SMPLX Motion Combiner

This script is designed for the purpose of combining multiple SMPLX motion parameters with the same shape parameters to construct a longer, seamless motion sequence. It achieves this by aligning the last frame of each individual motion with the first frame of the subsequent motion. Spherical Linear Interpolation (SLERP) is then applied to ensure a smooth transition between these consecutive motions.


