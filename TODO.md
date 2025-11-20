

CHANGE SCHEDULER TO AUC METRIC


- Zero-shot 
    - Evaluate zero-shot VGGT using cosine-similarity on HM3D and Gibson (pipeline done just evaluation)

- Trained 
    - Evaluate VGGT = backbone + CLS Head on HM3D and Gibson (Pairwise) 
        1. Chunked (pipeline ready just evaluating)
        2. Scratch (pipeline ready just evaluating)
    - Evaluate VGGT = backbone + CLS Head on HM3D and Gibson (Multiview)
        1. Chunked (pipeline ready just evaluating)
        2. Scratch

    - Addons:
        - Evaluate VGGT = Backbone + CLS Head + Graph strcturte in multiview to improve the results