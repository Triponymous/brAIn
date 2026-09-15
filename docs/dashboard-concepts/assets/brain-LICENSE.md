# Anatomy asset attribution

`brain-contours.glb` is adapted from Z-Anatomy / BodyParts3D.

- BodyParts3D, © The Database Center for Life Science (DBCLS), CC BY-SA 2.1 Japan: https://lifesciencedb.jp/bp3d/
- Z-Anatomy contributors, CC BY-SA 4.0: https://www.z-anatomy.com/
- Source GLB export by Dr. Murat Altun: https://github.com/DrMuratAltun/beyin-simulatoru
- Source attribution: https://github.com/DrMuratAltun/beyin-simulatoru/blob/main/ATTRIBUTION.md

This derived anatomical asset is licensed under Creative Commons
Attribution-ShareAlike 4.0 International: https://creativecommons.org/licenses/by-sa/4.0/

Changes: selected 89 structures; applied world transforms; centered and scaled;
converted surfaces to contour lines and spatial sample points. No source textures
or branding copied. Reproduction script: `3d/bake-brain.mjs` in the brAIn repository.
Named sample ranges and 12 source-derived anatomical landmark coordinates are
retained in the GLB metadata. Landmark labels are orientation aids, not complete
lobe boundaries or anatomical localization of the SNN's computational layers.

The rendered sample SNN connections are illustrative, not measured human neural tracts.
No clinical validation is claimed. The separate application source retains its own license.
