# Mixed Local Channel Attention for Object Detection

## Introduction
This project introduces a lightweight Mixed Local Channel Attention (MLCA) module that considers both channel-wise and spatial-wise information, combining local and global context to enhance network representation. Based on this module, we propose the MobileNet-Attention-YOLO (MAY) algorithm to compare the performance of various attention modules. On the Pascal VOC and SMID datasets, MLCA demonstrates a better balance between model representation effectiveness, performance, and complexity compared to other attention techniques. When compared to the Squeeze-and-Excitation (SE) attention mechanism on the PASCAL VOC dataset and the Coordinate Attention (CA) method on the SIMD dataset, MLCA achieves an improvement in mAP of 1.0% and 1.5%, respectively.

![Diagram of MLCA:](MLCA.png)

![Flowchart of MLCA:](MLCA-flow.png)

## Paper Link
- [Mixed Local Channel Attention for Object Detection](https://www.sciencedirect.com/science/article/abs/pii/S0952197623006267)

## Chinese Interpretation Link
- [Chinese Interpretation of Mixed Local Channel Attention](Chinese Interpretation Link) [TODO: Will be written and updated if necessary]

## Video Tutorial Link
- [Video Explanation and Innovative Solutions for Mixed Local Channel Attention](https://www.bilibili.com/video/BV1ju4y1c7ww/)

## Innovation Points Summary and Code Implementation (TODO)

## Citation Format
If this project and paper have been helpful to you, please cite the following paper:

@article{WAN2023106442,
title = {Mixed local channel attention for object detection},
journal = {Engineering Applications of Artificial Intelligence},
volume = {123},
pages = {106442},
year = {2023},
issn = {0952-1976},
doi = {https://doi.org/10.1016/j.engappai.2023.106442},
url = {https://www.sciencedirect.com/science/article/pii/S0952197623006267},
author = {Dahang Wan and Rongsheng Lu and Siyuan Shen and Ting Xu and Xianli Lang and Zhijie Ren},
}

For example:

D. Wan, R. Lu, S. Shen, T. Xu, X. Lang, Z. Ren. (2023). Mixed local channel attention for object detection. Engineering Applications of Artificial Intelligence, 123, 106442.

## References
- [Codebase using YOLOv5 framework](https://github.com/ultralytics/yolov5)
- [References for GradCAM visualization and partial modules](https://github.com/z1069614715/objectdetection_script)
- [ECA implementation](https://github.com/BangguWu/ECANet)
- [SqueezeNet repository](https://github.com/DeepScale/SqueezeNet)
- [Video tutorial for GradCAM visualization (No need to modify source code) (YOLOv5, YOLOv7, YOLOv8)](https://www.bilibili.com/video/BV1WP4y1v7gQ/)
- [Explanation video for GradCAM principle](https://www.bilibili.com/video/BV1PD4y1B77q/)

## Conclusion
Thank you for your interest and support for this project. The authors strive to provide the best quality and service, but there is always room for improvement. If you find any issues or have any suggestions, please let us know.
Additionally, this project is currently maintained by an individual and may contain errors. Your feedback and suggestions are welcome.

## Other Open-Source Projects
Other open-source projects will be organized and released gradually. Please check the author's homepage for downloads in the future.
[Homepage](https://github.com/wandahangFY)

## FAQ
1. README.md file addition (completed)
2. Heatmap visualization section addition, yolo-gradcam (completed, adapted from the objectdetection_script open-source project, detailed tutorials in the link, place yolov5_headmap.py in the root directory for normal usage, YOLOv7 and YOLOv8 likewise)
3. Project environment configuration (MLCA module is plug-and-play, the entire project is based on YOLOv5-6.1 version, refer to README-YOLOv5.md file and requirements.txt for configuration)
4. Folder correspondence explanation (consistency with YOLOv5-6.1, no hyperparameter changes) (TODO: Detailed explanation)
5. Innovation points summary and code implementation (TODO)
6. Paper figures (due to journal copyright issues, no PPT source files are provided, apologies):
   - Diagrams, network structure visuals, flowcharts: PPT (personal choice, can also use Visio, Edraw, AI, etc.)
   - Experimental comparisons: Origin (Matlab, Python, R, Excel all applicable)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=wandahangFY/MLCA&type=Date)](https://star-history.com/#wandahangFY/MLCA&Date)