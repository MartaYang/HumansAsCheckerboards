<div align="center">
  <h1>Humans as Checkerboards: Calibrating Camera Motion Scale for World-Coordinate Human Mesh Recovery  <br> (ICCV 2025)</h1>
</div>

<div align="center">
  <h3><a href=https://martayang.github.io/>Fengyuan Yang</a>, <a href=https://www.comp.nus.edu.sg/~keruigu/>Kerui Gu</a>, <a href=https://www.comp.nus.edu.sg/~hlinhn/> Ha Linh Nguyen</a>, <a href=https://eldentse.github.io/>Tze Ho Elden Tse</a>,<a href=https://www.comp.nus.edu.sg/~ayao/>Angela Yao</a></h3>
</div>

<div align="center">
  <h4> <a href=https://openaccess.thecvf.com/content/ICCV2025/papers/Yang_Humans_as_Checkerboards_Calibrating_Camera_Motion_Scale_for_World-Coordinate_Human_ICCV_2025_paper.pdf>[Paper]</a>, <a href=https://openaccess.thecvf.com/content/ICCV2025/supplemental/Yang_Humans_as_Checkerboards_ICCV_2025_supplemental.zip>[Supp]</a>, <a href=http://arxiv.org/abs/2407.00574>[arXiv]</a></h4>
</div>

## 1. Requirements
* Python 3.9
* PyTorch 1.11.0


## 2. Preparation

**HAC** calibrates **monocular SLAM scale** by leveraging **human priors from HMR**.  
It is **model-agnostic** to both HMR and SLAM, compatibility with various HMR and SLAM approaches as long as you can export the following inputs:

### 2.1 Required Inputs

#### (A) HMR predictions 
Provide a dict (or serialized file) like:
```python
{
    "pred_rotmat": ...,  # Predicted SMPL rotation matrix 
    "pred_shape":  ...,  # Predicted SMPL body shape parameters
    "pred_trans":  ...   # Predicted camera translation
}
```

#### (B) SLAM camera estimates and 3D point cloud
Provide a dict (or serialized file) like:
```python
{
    "traj":   ...,  # camera poses represented as [tx, ty, tz, qx, qy, qz, qw]
    "tstamp": ...,  # timestamps
    "disps":  ...   # disparity/depth proxy
}
```
together with SLAM‘s output 3D point cloud:
```python
  points.ply
```

### 2.2 Reference Configuration

For SOTA comparison/evaluation, we used [TRAM](https://github.com/yufu-wang/tram)'s setup:

* **HMR**: VIMO
* **SLAM**: Masked DroidSLAM

You can export the above inputs and points.ply from the source code and pass them directly to HAC.

## 3. Usage

* Test on EMDB 2
    ```python
    python eval_emdb2.py >logs/emdb2.logs 2>&1 
    ```
    * Corresponding output logs can found at [`logs/emdb2.logs`](logs/emdb2.logs)

* Test on EgoBody
    ```python
    python eval_egobody.py >logs/egobody.logs 2>&1 
    ```
    * Corresponding output logs can found at [`logs/egobody.logs`](logs/egobody.logs)

## Citation

If you find our paper or codes useful, please consider citing our paper:

```bibtex
@InProceedings{Yang_2025_ICCV,
    author    = {Yang, Fengyuan and Gu, Kerui and Nguyen, Ha Linh and Tse, Tze Ho Elden and Yao, Angela},
    title     = {Humans as Checkerboards: Calibrating Camera Motion Scale for World-Coordinate Human Mesh Recovery},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {6069-6079}
}
```

## Acknowledgments

Our codes are based on [WHAM](https://github.com/yohanshin/WHAM), [TRAM](https://github.com/yufu-wang/tram), and [SLAHMR](https://github.com/vye16/slahmr) and we really appreciate it. 
