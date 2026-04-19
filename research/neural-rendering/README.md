# Neural Rendering Research: 3D Gaussian Splatting
> **Project**: BootstrapAI Neural Rendering Reproduction  
> **Target**: Reproduce 3D Gaussian Splatting and approach paper metrics  
> **Started**: 2026-04-19  
> **Status**: Phase 1 - Literature Review & Setup

## 研究目标

复现 **3D Gaussian Splatting for Real-Time Radiance Field Rendering** (Kerbl et al., SIGGRAPH 2023)  
并在此基础上探索迭代改进，形成完整技术报告。

## 论文核心信息

| 属性 | 详情 |
|------|------|
| **标题** | 3D Gaussian Splatting for Real-Time Radiance Field Rendering |
| **作者** | Kerbl, Kopanas, Leimkühler, Drettakis |
| **发表** | ACM TOG (SIGGRAPH 2023) |
| **论文** | https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/ |
| **代码** | https://github.com/graphdeco-inria/gaussian-splatting |

## 核心创新点

1. **3D高斯表示**: 用3D高斯（而非NeRF的隐式神经网络）表示场景
2. **可微光栅化**: 高效的可微渲染管线，实现实时渲染（>30fps@1080p）
3. **各向异性高斯**: 支持视角依赖的外观
4. **自适应密度控制**: 自动的高斯克隆和分裂优化

## 技术指标目标

| 指标 | 论文结果 | 复现目标 |
|------|---------|---------|
| **渲染速度** | >30 fps @ 1080p | >25 fps |
| **PSNR** | ~30 dB (on synthetic) | >28 dB |
| **训练时间** | 几分钟 | <30分钟 |
| **存储成本** | 原始方法的几倍 | 接近原始 |

## 技术路线图

### Phase 1: 基础实现 (当前)
- [x] 文献调研
- [ ] 环境搭建 (CUDA, PyTorch)
- [ ] 3D高斯基类实现
- [ ] 投影变换数学推导
- [ ] 基础光栅化

### Phase 2: 可微渲染
- [ ] 球谐函数 (SH) 表示视角依赖颜色
- [ ] 可微光栅化完整实现
- [ ] 密度控制策略 (克隆/分裂)
- [ ] 训练循环

### Phase 3: 优化与扩展
- [ ] 性能优化 (CUDA kernel)
- [ ] 压缩方法探索
- [ ] 与NeRF对比实验
- [ ] 完整报告生成

## 数学基础

### 3D高斯定义
```
G(x) = exp(-0.5 * (x-μ)^T Σ^(-1) (x-μ))
```
其中：
- μ ∈ R³: 中心位置
- Σ ∈ R^(3×3): 协方差矩阵（决定形状和方向）
- 透明度 α ∈ [0,1]

### 投影到2D
世界空间到相机空间，再投影到图像平面。
协方差矩阵通过投影变换：
```
Σ' = JW Σ W^T J^T
```
其中J是投影变换的仿射近似雅可比矩阵，W是视图变换。

### α-混合渲染
按深度排序的高斯，从前到后混合：
```
C = Σ ci αi Gi(x) Ti
Ti = Π(1 - αj Gj(x))  // 透射率
```

## 实验数据集

1. **Synthetic NeRF** (Blender场景) - 基础测试
2. **Mip-NeRF 360** (真实场景) - 高级测试
3. **Tanks and Temples** (大规模场景) - 扩展测试

## 开发日志

### 2026-04-19 - 项目启动
- 完成文献调研
- 确定复现目标：3D Gaussian Splatting
- 创建研究项目结构

### Next Steps
1. 设置CUDA环境
2. 实现3D高斯数据结构
3. 开始投影变换数学实现

## 参考文献

1. Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering", SIGGRAPH 2023
2. Mildenhall et al., "NeRF: Representing Scenes as Neural Radiance Fields", ECCV 2020
3. Barron et al., "Mip-NeRF 360", CVPR 2022

## 相关资源

- 原始代码: https://github.com/graphdeco-inria/gaussian-splatting
- 复现教程: https://medium.com/@aminmus/reproducing-3d-gaussian-splatting-689d098c7c1d
- 中文综述: https://github.com/lennylxx/awesome-3d-gaussian-splatting

---

**Research Status**: 🟡 Phase 1 - Literature Review Complete  
**Last Updated**: 2026-04-19
