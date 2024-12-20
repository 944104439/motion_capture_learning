# OutLine

- SMPL:https://files.is.tue.mpg.de/black/papers/SMPL2015.pdf 

- SPIN: https://arxiv.org/abs/1909.12828 
- HMR:https://arxiv.org/abs/1712.06584 
- HMR2:https://arxiv.org/pdf/2305.20091 
- T2M-GPT: https://mael-zys.github.io/T2M-GPT/ 
- poseBERT: https://arxiv.org/abs/2208.10211
- Motion-X:https://arxiv.org/abs/2307.00818
- Neural Localizer Fields for Continuous 3D Human Pose and Shape Estimation:https://istvansarandi.com/nlf/
- TokenHMR: Advancing Human Mesh Recovery with a Tokenized Pose Representation

# 1. SMPL

A parameteric **model**  that can fitting into the human subject, the parameters can be divided into two types: Shape Parameters and Pose Parameters.

As the deep leaning stuffs should be summarize into two phrases: training and test. So we are gonna discuss the input data and output data first, and then discuss the optimization step. 

### Input and *Output via the model*. 

| Data type                      | Input shape | Output shape | Meaning of Data                                              |
| :----------------------------- | :---------- | :----------- | :----------------------------------------------------------- |
| Shape parameters  (β)          | (B, 10)     | -            | Control the Shape of human subject                           |
| Pose parameters (θ)            | (B, 72)     | -            | Control the rotation of human joints                         |
| Vertices (V)                   | -           | (B, 6890, 3) | Represent the positions of surface vertives of human subject |
| Joint Positions (J)            | -           | (B, 24, 3)   | Represent the positions of human skeletal joints             |
| Projected Keypoints (optional) | (B, 24, 3)  | (B, 24, 2)   | Project the joints into 2d space                             |

![image-20241216113248236](C:\Users\94410\AppData\Roaming\Typora\typora-user-images\image-20241216113248236.png)
$$
\overline{T} \ \text{is the template human subject}
$$

$$
\ B_S(\overline{\beta}) \text{ is a function that controls the body shape by } \overline{\beta} \text{ parameter}
$$

$$
J(\overline{\beta}) \text{ is a function that controls the joints by } \overline{\beta} \text{ parameter.}
$$

$$
T_p(\overline{\beta}, \overline{\theta}) \text{ is a function that blends the shape and pose parameters}
$$

## Optimizaton

## The design of loss functions

The total loss function can be divided into three parts: vertex loss, Joint loss and Regularization loss.

### Vertex Loss

$$
\mathcal{L}_{vertex} = \frac{1}{6890} \sum_{i=1}^{6890} \left\|V_i^{pred} - V_i^{gt}\right\|^2
$$

### Joint Loss

- 3D joint loss

$$
L_{joint}^{3D} = \frac{1}{24} \sum_{i=1}^{24} \left\| J_i^{pred} - J_i^{gt}  \right\|^2
$$

- 2D joint loss

$$
L_{joint}^{2D} = \frac{1}{24} \sum^{24}_{i=1} w_i \left\| J_{i,2d}^{pred} - J_{i,2d}^{gt} \right\|^2,\text{where} \ w_i\  \text{is the visible weight}
$$

### Regularization Loss

$$
L_{reg} = \lambda_{\beta} \left\| \beta \right\|^2 + \lambda_{\theta} \left\| \theta \right\|^2
$$

### Total Loss

$$
L_{total} = L_{vertex} + \alpha L_{joint}^{3D} + \beta L_{joint}^{2D} + L_{reg} \text{, where the} \ \alpha,\beta \ \text{here are the sclars}
$$

## Training and Test phrases (visualization)

### Training phrase

Simple example

```python
import torch
from smplx import SMPL

# 1. 加载 SMPL 模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
smpl = SMPL(model_path='path_to_smpl_model', gender='neutral').to(device)

# 2. 初始化优化器
optimizer = torch.optim.Adam(smpl.parameters(), lr=1e-4)

# 3. 假设我们有训练数据
batch_size = 4
betas_gt = torch.randn(batch_size, 10).to(device)  # 形状参数 (GT)
poses_gt = torch.randn(batch_size, 72).to(device)  # 姿态参数 (GT)
vertices_gt = torch.randn(batch_size, 6890, 3).to(device)  # 顶点 (GT)
joints_gt = torch.randn(batch_size, 24, 3).to(device)  # 关节位置 (GT)

# 4. 训练循环
for epoch in range(100):  # 假设训练 100 个 Epoch
    optimizer.zero_grad()

    # 输入随机初始化的 betas 和 poses
    betas_pred = torch.randn(batch_size, 10, requires_grad=True).to(device)
    poses_pred = torch.randn(batch_size, 72, requires_grad=True).to(device)

    # 前向传播
    output = smpl.forward(betas=betas_pred, body_pose=poses_pred[:, 3:], global_orient=poses_pred[:, :3])

    # 获取顶点和关节位置
    vertices_pred = output.vertices  # (B, 6890, 3)
    joints_pred = output.joints      # (B, 24, 3)

    # 计算顶点损失
    loss_vertices = torch.mean((vertices_pred - vertices_gt) ** 2)

    # 计算关节位置损失
    loss_joints = torch.mean((joints_pred - joints_gt) ** 2)

    # 总损失
    loss = loss_vertices + loss_joints

    # 反向传播和优化
    loss.backward()
    optimizer.step()

```

### Test phrase (Visualization)

Simple example

```python
import open3d as o3d
import torch
from smplx import SMPL

# 1. Load SMPL model and generate vertices
device = 'cuda' if torch.cuda.is_available() else 'cpu'
smpl = SMPL(model_path='path_to_smpl_model', gender='neutral').to(device)

# Example shape (betas) and pose parameters
# Shape parameters
betas = torch.zeros(1, 10).to(device) 
# pose parameters 69 + 3 = 72
body_pose = torch.zeros(1, 69).to(device)  # Body pose (69 excludes global orient)
global_orient = torch.zeros(1, 3).to(device)  # Global orientation

# SMPL forward pass: get vertices
output = smpl.forward(betas=betas, body_pose=body_pose, global_orient=global_orient)
vertices = output.vertices  # (1, 6890, 3)

# 2. Convert SMPL vertices and faces to Open3D mesh
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(vertices[0].detach().cpu().numpy())
mesh.triangles = o3d.utility.Vector3iVector(smpl.faces)  # SMPL provides fixed triangle faces

# 3. Visualize the mesh
mesh.compute_vertex_normals()  # Compute normals for shading
o3d.visualization.draw_geometries([mesh])
```

- **what's the use of Joint Positions** (Answered by chatgpt-4)

Joints position also output by the smpl model, and the represent the 3D locations of body joints (In my opinion, it's the skeletal (key) points in a human body). And the are mainly used for pose representation or skeleton visualization but not directly required for mesh rendering.**if you wanna visualize the skeleton (joint positions), you can overlay the joint positions on top of the mesh**.

## smpl model Pipeline

```python
class smpl(nn.Module):
	self.vertex_joint_selector = VertexJointSelector()
    def forward()
    vertices, joints = lbs(betas, full_pose, ...)
    joints = self.vertex_joint_selector(vertives, joints)
    
    return (vertices, joints)

def vertex_joint_selector(vertices, joints):
    extra_joints = torch.index_select(vertices, 1, self.extra_joints_idxs)
    joints = torch.cat([joints, extra_joints], dim=1)
    
    return joints

def lbs():
    # Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)
    # Get the joints
    # NxJx3 array
    J = vertices2joints(J_regressor, v_shaped)
    # 3. Add pose blend shapes
    # N x J x 3 x 3
    ident = torch.eye(3, dtype=dtype, device=device)
    if pose2rot:
        rot_mats = batch_rodrigues(pose.view(-1, 3)).view(
            [batch_size, -1, 3, 3])

        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        # (N x P) x (P, V * 3) -> N x V x 3
        pose_offsets = torch.matmul(
            pose_feature, posedirs).view(batch_size, -1, 3)
    else:
        pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
        rot_mats = pose.view(batch_size, -1, 3, 3)

        pose_offsets = torch.matmul(pose_feature.view(batch_size, -1),
                                    posedirs).view(batch_size, -1, 3)

    v_posed = pose_offsets + v_shaped
    # 4. Get the global joint location
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                               dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    return verts, J_transformed
```



# 2. SPIN (SMPL Optimization in the Loop)

We break this paper into four parts: problem definition, method framework and experiments and evaluation

## **Problem Definition**

The goal of this paper is to reconstruct an accurate 3d human model from 2d images and videos, including joint positions and body shape. 

## Method Framework

- SMPL model

$$
\text{input } \theta, \beta;\text{output }M \in \mathbb{R}^{N \times 3}
$$

input the pose parameters and shape parameters, and output the vertices of body mesh. 

And the major body joints X are defined as a linear conbination of the mesh vertives
$$
X \in \mathbb{R}^{N \times 3} = WM \text{, where } 	W \in \mathbb{R}^{N \times k}, M \in \mathbb{R}^{k \times 3}
$$

- Regression network

This type of networks is used to train a 2d regression network, which can output shape paremeters and pose parameters. The pipeline can be formulated below.

Input a image -> 2d regression network (need to be trained) -> getting the shape parameters and pose parameters -> output these parameters into smpl model (3D human model) -> getting the Joints prediction -> project the joints prediction into 2d space -> calculating the loss of difference between Joints predecition and joints ground truth. 

The reprojection loss can be formulated below,
$$
L_{2D} = \left\| J_{reg} - J_{gt} \right\|.
$$

- Optimization routine

The iterative fitting routine follows the SMPLift work, which fitting the SMPL model to a set of 2D keypoints using an optimization-based method. Instead of **training a 2D regression network**.

The objective function can be formulated below,
$$
E_J(\beta, \theta, K, J_{est}) + \lambda_{\theta}E_{\theta}(\theta) + \lambda_{\alpha}E_{\alpha}({\theta})+\lambda_{\beta}E_{\beta}(\beta)
$$
where J_est is the detected 2D joings, and K is the camera parameters. 

- SPIN

The SPIN builds on the insight that the previous two paradigms can form a tight collaboration to train a deep regressor for human pose and shape estimation. 

![image-20241216170952927](C:\Users\94410\AppData\Roaming\Typora\typora-user-images\image-20241216170952927.png)

- Loss function 

  parameters level  and mesh level. 

$$
L_{3D} = \left\| \theta_{reg} - \theta_{opt} \right\|, L_M = \left\| M_{reg} - M_{opt} \right\|
$$

## Experiments and Evaluations

- Human3.6M:

training: s1, s5, s6, s7, s8

test: s9, s11 

- MPI-INF-3DHP

training: subjest s1 to s8

test: as follows (maybe it's splited well) 

- LSP

A 2D human pose estimation. Only use the test set for evaluation 

- 3DPW

only for evaluation on its defined test set. 



# 3. HMR:End-to-end Recovery of Human Shape and Pose 

We describe this paper through three parts: motivation, technical approach,  experiment.

## Motivation 

An end-to-end framework for reconstructing a full 3D mesh of a human body from a single RGB image. In contrast to most current methods that compute 2D or 3D joint locations. 

## Technical approach 

![image-20241217135113250](C:\Users\94410\AppData\Roaming\Typora\typora-user-images\image-20241217135113250.png)

The loss can be formulated below.
$$
L = \lambda(L_{reproj} + \mathbb{1}L_{3D}) + L_{adv}
$$
Simple code generated by chatgpt, Note: it is incomplete

```python
# 定义HMR模型
class HMR(nn.Module):
    def __init__(self, smpl_model):
        super(HMR, self).__init__()
        self.smpl = smpl_model
        self.resnet = models.resnet50(pretrained=True)  # 使用预训练ResNet50提取图像特征
        self.fc = nn.Linear(2048, 256)  # 将特征映射到一个较小的空间

    def forward(self, x):
        # 图像特征提取
        features = self.resnet(x)
        features = self.fc(features)
        
        # SMPL参数预测
        body_params = self.smpl(features)  # 这里简化了SMPL参数预测的部分

        # 生成3D网格
        vertices, joints = self.smpl.forward(body_params)
        return vertices, joints

# SMPL模型初始化
smpl_model = SMPL(model_path='path_to_smpl_model', gender='neutral')
hmr_model = HMR(smpl_model)

for epoch in range(100):  # 设定训练epoch
    hmr_model.train()
    running_loss = 0.0
    for inputs, target_vertices, target_joints in dataloader:
        inputs = inputs.to(device)
        target_vertices = target_vertices.to(device)
        target_joints = target_joints.to(device)
        
        # 前向传播
        vertices, joints = hmr_model(inputs)
        
        # 计算损失
        loss = loss_function(vertices, joints, target_vertices, target_joints, target_joints)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
 

def loss_function(vertices, joints, target_vertices, target_joints, target_keypoints):
    # 3D网格的L2损失
    mesh_loss = torch.norm(vertices - target_vertices, p=2)

    # 3D关节位置的L2损失
    joint_loss = torch.norm(joints - target_joints, p=2)

    # 2D关键点的L2损失
    keypoint_loss = torch.norm(target_keypoints - project_to_2d(vertices), p=2)

    # 总损失是各个损失项的加权和
    total_loss = mesh_loss + joint_loss + keypoint_loss
    return total_loss
```

## Exprements 

**2D Datasets**: The in-the-wild images datasets annotated with 2D keypoints: LSP, LSP-extended, MPII, MS COCO. 

**3D Datasets:** Human3.6M, MPI-INF-3DHP. These both datasets are captured in a controlled environments and provide 150k training images with 3D joint annotations. 

**Unpaired data** used to train the adversarial prior: CMU, Human3.6M training set, PosePrior dataset.

## Question

In this past few years there has been rapid progress in single-view 3D pose prediction on images captured in a controlled environment. Although the performance on these benchmarks is starting to saturate, there has not been much progress on 3D human reconstruction from images in-the-wild.

*According to this paragarph, what's the difference between images captured in a controlled environment and images in-the-wild ?*

# 4. Humans in 4D: Reconstructing and Tracking Humans with Transformers

A system model for human mesh recovery 

## Method

![image-20241217164933930](C:\Users\94410\AppData\Roaming\Typora\typora-user-images\image-20241217164933930.png)

tracking model -> human subject -> HMR 2.0 -> human meshes

# 5. T2M-GPT: Generating Human Motion from Textual Descriptions with Discrete Representations

A framework that generation human motions by GPT with a corruption strategy. 

![image-20241220100010960](C:\Users\94410\AppData\Roaming\Typora\typora-user-images\image-20241220100010960.png)

