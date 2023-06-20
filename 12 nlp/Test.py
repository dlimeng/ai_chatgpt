import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 创建一个3D绘图 Axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# X, Y, Z 轴数据
X = [1, 2, 3, 4, 5]
Y = [2, 3, 1, 5, 4]
Z = [1, 2, 6, 4, 5]

# 绘制3D折线图
ax.plot(X, Y, Z, 'r-')

# 绘制3D散点图
ax.scatter(X, Y, Z, c='b', marker='o')

# 设置坐标轴标签
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# 设置视角
ax.view_init(30, 45)

# 显示图例
plt.show()